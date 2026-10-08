package ai_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func streamingCacheModel(model ai.Model) ai.Model {
	return streamingRequestModel{Model: model, stream: func(
		ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (iter.Seq2[ai.ModelStreamEvent, error], error) {
		response, err := model.Request(ctx, messages, params)
		if err != nil {
			return nil, err
		}
		return func(yield func(ai.ModelStreamEvent, error) bool) {
			yield(ai.FinishEvent{Parts: response.Parts, Usage: response.Usage, ModelName: response.ModelName,
				ProviderName: response.ProviderName, ProviderURL: response.ProviderURL,
				ProviderResponseID: response.ProviderResponseID, State: response.State}, nil)
		}, nil
	}}
}

func TestCacheHealthContinuations(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, mode := range []string{"capability", "decorator", "direct", "default"} {
			t.Run(mode+map[bool]string{false: "/request", true: "/stream"}[streaming], func(t *testing.T) {
				exporter, provider := cacheExporter(t)
				suspended := cacheResponse(14000, 500)
				suspended.State = ai.ModelResponseStateSuspended
				model := cacheModel(cacheResponse(0, 14000), suspended, cacheResponse(14500, 200), cacheResponse(14700, 0))
				if streaming {
					model = streamingCacheModel(model)
				}
				var capabilities []ai.Capability
				switch mode {
				case "capability":
					capabilities = []ai.Capability{ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))}
				case "decorator", "direct":
					model = ai.NewInstrumentedModel(model, ai.WithInstrumentationTracerProvider(provider))
				case "default":
					previous := otel.GetTracerProvider()
					otel.SetTracerProvider(provider)
					t.Cleanup(func() { otel.SetTracerProvider(previous) })
				}
				agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capabilities...))
				var history []ai.ModelMessage
				for range 3 {
					switch {
					case mode == "direct":
						if history == nil {
							history = []ai.ModelMessage{ai.ModelRequest{ConversationID: t.Name()}}
						}
						var response *ai.ModelResponse
						var err error
						if streaming {
							stream := ai.StreamModel(t.Context(), model, history, ai.ModelRequestParams{})
							for _, err := range stream.Events() {
								if err != nil {
									t.Fatal(err)
								}
							}
							response = stream.Response()
						} else {
							response, err = ai.RequestModel(t.Context(), model, history, ai.ModelRequestParams{})
						}
						if err != nil {
							t.Fatal(err)
						}
						history = append(history, *response)
					case streaming:
						stream := agent.RunStream(t.Context(), "turn", struct{}{}, ai.WithMessageHistory(history))
						for _, err := range stream.Events() {
							if err != nil {
								t.Fatal(err)
							}
						}
						history = stream.Result().Messages()
					default:
						result, err := agent.Run(t.Context(), "turn", struct{}{}, ai.WithMessageHistory(history))
						if err != nil {
							t.Fatal(err)
						}
						history = result.Messages()
					}
				}
				spans := cacheChatSpans(exporter)
				if len(spans) != 3 {
					t.Fatalf("continuations must share one logical span: %+v", spans)
				}
				for index, want := range []int64{14000, 14700, 14700} {
					attributes := cacheAttributes(spans[index])
					if attributes["pydantic_ai.cache.established_tokens"] != want || attributes["pydantic_ai.cache.collapsed"] != nil {
						t.Fatalf("request %d: %+v", index, attributes)
					}
				}
				if ratio := cacheAttributes(spans[1])["pydantic_ai.cache.hit_ratio"]; ratio != 0.725 {
					t.Fatalf("merged usage must not be treated as a prefix: %v", ratio)
				}
				if streaming && mode != "default" && instrumentationSpanAttributes(spans[1].Attributes)["gen_ai.client.operation.time_to_first_chunk"] == nil {
					t.Fatal("logical continuation lost streaming timing")
				}
			})
		}
	}
}

type rejectCacheResponse struct{ calls int }

func (*rejectCacheResponse) Setup(*ai.CapabilityRegistry) error { return nil }

func (hook *rejectCacheResponse) AfterModelRequest(
	_ context.Context, _ *ai.RunInfo, _ ai.ModelRequestContext, response *ai.ModelResponse,
) (*ai.ModelResponse, error) {
	hook.calls++
	if hook.calls == 1 {
		return nil, ai.Retryf("try again")
	}
	return response, nil
}

func TestCacheHealthCountsRejectedResponses(t *testing.T) {
	model := ai.NewProfiledModel(cacheModel(cacheResponse(0, 14000), cacheResponse(1000, 0)), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: time.Hour,
	})
	_, spans := runCacheTurns(t, model, 1, &rejectCacheResponse{})
	if len(spans) != 2 || cacheAttributes(spans[0])["pydantic_ai.cache.established_tokens"] != int64(14000) ||
		cacheAttributes(spans[1])["pydantic_ai.cache.collapse_reason"] != "unexpected" {
		t.Fatalf("served response was lost when its hook rejected it: %+v", spans)
	}
}

type dropFirstCacheRequest struct{ chats atomic.Int64 }

func (sampler *dropFirstCacheRequest) ShouldSample(params sdktrace.SamplingParameters) sdktrace.SamplingResult {
	decision := sdktrace.RecordAndSample
	if strings.HasPrefix(params.Name, "chat ") && sampler.chats.Add(1) == 1 {
		decision = sdktrace.Drop
	}
	return sdktrace.SamplingResult{Decision: decision}
}

func (*dropFirstCacheRequest) Description() string { return "drop first cache request" }

func TestCacheHealthSampledOutRequestEstablishesPrefix(t *testing.T) {
	exporter, provider := cacheExporter(t, sdktrace.WithSampler(&dropFirstCacheRequest{}))
	model := ai.NewProfiledModel(cacheModel(cacheResponse(0, 14000), cacheResponse(1000, 0)), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: time.Hour,
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))))
	first, err := agent.Run(t.Context(), "first", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(t.Context(), "second", struct{}{}, ai.WithMessageHistory(first.Messages())); err != nil {
		t.Fatal(err)
	}
	spans := cacheChatSpans(exporter)
	if len(spans) != 1 || cacheAttributes(spans[0])["pydantic_ai.cache.collapse_reason"] != "unexpected" {
		t.Fatalf("sampling lost cache state: %+v", spans)
	}
}

func TestCacheHealthDirectCallsWithoutConversationStayPrivate(t *testing.T) {
	exporter, provider := cacheExporter(t)
	first := cacheResponse(0, 14000)
	first.ProviderName, first.ModelName = "", ""
	first.Usage.InputTokens = 0
	model := ai.NewInstrumentedModel(cacheModel(first, cacheResponse(1000, 0)), ai.WithInstrumentationTracerProvider(provider))
	for range 2 {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
	}
	spans := cacheChatSpans(exporter)
	if cacheAttributes(spans[0])["pydantic_ai.cache.hit_ratio"] != 0.0 ||
		cacheAttributes(spans[1])["pydantic_ai.cache.collapsed"] != nil {
		t.Fatalf("unrelated direct requests shared marks: %+v", spans)
	}
}

func TestCacheHealthSuspendedResumeAndFailure(t *testing.T) {
	exporter, provider := cacheExporter(t)
	model := ai.NewInstrumentedModel(cacheModel(cacheResponse(14000, 0), cacheResponse(14000, 0)), ai.WithInstrumentationTracerProvider(provider))
	seed := cacheResponse(14000, 0)
	seed.State, seed.ConversationID = ai.ModelResponseStateSuspended, t.Name()
	seed.Usage.InputTokens = 20000
	response, err := ai.RequestModel(t.Context(), model, []ai.ModelMessage{*seed}, ai.ModelRequestParams{})
	if err != nil || response.Usage.CacheReadTokens != 28000 {
		t.Fatalf("resume = %+v, %v", response, err)
	}
	if _, err := ai.RequestModel(t.Context(), model, []ai.ModelMessage{*seed}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if got := cacheAttributes(cacheChatSpans(exporter)[0])["pydantic_ai.cache.established_tokens"]; got != int64(14000) {
		t.Fatalf("resume established merged prefix: %v", got)
	}
	failure := errors.New("provider failed")
	failed := ai.NewInstrumentedModel(&immediateContinuationModel{requestErr: failure}, ai.WithInstrumentationTracerProvider(provider))
	if _, err := ai.RequestModel(t.Context(), failed, nil, ai.ModelRequestParams{}); !errors.Is(err, failure) {
		t.Fatalf("failure = %v", err)
	}
}
