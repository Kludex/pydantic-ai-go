package ai_test

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func cacheResponse(read, write int) *ai.ModelResponse {
	return &ai.ModelResponse{
		Parts:        []ai.ResponsePart{ai.TextPart{Content: "done"}},
		Usage:        ai.Usage{InputTokens: 20000, CacheReadTokens: read, CacheWriteTokens: write},
		ProviderName: "test", ModelName: "cache-model",
	}
}

func cacheModel(responses ...*ai.ModelResponse) ai.Model {
	return fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		response := responses[0]
		responses = responses[1:]
		return response, nil
	})
}

func cacheExporter(t *testing.T, options ...sdktrace.TracerProviderOption) (*tracetest.InMemoryExporter, *sdktrace.TracerProvider) {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(append(options, sdktrace.WithSyncer(exporter))...)
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })
	return exporter, provider
}

func cacheChatSpans(exporter *tracetest.InMemoryExporter) tracetest.SpanStubs {
	var spans tracetest.SpanStubs
	for _, span := range exporter.GetSpans() {
		if strings.HasPrefix(span.Name, "chat ") {
			spans = append(spans, span)
		}
	}
	return spans
}

func cacheAttributes(span tracetest.SpanStub) map[string]any {
	attributes := map[string]any{}
	for key, value := range instrumentationSpanAttributes(span.Attributes) {
		if strings.HasPrefix(key, "pydantic_ai.cache.") {
			attributes[key] = value
		}
	}
	return attributes
}

func runCacheTurns(t *testing.T, model ai.Model, count int, capabilities ...ai.Capability) (*ai.RunResult[string], tracetest.SpanStubs) {
	t.Helper()
	exporter, provider := cacheExporter(t)
	capabilities = append([]ai.Capability{ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))}, capabilities...)
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capabilities...))
	var result *ai.RunResult[string]
	for range count {
		var options []ai.RunOption
		if result != nil {
			options = append(options, ai.WithConversation(result.Conversation()))
		}
		var err error
		result, err = agent.Run(t.Context(), "turn", struct{}{}, options...)
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, span := range exporter.GetSpans() {
		if !strings.HasPrefix(span.Name, "chat ") && len(cacheAttributes(span)) != 0 {
			t.Fatalf("cache health belongs on request spans only: %+v", span)
		}
	}
	return result, cacheChatSpans(exporter)
}

func TestCacheHealthClassification(t *testing.T) {
	for _, test := range []struct {
		name      string
		retention time.Duration
		read      int
		write     int
		compacts  bool
		reason    string
	}{
		{name: "warm", retention: time.Hour, read: 1000, reason: "unexpected"},
		{name: "missed token boundary", retention: time.Hour, read: 12000, reason: "unexpected"},
		{name: "expired", retention: time.Nanosecond, read: 1000, reason: "ttl_expired"},
		{name: "unknown", read: 1000, reason: "unknown"},
		{name: "unreported", retention: time.Hour, reason: "unreported"},
		{name: "compacted", retention: time.Hour, write: 3000, compacts: true, reason: "compacted"},
	} {
		t.Run(test.name, func(t *testing.T) {
			second := cacheResponse(test.read, test.write)
			if test.compacts {
				second.Parts = append(second.Parts, ai.CompactionPart{Content: "summary"})
			}
			model := ai.NewProfiledModel(cacheModel(cacheResponse(0, 14000), second), ai.ModelProfile{
				DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: test.retention,
			})
			_, spans := runCacheTurns(t, model, 2)
			want := map[string]any{
				"pydantic_ai.cache.hit_ratio":          float64(test.read) / 20000,
				"pydantic_ai.cache.established_tokens": int64(test.read + test.write),
				"pydantic_ai.cache.collapsed":          true, "pydantic_ai.cache.missed_tokens": int64(14000 - test.read),
				"pydantic_ai.cache.collapse_reason": test.reason,
			}
			if test.reason == "unreported" {
				want["pydantic_ai.cache.established_tokens"] = int64(14000)
			}
			if got := cacheAttributes(spans[1]); !reflect.DeepEqual(got, want) {
				t.Fatalf("health = %+v, want %+v", got, want)
			}
			if test.reason == "unexpected" {
				if len(spans[1].Events) != 1 || spans[1].Events[0].Name != "pydantic_ai.cache.collapse" {
					t.Fatalf("missing collapse event: %+v", spans[1].Events)
				}
				wantEvent := map[string]any{"established_tokens": int64(14000), "cache_read_tokens": int64(test.read),
					"missed_tokens": int64(14000 - test.read), "provider_name": "test", "model_name": "cache-model"}
				if got := instrumentationSpanAttributes(spans[1].Events[0].Attributes); !reflect.DeepEqual(got, wantEvent) {
					t.Fatalf("event = %+v", got)
				}
			} else if len(spans[1].Events) != 0 {
				t.Fatalf("explained collapse emitted an event: %+v", spans[1].Events)
			}
		})
	}
}

func TestCacheHealthThresholdsAndRecovery(t *testing.T) {
	responses := []*ai.ModelResponse{
		cacheResponse(0, 0), cacheResponse(0, 1900), cacheResponse(100, 0), cacheResponse(0, 100000),
		cacheResponse(96000, 0), cacheResponse(95000, 0), cacheResponse(90000, 0), cacheResponse(90000, 0),
		cacheResponse(0, 90000), cacheResponse(0, 90000), cacheResponse(90000, 0), cacheResponse(0, 90000),
	}
	model := ai.NewProfiledModel(cacheModel(responses...), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: time.Hour,
	})
	_, spans := runCacheTurns(t, model, len(responses))
	for index, want := range []bool{false, false, false, false, false, false, true, false, true, true, false, true} {
		if got := cacheAttributes(spans[index])["pydantic_ai.cache.collapsed"] == true; got != want {
			t.Fatalf("request %d collapsed = %v, want %v", index, got, want)
		}
	}
	if len(cacheAttributes(spans[0])) != 0 || cacheAttributes(spans[1])["pydantic_ai.cache.hit_ratio"] != 0.0 {
		t.Fatal("no-cache and establishing requests must differ")
	}
	for index, want := range []int{0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1} {
		if len(spans[index].Events) != want {
			t.Fatalf("request %d events = %+v", index, spans[index].Events)
		}
	}
}

func TestCacheHealthNativeUsage(t *testing.T) {
	for _, test := range []struct {
		name    string
		details map[string]int
		native  bool
		summed  bool
	}{
		{name: "native", native: true, summed: true},
		{name: "compaction passes", details: map[string]int{"message_iterations": 1, "compaction_iterations": 1}, summed: true},
		{name: "single pass", native: true, details: map[string]int{"message_iterations": 1}},
		{name: "separate tool prompts", native: true, details: map[string]int{"tool_use_prompt_tokens": 24000}},
	} {
		t.Run(test.name, func(t *testing.T) {
			native := cacheResponse(32800, 600)
			native.Usage.InputTokens = 40000
			native.Usage.Details = test.details
			if test.native {
				native.Parts = append(native.Parts, ai.NativeToolCallPart{ToolName: "web_search"}, ai.NativeToolReturnPart{ToolName: "web_search"})
			}
			_, spans := runCacheTurns(t, cacheModel(cacheResponse(0, 8000), native, cacheResponse(8400, 0)), 3)
			want := int64(33400)
			if test.summed {
				want = 8000
			}
			if got := cacheAttributes(spans[1])["pydantic_ai.cache.established_tokens"]; got != want {
				t.Fatalf("summed prefix = %v, want %v", got, want)
			}
			if collapsed := cacheAttributes(spans[2])["pydantic_ai.cache.collapsed"] == true; collapsed == test.summed {
				t.Fatalf("next request health = %+v", cacheAttributes(spans[2]))
			}
		})
	}
}

func TestCacheHealthResponseIdentityAndFallback(t *testing.T) {
	responses := []*ai.ModelResponse{cacheResponse(0, 14000), cacheResponse(0, 15000), cacheResponse(0, 16000), cacheResponse(1000, 0)}
	responses[1].ProviderURL = "https://other.example"
	responses[2].ModelName = "other-model"
	_, spans := runCacheTurns(t, ai.NewFallbackModel(cacheModel(responses...)), len(responses))
	for _, span := range spans[1:3] {
		if cacheAttributes(span)["pydantic_ai.cache.collapsed"] != nil {
			t.Fatalf("switch is not a collapse: %+v", cacheAttributes(span))
		}
	}
	if cacheAttributes(spans[3])["pydantic_ai.cache.collapse_reason"] != "unknown" || len(spans[3].Events) != 0 {
		t.Fatalf("fallback must not infer a serving child's retention: %+v", cacheAttributes(spans[3]))
	}
}
