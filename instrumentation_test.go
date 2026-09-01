package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"strconv"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func TestInstrumentedModelRequest(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = tracerProvider.Shutdown(t.Context())
		_ = meterProvider.Shutdown(t.Context())
	})

	cost := 0.25
	called := false
	base := &requestModel{name: "request-model", request: func(
		ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		called = true
		if !trace.SpanFromContext(ctx).SpanContext().IsValid() {
			t.Fatal("request did not receive the span context")
		}
		messages[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "mutated"}
		params.Tools[0].Schema["mutated"] = true
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{
				ai.TextPart{Content: "done"},
				ai.FilePart{Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}},
				ai.SpeechPart{
					Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("spoken"),
					Audio: &ai.BinaryContent{Data: []byte("voice"), MediaType: "audio/pcm"},
				},
				ai.ThinkingPart{Content: "thought"},
				ai.CompactionPart{Content: "summary"},
				ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call", Args: json.RawMessage(`{"city":"Paris"}`)},
				ai.NativeToolCallPart{ToolName: "search", ToolCallID: "native", Args: json.RawMessage("not-json")},
				ai.NativeToolReturnPart{ToolName: "search", ToolCallID: "native", Content: "result"},
				ai.NativeToolReturnPart{ToolName: "invalid", Content: func() {}},
				nil,
			},
			Usage: ai.Usage{
				InputTokens: 10, OutputTokens: 5, CacheReadTokens: 2, CacheWriteTokens: 1,
				Details: map[string]int{"reasoning_tokens": 3}, CostUSD: &cost,
			},
			ModelName: "response-model", ProviderName: "provider",
			ProviderURL: "https://example.com:8443/v1", ProviderResponseID: "response-id",
			FinishReason: ai.FinishReasonStop,
		}, nil
	}}
	model := ai.NewInstrumentedModel(
		base,
		ai.WithInstrumentationTracerProvider(tracerProvider),
		ai.WithInstrumentationMeterProvider(meterProvider),
	)
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "system"},
			ai.SpeechPart{
				Speaker: ai.SpeechSpeakerUser, Transcript: speechPointer("spoken input"),
				Audio: &ai.BinaryContent{Data: []byte("input voice"), MediaType: "audio/pcm"},
			},
			ai.UserPromptPart{Content: "plain user prompt"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "hello"}, ai.ImageURL{URL: "https://example.com/image.png"},
				ai.UploadedFile{FileID: "file-1", ProviderName: "openai", MediaType: "text/csv"},
				ai.BinaryContent{Data: []byte("secret"), MediaType: "audio/wav"}, ai.CachePoint{}, nil,
			}},
			ai.ToolReturnPart{ToolName: "prior", ToolCallID: "prior-id", Content: map[string]any{"ok": true}},
			ai.RetryPromptPart{Content: "retry", ToolName: "prior", ToolCallID: "prior-id"},
			ai.RetryPromptPart{Content: "plain retry"},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"new_tool"}},
			nil,
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "history"}}},
	}
	temperature := 0.5
	topP := 0.9
	seed := 3
	presence := 0.1
	frequency := 0.2
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		Instructions: "instructions",
		Tools: []ai.ToolDefinition{{
			Name: "lookup", Description: "Look up data", Schema: map[string]any{"type": "object"},
		}},
		OutputTool:   &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}},
		OutputSchema: map[string]any{"unsupported": func() {}},
		AllowText:    true,
		Settings: ai.ModelSettings{
			MaxTokens: 100, Temperature: &temperature, TopP: &topP, Seed: &seed,
			PresencePenalty: &presence, FrequencyPenalty: &frequency, StopSequences: []string{"stop"},
		},
	})
	if err != nil || !called || response.Text() != "done\n\nspoken" {
		t.Fatalf("unexpected instrumented response: response=%+v called=%v err=%v", response, called, err)
	}
	if ai.UnwrapModel(model) != base || model.Name() != "request-model" {
		t.Fatal("instrumented wrapper did not preserve model identity")
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Name != "chat request-model" || spans[0].SpanKind != trace.SpanKindClient {
		t.Fatalf("unexpected spans: %+v", spans)
	}
	attributes := instrumentationSpanAttributes(spans[0].Attributes)
	for key, want := range map[string]any{
		"gen_ai.operation.name": "chat", "gen_ai.request.model": "request-model",
		"gen_ai.response.model": "response-model", "gen_ai.provider.name": "provider",
		"gen_ai.system": "provider", "server.address": "example.com", "server.port": int64(8443),
		"gen_ai.response.id": "response-id", "gen_ai.usage.input_tokens": int64(10),
		"gen_ai.usage.output_tokens": int64(5), "operation.cost": 0.25,
		"gen_ai.usage.cache_creation.input_tokens": int64(1),
		"gen_ai.usage.cache_read.input_tokens":     int64(2),
		"gen_ai.usage.details.cache_write_tokens":  int64(1),
		"gen_ai.usage.details.cache_read_tokens":   int64(2),
		"gen_ai.usage.details.reasoning_tokens":    int64(3),
		"gen_ai.request.max_tokens":                int64(100), "gen_ai.request.temperature": 0.5,
	} {
		if got := attributes[key]; got != want {
			t.Fatalf("attribute %q = %#v, want %#v", key, got, want)
		}
	}
	input, _ := attributes["gen_ai.input.messages"].(string)
	output, _ := attributes["gen_ai.output.messages"].(string)
	definitions, _ := attributes["gen_ai.tool.definitions"].(string)
	parameters, _ := attributes["model_request_parameters"].(string)
	for _, fragment := range []string{
		"system", "spoken input", "aW5wdXQgdm9pY2U=", "plain user prompt", "hello", "c2VjcmV0",
		"prior-id", "plain retry", "new_tool", "history",
	} {
		if !strings.Contains(input, fragment) {
			t.Fatalf("input telemetry omitted %q: %s", fragment, input)
		}
	}
	for _, fragment := range []string{
		"done", "spoken", "dm9pY2U=", "thought", "summary", `"city":"Paris"`, "not-json", "result",
	} {
		if !strings.Contains(output, fragment) {
			t.Fatalf("output telemetry omitted %q: %s", fragment, output)
		}
	}
	if !strings.Contains(definitions, "lookup") || !strings.Contains(definitions, "final") ||
		strings.Contains(definitions, "mutated") || strings.Contains(parameters, "mutated") {
		t.Fatalf("request snapshots were not detached: definitions=%s parameters=%s", definitions, parameters)
	}

	var metrics metricdata.ResourceMetrics
	if err := reader.Collect(t.Context(), &metrics); err != nil {
		t.Fatal(err)
	}
	metricNames := map[string]bool{}
	for _, scope := range metrics.ScopeMetrics {
		for _, metric := range scope.Metrics {
			metricNames[metric.Name] = true
		}
	}
	for _, name := range []string{"gen_ai.client.token.usage", "operation.cost"} {
		if !metricNames[name] {
			t.Fatalf("missing metric %q: %+v", name, metricNames)
		}
	}
}

func TestInstrumentedModelCompaction(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	base := &explicitCompactionModel{compact: func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !trace.SpanFromContext(ctx).SpanContext().IsValid() {
			t.Fatal("compaction did not receive the span context")
		}
		return &ai.ModelResponse{
			Parts:        []ai.ResponsePart{ai.CompactionPart{Content: "Summary."}},
			Usage:        ai.Usage{InputTokens: 4, OutputTokens: 1},
			ModelName:    "compact-response-model",
			ProviderName: "provider",
		}, nil
	}}
	model := ai.NewInstrumentedModel(base, ai.WithInstrumentationTracerProvider(provider))
	response, err := ai.CompactModelMessages(t.Context(), model, []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}},
	}, ai.ModelRequestParams{Instructions: "Keep decisions."})
	if err != nil || response.Parts[0].(ai.CompactionPart).Content != "Summary." {
		t.Fatalf("unexpected compacted response=%+v err=%v", response, err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Name != "compact compact-model" || spans[0].SpanKind != trace.SpanKindClient {
		t.Fatalf("unexpected compaction spans: %+v", spans)
	}
	attributes := instrumentationSpanAttributes(spans[0].Attributes)
	for key, want := range map[string]any{
		"gen_ai.operation.name":     "compact",
		"gen_ai.request.model":      "compact-model",
		"gen_ai.response.model":     "compact-response-model",
		"gen_ai.provider.name":      "provider",
		"gen_ai.usage.input_tokens": int64(4),
	} {
		if got := attributes[key]; got != want {
			t.Fatalf("compaction attribute %q = %#v, want %#v", key, got, want)
		}
	}
	if input, _ := attributes["gen_ai.input.messages"].(string); !strings.Contains(input, "old") {
		t.Fatalf("compaction input was not traced: %s", input)
	}
	if output, _ := attributes["gen_ai.output.messages"].(string); !strings.Contains(output, "Summary.") {
		t.Fatalf("compaction output was not traced: %s", output)
	}

	compactor := any(model).(ai.ModelCompactor)
	if _, err := compactor.CompactMessages(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		RequestTimeout: -time.Second,
	}}); err == nil || err.Error() != "ai: request timeout must be non-negative, got -1s" {
		t.Fatalf("unexpected direct compaction validation error: %v", err)
	}
	base.compact = func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	if _, err := compactor.CompactMessages(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		RequestTimeout: time.Millisecond,
	}}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected direct compaction timeout: %v", err)
	}

	base.compact = func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{Content: "nested"}}}, nil
	}
	nested := ai.NewInstrumentedModel(model, ai.WithInstrumentationTracerProvider(provider))
	if _, err := ai.CompactModelMessages(t.Context(), nested, nil, ai.ModelRequestParams{}); err != nil {
		t.Fatalf("nested instrumented compaction failed: %v", err)
	}
}

type traceCompactionCapability struct{}

func (*traceCompactionCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (*traceCompactionCapability) BeforeModelRequest(
	ctx context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	response, err := ai.CompactModelMessages(ctx, request.Model, request.Messages, request.Params)
	if response != nil {
		request.AdditionalUsage.Add(response.Usage)
	}
	return request, err
}

func TestAgentInstrumentationTracesSideCompactionOnce(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	model := &explicitCompactionModel{compact: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.CompactionPart{Content: "Summary."}},
			Usage: ai.Usage{InputTokens: 3, OutputTokens: 1},
		}, nil
	}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider)),
		&traceCompactionCapability{},
	))
	result, err := agent.Run(t.Context(), "new", struct{}{})
	if err != nil || result.Output != "done" || result.Usage().InputTokens != 3 {
		t.Fatalf("unexpected instrumented run result=%+v err=%v", result, err)
	}
	model.compact = func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, nil
	}
	result, err = agent.Run(t.Context(), "again", struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected nil-side-response run result=%+v err=%v", result, err)
	}
	counts := map[string]int{}
	for _, span := range exporter.GetSpans() {
		counts[span.Name]++
	}
	for name, want := range map[string]int{
		"compact compact-model": 2,
		"chat compact-model":    2,
		"invoke_agent agent":    2,
	} {
		if counts[name] != want {
			t.Fatalf("span %q count = %d, want %d; all=%+v", name, counts[name], want, exporter.GetSpans())
		}
	}
}

func TestInstrumentedModelPrivacyControls(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	base := fallbackTextModel("private", "secret output", nil)
	_ = ai.NewInstrumentedModel(
		base,
		ai.WithInstrumentationTracerProvider(nil),
		ai.WithInstrumentationMeterProvider(nil),
	)
	model := ai.NewInstrumentedModel(
		base,
		ai.WithInstrumentationTracerProvider(provider),
		ai.WithInstrumentationContent(false),
		ai.WithInstrumentationBinaryContent(false),
		ai.WithInstrumentationModelRequestParameters(false),
	)
	if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "secret input"}, nil}},
	}}}, ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "visible"}}}); err != nil {
		t.Fatal(err)
	}
	attributes := instrumentationSpanAttributes(exporter.GetSpans()[0].Attributes)
	for _, key := range []string{"gen_ai.system_instructions", "model_request_parameters"} {
		if _, exists := attributes[key]; exists {
			t.Fatalf("private attribute %q was emitted: %+v", key, attributes)
		}
	}
	for _, key := range []string{"gen_ai.input.messages", "gen_ai.output.messages"} {
		messages, _ := attributes[key].(string)
		if strings.Contains(messages, "secret") || !strings.Contains(messages, `"type":"text"`) {
			t.Fatalf("message structure was not redacted for %q: %s", key, messages)
		}
	}
	if definitions, _ := attributes["gen_ai.tool.definitions"].(string); !strings.Contains(definitions, "visible") {
		t.Fatalf("tool definitions should remain visible: %s", definitions)
	}

	exporter.Reset()
	binaryModel := ai.NewInstrumentedModel(
		base, ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationBinaryContent(false),
	)
	if _, err := binaryModel.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SpeechPart{
			Speaker: ai.SpeechSpeakerUser, Transcript: speechPointer("speech text"),
			Audio: &ai.BinaryContent{Data: []byte("speech secret"), MediaType: "audio/pcm"},
		},
		ai.UserPromptPart{Contents: []ai.UserContent{
			ai.BinaryContent{Data: []byte("secret"), MediaType: "image/png"},
		}},
	}}}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	binaryInput, _ := instrumentationSpanAttributes(exporter.GetSpans()[0].Attributes)["gen_ai.input.messages"].(string)
	if !strings.Contains(binaryInput, "image/png") || !strings.Contains(binaryInput, "audio/pcm") ||
		!strings.Contains(binaryInput, "speech text") || strings.Contains(binaryInput, "c2VjcmV0") ||
		strings.Contains(binaryInput, "c3BlZWNoIHNlY3JldA==") {
		t.Fatalf("binary content was not redacted: %s", binaryInput)
	}

	if ai.InstrumentModel(model) != model {
		t.Fatal("already instrumented model was wrapped again")
	}
	if _, ok := ai.InstrumentModel(base).(*ai.InstrumentedModel); !ok {
		t.Fatal("plain model was not instrumented")
	}
}

func TestInstrumentationMessageVersions(t *testing.T) {
	for _, test := range []struct {
		version       int
		wantInput     []string
		unwantedInput []string
		wantOutput    []string
		unwantedOut   []string
	}{
		{
			version: 2,
			wantInput: []string{
				`"role":"user"`, `"type":"image-url"`, `"type":"video-url"`, `"type":"audio-url"`,
				`"type":"document-url"`, `"type":"binary"`,
			},
			unwantedInput: []string{`"role":"tool"`, `"type":"uri"`, `"type":"blob"`},
			unwantedOut:   []string{`"type":"reasoning"`},
		},
		{
			version: 3,
			wantInput: []string{
				`"role":"user"`, `"type":"image-url"`, `"type":"video-url"`, `"type":"audio-url"`,
				`"type":"document-url"`, `"type":"binary"`,
			},
			unwantedInput: []string{`"role":"tool"`, `"type":"uri"`, `"type":"blob"`},
			wantOutput:    []string{`"type":"reasoning"`},
		},
		{
			version: 4,
			wantInput: []string{
				`"role":"user"`, `"type":"uri"`, `"mime_type":"image/png"`, `"modality":"video"`,
				`"mime_type":"audio/mpeg"`, `"mime_type":"application/pdf"`, `"type":"blob"`,
			},
			unwantedInput: []string{`"role":"tool"`, `"type":"image-url"`},
			wantOutput:    []string{`"type":"reasoning"`},
		},
		{
			version: 5,
			wantInput: []string{
				`"role":"user"`, `"type":"uri"`, `"mime_type":"image/png"`, `"modality":"video"`,
				`"mime_type":"audio/mpeg"`, `"mime_type":"application/pdf"`, `"type":"blob"`,
			},
			unwantedInput: []string{`"role":"tool"`, `"type":"image-url"`},
			wantOutput:    []string{`"type":"reasoning"`},
		},
		{
			version: 6,
			wantInput: []string{
				`"role":"tool"`, `"type":"uri"`, `"modality":"video"`, `"modality":"audio"`,
				`"mime_type":"application/pdf"`, `"type":"blob"`,
			},
			unwantedInput: []string{`"type":"image-url"`},
			wantOutput:    []string{`"type":"reasoning"`},
		},
	} {
		t.Run("v"+strconv.Itoa(test.version), func(t *testing.T) {
			exporter := tracetest.NewInMemoryExporter()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
			t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
			model := ai.NewInstrumentedModel(
				requestModel{name: "versions", request: func(
					context.Context, []ai.ModelMessage, ai.ModelRequestParams,
				) (*ai.ModelResponse, error) {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{
						ai.ThinkingPart{Content: "private reasoning"}, ai.TextPart{Content: "done"},
					}}, nil
				}},
				ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationVersion(test.version),
			)
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{
					ai.ImageURL{URL: "https://example.com/image.png"}, ai.ImageURL{URL: "://invalid"},
					ai.VideoURL{URL: "https://example.com/video.mp4"}, ai.VideoURL{URL: "://invalid"},
					ai.AudioURL{URL: "https://example.com/audio.mp3"}, ai.AudioURL{URL: "://invalid"},
					ai.DocumentURL{URL: "https://example.com/report.pdf"}, ai.DocumentURL{URL: "://invalid"},
					ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
				}},
				ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call", Content: "result"},
				ai.RetryPromptPart{ToolName: "lookup", ToolCallID: "retry", Content: "try again"},
			}}}, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			attributes := instrumentationSpanAttributes(exporter.GetSpans()[0].Attributes)
			input, _ := attributes["gen_ai.input.messages"].(string)
			output, _ := attributes["gen_ai.output.messages"].(string)
			for _, fragment := range test.wantInput {
				if !strings.Contains(input, fragment) {
					t.Fatalf("version %d input omitted %q: %s", test.version, fragment, input)
				}
			}
			for _, fragment := range test.unwantedInput {
				if strings.Contains(input, fragment) {
					t.Fatalf("version %d input included %q: %s", test.version, fragment, input)
				}
			}
			for _, fragment := range test.wantOutput {
				if !strings.Contains(output, fragment) {
					t.Fatalf("version %d output omitted %q: %s", test.version, fragment, output)
				}
			}
			for _, fragment := range test.unwantedOut {
				if strings.Contains(output, fragment) {
					t.Fatalf("version %d output included %q: %s", test.version, fragment, output)
				}
			}
		})
	}

	defer func() {
		if recovered := recover(); recovered != "ai: instrumentation version must be between 2 and 6" {
			t.Fatalf("unexpected instrumentation version panic: %v", recovered)
		}
	}()
	ai.WithInstrumentationVersion(1)
}

func TestInstrumentedModelStream(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = tracerProvider.Shutdown(t.Context())
		_ = meterProvider.Shutdown(t.Context())
	})
	cost := 0.1
	base := streamingRequestModel{
		Model: fallbackTextModel("stream-model", "unused", nil),
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				events := []ai.ModelStreamEvent{
					ai.ResponseMetadataEvent{ModelName: "response-model", ProviderName: "provider"},
					ai.TextDeltaEvent{PartID: "text", Delta: "hel", ID: "text-id"},
					ai.TextDeltaEvent{PartID: "text", Delta: "lo"},
					ai.ThinkingDeltaEvent{Delta: "tho", SignatureDelta: "old"},
					ai.ThinkingDeltaEvent{Delta: "ught", SignatureDelta: "signature"},
					ai.CompactionEvent{PartID: "compact", Content: "summary", ID: "compact-id"},
					ai.ToolCallStartEvent{PartID: "call", ToolName: "lookup", ToolCallID: "call-id"},
					ai.ToolCallDeltaEvent{PartID: "call", ArgsDelta: `{"q":"x"}`},
					ai.ToolCallStartEvent{PartID: "native", ToolName: "search", Native: true},
					ai.ToolCallDeltaEvent{PartID: "native", ToolCallID: "native-id", ArgsDelta: `{}`},
					ai.NativeToolReturnEvent{PartID: "return", Part: ai.NativeToolReturnPart{
						ToolName: "search", ToolCallID: "native-id", Content: "found",
					}},
					ai.FinishEvent{
						Parts: []ai.ResponsePart{
							ai.TextPart{Content: "hello"}, ai.ThinkingPart{Content: "thought"},
							ai.CompactionPart{Content: "summary"},
							ai.ToolCallPart{ToolName: "lookup", Args: json.RawMessage(`{"q":"x"}`)},
							ai.NativeToolCallPart{ToolName: "search", Args: json.RawMessage(`{}`)},
							ai.NativeToolReturnPart{ToolName: "search", Content: "found"},
						},
						Usage:     ai.Usage{InputTokens: 4, OutputTokens: 2, CostUSD: &cost},
						ModelName: "response-model", ProviderName: "provider", FinishReason: ai.FinishReasonStop,
					},
				}
				for _, event := range events {
					if !yield(event, nil) {
						return
					}
				}
			}, nil
		},
	}
	model := ai.NewInstrumentedModel(
		base, ai.WithInstrumentationTracerProvider(tracerProvider), ai.WithInstrumentationMeterProvider(meterProvider),
	)
	events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 0 {
		t.Fatal("stream span ended before consumption")
	}
	count := 0
	for _, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		count++
	}
	if count != 12 {
		t.Fatalf("unexpected stream event count: %d", count)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("unexpected stream spans: %+v", spans)
	}
	attributes := instrumentationSpanAttributes(spans[0].Attributes)
	output, _ := attributes["gen_ai.output.messages"].(string)
	for _, fragment := range []string{"hello", "thought", "summary", "lookup", "search", "found"} {
		if !strings.Contains(output, fragment) {
			t.Fatalf("stream telemetry omitted %q: %s", fragment, output)
		}
	}
	if first, _ := attributes["gen_ai.client.operation.time_to_first_chunk"].(float64); first <= 0 {
		t.Fatalf("time to first chunk was not recorded: %#v", attributes)
	}
	var metrics metricdata.ResourceMetrics
	if err := reader.Collect(t.Context(), &metrics); err != nil {
		t.Fatal(err)
	}
	foundFirstChunk := false
	for _, scope := range metrics.ScopeMetrics {
		for _, metric := range scope.Metrics {
			foundFirstChunk = foundFirstChunk || metric.Name == "gen_ai.client.operation.time_to_first_chunk"
		}
	}
	if !foundFirstChunk {
		t.Fatal("time-to-first-chunk metric was not recorded")
	}
}

func TestInstrumentedModelStreamConsumerBreakAndAgentReuse(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		otel.SetTracerProvider(previous)
		_ = provider.Shutdown(t.Context())
	})
	base := streamingRequestModel{
		Model: fallbackTextModel("break-stream", "unused", nil),
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				if !yield(ai.TextDeltaEvent{Delta: "first"}, nil) {
					return
				}
				yield(ai.FinishEvent{}, nil)
			}, nil
		},
	}
	model := ai.NewInstrumentedModel(base, ai.WithInstrumentationTracerProvider(provider))
	events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
		break
	}
	if len(exporter.GetSpans()) != 1 {
		t.Fatalf("consumer break did not finish span: %+v", exporter.GetSpans())
	}
	exporter.Reset()

	outerModel := ai.NewInstrumentedModel(model, ai.WithInstrumentationTracerProvider(provider))
	streamed := ai.NewAgent[struct{}, string](outerModel).RunStream(t.Context(), "go", struct{}{})
	for _, err := range streamed.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	count := 0
	for _, span := range exporter.GetSpans() {
		if span.Name == "chat break-stream" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("agent streaming produced %d request spans", count)
	}
}

func TestInstrumentedModelCalculatedCostAndRequestError(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = provider.Shutdown(t.Context())
		_ = meterProvider.Shutdown(t.Context())
	})
	model := ai.NewInstrumentedModel(requestModel{name: "gpt-5", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, ModelName: "gpt-5", ProviderName: "openai",
			Usage: ai.Usage{InputTokens: 1}, ProviderURL: "://malformed",
		}, nil
	}}, ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationMeterProvider(meterProvider))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	attributes := instrumentationSpanAttributes(exporter.GetSpans()[0].Attributes)
	if cost, _ := attributes["operation.cost"].(float64); cost <= 0 {
		t.Fatalf("calculated cost was not recorded: %+v", attributes)
	}

	requestErr := errors.New("request failed")
	failed := ai.NewInstrumentedModel(requestModel{name: "failed", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, requestErr
	}}, ai.WithInstrumentationTracerProvider(provider))
	if _, err := failed.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, requestErr) {
		t.Fatalf("unexpected request error: %v", err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 2 || spans[1].Status.Code != codes.Error {
		t.Fatalf("request failure was not recorded: %+v", spans)
	}
}

func TestInstrumentedModelStreamFailures(t *testing.T) {
	for name, test := range map[string]struct {
		stream  func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error)
		consume bool
		want    string
	}{
		"open": {
			stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
				return nil, errors.New("open failed")
			},
			want: "open failed",
		},
		"nil": {
			stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
				return nil, nil
			},
			want: "model returned no stream",
		},
		"event": {
			stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
				return func(yield func(ai.ModelStreamEvent, error) bool) {
					yield(nil, errors.New("event failed"))
				}, nil
			},
			consume: true,
			want:    "event failed",
		},
	} {
		t.Run(name, func(t *testing.T) {
			exporter := tracetest.NewInMemoryExporter()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
			t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
			model := ai.NewInstrumentedModel(streamingRequestModel{
				Model: fallbackTextModel("failed-stream", "unused", nil), stream: test.stream,
			}, ai.WithInstrumentationTracerProvider(provider))
			events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if test.consume {
				for _, eventErr := range events {
					err = eventErr
				}
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected stream error: %v", err)
			}
			spans := exporter.GetSpans()
			if len(spans) != 1 || spans[0].Status.Code != codes.Error {
				t.Fatalf("stream error was not recorded: %+v", spans)
			}
		})
	}
}

type unwrappingRequestModel struct {
	ai.Model
	wrapped ai.Model
	cycle   bool
}

func (model *unwrappingRequestModel) UnwrapModel() ai.Model {
	if model.cycle {
		return model
	}
	return model.wrapped
}

func TestInstrumentedModelDoesNotDuplicateAgentRequestSpan(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		otel.SetTracerProvider(previous)
		_ = provider.Shutdown(t.Context())
	})
	innerModel := ai.NewInstrumentedModel(
		fallbackTextModel("wrapped", "done", nil), ai.WithInstrumentationTracerProvider(provider),
	)
	model := ai.NewInstrumentedModel(innerModel, ai.WithInstrumentationTracerProvider(provider))
	if _, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, span := range exporter.GetSpans() {
		if span.Name == "chat wrapped" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("instrumentation produced %d model request spans", count)
	}

	for name, wrapped := range map[string]ai.Model{
		"transparent": ai.WrapModel(innerModel),
		"nil unwrap": &unwrappingRequestModel{
			Model: fallbackTextModel("nil-unwrap", "done", nil),
		},
		"cycle": &unwrappingRequestModel{
			Model: fallbackTextModel("cycle", "done", nil), cycle: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			exporter.Reset()
			if _, err := ai.NewAgent[struct{}, string](wrapped).Run(t.Context(), "go", struct{}{}); err != nil {
				t.Fatal(err)
			}
			if len(exporter.GetSpans()) != 2 {
				t.Fatalf("unexpected run/request span count: %+v", exporter.GetSpans())
			}
		})
	}
}

func instrumentationSpanAttributes(values []attribute.KeyValue) map[string]any {
	attributes := make(map[string]any, len(values))
	for _, value := range values {
		attributes[string(value.Key)] = value.Value.AsInterface()
	}
	return attributes
}
