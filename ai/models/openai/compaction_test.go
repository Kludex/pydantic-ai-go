package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestResponsesCompactionCapabilityConfiguresContextManagement(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id":"response-1","model":"gpt-5","created_at":1,"status":"completed",
			"output":[{"id":"message-1","type":"message","content":[{"type":"output_text","text":"done"}]}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	})
	capability := openai.NewCompaction(openai.WithCompactionTokenThreshold(100_000))
	result, err := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability)).Run(
		t.Context(), "go", struct{}{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" {
		t.Fatalf("unexpected output: %q", result.Output)
	}
	want := []any{map[string]any{"type": "compaction", "compact_threshold": float64(100_000)}}
	if !reflect.DeepEqual(body["context_management"], want) {
		t.Fatalf("unexpected context management: %#v", body["context_management"])
	}
}

func TestResponsesStatelessCompactionReplacesHistoryAndCountsUsage(t *testing.T) {
	var paths []string
	var compactBody, responseBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, request *http.Request) {
		paths = append(paths, request.URL.Path)
		body := &responseBody
		response := `{
			"id":"response-1","model":"gpt-5","created_at":2,"status":"completed",
			"output":[{"id":"message-1","type":"message","content":[{"type":"output_text","text":"done"}]}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`
		if request.URL.Path == "/responses/compact" {
			body = &compactBody
			response = `{
				"id":"compact-response","model":"gpt-5","created_at":1,"status":"completed",
				"output":[{"id":"compaction-1","type":"compaction","encrypted_content":"opaque"}],
				"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}
			}`
		}
		if err := json.NewDecoder(request.Body).Decode(body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(response))
	})
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old question"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "old answer"}}},
	}
	capability := openai.NewCompaction(openai.WithCompactionMessageCountThreshold(2))
	result, err := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability)).Run(
		t.Context(), "new question", struct{}{}, ai.WithMessageHistory(history),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(paths, []string{"/responses/compact", "/responses"}) {
		t.Fatalf("unexpected request paths: %v", paths)
	}
	if compactBody["instructions"] != nil || compactBody["context_management"] != nil {
		t.Fatalf("unexpected compact request fields: %#v", compactBody)
	}
	input := responseBody["input"].([]any)
	if len(input) != 2 || input[0].(map[string]any)["type"] != "compaction" ||
		input[1].(map[string]any)["content"] != "new question" {
		t.Fatalf("unexpected post-compaction input: %#v", input)
	}
	messages := result.Messages()
	if len(messages) != 3 {
		t.Fatalf("compacted history has %d messages: %#v", len(messages), messages)
	}
	compacted := messages[0].(ai.ModelResponse).Parts[0].(ai.CompactionPart)
	if planted, _ := compacted.ProviderDetails[ai.StandingPromptPlantedKey].(bool); !planted {
		t.Fatalf("compaction did not retain standing-prompt provenance: %#v", compacted.ProviderDetails)
	}
	if messages[0].(ai.ModelResponse).RunID == "" || messages[0].(ai.ModelResponse).ConversationID == "" {
		t.Fatalf("compaction response lacks run identity: %#v", messages[0])
	}
	usage := result.Usage()
	if usage.Requests != 2 || usage.InputTokens != 11 || usage.OutputTokens != 3 {
		t.Fatalf("unexpected combined usage: %+v", usage)
	}
}

func TestResponsesCompactionPreservesExplicitContextManagement(t *testing.T) {
	capability := openai.NewCompaction()
	settings, err := capability.ModelSettings(t.Context(), nil, ai.ModelSettings{ExtraBody: map[string]any{
		"context_management": []any{map[string]any{"type": "custom"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if settings.ExtraBody != nil {
		t.Fatalf("compaction replaced explicit settings: %#v", settings.ExtraBody)
	}
}

func TestResponsesCompactionValidation(t *testing.T) {
	tests := []struct {
		name       string
		capability *openai.Compaction
		contains   string
	}{
		{name: "nil", capability: nil, contains: "must not be nil"},
		{
			name:       "negative token threshold",
			capability: openai.NewCompaction(openai.WithCompactionTokenThreshold(-1)),
			contains:   "token threshold must be non-negative",
		},
		{
			name:       "negative message threshold",
			capability: openai.NewCompaction(openai.WithCompactionMessageCountThreshold(-1)),
			contains:   "message count threshold must be non-negative",
		},
		{
			name:       "nil trigger",
			capability: openai.NewCompaction(openai.WithCompactionTrigger(nil)),
			contains:   "trigger must not be nil",
		},
		{
			name: "stateless token threshold",
			capability: openai.NewCompaction(
				openai.WithStatelessCompaction(), openai.WithCompactionTokenThreshold(1),
				openai.WithCompactionMessageCountThreshold(1),
			),
			contains: "token threshold is only valid for stateful compaction",
		},
		{
			name:       "stateless without trigger",
			capability: openai.NewCompaction(openai.WithStatelessCompaction()),
			contains:   "requires a message count threshold or trigger",
		},
		{
			name: "stateful message threshold",
			capability: openai.NewCompaction(
				openai.WithStatefulCompaction(), openai.WithCompactionMessageCountThreshold(1),
			),
			contains: "only valid for stateless compaction",
		},
		{
			name: "stateful trigger",
			capability: openai.NewCompaction(
				openai.WithStatefulCompaction(), openai.WithCompactionTrigger(func([]ai.ModelMessage) bool { return true }),
			),
			contains: "only valid for stateless compaction",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.capability.Setup(&ai.CapabilityRegistry{})
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected validation error: %v", err)
			}
		})
	}
	valid := openai.NewCompaction()
	if err := valid.Setup(&ai.CapabilityRegistry{}); err != nil {
		t.Fatal(err)
	}
	if _, err := valid.BeforeModelRequest(
		t.Context(), nil, ai.ModelRequestContext{},
	); err == nil || !strings.Contains(err.Error(), "requires ResponsesModel") {
		t.Fatalf("unexpected model validation error: %v", err)
	}
	if _, err := valid.BeforeModelRequest(t.Context(), nil, ai.ModelRequestContext{
		Model: ai.WrapModel(openai.NewResponsesModel("gpt-5")),
	}); err != nil {
		t.Fatalf("wrapped Responses model was rejected: %v", err)
	}
	settings, err := valid.ModelSettings(context.Background(), nil, ai.ModelSettings{})
	if err != nil {
		t.Fatal(err)
	}
	want := []any{map[string]any{"type": "compaction"}}
	if !reflect.DeepEqual(settings.ExtraBody["context_management"], want) {
		t.Fatalf("unexpected default context management: %#v", settings.ExtraBody)
	}
}

type compactionRoundTripFunc func(*http.Request) (*http.Response, error)

func (fn compactionRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

type compactionErrorBody struct{}

func (compactionErrorBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (compactionErrorBody) Close() error             { return nil }

func TestCompactMessagesErrors(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}}}
	t.Run("payload", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5")
		_, err := model.CompactMessages(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: "extreme"},
		}})
		if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
			t.Fatalf("unexpected payload error: %v", err)
		}
	})
	t.Run("request URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("://invalid"))
		_, err := model.CompactMessages(t.Context(), messages, ai.ModelRequestParams{})
		if err == nil {
			t.Fatal("invalid compaction URL succeeded")
		}
	})
	t.Run("request", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		model := newResponsesServer(t, func(http.ResponseWriter, *http.Request) {
			t.Fatal("canceled request reached server")
		})
		_, err := model.CompactMessages(ctx, messages, ai.ModelRequestParams{})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("unexpected request error: %v", err)
		}
	})
	t.Run("response read", func(t *testing.T) {
		client := &http.Client{Transport: compactionRoundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: compactionErrorBody{}, Header: http.Header{}}, nil
		})}
		model := openai.NewResponsesModel("gpt-5", openai.WithHTTPClient(client))
		_, err := model.CompactMessages(t.Context(), messages, ai.ModelRequestParams{})
		if err == nil || !strings.Contains(err.Error(), "read compaction response") {
			t.Fatalf("unexpected read error: %v", err)
		}
	})
	tests := []struct {
		name       string
		statusCode int
		body       string
		contains   string
	}{
		{name: "API", statusCode: http.StatusBadRequest, body: `bad`, contains: "status 400"},
		{name: "parse", statusCode: http.StatusOK, body: `{`, contains: "parse response"},
		{
			name: "empty output", statusCode: http.StatusOK,
			body:     `{"id":"compact","model":"gpt-5","status":"completed","output":[]}`,
			contains: "contained no output",
		},
		{
			name: "wrong output", statusCode: http.StatusOK,
			body:     `{"id":"compact","model":"gpt-5","status":"completed","output":[{"id":"m","type":"message","content":[{"type":"output_text","text":"wrong"}]}]}`,
			contains: "last compaction response item has type ai.TextPart",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.statusCode)
				_, _ = io.WriteString(w, test.body)
			})
			_, err := model.CompactMessages(t.Context(), messages, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected compact error: %v", err)
			}
		})
	}
}

func TestStatelessCompactionPropagatesEndpointError(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, "failed")
	})
	capability := openai.NewCompaction(openai.WithCompactionMessageCountThreshold(0))
	_, err := capability.BeforeModelRequest(t.Context(), &ai.RunInfo{}, ai.ModelRequestContext{
		Model: model,
		Messages: []ai.ModelMessage{
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}},
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "new"}}},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "status 500") {
		t.Fatalf("unexpected endpoint error: %v", err)
	}
}

func TestStatelessCompactionTriggerGetsDetachedHistory(t *testing.T) {
	model := openai.NewResponsesModel("gpt-5")
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "original"}}},
	}}}
	capability := openai.NewCompaction(openai.WithCompactionTrigger(func(cloned []ai.ModelMessage) bool {
		request := cloned[0].(ai.ModelRequest)
		request.Parts[0].(ai.UserPromptPart).Contents[0] = ai.TextContent{Text: "changed"}
		return false
	}))
	if err := capability.Setup(&ai.CapabilityRegistry{}); err != nil {
		t.Fatal(err)
	}
	request, err := capability.BeforeModelRequest(t.Context(), nil, ai.ModelRequestContext{
		Model: model, Messages: messages,
	})
	if err != nil {
		t.Fatal(err)
	}
	contents := request.Messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents
	if contents[0].(ai.TextContent).Text != "original" {
		t.Fatalf("trigger mutated request history: %#v", contents)
	}
	settings, err := capability.ModelSettings(t.Context(), nil, ai.ModelSettings{})
	if err != nil || settings.ExtraBody != nil {
		t.Fatalf("stateless compaction contributed stateful settings: %#v err=%v", settings, err)
	}
}
