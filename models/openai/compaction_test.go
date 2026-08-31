package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
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
	if err := (*openai.Compaction)(nil).Setup(&ai.CapabilityRegistry{}); err == nil ||
		!strings.Contains(err.Error(), "must not be nil") {
		t.Fatalf("unexpected nil validation error: %v", err)
	}
	capability := openai.NewCompaction(openai.WithCompactionTokenThreshold(-1))
	if err := capability.Setup(&ai.CapabilityRegistry{}); err == nil ||
		!strings.Contains(err.Error(), "must be non-negative") {
		t.Fatalf("unexpected threshold validation error: %v", err)
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
	settings, err := valid.ModelSettings(context.Background(), nil, ai.ModelSettings{})
	if err != nil {
		t.Fatal(err)
	}
	want := []any{map[string]any{"type": "compaction"}}
	if !reflect.DeepEqual(settings.ExtraBody["context_management"], want) {
		t.Fatalf("unexpected default context management: %#v", settings.ExtraBody)
	}
}
