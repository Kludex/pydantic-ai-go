package anthropic_test

import (
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func TestCompactionCapabilityConfiguresContextManagement(t *testing.T) {
	var body map[string]any
	var beta string
	model := newServer(t, func(w http.ResponseWriter, request *http.Request) {
		beta = request.Header.Get("anthropic-beta")
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id":"message-1","model":"claude-sonnet-4-5","type":"message","role":"assistant",
			"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	})
	capability := anthropic.NewCompaction(
		anthropic.WithCompactionTokenThreshold(100_000),
		anthropic.WithCompactionInstructions("Keep decisions."),
		anthropic.WithPauseAfterCompaction(),
	)
	result, err := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability)).Run(
		t.Context(), "go", struct{}{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" {
		t.Fatalf("unexpected output: %q", result.Output)
	}
	if beta != "compact-2026-01-12" {
		t.Fatalf("unexpected beta header: %q", beta)
	}
	contextManagement, ok := body["context_management"].(map[string]any)
	if !ok {
		t.Fatalf("missing context management: %#v", body)
	}
	want := []any{map[string]any{
		"type":                   "compact_20260112",
		"trigger":                map[string]any{"type": "input_tokens", "value": float64(100_000)},
		"instructions":           "Keep decisions.",
		"pause_after_compaction": true,
	}}
	if !reflect.DeepEqual(contextManagement["edits"], want) {
		t.Fatalf("unexpected compaction edits: %#v", contextManagement["edits"])
	}
}

func TestCompactionCapabilityAppendsExistingContextManagement(t *testing.T) {
	existing := map[string]any{
		"strategy": "custom",
		"edits":    []any{map[string]any{"type": "clear_tool_uses_20250919"}},
	}
	current := ai.ModelSettings{ExtraBody: map[string]any{"context_management": existing}}
	settings, err := anthropic.NewCompaction().ModelSettings(t.Context(), nil, current)
	if err != nil {
		t.Fatal(err)
	}
	contextManagement := settings.ExtraBody["context_management"].(map[string]any)
	if contextManagement["strategy"] != "custom" || len(contextManagement["edits"].([]any)) != 2 {
		t.Fatalf("existing context management was not preserved: %#v", contextManagement)
	}
	contextManagement["strategy"] = "changed"
	contextManagement["edits"].([]any)[0].(map[string]any)["type"] = "changed"
	if existing["strategy"] != "custom" || existing["edits"].([]any)[0].(map[string]any)["type"] != "clear_tool_uses_20250919" {
		t.Fatalf("returned settings alias input: %#v", existing)
	}
}

func TestCompactionCapabilityValidation(t *testing.T) {
	if err := (*anthropic.Compaction)(nil).Setup(&ai.CapabilityRegistry{}); err == nil ||
		!strings.Contains(err.Error(), "must not be nil") {
		t.Fatalf("unexpected nil validation error: %v", err)
	}
	capability := anthropic.NewCompaction(anthropic.WithCompactionTokenThreshold(49_999))
	if err := capability.Setup(&ai.CapabilityRegistry{}); err == nil ||
		!strings.Contains(err.Error(), "must be at least 50000") {
		t.Fatalf("unexpected threshold validation error: %v", err)
	}
	valid := anthropic.NewCompaction()
	if err := valid.Setup(&ai.CapabilityRegistry{}); err != nil {
		t.Fatal(err)
	}
	if _, err := valid.BeforeModelRequest(
		t.Context(), nil, ai.ModelRequestContext{},
	); err == nil || !strings.Contains(err.Error(), "requires Model") {
		t.Fatalf("unexpected model validation error: %v", err)
	}
}

func TestNonCompactionContextManagementDoesNotEnableBeta(t *testing.T) {
	tests := []struct {
		name  string
		edits any
	}{
		{name: "non-array edits", edits: "invalid"},
		{name: "other edits", edits: []any{"invalid", map[string]any{"type": "other"}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var beta string
			model := newServer(t, func(w http.ResponseWriter, request *http.Request) {
				beta = request.Header.Get("anthropic-beta")
				_, _ = w.Write([]byte(`{
					"id":"message-1","model":"claude-sonnet-4-5","type":"message","role":"assistant",
					"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn",
					"usage":{"input_tokens":1,"output_tokens":1}
				}`))
			})
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Content: "go"},
			}}}, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{
				"context_management": map[string]any{"edits": test.edits},
			}}})
			if err != nil {
				t.Fatal(err)
			}
			if beta != "" {
				t.Fatalf("unexpected compaction beta: %q", beta)
			}
		})
	}
}

func TestCompactionCapabilityRejectsMalformedExistingSettings(t *testing.T) {
	capability := anthropic.NewCompaction()
	tests := []struct {
		name  string
		value any
		want  string
	}{
		{name: "context management", value: "invalid", want: "context_management must be an object"},
		{name: "edits", value: map[string]any{"edits": "invalid"}, want: "context_management.edits must be an array"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := capability.ModelSettings(t.Context(), nil, ai.ModelSettings{ExtraBody: map[string]any{
				"context_management": test.value,
			}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
