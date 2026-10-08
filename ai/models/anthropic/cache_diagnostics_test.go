package anthropic_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/anthropic"
)

func TestAnthropicCacheDiagnostics(t *testing.T) {
	settings, err := (anthropic.Settings{CacheDiagnostics: true}).Build()
	if err != nil {
		t.Fatal(err)
	}
	for _, prior := range []string{"", "msg_prior", "gen_other"} {
		for _, streaming := range []bool{false, true} {
			var body map[string]any
			model := newNamedServer(t, "claude-sonnet-4", func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_current\",\"diagnostics\":{\"cache_miss_reason\":null}}}\n\ndata: {\"type\":\"message_stop\"}\n\n")
				} else {
					_, _ = fmt.Fprint(w, `{"id":"msg_current","content":[],"diagnostics":{"cache_miss_reason":null}}`)
				}
			})
			var history []ai.ModelMessage
			if prior != "" {
				history = []ai.ModelMessage{
					ai.ModelResponse{ProviderName: "anthropic", ProviderResponseID: prior},
					ai.ModelResponse{ProviderName: "other", ProviderResponseID: "msg_ignore"},
					ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "continue"}}},
				}
			}
			params := ai.ModelRequestParams{Settings: settings}
			var details map[string]any
			if streaming {
				stream, err := model.StreamRequest(t.Context(), history, params)
				if err != nil {
					t.Fatal(err)
				}
				for event, err := range stream {
					if err != nil {
						t.Fatal(err)
					}
					if finish, ok := event.(ai.FinishEvent); ok {
						details = finish.ProviderDetails
					}
				}
			} else {
				response, err := model.Request(t.Context(), history, params)
				if err != nil {
					t.Fatal(err)
				}
				details = response.ProviderDetails
			}
			diagnostics := body["diagnostics"].(map[string]any)
			var want any
			if prior == "msg_prior" {
				want = prior
			}
			if diagnostics["previous_message_id"] != want || details["cache_diagnostics"] == nil {
				t.Fatalf("request=%#v details=%#v", body, details)
			}
		}
	}
}

func TestAnthropicCacheDiagnosticsSettingsErrors(t *testing.T) {
	for _, settings := range []anthropic.Settings{
		{Common: ai.ModelSettings{ExtraBody: map[string]any{"anthropic_cache_diagnostics": true}}},
		{CacheDiagnostics: true, Common: ai.ModelSettings{ExtraBody: map[string]any{"diagnostics": map[string]any{}}}},
	} {
		if _, err := settings.Build(); err == nil {
			t.Fatal("conflicting diagnostics settings accepted")
		}
	}
	_, err := anthropic.NewModel("claude").Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{"anthropic_cache_diagnostics": "bad"}}})
	if err == nil || !strings.Contains(err.Error(), "must be a boolean") {
		t.Fatalf("invalid diagnostics error=%v", err)
	}
}
