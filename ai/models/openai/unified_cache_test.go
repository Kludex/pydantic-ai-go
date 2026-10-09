package openai_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestUnifiedOpenAICaching(t *testing.T) {
	for _, responses := range []bool{false, true} {
		var body map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body = nil
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if responses {
				_, _ = w.Write([]byte(`{"id":"r","model":"gpt-5.6","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
			} else {
				_, _ = w.Write([]byte(`{"model":"gpt-5.6","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
			}
		}))
		defer server.Close()
		noMessages := false
		for _, test := range []struct {
			name, model   string
			cache         *ai.CacheConfig
			local         bool
			mode          string
			modeResponses string
			continuation  bool
		}{
			{name: "default", model: "gpt-5.6", cache: &ai.CacheConfig{}, mode: "implicit"},
			{name: "snap up", model: "gpt-5.6", cache: &ai.CacheConfig{Retention: ai.CacheRetention5Minutes}, mode: "implicit"},
			{name: "snap down", model: "gpt-6.1-sol", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, mode: "implicit"},
			{name: "stable prefix", model: "gpt-5.6", cache: &ai.CacheConfig{Messages: &noMessages}, mode: "explicit"},
			// Chat Completions carry no server-state gate, so the instruction breakpoint is
			// placed and `mode='explicit'` stays; Responses cannot place the instruction
			// breakpoint on a continuation, so the request would cache nothing with
			// `mode='explicit'` and downgrades to the implicit breakpoint.
			{name: "server state", model: "gpt-5.6", cache: &ai.CacheConfig{Messages: &noMessages}, mode: "explicit", modeResponses: "implicit", continuation: true},
			{name: "older implicit model", model: "gpt-5", cache: &ai.CacheConfig{}},
			{name: "disabled", model: "gpt-5.6", cache: &ai.CacheConfig{Retention: ai.CacheRetentionDisabled}},
			{name: "provider disabled", model: "gpt-5.6", cache: &ai.CacheConfig{}, local: true},
		} {
			t.Run(test.name, func(t *testing.T) {
				settings := ai.ModelSettings{Cache: test.cache}
				if test.local {
					settings.ExtraBody = map[string]any{"openai_cache_instructions": false}
				}
				if test.continuation {
					settings.ExtraBody = map[string]any{"previous_response_id": "r0"}
				}
				var model ai.Model
				if responses {
					model = openai.NewResponsesModel(test.model, openai.WithBaseURL(server.URL))
				} else {
					model = openai.NewModel(test.model, openai.WithBaseURL(server.URL))
				}
				params := ai.ModelRequestParams{Instructions: "stable", Settings: settings}
				if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}, params); err != nil {
					t.Fatal(err)
				}
				wantMode := test.mode
				if responses && test.modeResponses != "" {
					wantMode = test.modeResponses
				}
				if wantMode == "" {
					if body["prompt_cache_options"] != nil {
						t.Fatalf("unsupported cache option: %+v", body)
					}
				} else {
					options := body["prompt_cache_options"].(map[string]any)
					if options["mode"] != wantMode || options["ttl"] != "30m" {
						t.Fatalf("options=%+v", options)
					}
					if responses && test.continuation && body["instructions"] != "stable" {
						t.Fatalf("continuation instructions changed: %+v", body)
					}
				}
				retention, known := ai.ResolveCacheRetention(model, &settings)
				wantKnown := wantMode != "" || test.local
				if known != wantKnown || known && retention != 30*time.Minute {
					t.Fatalf("retention=%s %v", retention, known)
				}
			})
		}
		var model ai.Model
		if responses {
			model = openai.NewResponsesModel("gpt-5.6", openai.WithBaseURL(server.URL))
		} else {
			model = openai.NewModel("gpt-5.6", openai.WithBaseURL(server.URL))
		}
		bad := ai.ModelSettings{Cache: &ai.CacheConfig{Retention: "week"}}
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: bad}); err == nil {
			t.Fatal("invalid cache accepted")
		}
		if _, known := ai.ResolveCacheRetention(model, &bad); known {
			t.Fatal("invalid retention claimed")
		}
		for _, settings := range []ai.ModelSettings{
			{ExtraBody: map[string]any{"openai_prompt_cache_options": openai.PromptCacheOptions{Mode: openai.PromptCacheModeExplicit}}},
			{ExtraBody: map[string]any{"openai_prompt_cache_options": openai.PromptCacheOptions{Mode: "invalid"}}},
		} {
			if _, known := ai.ResolveCacheRetention(model, &settings); known {
				t.Fatal("noncaching or invalid options claimed retention")
			}
		}
	}
}

func TestUnifiedCacheKeepsExplicitWhenBreakpointPlaced(t *testing.T) {
	for _, responses := range []bool{false, true} {
		var body map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body = nil
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if responses {
				_, _ = w.Write([]byte(`{"output":[]}`))
			} else {
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
			}
		}))
		defer server.Close()
		noMessages := false
		settings := ai.ModelSettings{Cache: &ai.CacheConfig{Messages: &noMessages}}
		var model ai.Model
		if responses {
			model = openai.NewResponsesModel("gpt-5.6", openai.WithBaseURL(server.URL))
		} else {
			model = openai.NewModel("gpt-5.6", openai.WithBaseURL(server.URL))
		}
		messages := []ai.ModelMessage{
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "previous"}}},
			ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}},
			ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{ai.CachePoint{}, ai.TextContent{Text: "next"}}},
			}},
		}
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{Settings: settings}); err != nil {
			t.Fatal(err)
		}
		options := body["prompt_cache_options"].(map[string]any)
		if options["mode"] != "explicit" || options["ttl"] != "30m" {
			t.Fatalf("user breakpoint should retain explicit mode: %+v", options)
		}
	}
}
