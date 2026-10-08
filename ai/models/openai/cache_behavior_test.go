package openai_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestInstructionCacheBoundaries(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, test := range []struct {
			name       string
			parts      []ai.InstructionPart
			standing   []ai.RequestPart
			extra      map[string]any
			model      string
			compaction bool
			want       bool
		}{
			{name: "static", want: true},
			{name: "dynamic suffix", parts: []ai.InstructionPart{{Content: "stable"}, {Content: "volatile", Dynamic: true}}, want: true},
			{name: "dynamic first", parts: []ai.InstructionPart{{Content: "volatile", Dynamic: true}, {Content: "stable"}}},
			{name: "standing only", standing: []ai.RequestPart{ai.SystemPromptPart{Content: "standing"}}, parts: []ai.InstructionPart{{Content: "volatile", Dynamic: true}}, want: true},
			{name: "dynamic system", standing: []ai.RequestPart{ai.SystemPromptPart{Content: "changing", DynamicRef: "dynamic"}}},
			{name: "legacy model", model: "gpt-4o"},
			{name: "previous response", extra: map[string]any{"previous_response_id": "auto"}, want: !responses},
			{name: "conversation", extra: map[string]any{"conversation": "conversation"}, want: !responses},
			{name: "compaction", compaction: true, want: !responses},
		} {
			t.Run(test.name+map[bool]string{false: " chat", true: " responses"}[responses], func(t *testing.T) {
				var body map[string]any
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
					if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if responses {
						_, _ = w.Write([]byte(`{"output":[]}`))
					} else {
						_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
					}
				}))
				defer server.Close()
				name := test.model
				if name == "" {
					name = "gpt-5.6"
				}
				options := []openai.Option{openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client())}
				var model ai.Model = openai.NewModel(name, options...)
				if responses {
					model = openai.NewResponsesModel(name, options...)
				}
				settings, err := (openai.Settings{Common: ai.ModelSettings{ExtraBody: test.extra}, CacheInstructions: true}).Build()
				if err != nil {
					t.Fatal(err)
				}
				messages := []ai.ModelMessage{ai.ModelRequest{Parts: test.standing}}
				if test.compaction {
					messages = append(messages, ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{Content: "summary"}}})
				}
				messages = append(messages, ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "question"}}})
				_, err = model.Request(t.Context(), messages, ai.ModelRequestParams{Instructions: "stable", InstructionParts: test.parts, Settings: settings})
				if err != nil {
					t.Fatal(err)
				}
				key := "messages"
				if responses {
					key = "input"
				}
				markers := 0
				for _, value := range body[key].([]any) {
					item := value.(map[string]any)
					content, _ := item["content"].([]any)
					for _, value := range content {
						part := value.(map[string]any)
						if part["prompt_cache_breakpoint"] != nil {
							markers++
							if part["text"] == "volatile" {
								t.Fatal("dynamic instructions entered the cached prefix")
							}
						}
					}
				}
				if (markers == 1) != test.want {
					t.Fatalf("cache boundaries=%d want=%t body=%#v", markers, test.want, body)
				}
				if responses && test.want && body["instructions"] != nil {
					t.Fatalf("cached instructions sent twice: %#v", body)
				}
			})
		}
	}
}

func TestResponsesPromptCacheDiagnostics(t *testing.T) {
	for _, test := range []struct {
		name     string
		history  []ai.ModelMessage
		disabled bool
		provider string
		want     string
	}{
		{name: "initial"},
		{name: "prior", history: []ai.ModelMessage{ai.ModelResponse{ProviderName: "openai", ProviderResponseID: "resp_prior"}}, want: "resp_prior"},
		{name: "other provider", history: []ai.ModelMessage{ai.ModelResponse{ProviderName: "other", ProviderResponseID: "resp_other"}}},
		{name: "invalid ID", history: []ai.ModelMessage{ai.ModelResponse{ProviderName: "openai", ProviderResponseID: "gen_prior"}}},
		{name: "compaction", history: []ai.ModelMessage{ai.ModelResponse{ProviderName: "openai", ProviderResponseID: "resp_prior", ProviderDetails: map[string]any{"compaction": true}}}},
		{name: "disabled", disabled: true, history: []ai.ModelMessage{ai.ModelResponse{ProviderName: "openai", ProviderResponseID: "resp_prior"}}},
		{name: "compatible", provider: "openrouter", history: []ai.ModelMessage{ai.ModelResponse{ProviderName: "openrouter", ProviderResponseID: "resp_prior"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"output":[],"prompt_cache_diagnostics":{"type":"cache_hit"}}`))
			}))
			defer server.Close()
			provider := test.provider
			if provider == "" {
				provider = "openai"
			}
			model := openai.NewResponsesModel("gpt-5.6", openai.WithProvider(openai.ProviderConfig{Name: provider, BaseURL: server.URL, HTTPClient: server.Client()}))
			settings := ai.ModelSettings{}
			if test.disabled {
				disabled := false
				var err error
				settings, err = (openai.Settings{PromptCacheDiagnostics: &disabled}).Build()
				if err != nil {
					t.Fatal(err)
				}
			}
			history := append([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "prior"}}}}, test.history...)
			response, err := model.Request(t.Context(), history, ai.ModelRequestParams{Settings: settings})
			if err != nil {
				t.Fatal(err)
			}
			options, _ := body["prompt_cache_options"].(map[string]any)
			comparison, _ := options["comparison_response_id"].(string)
			if comparison != test.want || response.ProviderDetails["prompt_cache_diagnostics"].(map[string]any)["type"] != "cache_hit" {
				t.Fatalf("comparison=%q body=%#v response=%+v", comparison, body, response)
			}
		})
	}
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.created","response":{"id":"resp","status":"in_progress","prompt_cache_diagnostics":{"type":"unavailable"}}}`,
		`{"type":"response.completed","response":{"id":"resp","status":"completed","service_tier":"flex","prompt_cache_diagnostics":{"type":"cache_hit"},"output":[]}}`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	finish := events[len(events)-1].(ai.FinishEvent)
	if finish.ProviderDetails["prompt_cache_diagnostics"].(map[string]any)["type"] != "cache_hit" || finish.ProviderDetails["service_tier"] != "flex" {
		t.Fatalf("terminal diagnostics=%#v", finish.ProviderDetails)
	}
}

func TestPromptCacheBooleanSettingsValidation(t *testing.T) {
	for _, field := range []string{"openai_cache_instructions", "openai_prompt_cache_diagnostics"} {
		if _, err := (openai.Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{field: true}}}).Build(); err == nil {
			t.Fatalf("reserved setting %s accepted", field)
		}
		_, err := openai.NewModel("gpt-5.6").Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{field: "bad"}}})
		if err == nil || !strings.Contains(err.Error(), "must be a boolean") {
			t.Fatalf("invalid setting %s error=%v", field, err)
		}
	}
}

func TestResponsesLeadingCachePointSkipsAssistantOutput(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"output":[]}`))
	}))
	defer server.Close()
	model := openai.NewResponsesModel(
		"gpt-5.6", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()),
	)
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "previous"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "assistant"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.CachePoint{}, ai.TextContent{Text: "next"},
		}}}},
	}
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	input := body["input"].([]any)
	previous := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if previous["prompt_cache_breakpoint"] == nil {
		t.Fatalf("cache point attached to assistant output: %#v", input)
	}
}

func TestLeadingCachePointsAttachToPriorContent(t *testing.T) {
	for _, responses := range []bool{false, true} {
		for _, prior := range []ai.ModelMessage{
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "previous"}}},
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "previous"}}}}},
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "previous"}, ai.ImageURL{URL: "https://example.com/image.png"},
			}}}},
			ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "work", ToolCallID: "call", Content: "previous"}}},
		} {
			for _, onlyPoint := range []bool{false, true} {
				var body map[string]any
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if responses {
						_, _ = w.Write([]byte(`{"output":[]}`))
					} else {
						_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
					}
				}))
				options := []openai.Option{openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client())}
				var model ai.Model = openai.NewModel("gpt-5.6", options...)
				if responses {
					model = openai.NewResponsesModel("gpt-5.6", options...)
				}
				content := []ai.UserContent{ai.CachePoint{}}
				if !onlyPoint {
					content = append(content, ai.TextContent{Text: "next"})
				}
				messages := []ai.ModelMessage{prior, ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: content}}}}
				_, err := model.Request(t.Context(), messages, ai.ModelRequestParams{})
				server.Close()
				if err != nil {
					t.Fatal(err)
				}
				encoded, _ := json.Marshal(body)
				if !strings.Contains(string(encoded), "prompt_cache_breakpoint") {
					t.Fatalf("leading marker lost: %s", encoded)
				}
			}
		}
	}
	model := openai.NewModel("gpt-5.6")
	_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.CachePoint{}}},
	}}}, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "must follow user content") {
		t.Fatalf("unexpected leading marker error: %v", err)
	}
}
