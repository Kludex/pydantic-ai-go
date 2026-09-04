package groq_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/groq"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestModelAndSettings(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("unexpected request: %s %v", request.URL.Path, request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{
			"id":"completion-1","model":"openai/gpt-oss-20b",
			"choices":[{"finish_reason":"stop","message":{"reasoning":"thinking","content":"hello"}}],
			"usage":{"prompt_tokens":3,"completion_tokens":2}
		}`))
	}))
	defer server.Close()

	common := ai.ModelSettings{ExtraBody: map[string]any{"custom": true}}
	settings, err := (groq.Settings{
		Common: common, ReasoningFormat: groq.ReasoningFormatParsed, ReasoningEffort: groq.ReasoningEffortHigh,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	common.ExtraBody["custom"] = false
	model := groq.NewModel(
		"openai/gpt-oss-20b", groq.WithBaseURL(server.URL), groq.WithAPIKey("secret"),
		groq.WithHTTPClient(server.Client()), groq.WithDefaultSettings(settings),
	)
	response, err := model.Request(context.Background(), nil, ai.ModelRequestParams{Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "openai/gpt-oss-20b" || model.ProviderName() != "groq" || model.ProviderURL() != server.URL ||
		response.Text() != "hello" || len(response.Parts) != 2 {
		t.Fatalf("unexpected model or response: model=%q provider=%q url=%q response=%#v",
			model.Name(), model.ProviderName(), model.ProviderURL(), response)
	}
	if body["reasoning_format"] != "parsed" || body["reasoning_effort"] != "high" || body["custom"] != true {
		t.Fatalf("unexpected Groq request body: %#v", body)
	}
	defaults := model.DefaultModelSettings()
	defaults.ExtraBody["custom"] = false
	if model.DefaultModelSettings().ExtraBody["custom"] != true {
		t.Fatal("default settings were not detached")
	}
}

func TestCompoundWebSearch(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		response.Header().Set("Content-Type", "application/json")
		if body["stream"] == true {
			_, _ = response.Write([]byte("data: {\"id\":\"completion\",\"model\":\"groq/compound\"," +
				"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"found\"}}]}\n\n" +
				"data: {\"id\":\"completion\",\"model\":\"groq/compound\"," +
				"\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n"))
			return
		}
		_, _ = response.Write([]byte(`{
			"id":"completion","model":"groq/compound",
			"choices":[{"finish_reason":"stop","message":{"content":"found"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1}
		}`))
	}))
	defer server.Close()

	model := groq.NewModel("groq/compound", groq.WithBaseURL(server.URL), groq.WithAPIKey("key"),
		groq.WithHTTPClient(server.Client()))
	search := ai.WebSearchTool{
		AllowedDomains: []string{"allowed.example"}, BlockedDomains: []string{"blocked.example"},
	}
	response, err := model.Request(context.Background(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{search},
	})
	if err != nil || response.Text() != "found" {
		t.Fatalf("unexpected compound response: %#v %v", response, err)
	}
	stream, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&search},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
	}
	if len(bodies) != 2 {
		t.Fatalf("unexpected request count: %d", len(bodies))
	}
	for _, body := range bodies {
		settings := body["search_settings"].(map[string]any)
		if settings["include_domains"].([]any)[0] != "allowed.example" ||
			settings["exclude_domains"].([]any)[0] != "blocked.example" {
			t.Fatalf("unexpected search settings: %#v", body)
		}
		if _, exists := body["tools"]; exists {
			t.Fatalf("implicit search emitted a tool declaration: %#v", body)
		}
	}
	if !model.SupportsNativeTool(search) || !model.SupportsNativeTool(&search) ||
		model.SupportsNativeTool(ai.CodeExecutionTool{}) {
		t.Fatal("unexpected compound native-tool support")
	}
	var nilSearch *ai.WebSearchTool
	if model.SupportsNativeTool(nilSearch) || groq.NewModel("model").SupportsNativeTool(search) {
		t.Fatal("unsupported search was accepted")
	}
}

func TestCompoundWebSearchValidation(t *testing.T) {
	compound := groq.NewModel("compound-beta")
	var nilSearch *ai.WebSearchTool
	tests := []struct {
		name   string
		model  *groq.Model
		params ai.ModelRequestParams
		match  string
	}{
		{name: "model", model: groq.NewModel("model"), params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
		}, match: "requires a compound model"},
		{name: "nil", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nil},
		}, match: "must not be nil"},
		{name: "typed nil", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nilSearch},
		}, match: "must not be nil"},
		{name: "tool", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}},
		}, match: "is not supported"},
		{name: "constraint", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{MaxUses: 1}},
		}, match: "only supports domain filters"},
		{name: "conflict", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
			Settings:    ai.ModelSettings{ExtraBody: map[string]any{"search_settings": map[string]any{}}},
		}, match: "conflicts"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.model.Request(context.Background(), nil, test.params)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("expected %q, got %v", test.match, err)
			}
			stream, streamErr := test.model.StreamRequest(context.Background(), nil, test.params)
			if streamErr == nil || stream != nil || !strings.Contains(streamErr.Error(), test.match) {
				t.Fatalf("expected streaming %q, got %v", test.match, streamErr)
			}
		})
	}
}

func TestProviderConfiguration(t *testing.T) {
	t.Setenv("GROQ_API_KEY", "environment-key")
	t.Setenv("GROQ_BASE_URL", "https://groq.example/v1")
	provider := groq.NewProviderConfig()
	if provider.Name != "groq" || provider.APIKey != "environment-key" || provider.BaseURL != "https://groq.example/v1" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
	t.Setenv("GROQ_BASE_URL", "")
	if provider := groq.NewProviderConfig(); provider.BaseURL != "https://api.groq.com/openai/v1" {
		t.Fatalf("unexpected default URL: %q", provider.BaseURL)
	}

	model := groq.NewModel("model", groq.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: "https://gateway.example/v1", APIKey: "key",
	}))
	if model.ProviderName() != "gateway" || model.ProviderURL() != "https://gateway.example/v1" {
		t.Fatalf("unexpected gateway identity: %q %q", model.ProviderName(), model.ProviderURL())
	}
}

func TestSettingsValidation(t *testing.T) {
	tests := []struct {
		name     string
		settings groq.Settings
		match    string
	}{
		{name: "format", settings: groq.Settings{ReasoningFormat: "invalid"}, match: "invalid reasoning format"},
		{name: "effort", settings: groq.Settings{ReasoningEffort: "invalid"}, match: "invalid reasoning effort"},
		{name: "format conflict", settings: groq.Settings{
			Common:          ai.ModelSettings{ExtraBody: map[string]any{"reasoning_format": "raw"}},
			ReasoningFormat: groq.ReasoningFormatParsed,
		}, match: "reasoning_format"},
		{name: "effort conflict", settings: groq.Settings{
			Common:          ai.ModelSettings{ExtraBody: map[string]any{"reasoning_effort": "low"}},
			ReasoningEffort: groq.ReasoningEffortHigh,
		}, match: "reasoning_effort"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	empty, err := (groq.Settings{}).Build()
	if err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v err=%v", empty, err)
	}
	for _, format := range []groq.ReasoningFormat{
		groq.ReasoningFormatHidden, groq.ReasoningFormatRaw, groq.ReasoningFormatParsed,
	} {
		if _, err := (groq.Settings{ReasoningFormat: format}).Build(); err != nil {
			t.Fatal(err)
		}
	}
	for _, effort := range []groq.ReasoningEffort{
		groq.ReasoningEffortNone, groq.ReasoningEffortDefault, groq.ReasoningEffortLow,
		groq.ReasoningEffortMedium, groq.ReasoningEffortHigh,
	} {
		if _, err := (groq.Settings{ReasoningEffort: effort}).Build(); err != nil {
			t.Fatal(err)
		}
	}
}
