package xai_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/xai"
)

func TestSettingsValidation(t *testing.T) {
	boolean := true
	integer := 1
	cases := []struct {
		name     string
		settings xai.Settings
		want     string
	}{
		{name: "effort", settings: xai.Settings{ReasoningEffort: "extreme"}, want: "invalid reasoning effort"},
		{name: "turns", settings: xai.Settings{MaxTurns: -1}, want: "max turns"},
		{name: "agents", settings: xai.Settings{AgentCount: -1}, want: "agent count"},
		{name: "typed conflict", settings: xai.Settings{
			Common: ai.ModelSettings{ExtraBody: map[string]any{"user": "raw"}}, User: "typed",
		}, want: "conflicts with typed settings"},
		{name: "logprobs conflict", settings: xai.Settings{
			Common: ai.ModelSettings{ExtraBody: map[string]any{"logprobs": false}}, Logprobs: &boolean,
		}, want: "conflicts with portable settings"},
		{name: "top logprobs", settings: xai.Settings{TopLogprobs: &integer}, want: "requires logprobs"},
		{name: "top logprobs range", settings: xai.Settings{
			Logprobs: &boolean, TopLogprobs: intPointer(21),
		}, want: "between 0 and 20"},
		{name: "reserved OpenAI setting", settings: xai.Settings{Common: ai.ModelSettings{
			ExtraBody: map[string]any{"openai_responses_include": []string{"value"}},
		}}, want: "reserved"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			settings, err := test.settings.Build()
			if err == nil {
				server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
				defer server.Close()
				_, err = xai.NewModel(
					"grok-4.3", xai.WithBaseURL(server.URL), xai.WithHTTPClient(server.Client()),
				).Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings})
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestThinkingProfiles(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if expected := request.Header.Get("X-Expected-Effort"); expected != "" {
			reasoning, ok := body["reasoning"].(map[string]any)
			if !ok || reasoning["effort"] != expected {
				t.Errorf("unexpected reasoning: %#v", body["reasoning"])
			}
		} else if _, exists := body["reasoning"]; exists {
			t.Errorf("unexpected reasoning: %#v", body["reasoning"])
		}
		_, _ = response.Write([]byte(`{"id":"id","model":"model","status":"completed","output":[]}`))
	}))
	defer server.Close()

	cases := []struct {
		name, model, expected string
		level                 ai.ThinkingLevel
	}{
		{name: "disabled", model: "grok-4.3", level: ai.ThinkingLevelDisabled, expected: "none"},
		{name: "enabled default", model: "grok-4.3", level: ai.ThinkingLevelEnabled},
		{name: "enabled always", model: "grok-4.5", level: ai.ThinkingLevelEnabled, expected: "medium"},
		{name: "minimal", model: "grok-4.5", level: ai.ThinkingLevelMinimal, expected: "low"},
		{name: "medium fallback", model: "grok-3-mini", level: ai.ThinkingLevelMedium, expected: "high"},
		{name: "high", model: "grok-4.6", level: ai.ThinkingLevelXHigh, expected: "high"},
		{name: "unsupported portable", model: "grok-2", level: ai.ThinkingLevelHigh},
		{name: "unsupported disabled", model: "grok-4.5", level: ai.ThinkingLevelDisabled},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			client := *server.Client()
			client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
				request.Header.Set("X-Expected-Effort", test.expected)
				return server.Client().Transport.RoundTrip(request)
			})
			model := xai.NewModel(
				test.model, xai.WithBaseURL(server.URL), xai.WithAPIKey("key"), xai.WithHTTPClient(&client),
			)
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
				Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: test.level}},
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestUnsupportedContent(t *testing.T) {
	cases := []struct {
		name    string
		content ai.UserContent
		want    string
	}{
		{name: "binary", content: ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/mpeg"}, want: "binary input"},
		{name: "audio URL", content: ai.AudioURL{URL: "https://example.com/audio.mp3"}, want: "audio URL"},
		{name: "video URL", content: ai.VideoURL{URL: "https://example.com/video.mp4"}, want: "video URL"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{test.content}},
			}}}
			model := xai.NewModel("grok-4.3")
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected request error: %v", err)
			}
			if _, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected stream error: %v", err)
			}
		})
	}
}

func TestThinkingErrors(t *testing.T) {
	budget := 100
	include := true
	explicit, err := (xai.Settings{ReasoningEffort: xai.ReasoningEffortNone}).Build()
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, model string
		settings    ai.ModelSettings
		want        string
	}{
		{name: "budget", model: "grok-4.3", settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{TokenBudget: &budget},
		}, want: "token budgets"},
		{name: "visibility", model: "grok-4.3", settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{IncludeThoughts: &include},
		}, want: "visibility"},
		{name: "explicit unsupported model", model: "grok-2", settings: explicit, want: "does not support reasoning"},
		{name: "explicit unsupported effort", model: "grok-4.5", settings: explicit, want: "does not support reasoning effort"},
		{name: "malformed marker", model: "grok-4.3", settings: ai.ModelSettings{
			ExtraBody: map[string]any{"xai_explicit_reasoning_effort": 1},
		}, want: "must be a string"},
		{name: "invalid explicit level", model: "grok-4.3", settings: ai.ModelSettings{
			Thinking:  &ai.ThinkingSettings{Level: "extreme"},
			ExtraBody: map[string]any{"xai_explicit_reasoning_effort": "extreme"},
		}, want: "does not support reasoning effort"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			model := xai.NewModel(test.model)
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: test.settings})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	unsupported := xai.NewModel("grok-2")
	invalid := ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.XSearchTool{
		AllowedXHandles: []string{"one"}, ExcludedXHandles: []string{"two"},
	}}}
	if _, err := unsupported.Request(t.Context(), nil, invalid); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("unexpected native-tool validation error: %v", err)
	}
	params := ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.XSearchTool{}}}
	if _, err := unsupported.Request(t.Context(), nil, params); err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("unexpected required native-tool error: %v", err)
	}
	if _, err := unsupported.StreamRequest(t.Context(), nil, params); err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("unexpected streamed native-tool error: %v", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}
