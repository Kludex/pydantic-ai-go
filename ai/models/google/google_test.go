package google_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
)

type unsupportedNativeTool struct{ optional bool }

func (tool unsupportedNativeTool) Kind() string                   { return "unsupported" }
func (tool unsupportedNativeTool) UniqueID() string               { return "unsupported" }
func (tool unsupportedNativeTool) IsOptional() bool               { return tool.optional }
func (tool unsupportedNativeTool) CloneNativeTool() ai.NativeTool { return tool }

func newServer(t *testing.T, handler http.HandlerFunc) *google.Model {
	t.Helper()
	return newNamedServer(t, "gemini-2.5-flash", handler)
}

func newNamedServer(t *testing.T, name string, handler http.HandlerFunc, extra ...google.Option) *google.Model {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	opts := []google.Option{
		google.WithAPIKey("test-key"),
		google.WithBaseURL(server.URL),
		google.WithHTTPClient(server.Client()),
	}
	return google.NewModel(name, append(opts, extra...)...)
}

func TestCachedContentSettings(t *testing.T) {
	temperature := 0.2
	common := ai.ModelSettings{Temperature: &temperature, ExtraBody: map[string]any{"custom": true}}
	settings, err := (google.Settings{Common: common, CachedContent: "cachedContents/example"}).Build()
	if err != nil {
		t.Fatal(err)
	}
	settings.ExtraBody["custom"] = false
	if common.ExtraBody["custom"] != true {
		t.Fatal("Google settings share caller-owned extra body")
	}
	var body map[string]any
	model := newNamedServer(t, "gemini-3-flash", func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]}}]}`))
	})
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
		Instructions: "ignored",
		Tools:        []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		NativeTools:  []ai.NativeTool{ai.WebSearchTool{}},
		Settings:     settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	if body["cachedContent"] != "cachedContents/example" || body["systemInstruction"] != nil ||
		body["tools"] != nil || body["toolConfig"] != nil {
		t.Fatalf("unexpected cached-content request: %#v", body)
	}
	empty, err := (google.Settings{CachedContent: "cachedContents/empty"}).Build()
	if err != nil || empty.ExtraBody["google_cached_content"] != "cachedContents/empty" {
		t.Fatalf("unexpected empty common settings: %#v %v", empty, err)
	}
	if _, err := (google.Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{
		"google_cached_content": "existing",
	}}, CachedContent: "duplicate"}).Build(); err == nil {
		t.Fatal("expected cached-content setting conflict")
	}
	plain, err := (google.Settings{Common: common}).Build()
	if err != nil || plain.ExtraBody["custom"] != true {
		t.Fatalf("unexpected plain settings: %#v %v", plain, err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraBody: map[string]any{"google_cached_content": 1},
	}})
	if err == nil || !strings.Contains(err.Error(), "non-empty string") {
		t.Fatalf("unexpected invalid cached content error: %v", err)
	}
}

func TestModelArmorSettings(t *testing.T) {
	config := &google.ModelArmorConfig{
		PromptTemplateName:   "projects/project/locations/global/templates/prompt",
		ResponseTemplateName: "projects/project/locations/global/templates/response",
	}
	settings, err := (google.Settings{ModelArmor: config}).Build()
	if err != nil {
		t.Fatal(err)
	}
	config.PromptTemplateName = "changed"

	var staticBody map[string]any
	vertex := newNamedServer(t, "gemini-2.5-flash", func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&staticBody); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]}}]}`))
	}, google.WithProvider(google.ProviderConfig{Transport: google.TransportVertexAI, Name: "google-cloud"}))
	if _, err := vertex.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err != nil {
		t.Fatal(err)
	}
	armor := staticBody["modelArmorConfig"].(map[string]any)
	if armor["promptTemplateName"] != "projects/project/locations/global/templates/prompt" ||
		armor["responseTemplateName"] != "projects/project/locations/global/templates/response" {
		t.Fatalf("unexpected Model Armor config: %#v", staticBody)
	}

	var streamBody map[string]any
	streaming := newNamedServer(t, "gemini-2.5-flash", func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&streamBody); err != nil {
			t.Error(err)
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = response.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]}}]}\n\n"))
	}, google.WithProvider(google.ProviderConfig{Transport: google.TransportVertexAI, Name: "google-cloud"}))
	stream, err := streaming.StreamRequest(t.Context(), nil, ai.ModelRequestParams{Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
	}
	if streamBody["modelArmorConfig"] != nil {
		t.Fatalf("streaming request included Model Armor: %#v", streamBody)
	}

	gemini := newServer(t, func(http.ResponseWriter, *http.Request) {})
	if _, err := gemini.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err == nil ||
		!strings.Contains(err.Error(), "only supported by Vertex") {
		t.Fatalf("unexpected Gemini Model Armor error: %v", err)
	}
	for name, invalid := range map[string]ai.ModelSettings{
		"encode": {ExtraBody: map[string]any{"google_model_armor_config": make(chan int)}},
		"decode": {ExtraBody: map[string]any{"google_model_armor_config": map[string]any{"promptTemplateName": 1}}},
		"empty":  {ExtraBody: map[string]any{"google_model_armor_config": map[string]any{}}},
	} {
		if _, err := gemini.Request(t.Context(), nil, ai.ModelRequestParams{Settings: invalid}); err == nil {
			t.Fatalf("invalid %s Model Armor config was accepted", name)
		}
	}
	if _, err := (google.Settings{ModelArmor: &google.ModelArmorConfig{}}).Build(); err == nil {
		t.Fatal("empty typed Model Armor config was accepted")
	}
	if _, err := (google.Settings{
		Common:     ai.ModelSettings{ExtraBody: map[string]any{"google_model_armor_config": map[string]any{}}},
		ModelArmor: config,
	}).Build(); err == nil {
		t.Fatal("conflicting Model Armor config was accepted")
	}
}

func TestGoogleCountTokensByTransport(t *testing.T) {
	for _, transport := range []google.Transport{google.TransportGeminiAPI, google.TransportVertexAI} {
		t.Run(string(transport), func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/models/gemini-3:countTokens" || request.Header.Get("x-goog-api-key") != "key" {
					t.Errorf("unexpected token count request: %s headers=%v", request.URL.String(), request.Header)
				}
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = response.Write([]byte(`{"totalTokens":23}`))
			}))
			defer server.Close()
			model := google.NewModel("gemini-3", google.WithProvider(google.ProviderConfig{
				Transport: transport, Name: "custom", BaseURL: server.URL, APIKey: "key", HTTPClient: server.Client(),
			}))
			usage, err := model.CountTokens(t.Context(), []ai.ModelMessage{ai.ModelRequest{
				Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}},
			}}, ai.ModelRequestParams{
				Instructions: "Be brief.",
				Tools:        []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
				NativeTools:  []ai.NativeTool{ai.WebSearchTool{}},
				Settings:     ai.ModelSettings{ExtraHeaders: map[string]string{"X-Test": "value"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if usage.InputTokens != 23 || usage.Requests != 0 {
				t.Fatalf("unexpected token count: %+v", usage)
			}
			if _, ok := body["contents"]; !ok {
				t.Fatalf("count body omitted contents: %v", body)
			}
			_, hasSystem := body["systemInstruction"]
			_, hasTools := body["tools"]
			if transport == google.TransportVertexAI &&
				(!hasSystem || !hasTools || body["generationConfig"] == nil || body["toolConfig"] != nil) {
				t.Fatalf("Vertex count has invalid generation context: %v", body)
			}
			if transport == google.TransportVertexAI {
				tools := body["tools"].([]any)
				if len(tools) != 2 || tools[0].(map[string]any)["googleSearch"] == nil {
					t.Fatalf("Vertex count omitted native tools: %#v", tools)
				}
			}
			if transport == google.TransportGeminiAPI && (hasSystem || hasTools) {
				t.Fatalf("Gemini API count included unsupported context: %v", body)
			}
		})
	}
}

type googleErrorBody struct{}

func (googleErrorBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (googleErrorBody) Close() error             { return nil }

func TestGoogleCountTokensErrors(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}
	t.Run("payload", func(t *testing.T) {
		model := google.NewModel("gemini")
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: "extreme"},
		}})
		if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
			t.Fatalf("unexpected payload error: %v", err)
		}
	})
	t.Run("marshal", func(t *testing.T) {
		included := true
		model := google.NewModel("gemini", google.WithProvider(google.ProviderConfig{
			Transport: google.TransportVertexAI, BaseURL: "http://example.test", APIKey: "key",
		}))
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
			Name: "tool", Schema: map[string]any{"type": "object"},
			ReturnSchema: map[string]any{"bad": make(chan struct{})}, IncludeReturnSchema: &included,
		}}})
		if err == nil || !strings.Contains(err.Error(), "marshal token count request") {
			t.Fatalf("unexpected marshal error: %v", err)
		}
	})
	t.Run("request", func(t *testing.T) {
		model := google.NewModel("gemini", google.WithBaseURL(":"))
		if _, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected request construction error")
		}
	})
	t.Run("prepare", func(t *testing.T) {
		prepareErr := errors.New("prepare failed")
		model := google.NewModel("gemini", google.WithProvider(google.ProviderConfig{
			BaseURL: "http://example.test", PrepareRequest: func(*http.Request) error { return prepareErr },
		}))
		if _, err := model.CountTokens(
			t.Context(), messages, ai.ModelRequestParams{},
		); !errors.Is(err, prepareErr) {
			t.Fatalf("unexpected prepare error: %v", err)
		}
	})
	for _, test := range []struct {
		name       string
		response   *http.Response
		requestErr error
		contains   string
	}{
		{name: "request failure", requestErr: errors.New("request failed"), contains: "token count request"},
		{name: "read failure", response: &http.Response{
			StatusCode: http.StatusOK, Body: googleErrorBody{}, Header: make(http.Header),
		}, contains: "read token count response"},
		{name: "status", response: &http.Response{
			StatusCode: http.StatusBadRequest, Body: io.NopCloser(strings.NewReader("bad")), Header: make(http.Header),
		}, contains: "status 400"},
		{name: "decode", response: &http.Response{
			StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{")), Header: make(http.Header),
		}, contains: "decode token count response"},
		{name: "missing", response: &http.Response{
			StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}")), Header: make(http.Header),
		}, contains: "omitted totalTokens"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return test.response, test.requestErr
			})}
			model := google.NewModel(
				"gemini", google.WithBaseURL("http://example.test"), google.WithHTTPClient(client),
			)
			if _, err := model.CountTokens(
				t.Context(), messages, ai.ModelRequestParams{},
			); err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected token count error: %v", err)
			}
		})
	}
}

func TestGeminiNativeToolReturnSchema(t *testing.T) {
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		tools := body["tools"].([]any)
		declarations := tools[0].(map[string]any)["functionDeclarations"].([]any)
		declaration := declarations[0].(map[string]any)
		responseSchema := declaration["responseJsonSchema"].(map[string]any)
		if responseSchema["type"] != "string" || declaration["description"] != "Lookup." {
			t.Errorf("unexpected native return schema: %v", declaration)
		}
		_, _ = response.Write([]byte(`{
			"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]
		}`))
	})
	included := true
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
		Name: "lookup", Description: "Lookup.", Schema: map[string]any{"type": "object"},
		ReturnSchema: map[string]any{"type": "string"}, IncludeReturnSchema: &included,
	}}})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDefaultSettingsAreDetached(t *testing.T) {
	stop := []string{"stop"}
	model := google.NewModel("gemini-test", google.WithDefaultSettings(ai.ModelSettings{
		MaxTokens: 42, StopSequences: stop,
	}))
	stop[0] = "changed"
	defaults := model.DefaultModelSettings()
	if defaults.MaxTokens != 42 || defaults.StopSequences[0] != "stop" {
		t.Fatalf("unexpected defaults: %+v", defaults)
	}
	defaults.StopSequences[0] = "mutated"
	if model.DefaultModelSettings().StopSequences[0] != "stop" {
		t.Fatal("model defaults were mutable")
	}
}

func TestThinkingSettings(t *testing.T) {
	for name, test := range map[string]struct {
		model      string
		settings   *ai.ThinkingSettings
		budget     *int
		level      string
		include    bool
		hasInclude bool
	}{
		"disabled budget": {
			model: "gemini-2.5-flash", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
			budget: func() *int { value := 0; return &value }(),
		},
		"enabled": {
			model: "gemini-2.5-flash", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelEnabled},
			include: true, hasInclude: true,
		},
		"effort budget": {
			model: "gemini-2.5-flash", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh},
			budget: func() *int { value := 24576; return &value }(), include: true, hasInclude: true,
		},
		"explicit budget": {
			model: "gemini-2.5-flash", settings: &ai.ThinkingSettings{
				TokenBudget: func() *int { value := 99; return &value }(),
			},
			budget: func() *int { value := 99; return &value }(), include: true, hasInclude: true,
		},
		"include override": {
			model: "gemini-2.5-flash", settings: &ai.ThinkingSettings{
				Level: ai.ThinkingLevelEnabled, IncludeThoughts: func() *bool { value := false; return &value }(),
			},
			hasInclude: true,
		},
		"disabled level": {
			model: "gemini-3-pro", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
			level: "MINIMAL",
		},
		"disabled explicit include": {
			model: "gemini-3-pro", settings: &ai.ThinkingSettings{
				Level: ai.ThinkingLevelDisabled, IncludeThoughts: func() *bool { value := false; return &value }(),
			},
			level: "MINIMAL", hasInclude: true,
		},
		"effort level": {
			model: "gemini-3-pro", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelXHigh},
			level: "HIGH", include: true, hasInclude: true,
		},
		"flash minimum snaps up": {
			model: "gemini-3.8-flash", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelMinimal},
			level: "LOW", include: true, hasInclude: true,
		},
		"pro medium tie snaps down": {
			model: "gemini-3-pro-preview", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelMedium},
			level: "LOW", include: true, hasInclude: true,
		},
		"flash lite image low snaps down": {
			model: "gemini-3.1-flash-lite-image", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelLow},
			level: "MINIMAL", include: true, hasInclude: true,
		},
		"flash lite image medium snaps up": {
			model: "gemini-3.1-flash-lite-image", settings: &ai.ThinkingSettings{Level: ai.ThinkingLevelMedium},
			level: "HIGH", include: true, hasInclude: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			model := newNamedServer(t, test.model, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
			})
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
				Thinking: test.settings,
			}})
			if err != nil {
				t.Fatal(err)
			}
			thinking := body["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
			include, hasInclude := thinking["includeThoughts"]
			if hasInclude != test.hasInclude || hasInclude && include != test.include {
				t.Fatalf("unexpected includeThoughts: %v", thinking)
			}
			if test.budget != nil && int(thinking["thinkingBudget"].(float64)) != *test.budget {
				t.Fatalf("unexpected thinking budget: %v", thinking)
			}
			if test.level != "" && thinking["thinkingLevel"] != test.level {
				t.Fatalf("unexpected thinking level: %v", thinking)
			}
		})
	}

	for name, modelName := range map[string]string{"budget model": "gemini-2.5-flash", "level model": "gemini-3-pro"} {
		t.Run("invalid "+name, func(t *testing.T) {
			model := google.NewModel(modelName)
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
				Thinking: &ai.ThinkingSettings{Level: "extreme"},
			}})
			if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
				t.Fatalf("unexpected invalid thinking error: %v", err)
			}
		})
	}
}

func TestServiceTierMapping(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: ai.ServiceTierFlex,
	}}); err != nil {
		t.Fatal(err)
	}
	if body["generationConfig"].(map[string]any)["serviceTier"] != "flex" {
		t.Fatalf("unexpected service tier payload: %v", body)
	}

	_, err := google.NewModel("gemini").Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: "expedited",
	}})
	if err == nil || err.Error() != `google: invalid service tier "expedited"` {
		t.Fatalf("unexpected service tier error: %v", err)
	}
}

func TestGooglePromptFeedbackBlock(t *testing.T) {
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{
			"responseId":"blocked","modelVersion":"gemini",
			"promptFeedback":{
				"blockReason":"PROHIBITED_CONTENT","blockReasonMessage":"The prompt was blocked.",
				"safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","blocked":true}]
			},
			"usageMetadata":{"trafficType":"PROVISIONED_THROUGHPUT"}
		}`))
	})
	_, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "blocked", struct{}{})
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) || filtered.Response().FinishReason != ai.FinishReasonContentFilter ||
		filtered.Response().ProviderDetails["block_reason"] != "PROHIBITED_CONTENT" ||
		filtered.Response().ProviderDetails["block_reason_message"] != "The prompt was blocked." ||
		filtered.Response().ProviderDetails["traffic_type"] != "PROVISIONED_THROUGHPUT" {
		t.Fatalf("unexpected prompt block: %v response=%+v", err, filtered)
	}
	ratings, ok := filtered.Response().ProviderDetails["safety_ratings"].([]map[string]any)
	if !ok || len(ratings) != 1 {
		t.Fatalf("unexpected prompt safety ratings: %#v", filtered.Response().ProviderDetails)
	}
	ratings[0]["blocked"] = false
	fresh := filtered.Response().ProviderDetails["safety_ratings"].([]map[string]any)
	if fresh[0]["blocked"] != true {
		t.Fatal("prompt safety ratings were not detached")
	}
}

func TestGoogleModelArmorAndSPIIFinishReasons(t *testing.T) {
	for _, reason := range []string{"MODEL_ARMOR", "SPII"} {
		t.Run(reason, func(t *testing.T) {
			model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(response, `{
					"candidates":[{"content":{"parts":[]},"finishReason":%q}]
				}`, reason)
			})
			_, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "blocked", struct{}{})
			var filtered *ai.ContentFilterError
			if !errors.As(err, &filtered) || filtered.Response().FinishReason != ai.FinishReasonContentFilter ||
				filtered.Response().ProviderDetails["finish_reason"] != reason {
				t.Fatalf("unexpected %s block: %v response=%+v", reason, err, filtered)
			}
		})
	}
}

func TestGoogleCandidateSafetyRatings(t *testing.T) {
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{
			"candidates":[{
				"content":{"parts":[{"text":"allowed"}]},"finishReason":"STOP",
				"safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"NEGLIGIBLE"}]
			}]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if ratings, ok := response.ProviderDetails["safety_ratings"].([]map[string]any); !ok || len(ratings) != 1 {
		t.Fatalf("unexpected candidate ratings: %#v", response.ProviderDetails)
	}
}

func TestRequestTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotKey, gotPath, gotCustom string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-gemini-service-tier", "PRIORITY")
		gotKey = r.Header.Get("x-goog-api-key")
		gotCustom = r.Header.Get("x-custom")
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"responseId": "response-1", "modelVersion": "gemini-2.5-flash",
			"candidates": [{
				"content": {"parts": [{"text": "Hello!", "thoughtSignature": "signature"}]},
				"finishReason": "STOP", "avgLogprobs": -0.25,
				"logprobsResult": {"chosenCandidates": [{"token": "Hello", "logProbability": -0.25}]}
			}],
			"usageMetadata": {
				"trafficType": "ON_DEMAND",
				"promptTokenCount": 12, "candidatesTokenCount": 3,
				"cachedContentTokenCount": 4, "thoughtsTokenCount": 2,
				"toolUsePromptTokenCount": 7,
				"promptTokensDetails": [
					{"modality":"AUDIO","tokenCount":2}, {"modality":"TEXT","tokenCount":10}
				],
				"cacheTokensDetails": [
					{"modality":"AUDIO","tokenCount":1}, {"modality":"TEXT","tokenCount":3}
				],
				"candidatesTokensDetails": [
					{"modality":"AUDIO","tokenCount":1}, {"modality":"TEXT","tokenCount":2}
				],
				"toolUsePromptTokensDetails": [{"modality":"TEXT","tokenCount":7}]
			}
		}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}
	temp := 0.5
	presencePenalty := 0.2
	frequencyPenalty := 0.3
	logprobs := true
	topLogprobs := 3
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{
		Instructions: "be brief",
		AllowText:    true,
		Settings: ai.ModelSettings{
			MaxTokens: 100, Temperature: &temp,
			PresencePenalty: &presencePenalty, FrequencyPenalty: &frequencyPenalty,
			Logprobs: &logprobs, TopLogprobs: &topLogprobs, ServiceTier: ai.ServiceTierDefault,
			ExtraHeaders: map[string]string{"x-custom": "value"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotKey != "test-key" || gotCustom != "value" {
		t.Fatalf("unexpected headers key=%q custom=%q", gotKey, gotCustom)
	}
	if !strings.HasSuffix(gotPath, "/models/gemini-2.5-flash:generateContent") {
		t.Fatalf("unexpected path %q", gotPath)
	}
	if gotBody["systemInstruction"] == nil {
		t.Fatal("system instruction not sent")
	}
	gen := gotBody["generationConfig"].(map[string]any)
	if gen["maxOutputTokens"].(float64) != 100 || gen["temperature"].(float64) != 0.5 ||
		gen["presencePenalty"].(float64) != presencePenalty ||
		gen["frequencyPenalty"].(float64) != frequencyPenalty || gen["responseLogprobs"] != true ||
		gen["logprobs"].(float64) != float64(topLogprobs) || gen["serviceTier"] != "standard" {
		t.Fatalf("generation config not sent: %v", gen)
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	textPart := resp.Parts[0].(ai.TextPart)
	if textPart.ProviderName != "google" || textPart.ProviderDetails["thought_signature"] != "signature" {
		t.Fatalf("thought signature metadata lost: %+v", textPart)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 5 || resp.Usage.Requests != 1 ||
		resp.Usage.CacheReadTokens != 4 || resp.Usage.ReasoningTokens != 2 ||
		resp.Usage.InputAudioTokens != 2 || resp.Usage.CacheAudioReadTokens != 1 ||
		resp.Usage.OutputAudioTokens != 1 || resp.Usage.Details["cached_content_tokens"] != 4 ||
		resp.Usage.Details["thoughts_tokens"] != 2 || resp.Usage.Details["tool_use_prompt_tokens"] != 7 ||
		resp.Usage.Details["audio_prompt_tokens"] != 2 || resp.Usage.Details["text_cache_tokens"] != 3 ||
		resp.Usage.Details["text_candidates_tokens"] != 2 ||
		resp.Usage.Details["text_tool_use_prompt_tokens"] != 7 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
	if resp.ModelName != "gemini-2.5-flash" || resp.ProviderName != "google" || resp.ProviderURL == "" ||
		resp.ProviderResponseID != "response-1" || resp.FinishReason != ai.FinishReasonStop ||
		resp.ProviderDetails["finish_reason"] != "STOP" || resp.ProviderDetails["service_tier"] != "priority" ||
		resp.ProviderDetails["traffic_type"] != "ON_DEMAND" || resp.ProviderDetails["avg_logprobs"] != -0.25 || resp.ProviderDetails["logprobs"] == nil {
		t.Fatalf("unexpected response metadata %+v", resp)
	}
}

func TestRequestFunctionCallRoundTrip(t *testing.T) {
	var gotBody map[string]any
	first := true
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-gemini-service-tier", "STANDARD")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"modelVersion": "gemini-2.5-flash",
				"candidates": [{"content": {"parts": [
					{"thought": true, "text": "checking"},
					{"functionCall": {"id": "call1", "name": "get_weather", "args": {"city": "SF"}}, "thoughtSignature": "tool-signature"}
				]}}],
				"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"modelVersion": "gemini-2.5-flash",
			"candidates": [{"content": {"parts": [{"text": "Sunny."}]}}],
			"usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4}
		}`))
	})

	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "get_weather", Schema: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
		}}},
		AllowText: true,
	}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "weather?"}}}}
	resp, err := model.Request(t.Context(), msgs, params)
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ToolName != "get_weather" || calls[0].ToolCallID != "call1" ||
		string(calls[0].Args) != `{"city":"SF"}` || calls[0].ProviderName != "google" ||
		calls[0].ProviderDetails["thought_signature"] != "tool-signature" {
		t.Fatalf("unexpected calls %+v", calls)
	}
	if _, ok := resp.Parts[0].(ai.ThinkingPart); !ok || resp.ProviderDetails["service_tier"] != "standard" {
		t.Fatalf("thought part or service tier lost: %+v", resp)
	}

	msgs = append(msgs, *resp, ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny", ToolCallID: "call1"},
		ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"archive"}, ToolCallID: "call1"},
	}})
	if _, err := model.Request(t.Context(), msgs, params); err != nil {
		t.Fatal(err)
	}
	contents := gotBody["contents"].([]any)
	modelTurn := contents[1].(map[string]any)
	if modelTurn["role"] != "model" {
		t.Fatalf("unexpected roles %v", contents)
	}
	modelParts := modelTurn["parts"].([]any)
	functionCall := modelParts[1].(map[string]any)
	if functionCall["thoughtSignature"] != "tool-signature" {
		t.Fatalf("thought signature was not round-tripped: %v", functionCall)
	}
	toolTurn := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)
	functionResponse := toolTurn["functionResponse"].(map[string]any)
	if functionResponse["id"] != "call1" {
		t.Fatalf("function response ID not sent: %v", toolTurn)
	}
	declared := gotBody["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	schema := declared["parametersJsonSchema"].(map[string]any)
	if schema["additionalProperties"] != false {
		t.Fatal("additionalProperties should be preserved in Gemini JSON Schema")
	}
}

func TestOutputToolForcesFunctionCalling(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"final_result","args":{}}}]}}],"usageMetadata":{}}`))
	})
	params := ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	mode := gotBody["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)["mode"]
	if mode != "ANY" {
		t.Fatalf("expected ANY mode, got %v", mode)
	}
}

func TestRetryAndSystemParts(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "sys"},
		ai.RetryPromptPart{Content: "bad args", ToolName: "t"},
		ai.RetryPromptPart{Content: "plain retry"},
		ai.ToolReturnPart{ToolName: "t", Content: "ok"},
		ai.ToolReturnPart{ToolName: "t", Content: "failed", Outcome: ai.ToolReturnOutcomeFailed},
		ai.ToolReturnPart{ToolName: "t", Content: "stopped", Outcome: ai.ToolReturnOutcomeInterrupted},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if parts[1].(map[string]any)["functionResponse"] == nil {
		t.Fatalf("tool retry should be a function response: %v", parts[1])
	}
	if text := parts[2].(map[string]any)["text"].(string); !strings.Contains(text, "Validation feedback:\nplain retry") ||
		!strings.HasSuffix(text, "Fix the errors and try again.") {
		t.Fatalf("plain retry should be formatted as validation feedback: %v", parts[2])
	}
	for index, key := range []string{"result", "error", "error"} {
		response := parts[index+3].(map[string]any)["functionResponse"].(map[string]any)["response"].(map[string]any)
		if _, ok := response[key]; !ok {
			t.Fatalf("tool outcome at %d did not use %q: %v", index, key, response)
		}
	}
}

func TestGoogleWebSearchGroundingMetadata(t *testing.T) {
	responses := []string{
		`{
			"responseId":"response","modelVersion":"gemini-3-flash","candidates":[{
				"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP",
				"groundingMetadata":{
					"webSearchQueries":[1,"Go news"],
					"groundingChunks":[1,{}, {"web":{"uri":"https://go.dev/blog","title":"Go Blog"}}]
				}
			}]
		}`,
		`{"candidates":[{"content":{"parts":[{"text":"answer"}]},"groundingMetadata":{"webSearchQueries":["query"]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"answer"}]},"groundingMetadata":{"webSearchQueries":[1]}}]}`,
	}
	index := 0
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(responses[index]))
		index++
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.Timestamp.IsZero() || len(response.Parts) != 3 ||
		response.ProviderDetails["grounding_metadata"] == nil {
		t.Fatalf("unexpected grounded response: %+v", response)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	results := returned.Content.([]map[string]any)
	if call.ToolKind != ai.ToolPartKindWebSearch || call.ToolCallID != "response:web_search" ||
		string(call.Args) != `{"queries":["Go news"]}` || returned.ToolCallID != call.ToolCallID ||
		returned.Timestamp != response.Timestamp || len(results) != 1 || results[0]["title"] != "Go Blog" {
		t.Fatalf("unexpected grounded tool parts: call=%+v return=%+v", call, returned)
	}
	response, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	results = response.Parts[1].(ai.NativeToolReturnPart).Content.([]map[string]any)
	if response.Parts[0].(ai.NativeToolCallPart).ToolCallID != "web_search" || len(results) != 0 {
		t.Fatalf("unexpected grounding without response ID or chunks: %#v", response.Parts)
	}
	response, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Parts) != 1 {
		t.Fatalf("non-string search queries produced native parts: %#v", response.Parts)
	}
}

func TestGoogleWebFetchURLContextMetadata(t *testing.T) {
	responses := []string{
		`{"responseId":"response","candidates":[{"content":{"parts":[{"text":"answer"}]},"urlContextMetadata":{"urlMetadata":[1,{"retrievedUrl":"https://go.dev","urlRetrievalStatus":"URL_RETRIEVAL_STATUS_SUCCESS"},{"urlRetrievalStatus":"URL_RETRIEVAL_STATUS_ERROR"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"answer"}]},"urlContextMetadata":{"urlMetadata":[{"urlRetrievalStatus":"URL_RETRIEVAL_STATUS_ERROR"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"answer"}]},"urlContextMetadata":{}}]}`,
	}
	index := 0
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(responses[index]))
		index++
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderDetails["url_context_metadata"] == nil || len(response.Parts) != 3 {
		t.Fatalf("unexpected URL context response: %+v", response)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	results := returned.Content.([]map[string]any)
	if call.ToolKind != ai.ToolPartKindWebFetch || call.ToolCallID != "response:web_fetch" ||
		string(call.Args) != `{"urls":["https://go.dev"]}` || returned.ToolKind != ai.ToolPartKindWebFetch ||
		len(results) != 2 || results[0]["retrievedUrl"] != "https://go.dev" {
		t.Fatalf("unexpected URL context parts: call=%+v return=%+v", call, returned)
	}
	response, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if call = response.Parts[0].(ai.NativeToolCallPart); call.ToolCallID != "web_fetch" || string(call.Args) != `{}` {
		t.Fatalf("unexpected URL context without URL or response ID: %+v", call)
	}
	response, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Parts) != 1 {
		t.Fatalf("empty URL context produced native parts: %#v", response.Parts)
	}
}

func TestGoogleFileSearchGroundingMetadata(t *testing.T) {
	responses := []string{
		`{"responseId":"response","candidates":[{"content":{"parts":[{"text":"Paris"}]},"groundingMetadata":{"groundingChunks":[1,{"retrievedContext":{"text":"Paris is the capital.","fileSearchStore":"fileSearchStores/store","customMetadata":{"source_url":"https://example.com/paris"}}}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"none"}]},"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com"}}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"none"}]},"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com"}}]}}]}`,
	}
	index := 0
	var body map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(responses[index]))
		index++
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.FileSearchTool{FileStoreIDs: []string{"fileSearchStores/store"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Parts) != 3 || response.ProviderDetails["grounding_metadata"] == nil {
		t.Fatalf("unexpected file search response: %+v", response)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	contexts := returned.Content.([]map[string]any)
	custom := contexts[0]["custom_metadata"].(map[string]any)
	if call.ToolKind != ai.ToolPartKindFileSearch || call.ToolCallID != "response:file_search" ||
		string(call.Args) != `{}` || returned.ToolCallID != call.ToolCallID || returned.Timestamp.IsZero() ||
		contexts[0]["text"] != "Paris is the capital." ||
		contexts[0]["file_search_store"] != "fileSearchStores/store" ||
		custom["source_url"] != "https://example.com/paris" {
		t.Fatalf("unexpected normalized file search: call=%+v return=%+v", call, returned)
	}
	custom["source_url"] = "changed"
	metadata := response.ProviderDetails["grounding_metadata"].(map[string]any)
	rawContext := metadata["groundingChunks"].([]any)[1].(map[string]any)["retrievedContext"].(map[string]any)
	if rawContext["customMetadata"].(map[string]any)["source_url"] != "https://example.com/paris" {
		t.Fatal("normalized file-search result aliases provider metadata")
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&ai.FileSearchTool{FileStoreIDs: []string{"fileSearchStores/store"}}},
	}); err != nil {
		t.Fatal(err)
	}
	parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["toolCall"].(map[string]any)["toolType"] != "FILE_SEARCH" ||
		parts[1].(map[string]any)["toolResponse"].(map[string]any)["toolType"] != "FILE_SEARCH" {
		t.Fatalf("unexpected file search replay: %#v", parts)
	}
	if response, err = model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.FileSearchTool{FileStoreIDs: []string{"fileSearchStores/store"}},
	}}); err != nil || len(response.Parts) != 1 {
		t.Fatalf("irrelevant grounding created file search parts: response=%+v err=%v", response, err)
	}
}

func TestGoogleExplicitFileSearchParts(t *testing.T) {
	var body map[string]any
	model := newNamedServer(t, "gemini-3-flash", func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"responseId":"response","candidates":[{"content":{"parts":[
			{"thoughtSignature":"call-signature","toolCall":{"id":"search","toolType":"FILE_SEARCH","args":{"query":"capital"}}},
			{"thoughtSignature":"return-signature","toolResponse":{"id":"search","toolType":"FILE_SEARCH"}},
			{"text":"Paris"}
		]},"groundingMetadata":{"webSearchQueries":["duplicate"],"groundingChunks":[{"retrievedContext":{"text":"Paris context","fileSearchStore":"fileSearchStores/store"}}]},"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://example.com"}]}}]}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.FileSearchTool{FileStoreIDs: []string{"fileSearchStores/store"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Parts) != 3 {
		t.Fatalf("explicit native parts were duplicated: %#v", response.Parts)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if call.ToolCallID != "search" || string(call.Args) != `{"query":"capital"}` ||
		call.ProviderDetails["thought_signature"] != "call-signature" || returned.ToolCallID != "search" ||
		returned.ProviderDetails["thought_signature"] != "return-signature" ||
		returned.Content.([]map[string]any)[0]["file_search_store"] != "fileSearchStores/store" {
		t.Fatalf("unexpected explicit file search: call=%+v return=%+v", call, returned)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["thoughtSignature"] != "call-signature" ||
		parts[1].(map[string]any)["thoughtSignature"] != "return-signature" {
		t.Fatalf("explicit file search signatures were not replayed: %#v", parts)
	}
}

func TestGoogleLegacyFileSearchExecutableCode(t *testing.T) {
	responses := []string{
		`{"responseId":"response","candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(file_search.query(query=\"capital of \\\"France\\\"\"))"}},{"text":"Paris"}]},"groundingMetadata":{"groundingChunks":[{"retrievedContext":{"text":"Paris"}}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"file_search.query(query='Eiffel\\'s location')"}}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"file_search.query(query=\"line\\nfeed\")"}}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}}]}}]}`,
	}
	index := 0
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(responses[index]))
		index++
	})
	params := ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.FileSearchTool{FileStoreIDs: []string{"fileSearchStores/store"}},
	}}
	response, err := model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Parts) != 3 {
		t.Fatalf("legacy file search was duplicated: %#v", response.Parts)
	}
	response, err = model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	if call.ToolKind != ai.ToolPartKindFileSearch || string(call.Args) != `{"query":"Eiffel's location"}` {
		t.Fatalf("unexpected legacy file search query: %+v", call)
	}
	response, err = model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	if call = response.Parts[0].(ai.NativeToolCallPart); string(call.Args) != `{"query":"line\\nfeed"}` {
		t.Fatalf("unexpected unknown query escape: %+v", call)
	}
	response, err = model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	if call = response.Parts[0].(ai.NativeToolCallPart); call.ToolKind != ai.ToolPartKindCodeExecution {
		t.Fatalf("ordinary executable code became file search: %+v", call)
	}
}

func TestGoogleNativeToolPartErrors(t *testing.T) {
	for name, nativePart := range map[string]string{
		"call":     `{"toolCall":{"toolType":"FUTURE","args":{}}}`,
		"response": `{"toolResponse":{"toolType":"FUTURE","response":{}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
				_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[` + nativePart + `]}}]}`))
			})
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
				!strings.Contains(err.Error(), "unknown native tool type") {
				t.Fatalf("unexpected native tool part error: %v", err)
			}
		})
	}
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"responseId":"response","candidates":[{"content":{"parts":[
			{"toolCall":{"toolType":"FILE_SEARCH","args":{}}},
			{"toolResponse":{"toolType":"FILE_SEARCH","response":[]}}
		]}}]}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if call.ToolCallID != returned.ToolCallID || call.ToolCallID != "response:file_search:0" {
		t.Fatalf("generated native IDs did not pair: call=%+v return=%+v", call, returned)
	}

	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: "file_search", ToolCallID: "empty", ToolKind: ai.ToolPartKindFileSearch,
			ProviderName: "google",
		},
		ai.NativeToolReturnPart{
			ToolName: "file_search", ToolCallID: "empty", ToolKind: ai.ToolPartKindFileSearch,
			ProviderName: "google", Content: []map[string]any{},
		},
		ai.NativeToolCallPart{
			ToolName: "foreign", ToolKind: ai.ToolPartKindFileSearch, ProviderName: "openai",
		},
		ai.NativeToolCallPart{
			ToolName: "unknown", ProviderName: "google",
		},
		ai.NativeToolReturnPart{
			ToolName: "foreign", ToolKind: ai.ToolPartKindFileSearch, ProviderName: "openai",
		},
		ai.NativeToolReturnPart{
			ToolName: "unknown", ProviderName: "google",
		},
	}}}
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	history = []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.NativeToolCallPart{
		ToolName: "file_search", ToolKind: ai.ToolPartKindFileSearch, ProviderName: "google",
		Args: json.RawMessage(`{`),
	}}}}
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "native tool call args") {
		t.Fatalf("unexpected malformed native history error: %v", err)
	}
}

func TestGoogleCodeExecutionResponse(t *testing.T) {
	responses := []string{
		`{"responseId":"response","candidates":[{"content":{"parts":[
			{"executableCode":{"language":"PYTHON","code":"print(1)"}},
			{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1\n"}},
			{"text":"done"}
		]}}]}`,
		`{"candidates":[{"content":{"parts":[{"codeExecutionResult":{"outcome":"OUTCOME_FAILED","output":"bad"}}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"pass"}}]}}]}`,
	}
	index := 0
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(responses[index]))
		index++
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if call.ToolKind != ai.ToolPartKindCodeExecution || call.ToolCallID != "response:code_execution:0" ||
		string(call.Args) != `{"code":"print(1)","language":"PYTHON"}` ||
		returned.ToolCallID != call.ToolCallID || returned.ToolKind != ai.ToolPartKindCodeExecution ||
		returned.Content.(map[string]any)["output"] != "1\n" || returned.Timestamp != response.Timestamp {
		t.Fatalf("unexpected code execution parts: call=%+v return=%+v", call, returned)
	}
	response, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	returned = response.Parts[0].(ai.NativeToolReturnPart)
	if returned.ToolCallID != "code_execution:0" || returned.Content.(map[string]any)["outcome"] != "OUTCOME_FAILED" {
		t.Fatalf("unexpected orphan code execution result: %+v", returned)
	}
	response, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if call = response.Parts[0].(ai.NativeToolCallPart); call.ToolCallID != "code_execution:0" {
		t.Fatalf("unexpected code call without response ID: %+v", call)
	}
}

func TestGoogleImageGeneration(t *testing.T) {
	var body map[string]any
	model := newNamedServer(t, "gemini-3-pro-image-preview", func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"responseId":"response","modelVersion":"gemini-3-pro-image-preview","candidates":[{"content":{"parts":[
			{"thought":true,"inlineData":{"mimeType":"image/png","data":"dGhvdWdodA=="}},
			{"thoughtSignature":"signature","inlineData":{"mimeType":"image/webp","data":"aW1hZ2U="}},
			{"text":"done"}
		]},"finishReason":"STOP"}]}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		AllowText: true,
		NativeTools: []ai.NativeTool{ai.ImageGenerationTool{
			AspectRatio: ai.ImageAspectRatio16x9, Size: ai.ImageGenerationSize2K,
			OutputFormat: ai.ImageGenerationOutputPNG,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	generation := body["generationConfig"].(map[string]any)
	image := generation["imageConfig"].(map[string]any)
	modalities := generation["responseModalities"].([]any)
	if body["tools"] != nil || image["aspectRatio"] != "16:9" || image["imageSize"] != "2K" ||
		image["outputMimeType"] != nil || len(modalities) != 2 || modalities[0] != "TEXT" || modalities[1] != "IMAGE" {
		t.Fatalf("unexpected Gemini image request: %#v", body)
	}
	if len(response.Parts) != 2 {
		t.Fatalf("thinking image was not omitted: %#v", response.Parts)
	}
	file := response.Parts[0].(ai.FilePart)
	if file.Content.MediaType != "image/webp" || string(file.Content.Data) != "image" ||
		file.ProviderName != "google" || file.ProviderDetails["thought_signature"] != "signature" ||
		response.Text() != "done" {
		t.Fatalf("unexpected generated image response: %+v", response)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	replayed := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0].(map[string]any)
	inline := replayed["inlineData"].(map[string]any)
	if inline["mimeType"] != "image/webp" || inline["data"] != "aW1hZ2U=" ||
		replayed["thoughtSignature"] != "signature" {
		t.Fatalf("unexpected generated image replay: %#v", replayed)
	}
}

func TestGoogleVertexImageGenerationConfig(t *testing.T) {
	compression := 85
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	}))
	defer server.Close()
	model := google.NewModel("gemini-3-pro-image-preview", google.WithProvider(google.ProviderConfig{
		Transport: google.TransportVertexAI, Name: "google-cloud", BaseURL: server.URL,
		APIKey: "key", HTTPClient: server.Client(),
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		&ai.ImageGenerationTool{
			OutputFormat: ai.ImageGenerationOutputJPEG, OutputCompression: &compression,
			Size: ai.ImageGenerationSize4K, AspectRatio: ai.ImageAspectRatio3x2,
		},
	}}); err != nil {
		t.Fatal(err)
	}
	image := body["generationConfig"].(map[string]any)["imageConfig"].(map[string]any)
	if image["outputMimeType"] != "image/jpeg" || image["outputCompressionQuality"] != float64(85) ||
		image["imageSize"] != "4K" || image["aspectRatio"] != "3:2" {
		t.Fatalf("unexpected Vertex image config: %#v", image)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.ImageGenerationTool{OutputCompression: &compression},
	}}); err != nil {
		t.Fatal(err)
	}
	image = body["generationConfig"].(map[string]any)["imageConfig"].(map[string]any)
	if image["outputMimeType"] != "image/jpeg" || image["outputCompressionQuality"] != float64(85) {
		t.Fatalf("compression did not default Vertex output to JPEG: %#v", image)
	}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
		Contents: []ai.UserContent{ai.UploadedFile{
			FileID: "gs://bucket/report.pdf", ProviderName: "google-cloud",
		}},
	}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	file := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0].(map[string]any)["fileData"].(map[string]any)
	if file["fileUri"] != "gs://bucket/report.pdf" || file["mimeType"] != "application/pdf" {
		t.Fatalf("unexpected Vertex uploaded file: %#v", file)
	}
	messages[0] = ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.UploadedFile{FileID: "https://example.com/file", ProviderName: "google-cloud"},
	}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "must use a gs:// URI") {
		t.Fatalf("unexpected Vertex uploaded file error: %v", err)
	}

	for _, test := range []struct {
		name string
		tool ai.NativeTool
		want string
	}{
		{name: "OpenAI size", tool: &ai.ImageGenerationTool{Size: ai.ImageGenerationSize1024x1024}, want: "unsupported image generation size"},
		{name: "auto size", tool: ai.ImageGenerationTool{Size: ai.ImageGenerationSizeAuto}, want: "unsupported image generation size"},
		{name: "compression format", tool: ai.ImageGenerationTool{
			OutputFormat: ai.ImageGenerationOutputPNG, OutputCompression: &compression,
		}, want: "requires JPEG format"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{test.tool}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected image generation error: %v", err)
			}
		})
	}

	gemini := newNamedServer(t, "gemini-2.5-flash-image", func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	})
	if _, err := gemini.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.ImageGenerationTool{OutputFormat: ai.ImageGenerationOutputWebP, OutputCompression: &compression},
	}}); err != nil {
		t.Fatal(err)
	}
	if image := body["generationConfig"].(map[string]any)["imageConfig"].(map[string]any); len(image) != 0 {
		t.Fatalf("Gemini API did not ignore output encoding settings: %#v", image)
	}
}

func TestGoogleImageGenerationCompatibility(t *testing.T) {
	handler := func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	}
	model := newServer(t, handler)
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.ImageGenerationTool{}},
	}); err == nil || !strings.Contains(err.Error(), "requires a model with image output support") {
		t.Fatalf("unexpected unsupported image model error: %v", err)
	}
	for _, optional := range []ai.NativeTool{
		ai.ImageGenerationTool{Optional: true}, &ai.ImageGenerationTool{Optional: true},
	} {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{optional},
		}); err != nil {
			t.Fatalf("optional image generation should be omitted: %v", err)
		}
	}
	imageModel := newNamedServer(t, "gemini-2.5-flash-image", handler)
	if _, err := imageModel.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.ImageGenerationTool{}},
		Tools:       []ai.ToolDefinition{{Name: "work", Schema: map[string]any{"type": "object"}}},
	}); err == nil || !strings.Contains(err.Error(), "does not support function and native tools together") {
		t.Fatalf("unexpected image/function combination error: %v", err)
	}
}

func TestGoogleImageResponseErrors(t *testing.T) {
	for name, inline := range map[string]string{
		"missing data":   `{"mimeType":"image/png"}`,
		"missing MIME":   `{"data":"aQ=="}`,
		"invalid base64": `{"mimeType":"image/png","data":"!"}`,
	} {
		t.Run(name, func(t *testing.T) {
			model := newNamedServer(t, "gemini-image", func(response http.ResponseWriter, _ *http.Request) {
				_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"inlineData":` + inline + `}]}}]}`))
			})
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected inline image error")
			}
		})
	}
}

func TestErrors(t *testing.T) {
	t.Run("native tools", func(t *testing.T) {
		var body map[string]any
		handler := func(w http.ResponseWriter, request *http.Request) {
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
		}
		model := newServer(t, handler)
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{
				ai.WebSearchTool{}, ai.WebFetchTool{}, ai.CodeExecutionTool{},
				ai.FileSearchTool{FileStoreIDs: []string{"fileSearchStores/store"}},
			},
		}); err != nil {
			t.Fatal(err)
		}
		tools := body["tools"].([]any)
		if len(tools) != 4 || tools[0].(map[string]any)["googleSearch"] == nil ||
			tools[0].(map[string]any)["functionDeclarations"] != nil ||
			tools[1].(map[string]any)["urlContext"] == nil ||
			tools[2].(map[string]any)["codeExecution"] == nil ||
			tools[3].(map[string]any)["fileSearch"].(map[string]any)["fileSearchStoreNames"].([]any)[0] !=
				"fileSearchStores/store" {
			t.Fatalf("unexpected Google web-search tool: %#v", tools)
		}
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{unsupportedNativeTool{}},
		}); err == nil || !strings.Contains(err.Error(), `native tool "unsupported" is not implemented`) {
			t.Fatalf("unexpected native-tool error: %v", err)
		}
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{unsupportedNativeTool{optional: true}},
		}); err != nil {
			t.Fatalf("optional native tool should be omitted: %v", err)
		}
		var nilTool *ai.WebSearchTool
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nilTool},
		}); err == nil || !strings.Contains(err.Error(), "native tool must not be nil") {
			t.Fatalf("unexpected nil native-tool error: %v", err)
		}
		combined := ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
			Tools:       []ai.ToolDefinition{{Name: "work", Schema: map[string]any{"type": "object"}}},
		}
		if _, err := model.Request(t.Context(), nil, combined); err == nil ||
			!strings.Contains(err.Error(), "does not support function and native tools together") {
			t.Fatalf("unexpected combined-tool error: %v", err)
		}
		gemini3 := newNamedServer(t, "gemini-3-flash", handler)
		if _, err := gemini3.Request(t.Context(), nil, combined); err != nil {
			t.Fatalf("Gemini 3 combined tools failed: %v", err)
		}
		if len(body["tools"].([]any)) != 2 ||
			body["toolConfig"].(map[string]any)["includeServerSideToolInvocations"] != true {
			t.Fatalf("Gemini 3 omitted combined tools or invocation context: %#v", body)
		}
	})
	t.Run("api error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
		})
		var apiErr *google.APIError
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests || !apiErr.IsModelAPIError() {
			t.Fatalf("expected fallback-eligible APIError 429, got %v", err)
		}
		if apiErr.Error() != "google: API returned status 429: rate limited" {
			t.Fatalf("unexpected message %q", apiErr.Error())
		}
	})
	t.Run("invalid json", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("not json")) })
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected parse error")
		}
	})
	t.Run("no candidates", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"candidates":[]}`)) })
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown message type", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		if _, err := model.Request(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown request part", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{nil}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("bad assistant tool args", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "t", Args: json.RawMessage(`not json`)},
		}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unserializable tool return", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "t", Content: make(chan int)},
		}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("transport error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := google.NewModel("m", google.WithBaseURL(server.URL))
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		var transportError *ai.ModelTransportError
		if !errors.As(err, &transportError) || transportError.ModelName != "m" ||
			transportError.ProviderName != "google" || transportError.Operation != "request" {
			t.Fatalf("unexpected transport error: %v", err)
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := google.NewModel("m", google.WithBaseURL("http://[::1"))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "1000")
			_, _ = w.Write([]byte(`{"model`))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestModelName(t *testing.T) {
	model := google.NewModel("gemini-2.5-flash")
	if model.Name() != "gemini-2.5-flash" || model.ProviderName() != "google" ||
		model.ProviderURL() != "https://generativelanguage.googleapis.com/v1beta" {
		t.Fatalf("unexpected model identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
}

func TestEndToEndAgentRun(t *testing.T) {
	first := true
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]}}],
				"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"candidates": [{"content": {"parts": [{"text": "It is sunny in SF."}]}}],
			"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 6}
		}`))
	})
	agent := ai.NewAgent[struct{}, string](model)
	ai.AddSimpleTool(agent, "get_weather", func(_ context.Context, args struct {
		City string `json:"city"`
	}) (string, error) {
		return "sunny in " + args.City, nil
	})
	result, err := agent.Run(t.Context(), "weather in SF?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "It is sunny in SF." || result.Usage().Requests != 2 {
		t.Fatalf("unexpected result %+v", result)
	}
}

func TestAssistantHistoryWithText(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.TextPart{
			Content: "previous answer", ProviderName: "other",
			ProviderDetails: map[string]any{"thought_signature": "foreign"},
		},
		ai.ToolCallPart{ToolName: "t"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "previous answer" ||
		parts[0].(map[string]any)["thoughtSignature"] != nil {
		t.Fatalf("text part lost: %v", parts)
	}
	if parts[1].(map[string]any)["functionCall"] == nil {
		t.Fatalf("empty-args tool call lost: %v", parts)
	}
}

func TestMultimodalUserPrompt(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"a cat"}]}}],"usageMetadata":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "what is this?"}, ai.CachePoint{},
		ai.BinaryContent{
			Data: []byte("hi"), MediaType: "image/png", VendorMetadata: map[string]any{"start_offset": "1s"},
		},
		ai.ImageURL{URL: "https://generativelanguage.googleapis.com/v1beta/files/cat.png"},
		ai.UploadedFile{
			FileID:       "https://generativelanguage.googleapis.com/v1beta/files/report",
			ProviderName: "google", MediaType: "application/pdf",
			VendorMetadata: map[string]any{"media_resolution": "MEDIA_RESOLUTION_LOW"},
		},
	}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if inline["mimeType"] != "image/png" || inline["data"] != "aGk=" ||
		parts[1].(map[string]any)["videoMetadata"].(map[string]any)["startOffset"] != "1s" {
		t.Fatalf("unexpected inline data %v", inline)
	}
	file := parts[2].(map[string]any)["fileData"].(map[string]any)
	if file["fileUri"] != "https://generativelanguage.googleapis.com/v1beta/files/cat.png" {
		t.Fatalf("unexpected file data %v", file)
	}
	uploaded := parts[3].(map[string]any)["fileData"].(map[string]any)
	if uploaded["fileUri"] != "https://generativelanguage.googleapis.com/v1beta/files/report" ||
		uploaded["mimeType"] != "application/pdf" ||
		parts[3].(map[string]any)["mediaResolution"] != "MEDIA_RESOLUTION_LOW" {
		t.Fatalf("unexpected uploaded file data %v", uploaded)
	}
}

func TestFileURLPrompt(t *testing.T) {
	fileServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/photo.png" {
			response.Header().Set("Content-Type", "image/webp")
		} else {
			response.Header().Set("Content-Type", "application/octet-stream")
		}
		_, _ = response.Write([]byte("file"))
	}))
	defer fileServer.Close()
	var gotBody map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	})
	metadata := map[string]any{"media_resolution": "MEDIA_RESOLUTION_HIGH", "ignored": true}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.ImageURL{
			URL: fileServer.URL + "/photo.png", ForceDownload: ai.FileDownloadAllowLocal,
			VendorMetadata: metadata,
		},
		ai.AudioURL{URL: fileServer.URL + "/speech.mp3", ForceDownload: ai.FileDownloadAllowLocal},
		ai.DocumentURL{URL: fileServer.URL + "/report.pdf", ForceDownload: ai.FileDownloadAllowLocal},
	}}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	image := parts[0].(map[string]any)
	if image["inlineData"].(map[string]any)["mimeType"] != "image/webp" ||
		image["inlineData"].(map[string]any)["data"] != "ZmlsZQ==" ||
		image["mediaResolution"] != "MEDIA_RESOLUTION_HIGH" || image["videoMetadata"] != nil {
		t.Fatalf("unexpected image URL part: %#v", image)
	}
	if metadata["media_resolution"] != "MEDIA_RESOLUTION_HIGH" || metadata["ignored"] != true {
		t.Fatalf("Google mutated image metadata: %#v", metadata)
	}
	if parts[1].(map[string]any)["inlineData"].(map[string]any)["mimeType"] != "audio/mpeg" ||
		parts[2].(map[string]any)["inlineData"].(map[string]any)["mimeType"] != "application/pdf" {
		t.Fatalf("unexpected audio or document URL parts: %#v", parts)
	}
	for name, content := range map[string]ai.UserContent{
		"blocked image": ai.ImageURL{URL: fileServer.URL + "/photo.png"},
		"invalid audio mode": ai.AudioURL{
			URL: "https://example.com/audio.mp3", ForceDownload: "invalid",
		},
		"unknown document media": ai.DocumentURL{URL: "https://example.com/document"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{content}},
			}}}, ai.ModelRequestParams{})
			if err == nil {
				t.Fatal("invalid Google file URL succeeded")
			}
		})
	}
}

func TestVideoURLPrompt(t *testing.T) {
	videoServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/typed.mp4" {
			response.Header().Set("Content-Type", "video/quicktime")
		} else {
			response.Header().Set("Content-Type", "application/octet-stream")
		}
		_, _ = response.Write([]byte("video"))
	}))
	defer videoServer.Close()
	var gotBody map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	})
	metadata := map[string]any{
		"start_offset": "1s", "end_offset": "2s", "media_resolution": "MEDIA_RESOLUTION_HIGH",
	}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.VideoURL{URL: "https://youtu.be/example", ForceDownload: ai.FileDownloadSafe, VendorMetadata: metadata},
		ai.VideoURL{URL: "https://generativelanguage.googleapis.com/v1beta/files/video", MediaType: "video/mp4"},
		ai.VideoURL{URL: videoServer.URL + "/clip.webm", ForceDownload: ai.FileDownloadAllowLocal},
		ai.VideoURL{URL: videoServer.URL + "/typed.mp4", ForceDownload: ai.FileDownloadAllowLocal},
	}}}}}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil || response.Text() != "done" {
		t.Fatalf("unexpected Google video response=%+v err=%v", response, err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	first := parts[0].(map[string]any)
	if metadata["start_offset"] != "1s" || metadata["startOffset"] != nil || metadata["media_resolution"] == nil {
		t.Fatalf("Google mutated video metadata: %#v", metadata)
	}
	if first["fileData"].(map[string]any)["fileUri"] != "https://youtu.be/example" ||
		first["mediaResolution"] != "MEDIA_RESOLUTION_HIGH" ||
		first["videoMetadata"].(map[string]any)["startOffset"] != "1s" ||
		first["videoMetadata"].(map[string]any)["endOffset"] != "2s" {
		t.Fatalf("unexpected YouTube part: %#v", first)
	}
	if parts[1].(map[string]any)["fileData"].(map[string]any)["fileUri"] !=
		"https://generativelanguage.googleapis.com/v1beta/files/video" {
		t.Fatalf("unexpected Files API video: %#v", parts[1])
	}
	inline := parts[2].(map[string]any)["inlineData"].(map[string]any)
	typed := parts[3].(map[string]any)["inlineData"].(map[string]any)
	if inline["mimeType"] != "video/webm" || inline["data"] != "dmlkZW8=" ||
		typed["mimeType"] != "video/quicktime" {
		t.Fatalf("unexpected inline videos: fallback=%#v typed=%#v", inline, typed)
	}

	for name, video := range map[string]ai.VideoURL{
		"blocked local": {URL: videoServer.URL + "/clip.webm"},
		"unknown media": {URL: "https://example.com/video"},
		"invalid mode":  {URL: "https://example.com/video.mp4", ForceDownload: "invalid"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{video}},
			}}}, ai.ModelRequestParams{})
			if err == nil {
				t.Fatal("invalid Google video request succeeded")
			}
		})
	}
}

func TestVertexVideoURLPrompt(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	}))
	defer server.Close()
	model := google.NewModel("gemini-3-pro", google.WithProvider(google.ProviderConfig{
		Transport: google.TransportVertexAI, Name: "google-cloud", BaseURL: server.URL,
		APIKey: "key", HTTPClient: server.Client(),
	}))
	_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{
			ai.VideoURL{URL: "gs://bucket/video.mp4", ForceDownload: ai.FileDownloadSafe},
			ai.DocumentURL{URL: "https://example.com/report.pdf"},
		}},
	}}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	video := parts[0].(map[string]any)["fileData"].(map[string]any)
	document := parts[1].(map[string]any)["fileData"].(map[string]any)
	if video["fileUri"] != "gs://bucket/video.mp4" || video["mimeType"] != "video/mp4" ||
		document["fileUri"] != "https://example.com/report.pdf" || document["mimeType"] != "application/pdf" {
		t.Fatalf("unexpected Vertex files: video=%#v document=%#v", video, document)
	}
}

func TestMultimodalUnknownContent(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
	for name, item := range map[string]ai.UserContent{
		"unknown": nil,
		"foreign upload": ai.UploadedFile{
			FileID: "https://example.com/file", ProviderName: "openai", MediaType: "application/pdf",
		},
		"invalid Gemini upload": ai.UploadedFile{
			FileID: "file", ProviderName: "google", MediaType: "application/pdf",
		},
		"invalid cache point": ai.CachePoint{TTL: "1d"},
	} {
		t.Run(name, func(t *testing.T) {
			msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{item}},
			}}}
			if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestNativeJSONOutputMode(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"{}"}]}}],"usageMetadata":{}}`))
	})
	params := ai.ModelRequestParams{
		AllowText: true,
		OutputSchema: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
		},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	gen := gotBody["generationConfig"].(map[string]any)
	if gen["responseMimeType"] != "application/json" {
		t.Fatalf("unexpected generation config %v", gen)
	}
	schema := gen["responseJsonSchema"].(map[string]any)
	if schema["additionalProperties"] != false {
		t.Fatal("additionalProperties should be preserved in Gemini JSON Schema")
	}
	params.OutputMode = ai.OutputModePrompted
	gotBody = nil
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if config, ok := gotBody["generationConfig"].(map[string]any); ok && config["responseMimeType"] != nil {
		t.Fatalf("prompted output enabled native response schema: %+v", gotBody)
	}
}

func TestStrictToolModes(t *testing.T) {
	for name, strict := range map[string]struct {
		value bool
		mode  string
	}{
		"enabled":  {value: true, mode: "VALIDATED"},
		"disabled": {value: false, mode: "AUTO"},
	} {
		t.Run(name, func(t *testing.T) {
			var gotBody map[string]any
			model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
			})
			value := strict.value
			params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
				Name: "search", Schema: map[string]any{"type": "object"}, Strict: &value,
			}}}
			if _, err := model.Request(t.Context(), nil, params); err != nil {
				t.Fatal(err)
			}
			mode := gotBody["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)["mode"]
			if mode != strict.mode {
				t.Fatalf("expected %s, got %v", strict.mode, mode)
			}
		})
	}
}

func TestStrictToolProfileDefaults(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		extra     []google.Option
		mode      string
	}{
		{name: "Gemini 2.5", modelName: "gemini-2.5-flash", mode: "VALIDATED"},
		{name: "Gemini 3", modelName: "gemini-3-pro", mode: "VALIDATED"},
		{name: "Gemini 2.0", modelName: "gemini-2.0-flash", mode: "AUTO"},
		{name: "image model", modelName: "gemini-3-pro-image-preview", mode: "AUTO"},
		{name: "disabled override", modelName: "gemini-2.5-flash", extra: []google.Option{google.WithStrictToolSupport(false)}, mode: "AUTO"},
		{name: "enabled override", modelName: "proxy-model", extra: []google.Option{google.WithStrictToolSupport(true)}, mode: "VALIDATED"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var gotBody map[string]any
			model := newNamedServer(t, test.modelName, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
			}, test.extra...)
			params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
				Name: "search", Schema: map[string]any{"type": "object"},
			}}}
			if _, err := model.Request(t.Context(), nil, params); err != nil {
				t.Fatal(err)
			}
			mode := gotBody["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)["mode"]
			if mode != test.mode {
				t.Fatalf("expected %s, got %v", test.mode, mode)
			}
		})
	}
}

func TestGeminiJSONSchemaTransform(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
	})
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
		Name: "inspect", Schema: map[string]any{
			"$schema": "https://json-schema.org/draft/2020-12/schema",
			"title":   "Input",
			"type":    "object",
			"properties": map[string]any{
				"title":   map[string]any{"type": "string", "title": "Display"},
				"when":    map[string]any{"type": "string", "format": "date-time", "description": "Start"},
				"empty":   map[string]any{"type": "string", "format": "email"},
				"id":      map[string]any{"const": "fixed", "examples": []any{"fixed"}},
				"typed":   map[string]any{"type": "string", "const": "fixed"},
				"unknown": map[string]any{"const": nil},
				"choice": map[string]any{"anyOf": []any{
					map[string]any{"type": "string", "title": "Text"}, map[string]any{"type": "integer"},
				}},
				"flag":  map[string]any{"const": true},
				"count": map[string]any{"const": 2},
				"ratio": map[string]any{"const": 1.5},
				"items": map[string]any{"type": "array", "items": map[string]any{
					"type": "number", "exclusiveMinimum": 0,
				}},
			},
			"additionalProperties": false,
			"discriminator":        map[string]any{"propertyName": "type"},
		},
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	declaration := gotBody["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	schema := declaration["parametersJsonSchema"].(map[string]any)
	if schema["additionalProperties"] != false || schema["title"] != nil || schema["$schema"] != nil || schema["discriminator"] != nil {
		t.Fatalf("unexpected root schema %v", schema)
	}
	properties := schema["properties"].(map[string]any)
	if properties["title"].(map[string]any)["title"] != nil {
		t.Fatalf("property named title was not preserved correctly: %v", properties)
	}
	when := properties["when"].(map[string]any)
	if when["format"] != nil || when["description"] != "Start (format: date-time)" {
		t.Fatalf("unexpected formatted string schema %v", when)
	}
	empty := properties["empty"].(map[string]any)
	if empty["format"] != nil || empty["description"] != "Format: email" {
		t.Fatalf("unexpected empty-description format schema %v", empty)
	}
	choice := properties["choice"].(map[string]any)["anyOf"].([]any)
	if choice[0].(map[string]any)["title"] != nil {
		t.Fatalf("nested slice schema was not transformed: %v", choice)
	}
	for name, wantType := range map[string]string{"id": "string", "flag": "boolean", "count": "integer", "ratio": "number"} {
		property := properties[name].(map[string]any)
		if property["type"] != wantType || property["const"] != nil || property["examples"] != nil {
			t.Fatalf("unexpected const schema for %s: %v", name, property)
		}
	}
	items := properties["items"].(map[string]any)["items"].(map[string]any)
	if items["exclusiveMinimum"] != nil {
		t.Fatalf("exclusive minimum should be removed: %v", items)
	}
}

func TestAnyLooseToolDisablesGoogleStrictMode(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
	})
	yes, no := true, false
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{
		{Name: "strict", Schema: map[string]any{"type": "object"}, Strict: &yes},
		{Name: "loose", Schema: map[string]any{"type": "object"}, Strict: &no},
	}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	mode := gotBody["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)["mode"]
	if mode != "AUTO" {
		t.Fatalf("expected AUTO, got %v", mode)
	}
}
