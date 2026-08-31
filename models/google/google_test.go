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

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

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

func TestGoogleCountTokensByTransport(t *testing.T) {
	for _, transport := range []google.Transport{google.TransportGeminiAPI, google.TransportVertexAI} {
		t.Run(string(transport), func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/models/gemini:countTokens" || request.Header.Get("x-goog-api-key") != "key" {
					t.Errorf("unexpected token count request: %s headers=%v", request.URL.String(), request.Header)
				}
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = response.Write([]byte(`{"totalTokens":23}`))
			}))
			defer server.Close()
			model := google.NewModel("gemini", google.WithProvider(google.ProviderConfig{
				Transport: transport, Name: "custom", BaseURL: server.URL, APIKey: "key", HTTPClient: server.Client(),
			}))
			usage, err := model.CountTokens(t.Context(), []ai.ModelMessage{ai.ModelRequest{
				Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}},
			}}, ai.ModelRequestParams{
				Instructions: "Be brief.",
				Tools:        []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
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
			}
		}`))
	})
	_, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "blocked", struct{}{})
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) || filtered.Response().FinishReason != ai.FinishReasonContentFilter ||
		filtered.Response().ProviderDetails["block_reason"] != "PROHIBITED_CONTENT" ||
		filtered.Response().ProviderDetails["block_reason_message"] != "The prompt was blocked." {
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
		resp.ProviderDetails["avg_logprobs"] != -0.25 || resp.ProviderDetails["logprobs"] == nil {
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

func TestErrors(t *testing.T) {
	t.Run("native tools", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
		}); err == nil || !strings.Contains(err.Error(), `native tool "web_search" is not implemented`) {
			t.Fatalf("unexpected native-tool error: %v", err)
		}
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{Optional: true}},
		}); err != nil {
			t.Fatalf("optional native tool should be omitted: %v", err)
		}
		var nilTool *ai.WebSearchTool
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nilTool},
		}); err == nil || !strings.Contains(err.Error(), "native tool must not be nil") {
			t.Fatalf("unexpected nil native-tool error: %v", err)
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
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
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
		ai.TextContent{Text: "what is this?"},
		ai.BinaryContent{Data: []byte("hi"), MediaType: "image/png"},
		ai.ImageURL{URL: "https://example.com/cat.png"},
	}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if inline["mimeType"] != "image/png" || inline["data"] != "aGk=" {
		t.Fatalf("unexpected inline data %v", inline)
	}
	file := parts[2].(map[string]any)["fileData"].(map[string]any)
	if file["fileUri"] != "https://example.com/cat.png" {
		t.Fatalf("unexpected file data %v", file)
	}
}

func TestMultimodalUnknownContent(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{nil}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected error")
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
