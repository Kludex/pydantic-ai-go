package xai_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	imagexai "github.com/Kludex/pydantic-ai-go/ai/images/xai"
	modelopenai "github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

var png = []byte("\x89PNG\r\n\x1a\nimage")

func TestGenerateBatchSettingsAndIdentity(t *testing.T) {
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/images/generations" || request.URL.Query().Get("region") != "west" {
			t.Errorf("unexpected URL: %s", request.URL)
		}
		if request.Header.Get("Authorization") != "Bearer secret" || request.Header.Get("X-Test") != "call" ||
			request.Header.Get("X-Dynamic") != "yes" {
			t.Errorf("unexpected headers: %#v", request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(response, `{
			"id":"response","model":"grok-imagine-image-quality","cost_usd":0.04,
			"data":[
				{"base64":"data:image/png;base64,`+base64.StdEncoding.EncodeToString(png)+`","respect_moderation":true},
				{"b64_json":"`+base64.StdEncoding.EncodeToString(png)+`"}
			],
			"usage":{"prompt_tokens":4,"completion_tokens":8,"reasoning_tokens":2,
				"prompt_text_tokens":3,"prompt_image_tokens":1,"cached_prompt_text_tokens":1,
				"cost_in_usd_ticks":400000}
		}`)
	}))
	defer server.Close()
	n := 2
	settings, err := (imagexai.Settings{
		Common: images.Settings{
			Dimensions:   &images.Dimensions{Width: 1280, Height: 720},
			ExtraHeaders: map[string]string{"X-Test": "call"}, ExtraBody: map[string]any{"custom": true},
		},
		N: &n, User: "user",
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithProvider(modelopenai.ProviderConfig{
		Name: "xai-custom", BaseURL: server.URL + "/v1/", APIKey: "secret", HTTPClient: server.Client(),
		Headers: http.Header{"X-Test": {"provider"}}, Query: url.Values{"region": {"west"}},
		PrepareRequest: func(request *http.Request) error { request.Header.Set("X-Dynamic", "yes"); return nil },
	}))
	result, err := model.Generate(t.Context(), "draw", []images.Input{
		ai.UploadedFile{FileID: "file-1", ProviderName: "xai-custom", MediaType: "image/png"},
		ai.BinaryContent{Data: png, MediaType: "image/png"},
	}, settings)
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "grok-imagine-image" || model.ProviderName() != "xai-custom" || model.ProviderURL() != server.URL+"/v1" ||
		len(result.Images) != 2 || result.ModelName != "grok-imagine-image-quality" || result.ProviderResponseID != "response" ||
		result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 8 || result.Usage.ReasoningTokens != 2 ||
		result.Usage.CacheReadTokens != 1 || result.Usage.CostUSD != nil || result.ProviderDetails["cost_usd"].(float64) != 0.04 ||
		result.ProviderDetails["cost_in_usd_ticks"].(int) != 400000 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if requestBody["n"].(float64) != 2 || requestBody["aspect_ratio"] != "16:9" || requestBody["resolution"] != "1k" ||
		len(requestBody["image_file_ids"].([]any)) != 1 || len(requestBody["image_urls"].([]any)) != 1 ||
		requestBody["custom"] != true {
		t.Fatalf("unexpected request: %#v", requestBody)
	}
}

func TestModerationInputsAndWarnings(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{"images":[
			{"base64":"data:image/png;base64,`+base64.StdEncoding.EncodeToString(png)+`","respect_moderation":false},
			{"image":"`+base64.StdEncoding.EncodeToString(png)+`","respect_moderation":true,"provider_details":{"value":1}}
		]}`)
	}))
	defer server.Close()
	settings, err := (imagexai.Settings{
		Common:      images.Settings{AspectRatio: images.AspectRatio16To9},
		AspectRatio: images.AspectRatio1To1, Resolution: imagexai.Resolution2K,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithBaseURL(server.URL), imagexai.WithHTTPClient(server.Client()))
	result, err := model.Generate(t.Context(), "draw", nil, settings)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 || len(result.Warnings) != 1 ||
		result.ProviderDetails["moderated_image_indices"].([]int)[0] != 0 {
		t.Fatalf("unexpected moderated result: %#v", result)
	}
	_, err = model.Generate(t.Context(), "edit", []images.Input{
		ai.ImageURL{URL: "https://example.com/image.png"},
		ai.UploadedFile{FileID: "file", ProviderName: "xai", MediaType: "image/png"},
	}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "uploaded files before") {
		t.Fatalf("unexpected order error: %v", err)
	}
	_, err = model.Generate(t.Context(), "edit", []images.Input{
		ai.UploadedFile{FileID: "file", ProviderName: "other", MediaType: "image/png"},
	}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "belongs") {
		t.Fatalf("unexpected provider error: %v", err)
	}
}

func TestSingleAndDownloadedReferences(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "image/png")
		_, _ = response.Write(png)
	}))
	defer imageServer.Close()
	var bodies []map[string]any
	api := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = io.WriteString(response, `{"data":[{"b64_json":"`+base64.StdEncoding.EncodeToString(png)+`"}]}`)
	}))
	defer api.Close()
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithBaseURL(api.URL), imagexai.WithHTTPClient(api.Client()))
	if _, err := model.Generate(t.Context(), "edit", []images.Input{ai.UploadedFile{
		FileID: "file", ProviderName: "xai", MediaType: "image/png",
	}}, images.Settings{}); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: imageServer.URL, ForceDownload: ai.FileDownloadAllowLocal,
	}}, images.Settings{}); err != nil {
		t.Fatal(err)
	}
	if bodies[0]["image_file_id"] != "file" || !strings.HasPrefix(bodies[1]["image_url"].(string), "data:image/png;base64,") {
		t.Fatalf("unexpected references: %#v", bodies)
	}
}

func TestProviderConfigurationAndFailures(t *testing.T) {
	if capturePanic(func() { imagexai.NewModel("model", imagexai.WithProvider(modelopenai.ProviderConfig{})) }) == nil {
		t.Fatal("empty provider URL accepted")
	}
	t.Setenv("XAI_API_KEY", "environment")
	model := imagexai.NewModel("model", imagexai.WithAPIKey("key"), imagexai.WithDefaultSettings(images.Settings{
		AspectRatio: images.AspectRatio1To1,
	}))
	if model.DefaultSettings().AspectRatio != images.AspectRatio1To1 {
		t.Fatal("default settings were lost")
	}
	prepareErr := errors.New("prepare")
	model = imagexai.NewModel("model", imagexai.WithProvider(modelopenai.ProviderConfig{
		BaseURL: "https://example.com", PrepareRequest: func(*http.Request) error { return prepareErr },
	}))
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{}); !errors.Is(err, prepareErr) {
		t.Fatalf("unexpected prepare error: %v", err)
	}
	for _, providerSettings := range []map[string]any{
		{"xai_n": "bad"}, {"xai_user": 1}, {"xai_aspect_ratio": "1:1"}, {"xai_resolution": "1k"},
	} {
		if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{ProviderSettings: providerSettings}); err == nil {
			t.Fatalf("invalid settings accepted: %#v", providerSettings)
		}
	}
	if _, err := imagexai.NewModel("model", imagexai.WithBaseURL("://bad")).Generate(t.Context(), "draw", nil, images.Settings{}); err == nil {
		t.Fatal("invalid endpoint accepted")
	}
	transport := imagexai.NewModel("model", imagexai.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("transport")
	})}))
	if _, err := transport.Generate(t.Context(), "draw", nil, images.Settings{}); err == nil || !strings.Contains(err.Error(), "transport") {
		t.Fatalf("unexpected transport error: %v", err)
	}
}

func TestValidationGeometryAndErrors(t *testing.T) {
	zero := 0
	if _, err := (imagexai.Settings{N: &zero}).Build(); err == nil {
		t.Fatal("invalid count accepted")
	}
	if _, err := (imagexai.Settings{Common: images.Settings{ProviderSettings: map[string]any{"xai_n": 1}}}).Build(); err == nil {
		t.Fatal("reserved setting accepted")
	}
	model := imagexai.NewModel("grok-imagine-image")
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{AspectRatio: images.AspectRatio1To4}); err == nil {
		t.Fatal("unsupported ratio accepted")
	}
	bad := images.Dimensions{Width: 1, Height: 1}
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{Dimensions: &bad}); err == nil {
		t.Fatal("unsupported dimensions accepted")
	}
	if _, err := imagexai.NewModel("grok-imagine-image-2.0").Generate(t.Context(), "draw", nil, images.Settings{Dimensions: &images.Dimensions{Width: 1024, Height: 1024}}); err == nil {
		t.Fatal("unknown dimensions mapping accepted")
	}

	cases := []struct {
		body     string
		filtered bool
	}{
		{body: `{}`}, {body: `{"data":[{"base64":"%%%"}]}`},
		{body: `{"data":[{"base64":"` + base64.StdEncoding.EncodeToString([]byte("bad")) + `"}]}`},
		{body: `{"data":[{"respect_moderation":false}]}`, filtered: true},
	}
	for _, test := range cases {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) { _, _ = io.WriteString(response, test.body) }))
		_, err := imagexai.NewModel("grok-imagine-image", imagexai.WithBaseURL(server.URL), imagexai.WithHTTPClient(server.Client())).Generate(t.Context(), "draw", nil, images.Settings{})
		server.Close()
		var filtered *ai.ContentFilterError
		var unexpected *ai.UnexpectedModelBehaviorError
		if test.filtered && !errors.As(err, &filtered) || !test.filtered && !errors.As(err, &unexpected) {
			t.Fatalf("unexpected error for %s: %v", test.body, err)
		}
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(response, "bad")
	}))
	defer server.Close()
	_, err := imagexai.NewModel("grok-imagine-image", imagexai.WithBaseURL(server.URL), imagexai.WithHTTPClient(server.Client())).Generate(t.Context(), "draw", nil, images.Settings{})
	var apiError *imagexai.APIError
	if !errors.As(err, &apiError) || !apiError.IsModelAPIError() {
		t.Fatalf("unexpected API error: %v", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
