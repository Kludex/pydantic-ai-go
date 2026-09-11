package google_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	imagegoogle "github.com/Kludex/pydantic-ai-go/ai/images/google"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

var jpeg = []byte("\xff\xd8\xffimage")

func TestGenerateEditSettingsAndIdentity(t *testing.T) {
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1beta/models/gemini-3.1-flash-image:generateContent" {
			t.Errorf("unexpected URL: %s", request.URL)
		}
		if request.Header.Get("x-goog-api-key") != "secret" || request.Header.Get("X-Test") != "call" ||
			request.Header.Get("X-Dynamic") != "yes" {
			t.Errorf("unexpected headers: %#v", request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(response, `{
			"responseId":"response","modelVersion":"gemini-image-version","createTime":"2026-01-01T00:00:00Z",
			"candidates":[{"finishReason":"STOP","safetyRatings":[{"category":"SAFE"}],"content":{"parts":[
				{"text":"ignored"},{"thought":true,"inlineData":{"mimeType":"image/png","data":"AA=="}},
				{"thoughtSignature":"signature","inlineData":{"mimeType":"image/jpeg","data":"`+base64.StdEncoding.EncodeToString(jpeg)+`"}}
			]}}],
			"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":8,"thoughtsTokenCount":2,
				"cachedContentTokenCount":1,"totalTokenCount":12,"trafficType":"ON_DEMAND"}
		}`)
	}))
	defer server.Close()
	quality := 80
	settings, err := (imagegoogle.Settings{
		Common: images.Settings{
			Dimensions:   &images.Dimensions{Width: 688, Height: 384},
			ExtraHeaders: map[string]string{"X-Test": "call"}, ExtraBody: map[string]any{"custom": true},
		},
		OutputMIMEType: "image/jpeg", OutputCompressionQuality: &quality,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := imagegoogle.NewModel("gemini-3.1-flash-image", imagegoogle.WithProvider(modelgoogle.ProviderConfig{
		Transport: modelgoogle.TransportGeminiAPI, Name: "google-custom", BaseURL: server.URL + "/v1beta/",
		APIKey: "secret", HTTPClient: server.Client(),
		PrepareRequest: func(request *http.Request) error { request.Header.Set("X-Dynamic", "yes"); return nil },
	}))
	result, err := model.Generate(t.Context(), "draw", []images.Input{
		ai.BinaryContent{Data: []byte("reference"), MediaType: "image/png", VendorMetadata: map[string]any{"media_resolution": "HIGH"}},
	}, settings)
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "gemini-3.1-flash-image" || model.ProviderName() != "google-custom" ||
		model.ProviderURL() != server.URL+"/v1beta" || model.Transport() != modelgoogle.TransportGeminiAPI ||
		result.ModelName != "gemini-image-version" || result.ProviderResponseID != "response" ||
		len(result.Images) != 1 || result.Images[0].OutputFormat != "jpeg" ||
		result.Images[0].ProviderDetails["has_thought_signature"] != true || result.Usage.InputTokens != 4 ||
		result.Usage.OutputTokens != 10 || result.Usage.ReasoningTokens != 2 || result.Usage.CacheReadTokens != 1 ||
		result.Usage.Details["thoughts_tokens"] != 2 || result.Usage.Details["cached_content_tokens"] != 1 {
		t.Fatalf("unexpected result: %#v", result)
	}
	config := requestBody["generationConfig"].(map[string]any)
	imageConfig := config["imageConfig"].(map[string]any)
	if imageConfig["aspectRatio"] != "16:9" || imageConfig["imageSize"] != "512" ||
		imageConfig["outputMimeType"] != "image/jpeg" || requestBody["custom"] != true {
		t.Fatalf("unexpected request: %#v", requestBody)
	}
}

func TestInputsVertexAndGeometryWarnings(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{"candidates":[{"content":{"parts":[{"inlineData":{"data":"`+
			base64.StdEncoding.EncodeToString([]byte("\x89PNGimage"))+`"}}]}}]}`)
	}))
	defer server.Close()
	providerRatio := "1:1"
	settings, err := (imagegoogle.Settings{
		Common: images.Settings{AspectRatio: images.AspectRatio16To9}, AspectRatio: providerRatio,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := imagegoogle.NewModel("gemini-3.1-flash-lite-image", imagegoogle.WithProvider(modelgoogle.ProviderConfig{
		Transport: modelgoogle.TransportGeminiAPI, Name: "google", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	result, err := model.Generate(t.Context(), "draw", []images.Input{
		ai.UploadedFile{FileID: "https://generativelanguage.googleapis.com/v1beta/files/a", ProviderName: "google-gla", MediaType: "image/png"},
	}, settings)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Warnings) != 1 || !strings.Contains(result.Warnings[0], "aspect_ratio") || result.Images[0].Content.MediaType != "image/png" {
		t.Fatalf("unexpected result: %#v", result)
	}
	vertex := imagegoogle.NewModel("gemini-3-pro-image", imagegoogle.WithProvider(modelgoogle.ProviderConfig{
		Transport: modelgoogle.TransportVertexAI, Name: "google-cloud", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	_, err = vertex.Generate(t.Context(), "draw", []images.Input{
		ai.UploadedFile{FileID: "gs://bucket/image.png", ProviderName: "google-cloud", MediaType: "image/png"},
	}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "does not accept uploaded") {
		t.Fatalf("unexpected Vertex error: %v", err)
	}
	bad := images.Dimensions{Width: 1, Height: 1}
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{Dimensions: &bad}); err == nil {
		t.Fatal("unsupported dimensions accepted")
	}
}

func TestDownloadAndFilesInputs(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "image/jpeg")
		_, _ = response.Write(jpeg)
	}))
	defer imageServer.Close()
	api := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
		if len(parts) != 2 || parts[1].(map[string]any)["inlineData"] == nil {
			t.Errorf("unexpected body: %#v", body)
		}
		_, _ = io.WriteString(response, `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/jpeg","data":"`+base64.StdEncoding.EncodeToString(jpeg)+`"}}]}}]}`)
	}))
	defer api.Close()
	model := imagegoogle.NewModel("gemini-2.5-flash-image", imagegoogle.WithBaseURL(api.URL), imagegoogle.WithHTTPClient(api.Client()))
	_, err := model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: imageServer.URL + "/image", ForceDownload: ai.FileDownloadAllowLocal,
	}}, images.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: "https://generativelanguage.googleapis.com/v1beta/files/no-extension",
	}}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "explicit media type") {
		t.Fatalf("unexpected file URL error: %v", err)
	}
}

func TestDetailedUsageAccounting(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{
			"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/jpeg","data":"`+
			base64.StdEncoding.EncodeToString(jpeg)+`"}}]}}],
			"usageMetadata":{
				"promptTokenCount":3,"candidatesTokenCount":5,"cachedContentTokenCount":2,
				"thoughtsTokenCount":2,"toolUsePromptTokenCount":4,
				"promptTokensDetails":[{"modality":"TEXT","tokenCount":1},{"modality":"IMAGE","tokenCount":2},{"modality":"AUDIO","tokenCount":1}],
				"cacheTokensDetails":[{"modality":"TEXT","tokenCount":2},{"modality":"AUDIO","tokenCount":1}],
				"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":5},{"modality":"AUDIO","tokenCount":2}],
				"toolUsePromptTokensDetails":[{"modality":"TEXT","tokenCount":4}]
			}
		}`)
	}))
	defer server.Close()
	result, err := imagegoogle.NewModel(
		"gemini-2.5-flash-image", imagegoogle.WithBaseURL(server.URL), imagegoogle.WithHTTPClient(server.Client()),
	).Generate(t.Context(), "draw", nil, images.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	usage := result.Usage
	if usage.InputTokens != 7 || usage.OutputTokens != 7 || usage.CacheReadTokens != 2 || usage.ReasoningTokens != 2 ||
		usage.InputAudioTokens != 1 || usage.CacheAudioReadTokens != 1 || usage.OutputAudioTokens != 2 ||
		usage.Details["text_prompt_tokens"] != 1 ||
		usage.Details["image_prompt_tokens"] != 2 || usage.Details["text_cache_tokens"] != 2 ||
		usage.Details["audio_cache_tokens"] != 1 || usage.Details["image_candidates_tokens"] != 5 ||
		usage.Details["audio_candidates_tokens"] != 2 || usage.Details["text_tool_use_prompt_tokens"] != 4 ||
		usage.Details["tool_use_prompt_tokens"] != 4 {
		t.Fatalf("unexpected detailed usage: %#v", usage)
	}
}

func TestProviderConfigurationAndSettingsTypes(t *testing.T) {
	if capturePanic(func() {
		imagegoogle.NewModel("model", imagegoogle.WithProvider(modelgoogle.ProviderConfig{
			Transport: "bad",
		}))
	}) == nil {
		t.Fatal("invalid transport accepted")
	}
	t.Setenv("GOOGLE_API_KEY", "environment")
	model := imagegoogle.NewModel("model", imagegoogle.WithAPIKey("key"), imagegoogle.WithDefaultSettings(images.Settings{
		AspectRatio: images.AspectRatio1To1,
	}))
	if model.DefaultSettings().AspectRatio != images.AspectRatio1To1 {
		t.Fatal("default settings were lost")
	}
	vertex, err := imagegoogle.NewVertexModel("model", modelgoogle.VertexConfig{
		APIKey: "key", Endpoint: "https://example.com",
	})
	if err != nil || vertex.ProviderName() != "google-cloud" {
		t.Fatalf("unexpected Vertex model: %#v %v", vertex, err)
	}
	for _, providerSettings := range []map[string]any{
		{"google_image_aspect_ratio": 1}, {"google_image_size": "1K"},
		{"google_image_output_mime_type": 1}, {"google_image_output_compression_quality": "bad"},
	} {
		if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{ProviderSettings: providerSettings}); err == nil {
			t.Fatalf("invalid settings accepted: %#v", providerSettings)
		}
	}
}

func TestGeometryInputAndRequestEdges(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "application/octet-stream")
		if request.URL.Path == "/text" {
			response.Header().Set("Content-Type", "text/plain")
			_, _ = response.Write([]byte("text"))
			return
		}
		if request.URL.Path == "/unknown" {
			_, _ = response.Write([]byte("unknown"))
			return
		}
		_, _ = response.Write(jpeg)
	}))
	defer imageServer.Close()
	api := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{"candidates":[{"content":{"parts":[{"inlineData":{"data":"`+
			base64.StdEncoding.EncodeToString([]byte("unknown"))+`"}}]}}],"usageMetadata":{}}`)
	}))
	defer api.Close()
	newModel := func(name string) *imagegoogle.Model {
		return imagegoogle.NewModel(name, imagegoogle.WithBaseURL(api.URL), imagegoogle.WithHTTPClient(api.Client()))
	}
	for _, call := range []struct {
		name     string
		settings images.Settings
	}{
		{"gemini-2.5-flash-image", images.Settings{Dimensions: &images.Dimensions{Width: 1024, Height: 1024}}},
		{"gemini-3-pro-image", images.Settings{AspectRatio: images.AspectRatio1To4}},
		{"gemini-3.1-flash-lite-image", images.Settings{AspectRatio: images.AspectRatio1To8}},
		{"future-image", images.Settings{AspectRatio: images.AspectRatio1To2}},
	} {
		result, err := newModel(call.name).Generate(t.Context(), "draw", nil, call.settings)
		if err != nil || result.Image().MediaType != "image/png" || result.ModelName != call.name || result.Usage.Details != nil {
			t.Fatalf("unexpected %s geometry result: %#v %v", call.name, result, err)
		}
	}
	providerSettings, err := (imagegoogle.Settings{
		Common:      images.Settings{Dimensions: &images.Dimensions{Width: 1024, Height: 1024}},
		AspectRatio: "16:9", ImageSize: imagegoogle.ImageSize4K,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err := newModel("gemini-2.5-flash-image").Generate(t.Context(), "draw", nil, providerSettings)
	if err != nil || len(result.Warnings) != 1 {
		t.Fatalf("unexpected geometry conflict: %#v %v", result, err)
	}

	model := newModel("gemini-2.5-flash-image")
	for _, input := range []images.Input{
		ai.UploadedFile{FileID: "file", ProviderName: "other", MediaType: "image/png"},
		ai.UploadedFile{FileID: "file", ProviderName: "google", MediaType: "image/png"},
		ai.ImageURL{URL: "https://example.com/image.png", ForceDownload: "invalid"},
		ai.ImageURL{URL: imageServer.URL + "/image", ForceDownload: ai.FileDownloadSafe},
		ai.ImageURL{URL: imageServer.URL + "/text", ForceDownload: ai.FileDownloadAllowLocal},
		ai.ImageURL{URL: imageServer.URL + "/unknown", ForceDownload: ai.FileDownloadAllowLocal},
	} {
		if _, err := model.Generate(t.Context(), "edit", []images.Input{input}, images.Settings{}); err == nil {
			t.Fatalf("invalid input accepted: %#v", input)
		}
	}
	if _, err := model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: "https://generativelanguage.googleapis.com/v1beta/files/file", MediaType: "image/png",
	}}, images.Settings{}); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: imageServer.URL + "/image", MediaType: "image/png", ForceDownload: ai.FileDownloadAllowLocal,
	}}, images.Settings{}); err != nil {
		t.Fatal(err)
	}

	for _, settings := range []images.Settings{
		{ExtraBody: map[string]any{"contents": true}},
		{ExtraBody: map[string]any{"custom": make(chan int)}},
	} {
		if _, err := model.Generate(t.Context(), "draw", nil, settings); err == nil {
			t.Fatalf("invalid request settings accepted: %#v", settings)
		}
	}
	if _, err := imagegoogle.NewModel("model", imagegoogle.WithBaseURL("://bad")).Generate(
		t.Context(), "draw", nil, images.Settings{},
	); err == nil {
		t.Fatal("invalid endpoint accepted")
	}
	prepareErr := errors.New("prepare")
	prepared := imagegoogle.NewModel("model", imagegoogle.WithProvider(modelgoogle.ProviderConfig{
		Transport: modelgoogle.TransportGeminiAPI, BaseURL: api.URL, HTTPClient: api.Client(),
		PrepareRequest: func(*http.Request) error { return prepareErr },
	}))
	if _, err := prepared.Generate(t.Context(), "draw", nil, images.Settings{}); !errors.Is(err, prepareErr) {
		t.Fatalf("unexpected prepare error: %v", err)
	}
	if _, err := model.Generate(t.Context(), " ", nil, images.Settings{}); err == nil {
		t.Fatal("empty prompt accepted")
	}
}

func TestConstructorAndSettingsEdges(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "fallback")
	if imagegoogle.NewModel("model").Name() != "model" {
		t.Fatal("fallback environment construction failed")
	}
	_, err := imagegoogle.NewVertexModel("model", modelgoogle.VertexConfig{
		APIKey: "key", TokenProvider: func(context.Context) (string, error) { return "token", nil },
	})
	if err == nil {
		t.Fatal("invalid Vertex authentication accepted")
	}
	if _, err := (imagegoogle.Settings{Common: images.Settings{
		Dimensions: &images.Dimensions{Width: 1, Height: 1}, AspectRatio: images.AspectRatio1To1,
	}}).Build(); err == nil {
		t.Fatal("invalid common geometry accepted")
	}
	settings, err := (imagegoogle.Settings{}).Build()
	if err != nil || settings.ProviderSettings != nil {
		t.Fatalf("unexpected empty settings: %#v %v", settings, err)
	}
}

func TestTransportErrorsAreInspectable(t *testing.T) {
	connectionErr := errors.New("connect")
	model := imagegoogle.NewModel("gemini-2.5-flash-image", imagegoogle.WithHTTPClient(&http.Client{
		Transport: googleRoundTripFunc(func(*http.Request) (*http.Response, error) { return nil, connectionErr }),
	}))
	_, err := model.Generate(t.Context(), "draw", nil, images.Settings{})
	var transportError *ai.ModelTransportError
	var apiError ai.ModelAPIError
	if !errors.As(err, &transportError) || !errors.As(err, &apiError) || !errors.Is(err, connectionErr) ||
		transportError.ModelName != "gemini-2.5-flash-image" || transportError.ProviderName != "google" ||
		transportError.Operation != "request" {
		t.Fatalf("unexpected connection error: %v", err)
	}
	readErr := errors.New("read")
	model = imagegoogle.NewModel("gemini-2.5-flash-image", imagegoogle.WithHTTPClient(&http.Client{
		Transport: googleRoundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: googleErrorBody{err: readErr}}, nil
		}),
	}))
	_, err = model.Generate(t.Context(), "draw", nil, images.Settings{})
	if !errors.As(err, &transportError) || !errors.Is(err, readErr) || transportError.Operation != "read response" {
		t.Fatalf("unexpected read error: %v", err)
	}
}

func TestResponseAndSettingsErrors(t *testing.T) {
	if _, err := (imagegoogle.Settings{Common: images.Settings{ProviderSettings: map[string]any{"google_image_size": imagegoogle.ImageSize1K}}}).Build(); err == nil {
		t.Fatal("reserved setting accepted")
	}
	malformed := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{`)
	}))
	_, err := imagegoogle.NewModel(
		"gemini-2.5-flash-image", imagegoogle.WithBaseURL(malformed.URL), imagegoogle.WithHTTPClient(malformed.Client()),
	).Generate(t.Context(), "draw", nil, images.Settings{})
	malformed.Close()
	if err == nil || !strings.Contains(err.Error(), "decode response") {
		t.Fatalf("unexpected malformed response error: %v", err)
	}
	cases := []struct {
		body     string
		filtered bool
	}{
		{body: `{}`},
		{body: `{"candidates":[{"finishReason":"NO_IMAGE"}]}`},
		{body: `{"candidates":[{"finishReason":"IMAGE_SAFETY"}]}`, filtered: true},
		{body: `{"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"blocked","safetyRatings":[{"category":"SAFE"}]}}`, filtered: true},
		{body: `{"candidates":[{"content":{"parts":[{"inlineData":{"data":"%%%"}}]}}]}`},
	}
	for _, test := range cases {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) { _, _ = io.WriteString(response, test.body) }))
		_, err = imagegoogle.NewModel("gemini-2.5-flash-image", imagegoogle.WithBaseURL(server.URL), imagegoogle.WithHTTPClient(server.Client())).Generate(t.Context(), "draw", nil, images.Settings{})
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
	_, err = imagegoogle.NewModel("gemini-2.5-flash-image", imagegoogle.WithBaseURL(server.URL), imagegoogle.WithHTTPClient(server.Client())).Generate(t.Context(), "draw", nil, images.Settings{})
	var apiError *modelgoogle.APIError
	if !errors.As(err, &apiError) {
		t.Fatalf("unexpected API error: %v", err)
	}
}

type googleRoundTripFunc func(*http.Request) (*http.Response, error)

func (function googleRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type googleErrorBody struct{ err error }

func (body googleErrorBody) Read([]byte) (int, error) { return 0, body.err }
func (googleErrorBody) Close() error                  { return nil }

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
