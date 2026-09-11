package openai_test

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
	imageopenai "github.com/Kludex/pydantic-ai-go/ai/images/openai"
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
			"id":"response","model":"gpt-image-2","created":123,"size":"1280x720","quality":"high",
			"data":[{"b64_json":"`+base64.StdEncoding.EncodeToString(png)+`","revised_prompt":"better"},
			{"b64_json":"`+base64.StdEncoding.EncodeToString(png)+`"}],
			"usage":{"input_tokens":4,"output_tokens":8,"total_tokens":12,
			"input_tokens_details":{"text_tokens":3,"image_tokens":1},
			"output_tokens_details":{"text_tokens":0,"image_tokens":8}}
		}`)
	}))
	defer server.Close()
	n := 2
	compression := 80
	settings, err := (imageopenai.Settings{
		Common: images.Settings{
			AspectRatio: images.AspectRatio16To9, ExtraHeaders: map[string]string{"X-Test": "call"},
			ExtraBody: map[string]any{"custom": true},
		},
		N: &n, OutputFormat: imageopenai.OutputFormatPNG, Quality: imageopenai.QualityHigh,
		Background: imageopenai.BackgroundOpaque, Moderation: imageopenai.ModerationLow,
		OutputCompression: &compression,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := imageopenai.NewModel("gpt-image-2", imageopenai.WithProvider(modelopenai.ProviderConfig{
		Name: "compatible", BaseURL: server.URL + "/v1/", APIKey: "secret", HTTPClient: server.Client(),
		Headers: http.Header{"X-Test": {"provider"}}, Query: url.Values{"region": {"west"}},
		PrepareRequest: func(request *http.Request) error { request.Header.Set("X-Dynamic", "yes"); return nil },
	}))
	result, err := model.Generate(t.Context(), "draw", nil, settings)
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "gpt-image-2" || model.ProviderName() != "compatible" || model.ProviderURL() != server.URL+"/v1" ||
		len(result.Images) != 2 || result.Images[0].RevisedPrompt != "better" || result.Images[0].OutputFormat != "png" ||
		result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 8 || result.Usage.Details["input_image_tokens"] != 1 ||
		result.ProviderResponseID != "response" || result.ProviderDetails["created"].(int64) != 123 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if requestBody["model"] != "gpt-image-2" || requestBody["size"] != "1280x720" || requestBody["n"].(float64) != 2 ||
		requestBody["custom"] != true || requestBody["output_compression"].(float64) != 80 {
		t.Fatalf("unexpected request body: %#v", requestBody)
	}
}

func TestEditMultipartAndWarnings(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/images/edits" {
			t.Errorf("unexpected URL: %s", request.URL)
		}
		if err := request.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			return
		}
		if request.FormValue("input_fidelity") != "high" || request.FormValue("size") != "1024x1024" ||
			request.FormValue("custom") != `{"nested":true}` {
			t.Errorf("unexpected form: %#v", request.MultipartForm.Value)
		}
		files := request.MultipartForm.File["image[]"]
		if len(files) != 1 || files[0].Filename != "image-0.png" {
			t.Errorf("unexpected files: %#v", files)
		}
		_, _ = io.WriteString(response, `{"data":[{"b64_json":"`+base64.StdEncoding.EncodeToString(png)+`"}]}`)
	}))
	defer server.Close()
	size := "1024x1024"
	settings, err := (imageopenai.Settings{
		Common: images.Settings{Dimensions: &images.Dimensions{Width: 1536, Height: 1024}, ExtraBody: map[string]any{"custom": map[string]any{"nested": true}}},
		Size:   &size, InputFidelity: imageopenai.InputFidelityHigh, Moderation: imageopenai.ModerationAuto,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err := imageopenai.NewModel("gpt-image-1", imageopenai.WithBaseURL(server.URL), imageopenai.WithHTTPClient(server.Client())).Generate(
		t.Context(), "edit", []images.Input{ai.BinaryContent{Data: png, MediaType: "image/png"}}, settings,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Warnings) != 2 || !strings.Contains(strings.Join(result.Warnings, " "), "moderation") ||
		!strings.Contains(strings.Join(result.Warnings, " "), "dimensions") {
		t.Fatalf("unexpected warnings: %#v", result.Warnings)
	}
}

func TestValidationAndProviderErrors(t *testing.T) {
	for _, name := range []string{"dall-e-2", "dall-e-3"} {
		if capturePanic(func() { imageopenai.NewModel(name) }) == nil {
			t.Fatalf("%s did not panic", name)
		}
	}
	zero := 0
	if _, err := (imageopenai.Settings{N: &zero}).Build(); err == nil {
		t.Fatal("invalid count accepted")
	}
	if _, err := (imageopenai.Settings{Common: images.Settings{ProviderSettings: map[string]any{"openai_n": 1}}}).Build(); err == nil {
		t.Fatal("reserved setting accepted")
	}
	model := imageopenai.NewModel("gpt-image-2")
	for _, dimensions := range []images.Dimensions{{Width: 1, Height: 1}, {Width: 3904, Height: 1024}, {Width: 3200, Height: 512}} {
		_, err := model.Generate(t.Context(), "draw", nil, images.Settings{Dimensions: &dimensions})
		if err == nil {
			t.Fatalf("invalid dimensions accepted: %#v", dimensions)
		}
	}
	if _, err := imageopenai.NewModel("gpt-image-1").Generate(t.Context(), "draw", nil, images.Settings{AspectRatio: images.AspectRatio16To9}); err == nil {
		t.Fatal("unsupported legacy ratio accepted")
	}
	if _, err := imageopenai.NewModel("future").Generate(t.Context(), "draw", nil, images.Settings{AspectRatio: images.AspectRatio1To1}); err == nil {
		t.Fatal("unknown ratio mapping accepted")
	}

	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(response, `{"error":{"code":"moderation_blocked"}}`)
	}))
	defer server.Close()
	_, err := imageopenai.NewModel("gpt-image-2", imageopenai.WithBaseURL(server.URL), imageopenai.WithHTTPClient(server.Client())).Generate(t.Context(), "draw", nil, images.Settings{})
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProviderConfigurationAndSettingsErrors(t *testing.T) {
	if capturePanic(func() {
		imageopenai.NewModel("model", imageopenai.WithProvider(modelopenai.ProviderConfig{BaseURL: "https://example.com"}))
	}) == nil {
		t.Fatal("empty provider name accepted")
	}
	if capturePanic(func() {
		imageopenai.NewModel("model", imageopenai.WithProvider(modelopenai.ProviderConfig{Name: "test"}))
	}) == nil {
		t.Fatal("empty provider URL accepted")
	}
	t.Setenv("OPENAI_BASE_URL", "https://environment.example/v1")
	model := imageopenai.NewModel("model", imageopenai.WithAPIKey("key"), imageopenai.WithDefaultSettings(images.Settings{
		AspectRatio: images.AspectRatio1To1,
	}))
	if model.ProviderURL() != "https://environment.example/v1" || model.DefaultSettings().AspectRatio != images.AspectRatio1To1 {
		t.Fatalf("unexpected defaults: %s %#v", model.ProviderURL(), model.DefaultSettings())
	}
	badSettings := []map[string]any{
		{"openai_n": "bad"}, {"openai_size": 1}, {"openai_output_compression": "bad"}, {"openai_user": 1},
		{"openai_output_format": "png"}, {"openai_quality": "high"}, {"openai_background": "auto"},
		{"openai_input_fidelity": "high"}, {"openai_moderation": "auto"},
	}
	for _, providerSettings := range badSettings {
		if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{ProviderSettings: providerSettings}); err == nil {
			t.Fatalf("invalid settings accepted: %#v", providerSettings)
		}
	}
}

func TestResponseAndInputErrors(t *testing.T) {
	responses := []string{
		`{}`, `{"data":[{}]}`, `{"data":[{"b64_json":"not base64"}]}`,
		`{"data":[{"b64_json":"` + base64.StdEncoding.EncodeToString([]byte("unknown")) + `"}]}`,
	}
	for _, body := range responses {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) { _, _ = io.WriteString(response, body) }))
		_, err := imageopenai.NewModel("gpt-image-2", imageopenai.WithBaseURL(server.URL), imageopenai.WithHTTPClient(server.Client())).Generate(t.Context(), "draw", nil, images.Settings{})
		server.Close()
		var unexpected *ai.UnexpectedModelBehaviorError
		if !errors.As(err, &unexpected) {
			t.Fatalf("unexpected response error: %v", err)
		}
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `not json`)
	}))
	_, err := imageopenai.NewModel(
		"gpt-image-2", imageopenai.WithBaseURL(server.URL), imageopenai.WithHTTPClient(server.Client()),
	).Generate(t.Context(), "draw", nil, images.Settings{})
	server.Close()
	if err == nil || !strings.Contains(err.Error(), "decode response") {
		t.Fatalf("unexpected decode error: %v", err)
	}

	model := imageopenai.NewModel("gpt-image-2")
	_, err = model.Generate(t.Context(), "edit", []images.Input{ai.BinaryContent{Data: []byte("x"), MediaType: "image/gif"}}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "PNG, JPEG, or WebP") {
		t.Fatalf("unexpected media error: %v", err)
	}
	_, err = model.Generate(t.Context(), "edit", []images.Input{ai.UploadedFile{FileID: "file", ProviderName: "other", MediaType: "image/png"}}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "belongs") {
		t.Fatalf("unexpected upload error: %v", err)
	}

	server = httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(response, "limited")
	}))
	defer server.Close()
	_, err = imageopenai.NewModel("gpt-image-2", imageopenai.WithBaseURL(server.URL), imageopenai.WithHTTPClient(server.Client())).Generate(t.Context(), "draw", nil, images.Settings{})
	var apiError *modelopenai.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("unexpected API error: %v", err)
	}
}

func TestGeometryAndSettingsPrecedence(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = io.WriteString(response, `{"background":"transparent","data":[{"b64_json":"`+
			base64.StdEncoding.EncodeToString(png)+`"}]}`)
	}))
	defer server.Close()
	model := imageopenai.NewModel("gpt-image-2", imageopenai.WithBaseURL(server.URL), imageopenai.WithHTTPClient(server.Client()))
	valid := images.Dimensions{Width: 1024, Height: 1024}
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{Dimensions: &valid}); err != nil {
		t.Fatal(err)
	}
	size := "auto"
	settings, err := (imageopenai.Settings{
		Common: images.Settings{AspectRatio: images.AspectRatio1To4}, Size: &size,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Generate(t.Context(), "draw", nil, settings)
	if err != nil || len(result.Warnings) != 1 || result.ProviderDetails["background"] != "transparent" {
		t.Fatalf("unexpected provider-size result: %#v %v", result, err)
	}
	if bodies[0]["size"] != "1024x1024" || bodies[1]["size"] != "auto" {
		t.Fatalf("unexpected geometry bodies: %#v", bodies)
	}
	if _, err := (imageopenai.Settings{Common: images.Settings{
		Dimensions: &images.Dimensions{Width: 1, Height: 1}, AspectRatio: images.AspectRatio1To1,
	}}).Build(); err == nil {
		t.Fatal("invalid common settings accepted")
	}
	if settings, err := (imageopenai.Settings{}).Build(); err != nil || settings.ProviderSettings != nil {
		t.Fatalf("unexpected empty settings: %#v %v", settings, err)
	}
}

func TestRequestFailuresAndReferenceTypes(t *testing.T) {
	if _, err := imageopenai.NewModel("gpt-image-2").Generate(t.Context(), " ", nil, images.Settings{}); err == nil {
		t.Fatal("empty prompt accepted")
	}
	if _, err := imageopenai.NewModel("gpt-image-2").Generate(t.Context(), "draw", nil, images.Settings{
		ProviderSettings: map[string]any{"openai_n": 0},
	}); err == nil {
		t.Fatal("invalid direct count accepted")
	}
	prepareErr := errors.New("prepare")
	model := imageopenai.NewModel("gpt-image-2", imageopenai.WithProvider(modelopenai.ProviderConfig{
		Name: "openai", BaseURL: "https://example.com/v1",
		PrepareRequest: func(*http.Request) error { return prepareErr },
	}))
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{}); !errors.Is(err, prepareErr) {
		t.Fatalf("unexpected prepare error: %v", err)
	}
	model = imageopenai.NewModel("gpt-image-2", imageopenai.WithBaseURL("://bad"))
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{}); err == nil {
		t.Fatal("invalid endpoint accepted")
	}
	model = imageopenai.NewModel("gpt-image-2", imageopenai.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("transport")
	})}))
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{}); err == nil || !strings.Contains(err.Error(), "transport") {
		t.Fatalf("unexpected transport error: %v", err)
	}
	model = imageopenai.NewModel("gpt-image-2", imageopenai.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: errorBody{}}, nil
	})}))
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{}); err == nil || !strings.Contains(err.Error(), "read response") {
		t.Fatalf("unexpected read error: %v", err)
	}
	if _, err := imageopenai.NewModel("gpt-image-2").Generate(t.Context(), "draw", nil, images.Settings{
		ExtraBody: map[string]any{"model": "other"},
	}); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected conflict error: %v", err)
	}
	if _, err := imageopenai.NewModel("gpt-image-2").Generate(t.Context(), "draw", nil, images.Settings{
		ExtraBody: map[string]any{"custom": make(chan int)},
	}); err == nil || !strings.Contains(err.Error(), "encode request") {
		t.Fatalf("unexpected encode error: %v", err)
	}

	imageServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "image/jpeg")
		_, _ = response.Write([]byte("\xff\xd8\xffimage"))
	}))
	defer imageServer.Close()
	api := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := request.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(response, `{"data":[{"b64_json":"`+base64.StdEncoding.EncodeToString(png)+`"}]}`)
	}))
	defer api.Close()
	model = imageopenai.NewModel("gpt-image-2", imageopenai.WithBaseURL(api.URL), imageopenai.WithHTTPClient(api.Client()))
	if _, err := model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: imageServer.URL, ForceDownload: ai.FileDownloadAllowLocal,
	}}, images.Settings{}); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(t.Context(), "edit", []images.Input{ai.UploadedFile{
		FileID: "file", ProviderName: "openai", MediaType: "image/png",
	}}, images.Settings{}); err == nil || !strings.Contains(err.Error(), "does not accept") {
		t.Fatalf("unexpected upload error: %v", err)
	}
	if _, err := model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: imageServer.URL, ForceDownload: "bad",
	}}, images.Settings{}); err == nil {
		t.Fatal("invalid download mode accepted")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type errorBody struct{}

func (errorBody) Read([]byte) (int, error) { return 0, errors.New("read") }
func (errorBody) Close() error             { return nil }

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
