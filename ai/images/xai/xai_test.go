package xai_test

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	imagexai "github.com/Kludex/pydantic-ai-go/ai/images/xai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

var png = []byte("\x89PNG\r\n\x1a\nimage")

type grpcClient struct {
	method             string
	request            map[string]any
	requestRaw         []byte
	expectedRequestRaw []byte
	requestMatches     bool
	metadata           metadata.MD
	response           string
	responseRaw        []byte
	err                error
}

func (client *grpcClient) Invoke(
	ctx context.Context, method string, args any, reply any, _ ...grpc.CallOption,
) error {
	client.method = method
	client.metadata, _ = metadata.FromOutgoingContext(ctx)
	requestRaw, err := proto.Marshal(args.(proto.Message))
	if err != nil {
		return err
	}
	client.requestRaw = requestRaw
	if client.expectedRequestRaw != nil {
		expected := proto.Clone(args.(proto.Message))
		proto.Reset(expected)
		if err := proto.Unmarshal(client.expectedRequestRaw, expected); err != nil {
			return err
		}
		client.requestMatches = proto.Equal(args.(proto.Message), expected)
	}
	encoded, err := protojson.Marshal(args.(proto.Message))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(encoded, &client.request); err != nil {
		return err
	}
	if client.err != nil {
		return client.err
	}
	if client.responseRaw != nil {
		return proto.Unmarshal(client.responseRaw, reply.(proto.Message))
	}
	return protojson.Unmarshal([]byte(client.response), reply.(proto.Message))
}

func (*grpcClient) NewStream(
	context.Context, *grpc.StreamDesc, string, ...grpc.CallOption,
) (grpc.ClientStream, error) {
	return nil, errors.New("unexpected stream")
}

func imageData(data []byte) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
}

func TestGenerateBatchUsesOfficialGRPCWire(t *testing.T) {
	// These bytes come from xai-sdk-python v1.18.0 for this exact sample_batch call and response.
	responseRaw, err := hex.DecodeString(
		"0a2e0a2a646174613a696d6167652f706e673b6261736536342c6956424f5277304b476770706257466e5a513d3d2001" +
			"0a2e0a2a646174613a696d6167652f706e673b6261736536342c6956424f5277304b476770706257466e5a513d3d2001" +
			"121a67726f6b2d696d6167696e652d696d6167652d7175616c6974791a12080b10072004280330033802588088debe01",
	)
	if err != nil {
		t.Fatal(err)
	}
	requestRaw, err := hex.DecodeString(
		"0a137265706c61636520746865207375626a656374121267726f6b2d696d6167696e652d696d616765" +
			"18022208757365722d3132335801700578018a010c10011a0866696c652d3132338a012e0a2a646174613a696d" +
			"6167652f706e673b6261736536342c6956424f5277304b476770706257466e5a513d3d1001",
	)
	if err != nil {
		t.Fatal(err)
	}
	client := &grpcClient{responseRaw: responseRaw, expectedRequestRaw: requestRaw}
	n := 2
	settings, err := (imagexai.Settings{
		Common: images.Settings{
			Dimensions:   &images.Dimensions{Width: 1280, Height: 720},
			ExtraHeaders: map[string]string{"X-Test": "ignored"}, ExtraBody: map[string]any{"seed": 1},
		},
		N: &n, User: "user-123",
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := imagexai.NewModel(
		"grok-imagine-image", imagexai.WithAPIKey("secret"), imagexai.WithClient(client),
		imagexai.WithProviderURL("https://images.example.com/v1/"),
	)
	result, err := model.Generate(t.Context(), "replace the subject", []images.Input{
		ai.UploadedFile{FileID: "file-123", ProviderName: "xai", MediaType: "image/jpeg"},
		ai.BinaryContent{Data: png, MediaType: "image/png"},
	}, settings)
	if err != nil {
		t.Fatal(err)
	}
	if client.method != "/xai_api.Image/GenerateImage" || client.metadata.Get("authorization")[0] != "Bearer secret" ||
		!client.requestMatches || client.request["prompt"] != "replace the subject" ||
		client.request["model"] != "grok-imagine-image" || client.request["n"].(float64) != 2 ||
		client.request["format"] != "IMG_FORMAT_BASE64" || client.request["aspectRatio"] != "IMG_ASPECT_RATIO_16_9" ||
		client.request["resolution"] != "IMG_RESOLUTION_1K" || client.request["user"] != "user-123" {
		t.Fatalf("unexpected GenerateImage request: method=%q raw=%x body=%#v", client.method, client.requestRaw, client.request)
	}
	references := client.request["images"].([]any)
	if references[0].(map[string]any)["fileId"] != "file-123" ||
		!strings.HasPrefix(references[1].(map[string]any)["imageUrl"].(string), "data:image/png;base64,") {
		t.Fatalf("unexpected ordered references: %#v", references)
	}
	if len(result.Images) != 2 || result.ModelName != "grok-imagine-image-quality" ||
		result.ProviderURL != "https://images.example.com/v1" || result.Usage.InputTokens != 7 ||
		result.Usage.OutputTokens != 11 || result.Usage.ReasoningTokens != 3 || result.Usage.CacheReadTokens != 2 ||
		result.Usage.Details["input_text_tokens"] != 4 || result.Usage.Details["input_image_tokens"] != 3 ||
		result.ProviderDetails["cost_in_usd_ticks"].(int64) != 400000000 ||
		result.ProviderDetails["cost_usd"].(float64) != 0.04 || len(result.Warnings) != 2 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestGenerateSingleUsesSingularReference(t *testing.T) {
	client := &grpcClient{response: `{"images":[
		{"base64":"` + imageData(png) + `","respectModeration":true},
		{"base64":"` + imageData([]byte("ignored")) + `","respectModeration":true}
	]}`}
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithClient(client))
	result, err := model.Generate(t.Context(), "edit", []images.Input{ai.UploadedFile{
		FileID: "file", ProviderName: "xai", MediaType: "image/png",
	}}, images.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	image := client.request["image"].(map[string]any)
	if client.request["n"].(float64) != 1 || image["fileId"] != "file" || client.request["images"] != nil ||
		len(result.Images) != 1 || result.ModelName != "grok-imagine-image" || result.Usage.Requests != 1 ||
		result.ProviderDetails != nil {
		t.Fatalf("unexpected single request or result: request=%#v result=%#v", client.request, result)
	}

	client.response = `{"images":[{"base64":"` + imageData(png) + `","respectModeration":true}]}`
	_, err = model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{URL: "https://example.com/image.png"}}, images.Settings{})
	if err != nil || client.request["image"].(map[string]any)["imageUrl"] != "https://example.com/image.png" {
		t.Fatalf("unexpected URL request: %#v %v", client.request, err)
	}
}

func TestModerationAndInputValidation(t *testing.T) {
	client := &grpcClient{response: `{"images":[
		{"respectModeration":false},
		{"base64":"` + imageData(png) + `","respectModeration":true}
	]}`}
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithClient(client))
	count := 2
	settings, err := (imagexai.Settings{N: &count}).Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Generate(t.Context(), "draw", nil, settings)
	if err != nil || len(result.Images) != 1 || result.ProviderDetails["moderated_image_indices"].([]int)[0] != 0 {
		t.Fatalf("unexpected moderated result: %#v %v", result, err)
	}
	client.response = `{"images":[{"respectModeration":false}]}`
	_, err = model.Generate(t.Context(), "draw", nil, images.Settings{})
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) {
		t.Fatalf("unexpected moderation error: %v", err)
	}

	_, err = model.Generate(t.Context(), "edit", []images.Input{
		ai.ImageURL{URL: "https://example.com/image.png"},
		ai.UploadedFile{FileID: "file", ProviderName: "xai", MediaType: "image/png"},
	}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "uploaded files before") {
		t.Fatalf("unexpected order error: %v", err)
	}
	_, err = model.Generate(t.Context(), "edit", []images.Input{ai.UploadedFile{
		FileID: "file", ProviderName: "other", MediaType: "image/png",
	}}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "belongs") {
		t.Fatalf("unexpected provider error: %v", err)
	}
}

func TestDownloadedReference(t *testing.T) {
	client := &grpcClient{response: `{"images":[{"base64":"` + imageData(png) + `","respectModeration":true}]}`}
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithClient(client))
	_, err := model.Generate(t.Context(), "edit", []images.Input{ai.ImageURL{
		URL: "http://127.0.0.1/image.png", ForceDownload: ai.FileDownloadSafe,
	}}, images.Settings{})
	if err == nil || !strings.Contains(err.Error(), "download reference image") {
		t.Fatalf("unexpected protected download error: %v", err)
	}
}

func TestDownloadedInputAndGeometryConflicts(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "application/octet-stream")
		if request.URL.Path == "/unknown" {
			_, _ = response.Write([]byte("unknown"))
			return
		}
		if request.URL.Path == "/text" {
			response.Header().Set("Content-Type", "text/plain")
			_, _ = response.Write([]byte("text"))
			return
		}
		_, _ = response.Write(png)
	}))
	defer imageServer.Close()
	client := &grpcClient{response: `{"images":[{"base64":"` + imageData(png) + `","respectModeration":true}]}`}
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithClient(client))
	for _, input := range []images.Input{
		ai.ImageURL{URL: imageServer.URL + "/image", ForceDownload: ai.FileDownloadAllowLocal},
		ai.ImageURL{URL: imageServer.URL + "/image", MediaType: "image/png", ForceDownload: ai.FileDownloadAllowLocal},
	} {
		if _, err := model.Generate(t.Context(), "edit", []images.Input{input}, images.Settings{}); err != nil {
			t.Fatalf("valid downloaded input failed: %#v %v", input, err)
		}
	}
	for _, input := range []images.Input{
		ai.ImageURL{URL: imageServer.URL + "/image", ForceDownload: ai.FileDownloadSafe},
		ai.ImageURL{URL: imageServer.URL + "/text", ForceDownload: ai.FileDownloadAllowLocal},
		ai.ImageURL{URL: imageServer.URL + "/unknown", ForceDownload: ai.FileDownloadAllowLocal},
		ai.ImageURL{URL: imageServer.URL + "/image", ForceDownload: "invalid"},
	} {
		if _, err := model.Generate(t.Context(), "edit", []images.Input{input}, images.Settings{}); err == nil {
			t.Fatalf("invalid downloaded input accepted: %#v", input)
		}
	}

	settings, err := (imagexai.Settings{
		Common:      images.Settings{Dimensions: &images.Dimensions{Width: 1280, Height: 720}},
		AspectRatio: images.AspectRatio1To1, Resolution: imagexai.Resolution2K,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Generate(t.Context(), "draw", nil, settings)
	if err != nil || len(result.Warnings) != 1 || client.request["aspectRatio"] != "IMG_ASPECT_RATIO_1_1" ||
		client.request["resolution"] != "IMG_RESOLUTION_2K" {
		t.Fatalf("unexpected dimensions conflict: request=%#v result=%#v err=%v", client.request, result, err)
	}
	settings, err = (imagexai.Settings{
		Common: images.Settings{AspectRatio: images.AspectRatio16To9}, AspectRatio: images.AspectRatio1To1,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err = model.Generate(t.Context(), "draw", nil, settings)
	if err != nil || len(result.Warnings) != 1 {
		t.Fatalf("unexpected aspect conflict: %#v %v", result, err)
	}
}

func TestSettingsGeometryAndFailures(t *testing.T) {
	zero := 0
	if _, err := (imagexai.Settings{N: &zero}).Build(); err == nil {
		t.Fatal("invalid count accepted")
	}
	if _, err := (imagexai.Settings{Common: images.Settings{
		ProviderSettings: map[string]any{"xai_n": 1},
	}}).Build(); err == nil {
		t.Fatal("reserved setting accepted")
	}
	if _, err := (imagexai.Settings{Common: images.Settings{
		Dimensions: &images.Dimensions{Width: 1, Height: 1}, AspectRatio: images.AspectRatio1To1,
	}}).Build(); err == nil {
		t.Fatal("invalid common geometry accepted")
	}
	if settings, err := (imagexai.Settings{}).Build(); err != nil || settings.ProviderSettings != nil {
		t.Fatalf("unexpected empty settings: %#v %v", settings, err)
	}
	for _, providerSettings := range []map[string]any{
		{"xai_n": "bad"}, {"xai_user": 1}, {"xai_aspect_ratio": "1:1"}, {"xai_resolution": "1k"},
	} {
		model := imagexai.NewModel("grok-imagine-image", imagexai.WithClient(&grpcClient{}))
		if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{ProviderSettings: providerSettings}); err == nil {
			t.Fatalf("invalid settings accepted: %#v", providerSettings)
		}
	}
	model := imagexai.NewModel("grok-imagine-image", imagexai.WithClient(&grpcClient{}))
	if _, err := model.Generate(t.Context(), " ", nil, images.Settings{}); err == nil {
		t.Fatal("empty prompt accepted")
	}
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{
		ProviderSettings: map[string]any{"xai_n": int(^uint32(0)>>1) + 1},
	}); err == nil {
		t.Fatal("oversized image count accepted")
	}
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{AspectRatio: images.AspectRatio1To4}); err == nil {
		t.Fatal("unsupported ratio accepted")
	}
	if _, err := model.Generate(t.Context(), "draw", nil, images.Settings{
		Dimensions: &images.Dimensions{Width: 1, Height: 1},
	}); err == nil {
		t.Fatal("unsupported dimensions accepted")
	}
	if _, err := imagexai.NewModel("unknown", imagexai.WithClient(&grpcClient{})).Generate(
		t.Context(), "draw", nil, images.Settings{Dimensions: &images.Dimensions{Width: 1024, Height: 1024}},
	); err == nil {
		t.Fatal("unknown dimensions mapping accepted")
	}
}

func TestGRPCErrorsAreInspectable(t *testing.T) {
	for _, test := range []struct {
		code codes.Code
		want int
	}{
		{codes.InvalidArgument, 400}, {codes.Unauthenticated, 401}, {codes.PermissionDenied, 403},
		{codes.NotFound, 404}, {codes.ResourceExhausted, 429}, {codes.Internal, 500},
		{codes.Unavailable, 503}, {codes.DeadlineExceeded, 504}, {codes.Canceled, 0},
	} {
		client := &grpcClient{err: status.Error(test.code, "boom")}
		_, err := imagexai.NewModel("model", imagexai.WithClient(client)).Generate(
			t.Context(), "draw", nil, images.Settings{},
		)
		var apiError ai.ModelAPIError
		var xaiError *imagexai.APIError
		if !errors.As(err, &apiError) || !apiError.IsModelAPIError() || !errors.As(err, &xaiError) ||
			xaiError.StatusCode != test.want || xaiError.Code != test.code || !errors.Is(err, client.err) ||
			xaiError.Error() == "" {
			t.Fatalf("unexpected error for %s: %#v", test.code, err)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := imagexai.NewModel("model", imagexai.WithClient(&grpcClient{
		err: status.Error(codes.Canceled, "canceled"),
	})).Generate(ctx, "draw", nil, images.Settings{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected cancellation error: %v", err)
	}

	plain := errors.New("connection failed")
	_, err = imagexai.NewModel("model", imagexai.WithClient(&grpcClient{err: plain})).Generate(
		t.Context(), "draw", nil, images.Settings{},
	)
	var transport *ai.ModelTransportError
	if !errors.As(err, &transport) || !errors.Is(err, plain) || transport.ModelName != "model" ||
		transport.ProviderName != "xai" || transport.Operation != "GenerateImage" {
		t.Fatalf("unexpected transport error: %v", err)
	}
}

func TestResponseValidationAndConfiguration(t *testing.T) {
	count := 2
	batchSettings, err := (imagexai.Settings{N: &count}).Build()
	if err != nil {
		t.Fatal(err)
	}
	_, err = imagexai.NewModel("model", imagexai.WithClient(&grpcClient{response: `{"images":[
		{"base64":"` + imageData(png) + `","respectModeration":true}
	]}`})).Generate(t.Context(), "draw", nil, batchSettings)
	var unexpected *ai.UnexpectedModelBehaviorError
	if !errors.As(err, &unexpected) || !strings.Contains(err.Error(), "expected 2") {
		t.Fatalf("unexpected short batch error: %v", err)
	}
	for _, body := range []string{
		`{}`, `{"images":[{"base64":"data:image/png;base64,","respectModeration":true}]}`,
		`{"images":[{"base64":"%%%","respectModeration":true}]}`,
		`{"images":[{"base64":"aGVsbG8=","respectModeration":true}]}`,
	} {
		model := imagexai.NewModel("model", imagexai.WithClient(&grpcClient{response: body}))
		_, err := model.Generate(t.Context(), "draw", nil, images.Settings{})
		if !errors.As(err, &unexpected) {
			t.Fatalf("unexpected error for %s: %v", body, err)
		}
	}
	if panicValue := capturePanic(func() { imagexai.NewModel("model", imagexai.WithTarget(" ")) }); panicValue == nil {
		t.Fatal("empty target accepted")
	}
	if panicValue := capturePanic(func() { imagexai.NewModel("model", imagexai.WithClient(nil)) }); panicValue == nil {
		t.Fatal("nil client accepted")
	}
	invalid := imagexai.NewModel("model", imagexai.WithTarget("dns:///%"))
	if _, err := invalid.Generate(t.Context(), "draw", nil, images.Settings{}); err == nil ||
		!strings.Contains(err.Error(), "create gRPC client") {
		t.Fatalf("unexpected invalid target error: %v", err)
	}

	t.Setenv("XAI_API_KEY", "environment")
	model := imagexai.NewModel(
		"model", imagexai.WithTarget("passthrough:///invalid"), imagexai.WithDefaultSettings(images.Settings{
			AspectRatio: images.AspectRatio1To1,
		}),
	)
	if model.Name() != "model" || model.ProviderName() != "xai" || model.ProviderURL() != "https://api.x.ai/v1" ||
		model.DefaultSettings().AspectRatio != images.AspectRatio1To1 {
		t.Fatalf("unexpected model configuration: %#v", model)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := model.Generate(ctx, "draw", nil, images.Settings{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected default client error: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
}

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
