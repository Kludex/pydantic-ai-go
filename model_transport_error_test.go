package ai_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

type transportFallbackModel struct{}

type namedTransportModel struct{}

func (namedTransportModel) Name() string { return "custom" }

func (namedTransportModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return nil, nil
}

func (transportFallbackModel) Name() string         { return "secondary" }
func (transportFallbackModel) ProviderName() string { return "provider" }
func (transportFallbackModel) ProviderURL() string  { return "https://provider.test" }

func (transportFallbackModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "fallback"}}}, nil
}

func TestModelTransportError(t *testing.T) {
	cause := errors.New("connection reset")
	err := ai.NewModelTransportError(t.Context(), transportFallbackModel{}, "read response", cause)
	var transportError *ai.ModelTransportError
	var apiError ai.ModelAPIError
	if !errors.As(err, &transportError) || !errors.As(err, &apiError) || !errors.Is(err, cause) {
		t.Fatalf("unexpected transport error contract: %v", err)
	}
	if transportError.ModelName != "secondary" || transportError.ProviderName != "provider" ||
		transportError.Operation != "read response" || transportError.Err != cause ||
		err.Error() != "provider: read response: connection reset" {
		t.Fatalf("unexpected transport error: %#v", transportError)
	}
	if !apiError.IsModelAPIError() {
		t.Fatal("transport error was not eligible for fallback")
	}
	if err := ai.NewModelTransportError(t.Context(), transportFallbackModel{}, "request", nil); err != nil {
		t.Fatalf("nil failure returned an error: %v", err)
	}
}

func TestModelTransportErrorPreservesCancellation(t *testing.T) {
	for name, cause := range map[string]error{
		"canceled": context.Canceled,
		"deadline": context.DeadlineExceeded,
	} {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(t.Context())
			cancel(cause)
			err := ai.NewModelTransportError(ctx, transportFallbackModel{}, "request", cause)
			var apiError ai.ModelAPIError
			if !errors.Is(err, cause) || errors.As(err, &apiError) ||
				!strings.HasPrefix(err.Error(), "provider: request: ") {
				t.Fatalf("caller cancellation was misclassified: %v", err)
			}
		})
	}
}

func TestProviderTransportErrorTriggersFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	server.Close()
	primary := openai.NewModel("gpt-5", openai.WithAPIKey("test"), openai.WithBaseURL(server.URL))
	model := ai.NewFallbackModel(primary, ai.WithFallbackModels(transportFallbackModel{}))
	response, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "fallback" || response.ModelName != "secondary" {
		t.Fatalf("unexpected fallback response: %#v", response)
	}
}

func TestModelTransportErrorDefaults(t *testing.T) {
	cause := errors.New("offline")
	err := ai.NewModelTransportError(context.Background(), nil, "", cause)
	if err.Error() != "model: request: offline" {
		t.Fatalf("unexpected default error: %v", err)
	}
	withoutCause := &ai.ModelTransportError{}
	if withoutCause.Error() != "model: request failed" || withoutCause.Unwrap() != nil {
		t.Fatalf("unexpected empty transport error: %v", withoutCause)
	}
	err = ai.NewModelTransportError(context.Background(), namedTransportModel{}, "connect", cause)
	if err.Error() != "custom: connect: offline" {
		t.Fatalf("unexpected model-only error: %v", err)
	}
}
