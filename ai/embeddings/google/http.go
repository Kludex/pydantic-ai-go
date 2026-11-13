package google

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

func (model *Model) doRequest(
	ctx context.Context, endpoint string, body []byte, extraHeaders map[string]string,
) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("google embeddings: build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	if model.apiKey != "" {
		request.Header.Set("x-goog-api-key", model.apiKey)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, fmt.Errorf("google embeddings: prepare request: %w", err)
		}
	}
	for name, value := range extraHeaders {
		request.Header.Set(name, value)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, model.transportError(ctx, "request", err)
	}
	defer func() { _ = response.Body.Close() }()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, model.transportError(ctx, "read response", err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, &modelgoogle.APIError{StatusCode: response.StatusCode, Body: string(responseBody)}
	}
	return responseBody, nil
}

func (model *Model) transportError(ctx context.Context, operation string, err error) error {
	if ctx.Err() != nil {
		return fmt.Errorf("google embeddings: %s: %w", operation, err)
	}
	return &ai.ModelTransportError{ModelName: model.name, ProviderName: "google embeddings", Operation: operation, Err: err}
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
