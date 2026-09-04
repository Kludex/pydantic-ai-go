package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	openaimodels "github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

// Embed creates one embedding for each input.
func (model *Model) Embed(
	ctx context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings,
) (*embeddings.Result, error) {
	settings = embeddings.MergeSettings(model.settings, settings)
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	body, err := model.requestBody(inputs, settings)
	if err != nil {
		return nil, err
	}
	endpoint := model.baseURL + "/embeddings"
	if query := model.query.Encode(); query != "" {
		endpoint += "?" + query
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: build request: %w", err)
	}
	model.configureRequest(request, settings.ExtraHeaders)
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, fmt.Errorf("openai embeddings: prepare request: %w", err)
		}
		setHeaders(request, settings.ExtraHeaders)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: request: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: read response: %w", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, &openaimodels.APIError{
			StatusCode: response.StatusCode, Body: string(responseBody), ProviderName: model.providerName,
		}
	}
	return model.parseResponse(responseBody, inputs, inputType)
}

func (model *Model) requestBody(inputs []string, settings embeddings.Settings) ([]byte, error) {
	body := maps.Clone(settings.ExtraBody)
	if body == nil {
		body = map[string]any{}
	}
	for _, key := range []string{"model", "input", "dimensions"} {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("openai embeddings: extra body field %q conflicts with typed settings", key)
		}
	}
	body["model"] = model.name
	body["input"] = slices.Clone(inputs)
	if settings.Dimensions != nil {
		body["dimensions"] = *settings.Dimensions
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: encode request: %w", err)
	}
	return encoded, nil
}

func (model *Model) configureRequest(request *http.Request, extraHeaders map[string]string) {
	request.Header.Set("Content-Type", "application/json")
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	for name, values := range model.headers {
		request.Header[name] = slices.Clone(values)
	}
	setHeaders(request, extraHeaders)
}

func setHeaders(request *http.Request, headers map[string]string) {
	for name, value := range headers {
		request.Header.Set(name, value)
	}
}
