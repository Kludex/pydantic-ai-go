package voyageai

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
)

// Embed creates one embedding for each input.
func (model *Model) Embed(
	ctx context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings,
) (*embeddings.Result, error) {
	settings = mergeSettings(model.settings, settings)
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	settings, voyageInputType, err := extractSettings(settings)
	if err != nil {
		return nil, err
	}
	body, err := model.requestBody(inputs, inputType, settings, voyageInputType)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, model.baseURL+"/embeddings", bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("voyageai embeddings: build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	for name, values := range model.headers {
		request.Header[name] = slices.Clone(values)
	}
	setHeaders(request, settings.ExtraHeaders)
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, fmt.Errorf("voyageai embeddings: prepare request: %w", err)
		}
		setHeaders(request, settings.ExtraHeaders)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("voyageai embeddings: request: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("voyageai embeddings: read response: %w", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, &APIError{
			StatusCode: response.StatusCode, Body: string(responseBody), Headers: response.Header.Clone(),
			ProviderName: model.providerName,
		}
	}
	return model.parseResponse(responseBody, inputs, inputType)
}

func (model *Model) requestBody(
	inputs []string, inputType embeddings.InputType, settings embeddings.Settings, voyageInputType InputType,
) ([]byte, error) {
	body := maps.Clone(settings.ExtraBody)
	if body == nil {
		body = map[string]any{}
	}
	for _, key := range []string{
		"input", "model", "input_type", "truncation", "output_dimension", "output_dtype", "encoding_format",
	} {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("voyageai embeddings: extra body field %q conflicts with typed settings", key)
		}
	}
	if voyageInputType == "" {
		voyageInputType = InputTypeQuery
		if inputType == embeddings.InputTypeDocument {
			voyageInputType = InputTypeDocument
		}
	}
	var wireInputType any = voyageInputType
	if voyageInputType == InputTypeNone {
		wireInputType = nil
	}
	truncate := false
	if settings.Truncate != nil {
		truncate = *settings.Truncate
	}
	body["input"] = slices.Clone(inputs)
	body["model"] = model.name
	body["input_type"] = wireInputType
	body["truncation"] = truncate
	body["output_dimension"] = settings.Dimensions
	body["output_dtype"] = nil
	body["encoding_format"] = "base64"
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("voyageai embeddings: encode request: %w", err)
	}
	return encoded, nil
}

func setHeaders(request *http.Request, headers map[string]string) {
	for name, value := range headers {
		request.Header.Set(name, value)
	}
}
