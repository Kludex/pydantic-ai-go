package cohere

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
	modelcohere "github.com/Kludex/pydantic-ai-go/ai/models/cohere"
)

// Embed creates one embedding for each input.
func (model *Model) Embed(
	ctx context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings,
) (*embeddings.Result, error) {
	settings = mergeSettings(model.settings, settings)
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	settings, local, err := extractSettings(settings)
	if err != nil {
		return nil, err
	}
	body, err := model.requestBody(inputs, inputType, settings, local)
	if err != nil {
		return nil, err
	}
	responseBody, err := model.doRequest(ctx, model.baseURL+"/v2/embed", body, settings.ExtraHeaders)
	if err != nil {
		return nil, err
	}
	return model.parseResponse(responseBody, inputs, inputType)
}

// CountTokens counts text with Cohere's v1 tokenizer endpoint.
func (model *Model) CountTokens(ctx context.Context, text string) (int, error) {
	body, _ := json.Marshal(map[string]any{"model": model.name, "text": text})
	responseBody, err := model.doRequest(ctx, model.baseURL+"/v1/tokenize", body, nil)
	if err != nil {
		return 0, err
	}
	var response struct {
		Tokens *[]int `json:"tokens"`
	}
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return 0, fmt.Errorf("cohere embeddings: decode token count response: %w", err)
	}
	if response.Tokens == nil {
		return 0, fmt.Errorf("cohere embeddings: token count response omitted tokens")
	}
	return len(*response.Tokens), nil
}

func (model *Model) requestBody(
	inputs []string, inputType embeddings.InputType, settings embeddings.Settings, local localSettings,
) ([]byte, error) {
	body := maps.Clone(settings.ExtraBody)
	if body == nil {
		body = map[string]any{}
	}
	for _, key := range []string{
		"model", "texts", "output_dimension", "input_type", "max_tokens", "truncate", "embedding_types",
	} {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("cohere embeddings: extra body field %q conflicts with typed settings", key)
		}
	}
	cohereInputType := local.inputType
	if cohereInputType == "" {
		cohereInputType = InputTypeSearchQuery
		if inputType == embeddings.InputTypeDocument {
			cohereInputType = InputTypeSearchDocument
		}
	}
	truncate := local.truncate
	if !local.truncateSet {
		truncate = TruncationNone
		if settings.Truncate != nil && *settings.Truncate {
			truncate = TruncationEnd
		}
	}
	body["model"] = model.name
	body["texts"] = slices.Clone(inputs)
	body["output_dimension"] = settings.Dimensions
	body["input_type"] = cohereInputType
	body["max_tokens"] = clonePointer(local.maxTokens)
	body["truncate"] = truncate
	body["embedding_types"] = []string{"float"}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("cohere embeddings: encode request: %w", err)
	}
	return encoded, nil
}

func (model *Model) doRequest(
	ctx context.Context, endpoint string, body []byte, extraHeaders map[string]string,
) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("cohere embeddings: build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	for name, values := range model.headers {
		request.Header[name] = slices.Clone(values)
	}
	setHeaders(request, extraHeaders)
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, fmt.Errorf("cohere embeddings: prepare request: %w", err)
		}
		setHeaders(request, extraHeaders)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("cohere embeddings: request: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("cohere embeddings: read response: %w", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, &modelcohere.APIError{
			StatusCode: response.StatusCode, Body: string(responseBody), Headers: response.Header.Clone(),
			ProviderName: model.providerName,
		}
	}
	return responseBody, nil
}

func setHeaders(request *http.Request, headers map[string]string) {
	for name, value := range headers {
		request.Header.Set(name, value)
	}
}
