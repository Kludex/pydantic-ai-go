// Package mistral implements ai.Model against Mistral's native Chat Completions API.
package mistral

import (
	"bytes"
	"context"
	"io"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go"
)

const defaultBaseURL = "https://api.mistral.ai/v1"

// Model calls Mistral's native Chat Completions API.
type Model struct {
	name            string
	providerName    string
	baseURL         string
	apiKey          string
	httpClient      *http.Client
	headers         http.Header
	prepareRequest  RequestPreparationFunc
	defaultSettings ai.ModelSettings
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	payload, err := model.buildRequest(ctx, messages, params, false)
	if err != nil {
		return nil, err
	}
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
	if err != nil {
		return nil, err
	}
	request, err := model.newRequest(ctx, body, params.Settings.ExtraHeaders, "application/json")
	if err != nil {
		return nil, err
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "request", err)
	}
	defer func() { _ = response.Body.Close() }()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "read response", err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, &APIError{
			StatusCode: response.StatusCode, Body: string(data), Headers: response.Header.Clone(),
			ProviderName: model.providerName,
		}
	}
	return model.parseResponse(data)
}

func (model *Model) newRequest(
	ctx context.Context, body []byte, headers map[string]string, accept string,
) (*http.Request, error) {
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, model.baseURL+"/chat/completions", bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}
	request.Header = model.headers.Clone()
	if request.Header == nil {
		request.Header = make(http.Header)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", accept)
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, err
		}
	}
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	return request, nil
}
