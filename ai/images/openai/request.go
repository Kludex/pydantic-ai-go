package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	modelopenai "github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

// Generate creates or edits images through the OpenAI Images API.
func (model *Model) Generate(
	ctx context.Context, prompt string, inputs []images.Input, settings images.Settings,
) (*images.Result, error) {
	settings = images.MergeSettings(model.settings, settings)
	prompt, inputs, settings, err := images.PrepareRequest(prompt, inputs, settings)
	if err != nil {
		return nil, err
	}
	settings, provider, err := extractSettings(settings)
	if err != nil {
		return nil, err
	}
	if provider.n != nil && *provider.n <= 0 {
		return nil, fmt.Errorf("openai images: image count must be greater than zero")
	}
	size, conflicts, err := resolveGeometry(model.name, settings, provider.size)
	if err != nil {
		return nil, err
	}
	warnings := conflictWarnings(conflicts)
	if len(inputs) > 0 && provider.moderation != "" {
		warnings = append(warnings, "openai images: ignored moderation for image editing")
	}
	if len(inputs) == 0 && provider.inputFidelity != "" {
		warnings = append(warnings, "openai images: ignored input fidelity without reference images")
	}
	var request *http.Request
	if len(inputs) == 0 {
		request, err = model.generationRequest(ctx, prompt, size, provider, settings)
	} else {
		request, err = model.editRequest(ctx, prompt, inputs, size, provider, settings)
	}
	if err != nil {
		return nil, err
	}
	if err := model.configureRequest(request, settings.ExtraHeaders); err != nil {
		return nil, err
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, images.NewModelTransportError(ctx, model, "request", err)
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, images.NewModelTransportError(ctx, model, "read response", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		if moderationBlocked(body) {
			return nil, &ai.ContentFilterError{Message: "OpenAI image generation was blocked for content moderation"}
		}
		return nil, &modelopenai.APIError{
			StatusCode: response.StatusCode, Body: string(body), ProviderName: model.providerName,
		}
	}
	result, err := model.parseResponse(prompt, body)
	if err != nil {
		return nil, err
	}
	result.Warnings = warnings
	return result, nil
}

func (model *Model) generationRequest(
	ctx context.Context, prompt, size string, provider providerSettings, settings images.Settings,
) (*http.Request, error) {
	body := maps.Clone(settings.ExtraBody)
	if body == nil {
		body = map[string]any{}
	}
	for _, key := range []string{
		"model", "prompt", "n", "size", "output_format", "quality", "background", "moderation",
		"output_compression", "user",
	} {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("openai images: extra body field %q conflicts with typed settings", key)
		}
	}
	body["model"] = model.name
	body["prompt"] = prompt
	setProviderBody(body, size, provider, false)
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("openai images: encode request: %w", err)
	}
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, model.endpoint("/images/generations"), bytes.NewReader(encoded),
	)
	if err == nil {
		request.Header.Set("Content-Type", "application/json")
	}
	return request, err
}

func setProviderBody(body map[string]any, size string, provider providerSettings, edit bool) {
	if provider.n != nil {
		body["n"] = *provider.n
	}
	if size != "" {
		body["size"] = size
	}
	if provider.outputFormat != "" {
		body["output_format"] = provider.outputFormat
	}
	if provider.quality != "" {
		body["quality"] = provider.quality
	}
	if provider.background != "" {
		body["background"] = provider.background
	}
	if edit && provider.inputFidelity != "" {
		body["input_fidelity"] = provider.inputFidelity
	}
	if !edit && provider.moderation != "" {
		body["moderation"] = provider.moderation
	}
	if provider.outputCompression != nil {
		body["output_compression"] = *provider.outputCompression
	}
	if provider.user != nil {
		body["user"] = *provider.user
	}
}
