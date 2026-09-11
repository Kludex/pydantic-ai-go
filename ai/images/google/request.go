package google

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

type content struct {
	Role  string `json:"role"`
	Parts []part `json:"parts"`
}

type part struct {
	Text            string      `json:"text,omitempty"`
	InlineData      *inlineData `json:"inlineData,omitempty"`
	FileData        *fileData   `json:"fileData,omitempty"`
	MediaResolution any         `json:"mediaResolution,omitempty"`
}

type inlineData struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"`
}
type fileData struct {
	MIMEType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type imageConfig struct {
	AspectRatio              string    `json:"aspectRatio,omitempty"`
	ImageSize                ImageSize `json:"imageSize,omitempty"`
	OutputMIMEType           string    `json:"outputMimeType,omitempty"`
	OutputCompressionQuality *int      `json:"outputCompressionQuality,omitempty"`
}

type generationConfig struct {
	ResponseModalities []string     `json:"responseModalities"`
	ImageConfig        *imageConfig `json:"imageConfig,omitempty"`
}

// Generate creates or edits images through Google generateContent.
func (model *Model) Generate(
	ctx context.Context, prompt string, inputs []images.Input, settings images.Settings,
) (*images.Result, error) {
	settings = images.MergeSettings(model.settings, settings)
	prompt, inputs, settings, err := images.PrepareRequest(prompt, inputs, settings)
	if err != nil {
		return nil, err
	}
	provider, err := extractSettings(settings)
	if err != nil {
		return nil, err
	}
	aspectRatio, imageSize, conflicts, err := resolveGeometry(model.name, settings, provider)
	if err != nil {
		return nil, err
	}
	parts := []part{{Text: prompt}}
	for _, input := range inputs {
		mapped, err := model.mapInput(ctx, input)
		if err != nil {
			return nil, err
		}
		parts = append(parts, mapped)
	}
	config := &imageConfig{
		AspectRatio: aspectRatio, ImageSize: imageSize, OutputMIMEType: provider.outputMIMEType,
		OutputCompressionQuality: provider.compressionQuality,
	}
	if *config == (imageConfig{}) {
		config = nil
	}
	body := maps.Clone(settings.ExtraBody)
	if body == nil {
		body = map[string]any{}
	}
	for _, key := range []string{"contents", "generationConfig"} {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("google images: extra body field %q conflicts with typed settings", key)
		}
	}
	body["contents"] = []content{{Role: "user", Parts: parts}}
	body["generationConfig"] = generationConfig{ResponseModalities: []string{"IMAGE"}, ImageConfig: config}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("google images: encode request: %w", err)
	}
	endpoint := fmt.Sprintf("%s/models/%s:generateContent", model.baseURL, model.name)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	if model.apiKey != "" {
		request.Header.Set("x-goog-api-key", model.apiKey)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, fmt.Errorf("google images: prepare request: %w", err)
		}
	}
	for name, value := range settings.ExtraHeaders {
		request.Header.Set(name, value)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, images.NewModelTransportError(ctx, model, "request", err)
	}
	defer func() { _ = response.Body.Close() }()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, images.NewModelTransportError(ctx, model, "read response", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, &modelgoogle.APIError{StatusCode: response.StatusCode, Body: string(responseBody)}
	}
	result, err := model.parseResponse(prompt, responseBody)
	if err != nil {
		return nil, err
	}
	for _, conflict := range conflicts {
		result.Warnings = append(
			result.Warnings, "google images: used provider-specific geometry instead of portable "+conflict,
		)
	}
	return result, nil
}
