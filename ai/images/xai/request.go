package xai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

type inputReferences struct {
	imageURL     string
	imageFileID  string
	imageURLs    []string
	imageFileIDs []string
}

// Generate creates, edits, or batches images through xAI's image endpoint.
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
	if provider.n != nil && *provider.n <= 0 {
		return nil, fmt.Errorf("xai images: image count must be greater than zero")
	}
	ratio, resolution, conflicts, err := resolveGeometry(model.name, settings, provider)
	if err != nil {
		return nil, err
	}
	references, err := model.mapInputs(ctx, inputs)
	if err != nil {
		return nil, err
	}
	body := maps.Clone(settings.ExtraBody)
	if body == nil {
		body = map[string]any{}
	}
	for _, key := range []string{
		"model", "prompt", "n", "response_format", "user", "aspect_ratio", "resolution",
		"image_url", "image_file_id", "image_urls", "image_file_ids",
	} {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("xai images: extra body field %q conflicts with typed settings", key)
		}
	}
	body["model"] = model.name
	body["prompt"] = prompt
	body["response_format"] = "b64_json"
	if provider.n != nil {
		body["n"] = *provider.n
	}
	if provider.user != "" {
		body["user"] = provider.user
	}
	if ratio != "" {
		body["aspect_ratio"] = ratio
	}
	if resolution != "" {
		body["resolution"] = resolution
	}
	if references.imageURL != "" {
		body["image_url"] = references.imageURL
	}
	if references.imageFileID != "" {
		body["image_file_id"] = references.imageFileID
	}
	if references.imageURLs != nil {
		body["image_urls"] = references.imageURLs
	}
	if references.imageFileIDs != nil {
		body["image_file_ids"] = references.imageFileIDs
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("xai images: encode request: %w", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, model.endpoint(), bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	if err := model.configureRequest(request, settings.ExtraHeaders); err != nil {
		return nil, err
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("xai images: request: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("xai images: read response: %w", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, &APIError{StatusCode: response.StatusCode, Body: string(responseBody)}
	}
	result, err := model.parseResponse(prompt, responseBody)
	if err != nil {
		return nil, err
	}
	for _, conflict := range conflicts {
		result.Warnings = append(result.Warnings, "xai images: used provider-specific geometry instead of portable "+conflict)
	}
	return result, nil
}

// APIError is a non-success response from xAI's image endpoint.
type APIError struct {
	StatusCode int
	Body       string
}

// Error formats the provider status and body.
func (error *APIError) Error() string {
	return fmt.Sprintf("xai images: API returned status %d: %s", error.StatusCode, error.Body)
}

// IsModelAPIError marks provider API responses as eligible for fallback decisions.
func (*APIError) IsModelAPIError() bool { return true }
