package openai

import (
	"bytes"
	"context"
	"fmt"
	"mime/multipart"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/internal/download"
)

func (model *Model) editRequest(
	ctx context.Context, prompt string, inputs []images.Input, size string,
	provider providerSettings, settings images.Settings,
) (*http.Request, error) {
	fields := map[string]any{"model": model.name, "prompt": prompt}
	setProviderBody(fields, size, provider, true)
	for key, value := range settings.ExtraBody {
		if _, exists := fields[key]; exists {
			return nil, fmt.Errorf("openai images: extra body field %q conflicts with typed settings", key)
		}
		fields[key] = value
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for key, value := range fields {
		text, err := formValue(value)
		if err != nil {
			return nil, fmt.Errorf("openai images: encode multipart field %q: %w", key, err)
		}
		_ = writer.WriteField(key, text)
	}
	for index, input := range inputs {
		data, mediaType, err := openAIInput(ctx, input, model.providerName)
		if err != nil {
			return nil, err
		}
		extension := mediaExtension(mediaType)
		if extension == "" {
			return nil, fmt.Errorf(
				"openai images: editing supports PNG, JPEG, or WebP inputs, got %q", mediaType,
			)
		}
		part, _ := writer.CreatePart(fileHeader("image[]", fmt.Sprintf("image-%d.%s", index, extension), mediaType))
		_, _ = part.Write(data)
	}
	_ = writer.Close()
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, model.endpoint("/images/edits"), &body,
	)
	if err == nil {
		request.Header.Set("Content-Type", writer.FormDataContentType())
	}
	return request, err
}

func openAIInput(ctx context.Context, input images.Input, providerName string) ([]byte, string, error) {
	if binary, ok := input.(ai.BinaryContent); ok {
		return append([]byte(nil), binary.Data...), binary.MediaType, nil
	}
	if imageURL, ok := input.(ai.ImageURL); ok {
		if err := imageURL.ForceDownload.Validate(); err != nil {
			return nil, "", err
		}
		downloaded, err := download.Fetch(ctx, imageURL.URL, imageURL.ForceDownload == ai.FileDownloadAllowLocal)
		if err != nil {
			return nil, "", fmt.Errorf("openai images: download reference image: %w", err)
		}
		mediaType := imageURL.MediaType
		if mediaType == "" {
			mediaType = downloaded.MediaType
		}
		if mediaType == "" {
			mediaType, err = imageURL.ResolvedMediaType()
			if err != nil {
				return nil, "", err
			}
		}
		return downloaded.Data, mediaType, nil
	}
	uploaded := input.(ai.UploadedFile)
	if uploaded.ProviderName != providerName {
		return nil, "", fmt.Errorf(
			"openai images: uploaded file %q belongs to provider %q", uploaded.FileID, uploaded.ProviderName,
		)
	}
	return nil, "", fmt.Errorf(
		"openai images: editing requires file content and does not accept uploaded file IDs",
	)
}
