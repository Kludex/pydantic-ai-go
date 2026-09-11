package google

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/internal/download"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

func (model *Model) mapInput(ctx context.Context, input images.Input) (part, error) {
	var mapped part
	switch input := input.(type) {
	case ai.BinaryContent:
		mapped.InlineData = &inlineData{MIMEType: input.MediaType, Data: base64.StdEncoding.EncodeToString(input.Data)}
		mapped.MediaResolution = input.VendorMetadata["media_resolution"]
	case ai.UploadedFile:
		if model.transport == modelgoogle.TransportVertexAI {
			return part{}, fmt.Errorf("google images: Vertex AI does not accept uploaded file references")
		}
		if input.ProviderName != model.providerName && input.ProviderName != "google" &&
			input.ProviderName != "google-gla" {
			return part{}, fmt.Errorf(
				"google images: uploaded file %q belongs to provider %q", input.FileID, input.ProviderName,
			)
		}
		if !strings.HasPrefix(input.FileID, "https://") {
			return part{}, fmt.Errorf("google images: uploaded file ID must be an HTTPS Files API URI")
		}
		mapped.FileData = &fileData{MIMEType: input.ResolvedMediaType(), FileURI: input.FileID}
		mapped.MediaResolution = input.VendorMetadata["media_resolution"]
	case ai.ImageURL:
		if err := input.ForceDownload.Validate(); err != nil {
			return part{}, err
		}
		if input.ForceDownload == ai.FileDownloadNever && model.transport != modelgoogle.TransportVertexAI &&
			strings.HasPrefix(input.URL, "https://generativelanguage.googleapis.com/v1beta/files") {
			mediaType, err := input.ResolvedMediaType()
			if err != nil {
				return part{}, fmt.Errorf("google images: Files API image URL needs an explicit media type: %w", err)
			}
			mapped.FileData = &fileData{MIMEType: mediaType, FileURI: input.URL}
		} else {
			downloaded, err := download.Fetch(ctx, input.URL, input.ForceDownload == ai.FileDownloadAllowLocal)
			if err != nil {
				return part{}, fmt.Errorf("google images: download reference image: %w", err)
			}
			mediaType := input.MediaType
			if mediaType == "" {
				mediaType = downloaded.MediaType
			}
			if mediaType == "" {
				mediaType = images.MediaTypeFromBytes(downloaded.Data)
			}
			if mediaType == "" {
				return part{}, fmt.Errorf("google images: cannot determine reference image media type")
			}
			mapped.InlineData = &inlineData{MIMEType: mediaType, Data: base64.StdEncoding.EncodeToString(downloaded.Data)}
		}
		mapped.MediaResolution = input.VendorMetadata["media_resolution"]
	default:
		return part{}, fmt.Errorf("google images: unsupported input type %T", input)
	}
	return mapped, nil
}
