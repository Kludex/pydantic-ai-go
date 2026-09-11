package xai

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/internal/download"
)

func (model *Model) mapInputs(ctx context.Context, inputs []images.Input) (inputReferences, error) {
	urls := []string{}
	fileIDs := []string{}
	seenURL := false
	orderViolated := false
	for _, input := range inputs {
		if uploaded, ok := input.(ai.UploadedFile); ok {
			if uploaded.ProviderName != model.ProviderName() {
				return inputReferences{}, fmt.Errorf(
					"xai images: uploaded file %q belongs to provider %q", uploaded.FileID, uploaded.ProviderName,
				)
			}
			if seenURL {
				orderViolated = true
			}
			fileIDs = append(fileIDs, uploaded.FileID)
			continue
		}
		if binary, ok := input.(ai.BinaryContent); ok {
			urls = append(urls, "data:"+binary.MediaType+";base64,"+base64.StdEncoding.EncodeToString(binary.Data))
			seenURL = true
			continue
		}
		imageURL := input.(ai.ImageURL)
		if err := imageURL.ForceDownload.Validate(); err != nil {
			return inputReferences{}, err
		}
		value := imageURL.URL
		if imageURL.ForceDownload != ai.FileDownloadNever {
			downloaded, err := download.Fetch(ctx, imageURL.URL, imageURL.ForceDownload == ai.FileDownloadAllowLocal)
			if err != nil {
				return inputReferences{}, fmt.Errorf("xai images: download reference image: %w", err)
			}
			mediaType := imageURL.MediaType
			if mediaType == "" {
				mediaType = downloaded.MediaType
			}
			if mediaType == "" {
				mediaType = images.MediaTypeFromBytes(downloaded.Data)
			}
			if mediaType == "" {
				return inputReferences{}, fmt.Errorf("xai images: cannot determine reference image media type")
			}
			if !strings.HasPrefix(strings.ToLower(mediaType), "image/") {
				return inputReferences{}, fmt.Errorf("xai images: reference content must have an image media type, got %q", mediaType)
			}
			value = "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(downloaded.Data)
		}
		urls = append(urls, value)
		seenURL = true
	}
	if orderViolated {
		return inputReferences{}, fmt.Errorf("xai images: place uploaded files before URL or binary inputs to preserve order")
	}
	if len(inputs) == 1 {
		if len(fileIDs) == 1 {
			return inputReferences{imageFileID: fileIDs[0]}, nil
		}
		return inputReferences{imageURL: urls[0]}, nil
	}
	result := inputReferences{}
	if len(urls) > 0 {
		result.imageURLs = urls
	}
	if len(fileIDs) > 0 {
		result.imageFileIDs = fileIDs
	}
	return result, nil
}
