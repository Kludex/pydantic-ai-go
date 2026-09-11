package xai

import (
	"context"
	"encoding/base64"
	"fmt"

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
		switch input := input.(type) {
		case ai.UploadedFile:
			if input.ProviderName != model.providerName && input.ProviderName != "xai" {
				return inputReferences{}, fmt.Errorf(
					"xai images: uploaded file %q belongs to provider %q", input.FileID, input.ProviderName,
				)
			}
			if seenURL {
				orderViolated = true
			}
			fileIDs = append(fileIDs, input.FileID)
		case ai.BinaryContent:
			urls = append(urls, "data:"+input.MediaType+";base64,"+base64.StdEncoding.EncodeToString(input.Data))
			seenURL = true
		case ai.ImageURL:
			if err := input.ForceDownload.Validate(); err != nil {
				return inputReferences{}, err
			}
			value := input.URL
			if input.ForceDownload != ai.FileDownloadNever {
				downloaded, err := download.Fetch(ctx, input.URL, input.ForceDownload == ai.FileDownloadAllowLocal)
				if err != nil {
					return inputReferences{}, fmt.Errorf("xai images: download reference image: %w", err)
				}
				mediaType := input.MediaType
				if mediaType == "" {
					mediaType = downloaded.MediaType
				}
				if mediaType == "" {
					mediaType = images.MediaTypeFromBytes(downloaded.Data)
				}
				if mediaType == "" {
					return inputReferences{}, fmt.Errorf("xai images: cannot determine reference image media type")
				}
				value = "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(downloaded.Data)
			}
			urls = append(urls, value)
			seenURL = true
		default:
			return inputReferences{}, fmt.Errorf("xai images: unsupported input type %T", input)
		}
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
