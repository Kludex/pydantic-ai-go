package openai

import (
	"context"
	"fmt"
	"mime"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/download"
)

func liveToolContent(ctx context.Context, items []ai.UserContent) ([]any, error) {
	var parts []any
	for _, item := range items {
		var part map[string]any
		switch item := item.(type) {
		case ai.CachePoint:
			continue
		case ai.TextContent:
			part = map[string]any{"type": "input_text", "text": item.Text}
		case ai.BinaryContent:
			var err error
			part, err = liveBinaryPart(item)
			if err != nil {
				return nil, err
			}
		case ai.UploadedFile:
			if item.ProviderName != "openai" || item.FileID == "" {
				return nil, fmt.Errorf("openai GPT-Live: uploaded file must belong to OpenAI and have an ID")
			}
			media := item.ResolvedMediaType()
			if strings.HasPrefix(media, "audio/") || strings.HasPrefix(media, "video/") {
				return nil, fmt.Errorf("openai GPT-Live: unsupported uploaded media %q", media)
			}
			part = map[string]any{"type": "input_file", "file_id": item.FileID}
			if strings.HasPrefix(media, "image/") {
				part["type"] = "input_image"
				part["detail"] = liveImageDetail(item.VendorMetadata)
			}
		case ai.ImageURL:
			if err := item.ForceDownload.Validate(); err != nil {
				return nil, err
			}
			part = map[string]any{"type": "input_image", "image_url": item.URL, "detail": liveImageDetail(item.VendorMetadata)}
			if item.ForceDownload != ai.FileDownloadNever {
				downloaded, err := download.Fetch(ctx, item.URL, item.ForceDownload == ai.FileDownloadAllowLocal)
				if err != nil {
					return nil, err
				}
				media := downloaded.MediaType
				if media == "" {
					media, err = item.ResolvedMediaType()
					if err != nil {
						return nil, err
					}
				}
				if !strings.HasPrefix(media, "image/") {
					return nil, fmt.Errorf("openai GPT-Live: downloaded image has media type %q", media)
				}
				part["image_url"] = liveDataURL(ai.BinaryContent{Data: downloaded.Data, MediaType: media})
			}
		case ai.DocumentURL:
			if err := item.ForceDownload.Validate(); err != nil {
				return nil, err
			}
			media, mediaErr := item.ResolvedMediaType()
			if strings.HasPrefix(media, "audio/") || strings.HasPrefix(media, "video/") {
				return nil, fmt.Errorf("openai GPT-Live: unsupported document media %q", media)
			}
			part = map[string]any{"type": "input_file", "file_url": item.URL}
			if item.ForceDownload != ai.FileDownloadNever {
				downloaded, err := download.Fetch(ctx, item.URL, item.ForceDownload == ai.FileDownloadAllowLocal)
				if err != nil {
					return nil, err
				}
				if downloaded.MediaType != "" {
					media = downloaded.MediaType
				} else if mediaErr != nil {
					return nil, mediaErr
				}
				part, err = liveBinaryPart(ai.BinaryContent{Data: downloaded.Data, MediaType: media})
				if err != nil {
					return nil, err
				}
			}
		default:
			return nil, fmt.Errorf("openai GPT-Live: backend tool content supports only text, images, and documents, got %T", item)
		}
		parts = append(parts, part)
	}
	return parts, nil
}

func liveBinaryPart(content ai.BinaryContent) (map[string]any, error) {
	if strings.HasPrefix(content.MediaType, "image/") {
		return map[string]any{"type": "input_image", "image_url": liveDataURL(content), "detail": liveImageDetail(content.VendorMetadata)}, nil
	}
	if strings.HasPrefix(content.MediaType, "audio/") || strings.HasPrefix(content.MediaType, "video/") {
		return nil, fmt.Errorf("openai GPT-Live: unsupported backend media %q", content.MediaType)
	}
	extensions, _ := mime.ExtensionsByType(content.MediaType)
	if len(extensions) == 0 {
		return nil, fmt.Errorf("openai GPT-Live: unsupported document media %q", content.MediaType)
	}
	return map[string]any{"type": "input_file", "file_data": liveDataURL(content), "filename": "filename" + extensions[0]}, nil
}

func liveImageDetail(metadata map[string]any) string {
	detail := stringValue(metadata["detail"])
	if detail == "" {
		detail = "auto"
	}
	return detail
}
