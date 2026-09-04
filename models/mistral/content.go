package mistral

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/internal/download"
)

func (model *Model) userContent(ctx context.Context, part ai.UserPromptPart) (any, error) {
	if len(part.Contents) == 0 {
		return part.Content, nil
	}
	content := make([]contentChunk, 0, len(part.Contents))
	for _, item := range part.Contents {
		chunks, err := model.userContentItem(ctx, item)
		if err != nil {
			return nil, err
		}
		content = append(content, chunks...)
	}
	return content, nil
}

func (model *Model) userContentItem(ctx context.Context, item ai.UserContent) ([]contentChunk, error) {
	unsupported := fmt.Sprintf("mistral: user content %T is not supported", item)
	switch item := item.(type) {
	case ai.TextContent:
		return []contentChunk{{Type: "text", Text: item.Text}}, nil
	case ai.CachePoint:
		return nil, nil
	case ai.ImageURL:
		location := item.URL
		if item.ForceDownload != ai.FileDownloadNever {
			data, mediaType, err := downloadURL(ctx, item.URL, item.ForceDownload)
			if err != nil {
				return nil, err
			}
			if mediaType == "" {
				mediaType, err = item.ResolvedMediaType()
				if err != nil {
					return nil, err
				}
			}
			location = dataURI(mediaType, data)
		}
		return []contentChunk{{Type: "image_url", ImageURL: &imageURL{
			URL: location, Detail: imageDetail(item.VendorMetadata),
		}}}, nil
	case ai.BinaryContent:
		return binaryContentChunks(item)
	case ai.DocumentURL:
		mediaType, err := item.ResolvedMediaType()
		if err != nil {
			return nil, err
		}
		if isTextLike(mediaType) {
			mode := item.ForceDownload
			if mode == ai.FileDownloadNever {
				mode = ai.FileDownloadSafe
			}
			data, downloadedType, err := downloadURL(ctx, item.URL, mode)
			if err != nil {
				return nil, err
			}
			if downloadedType != "" {
				mediaType = downloadedType
			}
			return []contentChunk{{
				Type: "text", Text: inlineTextFile(data, mediaType, item.ResolvedIdentifier()),
			}}, nil
		}
		if mediaType != "application/pdf" {
			return nil, fmt.Errorf("mistral: document media type %q is not supported", mediaType)
		}
		location := item.URL
		if item.ForceDownload != ai.FileDownloadNever {
			data, downloadedType, err := downloadURL(ctx, item.URL, item.ForceDownload)
			if err != nil {
				return nil, err
			}
			if downloadedType != "" {
				mediaType = downloadedType
			}
			location = dataURI(mediaType, data)
		}
		return []contentChunk{{Type: "document_url", DocumentURL: location}}, nil
	case ai.AudioURL:
		unsupported = "mistral: audio URL input is not supported"
	case ai.VideoURL:
		unsupported = "mistral: video URL input is not supported"
	case ai.UploadedFile:
		unsupported = "mistral: uploaded-file input is not supported"
	}
	return nil, fmt.Errorf("%s", unsupported)
}

func binaryContentChunks(item ai.BinaryContent) ([]contentChunk, error) {
	mediaType := strings.ToLower(item.MediaType)
	switch {
	case isTextLike(mediaType):
		return []contentChunk{{
			Type: "text", Text: inlineTextFile(item.Data, mediaType, item.ResolvedIdentifier()),
		}}, nil
	case strings.HasPrefix(mediaType, "image/"):
		return []contentChunk{{Type: "image_url", ImageURL: &imageURL{
			URL: dataURI(mediaType, item.Data), Detail: imageDetail(item.VendorMetadata),
		}}}, nil
	case mediaType == "application/pdf":
		return []contentChunk{{Type: "document_url", DocumentURL: dataURI(mediaType, item.Data)}}, nil
	default:
		return nil, fmt.Errorf("mistral: binary media type %q is not supported", item.MediaType)
	}
}

func imageDetail(metadata map[string]any) string {
	if len(metadata) == 0 {
		return ""
	}
	detail, _ := metadata["detail"].(string)
	if detail == "" {
		return "auto"
	}
	return detail
}

func dataURI(mediaType string, data []byte) string {
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func isTextLike(mediaType string) bool {
	mediaType = strings.ToLower(mediaType)
	return strings.HasPrefix(mediaType, "text/") || mediaType == "application/json" ||
		strings.HasSuffix(mediaType, "+json") || mediaType == "application/xml" ||
		strings.HasSuffix(mediaType, "+xml") || mediaType == "application/yaml" || mediaType == "application/x-yaml"
}

func inlineTextFile(data []byte, mediaType, identifier string) string {
	return fmt.Sprintf(
		"-----BEGIN FILE id=\"%s\" type=\"%s\"-----\n%s\n-----END FILE id=\"%s\"-----",
		identifier, mediaType, data, identifier,
	)
}

func downloadURL(ctx context.Context, rawURL string, mode ai.FileDownloadMode) ([]byte, string, error) {
	if err := mode.Validate(); err != nil {
		return nil, "", err
	}
	result, err := download.Fetch(ctx, rawURL, mode == ai.FileDownloadAllowLocal)
	if err != nil {
		return nil, "", err
	}
	return result.Data, result.MediaType, nil
}
