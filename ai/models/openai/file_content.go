package openai

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/download"
)

type downloadedFileContent struct {
	data      []byte
	dataURI   string
	mediaType string
}

func downloadFileContent(
	ctx context.Context,
	rawURL string,
	resolveMediaType func() (string, error),
	mode ai.FileDownloadMode,
) (downloadedFileContent, error) {
	if err := mode.Validate(); err != nil {
		return downloadedFileContent{}, err
	}
	downloaded, err := download.Fetch(ctx, rawURL, mode == ai.FileDownloadAllowLocal)
	if err != nil {
		return downloadedFileContent{}, err
	}
	mediaType := downloaded.MediaType
	if mediaType == "" {
		mediaType, err = resolveMediaType()
		if err != nil {
			return downloadedFileContent{}, err
		}
	}
	return downloadedFileContent{
		data:      downloaded.Data,
		dataURI:   "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(downloaded.Data),
		mediaType: mediaType,
	}, nil
}

var fileExtensions = map[string]string{
	"application/msword":       "doc",
	"application/pdf":          "pdf",
	"application/vnd.ms-excel": "xls",
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":       "xlsx",
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
	"audio/aac":     "aac",
	"audio/aiff":    "aiff",
	"audio/flac":    "flac",
	"audio/mpeg":    "mp3",
	"audio/ogg":     "oga",
	"audio/wav":     "wav",
	"image/gif":     "gif",
	"image/jpeg":    "jpeg",
	"image/png":     "png",
	"image/webp":    "webp",
	"text/csv":      "csv",
	"text/html":     "html",
	"text/markdown": "md",
	"text/plain":    "txt",
}

func fileExtension(mediaType string) (string, error) {
	extension := fileExtensions[strings.ToLower(mediaType)]
	if extension == "" {
		return "", fmt.Errorf("openai: unsupported file media type %q", mediaType)
	}
	return extension, nil
}

func (model *Model) chatAudioPart(dataURI, mediaType string) (contentPart, error) {
	format, err := fileExtension(mediaType)
	if err != nil || format != "mp3" && format != "wav" {
		return contentPart{}, fmt.Errorf("%s: unsupported Chat Completions audio media type %q", model.providerName, mediaType)
	}
	data := dataURI
	if !model.chatCompatibility.AudioInputDataURI {
		_, data, _ = strings.Cut(dataURI, ",")
	}
	return contentPart{Type: "input_audio", InputAudio: &inputAudio{Data: data, Format: format}}, nil
}

func (model *Model) chatDocumentPart(
	data []byte, dataURI, mediaType, identifier string,
) (contentPart, error) {
	if isTextLikeMediaType(mediaType) {
		return contentPart{
			Type: "text",
			Text: fmt.Sprintf(
				"-----BEGIN FILE id=\"%s\" type=\"%s\"-----\n%s\n-----END FILE id=\"%s\"-----",
				identifier, mediaType, data, identifier,
			),
		}, nil
	}
	if model.chatCompatibility.DisableDocumentInput {
		return contentPart{}, fmt.Errorf(
			"%s: Chat Completions does not support document input", model.providerName,
		)
	}
	extension, err := fileExtension(mediaType)
	if err != nil {
		return contentPart{}, err
	}
	return contentPart{
		Type: "file", File: &chatFile{FileData: dataURI, Filename: "filename." + extension},
	}, nil
}

func chatImageDetail(metadata map[string]any) string {
	if len(metadata) == 0 {
		return ""
	}
	return imageDetail(metadata)
}

func imageDetail(metadata map[string]any) string {
	detail, _ := metadata["detail"].(string)
	if detail == "" {
		return "auto"
	}
	return detail
}

func isTextLikeMediaType(mediaType string) bool {
	mediaType = strings.ToLower(mediaType)
	return strings.HasPrefix(mediaType, "text/") || mediaType == "application/json" ||
		strings.HasSuffix(mediaType, "+json") || mediaType == "application/xml" ||
		strings.HasSuffix(mediaType, "+xml") || mediaType == "application/x-yaml" || mediaType == "application/yaml"
}

func isImageMediaType(mediaType string) bool {
	return strings.HasPrefix(strings.ToLower(mediaType), "image/")
}

func isAudioMediaType(mediaType string) bool {
	return strings.HasPrefix(strings.ToLower(mediaType), "audio/")
}

func isVideoMediaType(mediaType string) bool {
	return strings.HasPrefix(strings.ToLower(mediaType), "video/")
}
