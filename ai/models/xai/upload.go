package xai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"path"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/download"
)

func (model *Model) prepareMessages(ctx context.Context, messages []ai.ModelMessage) ([]ai.ModelMessage, error) {
	prepared := ai.ModelRequestContext{Messages: messages}.Clone().Messages
	for messageIndex, message := range prepared {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for partIndex, part := range request.Parts {
			prompt, ok := part.(ai.UserPromptPart)
			if !ok || len(prompt.Contents) == 0 {
				continue
			}
			for contentIndex, content := range prompt.Contents {
				var data []byte
				var filename, mediaType string
				switch content := content.(type) {
				case ai.BinaryContent:
					if !isDocumentMediaType(content.MediaType) {
						continue
					}
					data = append([]byte(nil), content.Data...)
					mediaType = content.MediaType
					filename = content.Identifier
				case ai.DocumentURL:
					downloaded, err := download.Fetch(ctx, content.URL, content.ForceDownload == ai.FileDownloadAllowLocal)
					if err != nil {
						return nil, fmt.Errorf("xai: download document: %w", err)
					}
					data = downloaded.Data
					mediaType = content.MediaType
					if mediaType == "" {
						mediaType = downloaded.MediaType
					}
					filename = content.Identifier
					if filename == "" {
						filename = filenameFromURL(downloaded.URL)
					}
				default:
					continue
				}
				if filename == "" {
					filename = documentFilename(mediaType)
				}
				fileID, err := model.uploadFile(ctx, data, filename, mediaType)
				if err != nil {
					return nil, err
				}
				prompt.Contents[contentIndex] = ai.UploadedFile{
					FileID: fileID, ProviderName: "xai", MediaType: mediaType,
				}
			}
			request.Parts[partIndex] = prompt
		}
		prepared[messageIndex] = request
	}
	return prepared, nil
}

func (model *Model) uploadFile(ctx context.Context, data []byte, filename, mediaType string) (string, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	headers := make(textproto.MIMEHeader)
	headers.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename=%q`, filename))
	headers.Set("Content-Type", mediaType)
	part, _ := writer.CreatePart(headers)
	_, _ = part.Write(data)
	_ = writer.Close()
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, strings.TrimRight(model.baseURL, "/")+"/files", &body,
	)
	if err != nil {
		return "", err
	}
	request.Header.Set("Content-Type", writer.FormDataContentType())
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	for name, values := range model.headers {
		request.Header.Del(name)
		for _, value := range values {
			request.Header.Add(name, value)
		}
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return "", fmt.Errorf("xai: prepare file upload: %w", err)
		}
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return "", ai.NewModelTransportError(ctx, model, "upload file", err)
	}
	defer func() { _ = response.Body.Close() }()
	encoded, err := io.ReadAll(response.Body)
	if err != nil {
		return "", ai.NewModelTransportError(ctx, model, "read file upload", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return "", fmt.Errorf("xai: file upload returned %s: %s", response.Status, strings.TrimSpace(string(encoded)))
	}
	var uploaded struct {
		ID   string `json:"id"`
		File *struct {
			ID string `json:"id"`
		} `json:"file"`
	}
	if err := json.Unmarshal(encoded, &uploaded); err != nil {
		return "", fmt.Errorf("xai: decode file upload: %w", err)
	}
	if uploaded.ID == "" && uploaded.File != nil {
		uploaded.ID = uploaded.File.ID
	}
	if uploaded.ID == "" {
		return "", fmt.Errorf("xai: file upload response contained no file ID")
	}
	return uploaded.ID, nil
}

func isDocumentMediaType(mediaType string) bool {
	mediaType = strings.ToLower(mediaType)
	return strings.HasPrefix(mediaType, "application/") || strings.HasPrefix(mediaType, "text/")
}

func filenameFromURL(rawURL string) string {
	parsed, _ := url.Parse(rawURL)
	filename := path.Base(parsed.Path)
	if filename == "." || filename == "/" {
		return ""
	}
	return filename
}

func documentFilename(mediaType string) string {
	extensions, _ := mime.ExtensionsByType(mediaType)
	if len(extensions) > 0 {
		return "document" + extensions[0]
	}
	return "document"
}
