package vercel

import (
	"encoding/base64"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func userFile(part UIMessagePart) (ai.UserContent, error) {
	if part.URL == "" {
		return nil, fmt.Errorf("vercel: file URL must not be empty")
	}
	metadata := loadPartMetadata(part.ProviderMetadata)
	if metadata.fileID != "" && metadata.providerName != "" {
		return ai.UploadedFile{
			FileID: metadata.fileID, ProviderName: metadata.providerName, MediaType: part.MediaType,
			Identifier: metadata.identifier, VendorMetadata: cloneMap(metadata.vendorMetadata),
		}, nil
	}
	if strings.HasPrefix(part.URL, "data:") {
		file, err := binaryFile(part.URL)
		file.Identifier = metadata.identifier
		file.VendorMetadata = cloneMap(metadata.vendorMetadata)
		return file, err
	}
	switch strings.ToLower(strings.SplitN(part.MediaType, "/", 2)[0]) {
	case "image":
		return ai.ImageURL{
			URL: part.URL, MediaType: part.MediaType, Identifier: metadata.identifier,
			ForceDownload: metadata.forceDownload, VendorMetadata: cloneMap(metadata.vendorMetadata),
		}, nil
	case "video":
		return ai.VideoURL{
			URL: part.URL, MediaType: part.MediaType, Identifier: metadata.identifier,
			ForceDownload: metadata.forceDownload, VendorMetadata: cloneMap(metadata.vendorMetadata),
		}, nil
	case "audio":
		return ai.AudioURL{
			URL: part.URL, MediaType: part.MediaType, Identifier: metadata.identifier,
			ForceDownload: metadata.forceDownload, VendorMetadata: cloneMap(metadata.vendorMetadata),
		}, nil
	default:
		return ai.DocumentURL{
			URL: part.URL, MediaType: part.MediaType, Identifier: metadata.identifier,
			ForceDownload: metadata.forceDownload, VendorMetadata: cloneMap(metadata.vendorMetadata),
		}, nil
	}
}

func binaryFile(dataURI string) (ai.BinaryContent, error) {
	header, payload, found := strings.Cut(strings.TrimPrefix(dataURI, "data:"), ",")
	if !found || !strings.HasSuffix(header, ";base64") {
		return ai.BinaryContent{}, fmt.Errorf("vercel: file data URL must contain base64 data")
	}
	mediaType := strings.TrimSuffix(header, ";base64")
	if mediaType == "" {
		return ai.BinaryContent{}, fmt.Errorf("vercel: file data URL must contain a media type")
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return ai.BinaryContent{}, fmt.Errorf("vercel: decode file data URL: %w", err)
	}
	return ai.BinaryContent{Data: data, MediaType: mediaType}, nil
}

func normalizeClientToolReturnContent(value any) any {
	switch value := value.(type) {
	case []any:
		normalized := make([]any, len(value))
		for index, item := range value {
			normalized[index] = normalizeClientToolReturnContent(item)
		}
		return normalized
	case map[string]any:
		normalized := make(map[string]any, len(value))
		for key, item := range value {
			normalized[key] = normalizeClientToolReturnContent(item)
		}
		kind, _ := normalized["kind"].(string)
		if kind == "binary" {
			if mediaType, ok := normalized["media_type"].(string); ok && mediaType != "" {
				if data, ok := jsBinaryBytes(normalized["data"]); ok {
					normalized["data"] = base64.URLEncoding.EncodeToString(data)
				}
			}
		} else if mediaType, exists := normalized["media_type"]; !exists || mediaType == nil || mediaType == "" {
			if inferred, ok := inferredURLMediaType(kind, normalized["url"]); ok {
				normalized["media_type"] = inferred
			}
		}
		return normalized
	default:
		return value
	}
}

func inferredURLMediaType(kind string, value any) (string, bool) {
	rawURL, ok := value.(string)
	if !ok || rawURL == "" {
		return "", false
	}
	var mediaType string
	var err error
	switch kind {
	case "image-url":
		mediaType, err = (ai.ImageURL{URL: rawURL}).ResolvedMediaType()
	case "video-url":
		mediaType, err = (ai.VideoURL{URL: rawURL}).ResolvedMediaType()
	case "audio-url":
		mediaType, err = (ai.AudioURL{URL: rawURL}).ResolvedMediaType()
	case "document-url":
		mediaType, err = (ai.DocumentURL{URL: rawURL}).ResolvedMediaType()
	default:
		return "", false
	}
	return mediaType, err == nil && mediaType != ""
}

func jsBinaryBytes(value any) ([]byte, bool) {
	mapping, ok := value.(map[string]any)
	if !ok {
		return nil, false
	}
	if mapping["type"] == "Buffer" {
		values, ok := mapping["data"].([]any)
		if !ok {
			return nil, false
		}
		return jsonBytes(values)
	}
	if len(mapping) == 0 {
		return nil, false
	}
	values := make([]any, len(mapping))
	for index := range values {
		value, ok := mapping[fmt.Sprint(index)]
		if !ok {
			return nil, false
		}
		values[index] = value
	}
	return jsonBytes(values)
}

func jsonBytes(values []any) ([]byte, bool) {
	result := make([]byte, len(values))
	for index, value := range values {
		number, ok := value.(float64)
		if !ok || number < 0 || number > 255 || number != float64(byte(number)) {
			return nil, false
		}
		result[index] = byte(number)
	}
	return result, true
}

func fileChunk(part ai.FilePart) Chunk {
	return Chunk{
		Type:      ChunkFile,
		URL:       "data:" + part.Content.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Content.Data),
		MediaType: part.Content.MediaType,
		ProviderMetadata: dumpPartMetadata(
			part.ID, "", part.ProviderName, part.ProviderDetails, "", part.Content.VendorMetadata,
		),
	}
}
