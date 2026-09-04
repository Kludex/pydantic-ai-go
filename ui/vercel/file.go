package vercel

import (
	"encoding/base64"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

func userFile(part UIMessagePart) (ai.UserContent, error) {
	if part.URL == "" {
		return nil, fmt.Errorf("vercel: file URL must not be empty")
	}
	if strings.HasPrefix(part.URL, "data:") {
		return binaryFile(part.URL)
	}
	switch strings.ToLower(strings.SplitN(part.MediaType, "/", 2)[0]) {
	case "image":
		return ai.ImageURL{URL: part.URL, MediaType: part.MediaType}, nil
	case "video":
		return ai.VideoURL{URL: part.URL, MediaType: part.MediaType}, nil
	case "audio":
		return ai.AudioURL{URL: part.URL, MediaType: part.MediaType}, nil
	default:
		return ai.DocumentURL{URL: part.URL, MediaType: part.MediaType}, nil
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

func fileChunk(part ai.FilePart) Chunk {
	return Chunk{
		Type:      ChunkFile,
		URL:       "data:" + part.Content.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Content.Data),
		MediaType: part.Content.MediaType,
	}
}
