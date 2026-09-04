package xai

import (
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

func validateMessages(messages []ai.ModelMessage) error {
	for _, message := range messages {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			prompt, ok := part.(ai.UserPromptPart)
			if !ok {
				continue
			}
			for _, content := range prompt.Contents {
				switch content := content.(type) {
				case ai.TextContent, ai.ImageURL, ai.UploadedFile, ai.CachePoint:
				case ai.BinaryContent:
					if !strings.HasPrefix(strings.ToLower(content.MediaType), "image/") {
						return fmt.Errorf("xai: binary input media type %q is not supported", content.MediaType)
					}
				case ai.AudioURL:
					return fmt.Errorf("xai: audio URL input is not supported")
				case ai.VideoURL:
					return fmt.Errorf("xai: video URL input is not supported")
				}
			}
		}
	}
	return nil
}
