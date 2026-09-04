package mistral

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func toolResultContent(value any) (string, []ai.UserContent, error) {
	var trailing []ai.UserContent
	switch rich := value.(type) {
	case ai.ToolReturn:
		value, trailing = rich.ReturnValue, slices.Clone(rich.Content)
	case *ai.ToolReturn:
		if rich != nil {
			value, trailing = rich.ReturnValue, slices.Clone(rich.Content)
		}
	}
	content, err := stringify(value)
	if err != nil {
		return "", nil, fmt.Errorf("mistral: marshal tool result: %w", err)
	}
	if len(trailing) == 0 {
		return content, nil, nil
	}
	expanded := make([]ai.UserContent, 0, len(trailing)*2)
	for _, item := range trailing {
		identifier := userContentIdentifier(item)
		if identifier != "" {
			content += "\nSee file " + identifier
			expanded = append(expanded, ai.TextContent{Text: "This is file " + identifier + ":"})
		}
		expanded = append(expanded, item)
	}
	return content, expanded, nil
}

func stringify(value any) (string, error) {
	if text, ok := value.(string); ok {
		return text, nil
	}
	data, err := json.Marshal(value)
	return string(data), err
}

func userContentIdentifier(content ai.UserContent) string {
	identifier := ""
	switch content := content.(type) {
	case ai.ImageURL:
		identifier = content.ResolvedIdentifier()
	case ai.DocumentURL:
		identifier = content.ResolvedIdentifier()
	case ai.AudioURL:
		identifier = content.ResolvedIdentifier()
	case ai.VideoURL:
		identifier = content.ResolvedIdentifier()
	case ai.BinaryContent:
		identifier = content.ResolvedIdentifier()
	case ai.UploadedFile:
		identifier = content.Identifier
	}
	return identifier
}

func generatedToolCallID(name string, arguments json.RawMessage) string {
	digest := sha256.Sum256(append(append([]byte(name), 0), arguments...))
	return "mistral-" + hex.EncodeToString(digest[:8])
}
