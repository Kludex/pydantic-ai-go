package ai

import "strings"

func frameToolReturnContent(toolName, toolCallID string, contents []UserContent) []UserContent {
	framed := make([]UserContent, 0, len(contents)*3)
	for _, content := range contents {
		identifier, ok := userContentIdentifier(content)
		if !ok {
			framed = append(framed, content)
			continue
		}
		framed = append(framed,
			TextContent{Text: `<tool_result tool_name="` + escapeToolResultAttribute(toolName) +
				`" tool_call_id="` + escapeToolResultAttribute(toolCallID) +
				`" file_id="` + escapeToolResultAttribute(identifier) + `">`},
			content,
			TextContent{Text: "</tool_result>"},
		)
	}
	return framed
}

func userContentIdentifier(content UserContent) (string, bool) {
	switch content := content.(type) {
	case ImageURL:
		return content.ResolvedIdentifier(), true
	case VideoURL:
		return content.ResolvedIdentifier(), true
	case AudioURL:
		return content.ResolvedIdentifier(), true
	case DocumentURL:
		return content.ResolvedIdentifier(), true
	case BinaryContent:
		return content.ResolvedIdentifier(), true
	case UploadedFile:
		return content.ResolvedIdentifier(), true
	default:
		return "", false
	}
}

func escapeToolResultAttribute(value string) string {
	return strings.NewReplacer(
		"&", "&amp;",
		`"`, "&quot;",
		"'", "&#x27;",
		"<", "&lt;",
		">", "&gt;",
	).Replace(value)
}
