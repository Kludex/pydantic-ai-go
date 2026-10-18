// Package thinking renders reasoning that a provider cannot replay through its native channel.
package thinking

import (
	"regexp"
	"strings"
)

var (
	closingTag      = regexp.MustCompile(`(?i)<\s*/\s*assistant_thinking[\s/>]`)
	attributeEscape = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&#x27;")
)

// Carried wraps reasoning sent in a user turn so it reads as the named model's, not the user's.
// Empty attributes are omitted. A closing tag inside the content is escaped so it cannot end the block.
func Carried(content, modelName, providerName string) string {
	var text strings.Builder
	text.WriteString("<assistant_thinking")
	for _, attribute := range [][2]string{{"model", modelName}, {"provider", providerName}} {
		if attribute[1] != "" {
			text.WriteString(" " + attribute[0] + `="` + attributeEscape.Replace(attribute[1]) + `"`)
		}
	}
	text.WriteString(">\n")
	text.WriteString(closingTag.ReplaceAllStringFunc(content, func(tag string) string { return "&lt;" + tag[1:] }))
	text.WriteString("\n</assistant_thinking>")
	return text.String()
}
