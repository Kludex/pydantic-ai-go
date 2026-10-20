package elevenlabs

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// audioTag matches an expressive audio tag such as [laughs] or [strong French
// accent]: words in square brackets. A match followed by "(" is a Markdown link
// and is kept. Tags are lowercase by convention, so [Your Name] is kept too.
var audioTag = regexp.MustCompile(`\[(\pL(?:\pL|[ '\x{2019}-]){1,49})\]`)

// audioTagStart matches what the rest of an audio tag could still follow.
var audioTagStart = regexp.MustCompile(`^\[(?:\pL(?:\pL|[ '\x{2019}-]){0,49}\]?)?$`)

// stripAudioTags removes audio tags, returning the text left and the tags
// without brackets. A tag goes with the whitespace before it, or after it at
// the start of text that has not started yet, so "Hi [laughs] there" becomes
// "Hi there" and "[laughs] Hi" becomes "Hi".
func stripAudioTags(text string, started bool) (string, []string) {
	var tags []string
	var result strings.Builder
	position := 0
	for _, match := range audioTag.FindAllStringSubmatchIndex(text, -1) {
		tag := text[match[2]:match[3]]
		first, _ := utf8.DecodeRuneInString(tag)
		if unicode.IsUpper(first) || strings.HasPrefix(text[match[1]:], "(") {
			continue
		}
		kept := strings.TrimRightFunc(result.String()+text[position:match[0]], unicode.IsSpace)
		result.Reset()
		result.WriteString(kept)
		tags = append(tags, tag)
		position = match[1]
	}
	result.WriteString(text[position:])
	if started {
		return result.String(), tags
	}
	return strings.TrimLeftFunc(result.String(), unicode.IsSpace), tags
}

// audioTagStripper strips tags from streamed text so the pieces add up to the
// whole text stripped at once. Text from a bracket that may still open a tag,
// and trailing whitespace a following tag would remove, are held back.
type audioTagStripper struct {
	held    string
	started bool
}

func (stripper *audioTagStripper) feed(text string) string {
	buffer := stripper.held + text
	end := strings.LastIndex(buffer, "[")
	if end == -1 || !audioTagStart.MatchString(buffer[end:]) {
		end = len(buffer)
	}
	head := strings.TrimRightFunc(buffer[:end], unicode.IsSpace)
	stripper.held = buffer[len(head):]
	stripped, _ := stripAudioTags(head, stripper.started)
	stripper.started = stripper.started || stripped != ""
	return stripped
}
