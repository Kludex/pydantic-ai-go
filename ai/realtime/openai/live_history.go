package openai

import (
	"fmt"
	"regexp"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/tiktoken-go/tokenizer"
)

func seedLiveItems(messages []ai.ModelMessage, replay bool) ([]map[string]any, error) {
	var items []map[string]any
	add := func(role, text string) {
		text = strings.TrimSpace(text)
		if text == "" {
			return
		}
		kind := "input_text"
		if role == "assistant" {
			kind = "output_text"
		}
		items = append(items, map[string]any{"role": role, "content": []any{map[string]any{"type": kind, "text": text}}})
	}
	for _, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.UserPromptPart:
					text := part.Content
					if len(part.Contents) > 0 {
						var texts []string
						for _, content := range part.Contents {
							if content, ok := content.(ai.TextContent); ok {
								texts = append(texts, content.Text)
							} else if !replay {
								return nil, fmt.Errorf("openai GPT-Live: seed history accepts text only, got %T", content)
							}
						}
						text = strings.Join(texts, "\n")
					}
					add("user", text)
				case ai.SpeechPart:
					if part.Transcript != nil {
						add("user", *part.Transcript)
					} else if part.Audio != nil && !replay {
						return nil, fmt.Errorf("openai GPT-Live: seeded audio needs a transcript")
					}
				case ai.ToolReturnPart:
					add("user", fmt.Sprintf("Result of `%s`: %s", part.ToolName, render(part.Content)))
				case ai.RetryPromptPart:
					attempt := "The previous attempt failed"
					if part.ToolName != "" {
						attempt = "`" + part.ToolName + "` failed"
					}
					add("user", attempt+": "+render(part.Content))
				}
			}
		case ai.ModelResponse:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.TextPart:
					add("assistant", part.Content)
				case ai.ThinkingPart:
					add("assistant", part.Content)
				case ai.SpeechPart:
					if part.Transcript != nil {
						add("assistant", *part.Transcript)
					} else if part.Audio != nil && !replay {
						return nil, fmt.Errorf("openai GPT-Live: seeded audio needs a transcript")
					}
				case ai.ToolCallPart:
					add("assistant", fmt.Sprintf("Called `%s` with %s.", part.ToolName, part.Args))
				}
			}
		}
	}
	if !replay {
		return items, nil
	}
	tokens := 0
	start := len(items)
	for index := len(items) - 1; index >= 0; index-- {
		cost := liveTokenCount(stringValue(object(items[index]["content"].([]any)[0])["text"])) + 8
		if cost > 8192 {
			items = append(items[:index], items[index+1:]...)
			start--
			continue
		}
		if tokens+cost > 8192 || len(items)-start >= 128 {
			break
		}
		tokens += cost
		start = index
	}
	return items[start:], nil
}

var liveSpecialTokens = regexp.MustCompile(`<\|endof(?:text|prompt)\|>`)

func liveTokenCount(text string) int {
	// The installed tokenizer does not recognize special tokens.
	codec, _ := tokenizer.Get(tokenizer.O200kBase)
	count := len(liveSpecialTokens.FindAllStringIndex(text, -1))
	for _, part := range liveSpecialTokens.Split(text, -1) {
		tokens, _ := codec.Count(part)
		count += tokens
	}
	return count
}
