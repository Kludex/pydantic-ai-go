package openai

import (
	"context"
	"fmt"
	"strings"
)

// ChatPromptCacheMarkerStyle selects the provider wire format for explicit markers.
type ChatPromptCacheMarkerStyle string

const (
	// ChatPromptCacheMarkerControl emits Anthropic or Gemini-style cache controls.
	ChatPromptCacheMarkerControl ChatPromptCacheMarkerStyle = "cache_control"
	// ChatPromptCacheMarkerBreakpoint emits OpenAI explicit breakpoint objects.
	ChatPromptCacheMarkerBreakpoint ChatPromptCacheMarkerStyle = "openai"
)

// ChatPromptCache configures explicit cache breakpoints for one compatible
// Chat Completions request. Provider packages should ignore unsupported fields.
type ChatPromptCache struct {
	// InstructionsTTL caches the final stable instruction boundary.
	InstructionsTTL string
	// MessagesTTL caches recent message boundaries.
	MessagesTTL string
	// ToolsTTL caches the final function-tool definition.
	ToolsTTL string
	// IncludeTTL sends retention values with cache controls.
	IncludeTTL bool
	// SupportsDynamicInstructions permits a boundary before dynamic instructions.
	SupportsDynamicInstructions bool
	// ExplicitMarkerStyle selects the downstream wire representation.
	ExplicitMarkerStyle ChatPromptCacheMarkerStyle
	// MaxPoints limits explicit cache boundaries. Zero uses the provider default.
	MaxPoints int
}

type chatPromptCacheContextKey struct{}

// WithChatPromptCache attaches provider-resolved cache behavior to one request context.
// It is intended for provider packages that build on the OpenAI-compatible transport.
func WithChatPromptCache(ctx context.Context, cache ChatPromptCache) context.Context {
	return context.WithValue(ctx, chatPromptCacheContextKey{}, cache)
}

func chatPromptCacheFromContext(ctx context.Context) ChatPromptCache {
	cache, _ := ctx.Value(chatPromptCacheContextKey{}).(ChatPromptCache)
	return cache
}

type chatCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type openAIPromptCacheBreakpoint struct {
	Mode string `json:"mode"`
}

func supportsOpenAIPromptCache(modelName string) bool {
	modelName = strings.TrimPrefix(strings.ToLower(modelName), "openai.")
	return strings.HasPrefix(modelName, "gpt-5.6")
}

func newChatCacheControl(ttl string, includeTTL bool) *chatCacheControl {
	control := &chatCacheControl{Type: "ephemeral"}
	if includeTTL {
		control.TTL = ttl
	}
	return control
}

func addChatMessageCache(message *chatMessage, ttl string, includeTTL bool) {
	control := newChatCacheControl(ttl, includeTTL)
	switch content := message.Content.(type) {
	case string:
		if content == "" {
			return
		}
		message.Content = []contentPart{{Type: "text", Text: content, CacheControl: control}}
	case []contentPart:
		content[len(content)-1].CacheControl = control
		message.Content = content
	}
}

func limitChatCachePoints(messages []chatMessage, tools []any, maximum int) error {
	if maximum == 0 {
		return nil
	}
	used := 0
	for _, tool := range tools {
		if functionTool, ok := tool.(chatTool); ok && functionTool.CacheControl != nil {
			used++
		}
	}
	for _, message := range messages {
		if message.Role == "system" {
			used += chatMessageCachePoints(message)
		}
	}
	if used > maximum {
		return fmt.Errorf(
			"openai: tool and system cache points use %d slots, exceeding the maximum of %d", used, maximum,
		)
	}
	remaining := maximum - used
	for messageIndex := len(messages) - 1; messageIndex >= 0; messageIndex-- {
		if messages[messageIndex].Role == "system" {
			continue
		}
		content, ok := messages[messageIndex].Content.([]contentPart)
		if !ok {
			continue
		}
		for partIndex := len(content) - 1; partIndex >= 0; partIndex-- {
			if content[partIndex].CacheControl == nil {
				continue
			}
			if remaining > 0 {
				remaining--
			} else {
				content[partIndex].CacheControl = nil
			}
		}
		messages[messageIndex].Content = content
	}
	return nil
}

func chatMessageCachePoints(message chatMessage) int {
	content, ok := message.Content.([]contentPart)
	if !ok {
		return 0
	}
	count := 0
	for _, part := range content {
		if part.CacheControl != nil {
			count++
		}
	}
	return count
}
