package openai

import "context"

// ChatPromptCache configures explicit cache breakpoints for one compatible
// Chat Completions request. Provider packages should ignore unsupported fields.
type ChatPromptCache struct {
	InstructionsTTL             string
	MessagesTTL                 string
	ToolsTTL                    string
	IncludeTTL                  bool
	SupportsDynamicInstructions bool
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
