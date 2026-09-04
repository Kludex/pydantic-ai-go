package mistral

import (
	"encoding/json"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type chatRequest struct {
	Model             string           `json:"model"`
	Messages          []mistralMessage `json:"messages"`
	Stream            bool             `json:"stream"`
	N                 *int             `json:"n,omitempty"`
	Tools             []mistralTool    `json:"tools,omitempty"`
	ToolChoice        ToolChoice       `json:"tool_choice,omitempty"`
	MaxTokens         int              `json:"max_tokens,omitempty"`
	Temperature       *float64         `json:"temperature,omitempty"`
	TopP              *float64         `json:"top_p,omitempty"`
	RandomSeed        *int             `json:"random_seed,omitempty"`
	PresencePenalty   *float64         `json:"presence_penalty,omitempty"`
	FrequencyPenalty  *float64         `json:"frequency_penalty,omitempty"`
	Stop              []string         `json:"stop,omitempty"`
	ReasoningEffort   string           `json:"reasoning_effort,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	PromptCacheKey    string           `json:"prompt_cache_key,omitempty"`
}

type mistralMessage struct {
	Role       string            `json:"role"`
	Content    any               `json:"content,omitempty"`
	ToolCalls  []mistralToolCall `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
	Prefix     *bool             `json:"prefix,omitempty"`
}

type contentChunk struct {
	Type        string            `json:"type"`
	Text        string            `json:"text,omitempty"`
	Thinking    []thinkingContent `json:"thinking,omitempty"`
	ImageURL    *imageURL         `json:"image_url,omitempty"`
	DocumentURL string            `json:"document_url,omitempty"`
}

type thinkingContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type imageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type mistralTool struct {
	Function mistralFunction `json:"function"`
}

type mistralFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  map[string]any  `json:"parameters,omitempty"`
	Arguments   json.RawMessage `json:"arguments,omitempty"`
}

type mistralToolCall struct {
	ID       string          `json:"id,omitempty"`
	Index    *int            `json:"index,omitempty"`
	Type     string          `json:"type,omitempty"`
	Function mistralFunction `json:"function"`
}

type chatResponse struct {
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Created int64           `json:"created"`
	Choices []mistralChoice `json:"choices"`
	Usage   mistralUsage    `json:"usage"`
}

type mistralChoice struct {
	Index        int             `json:"index"`
	Message      responseMessage `json:"message"`
	Delta        responseMessage `json:"delta"`
	FinishReason string          `json:"finish_reason"`
}

type responseMessage struct {
	Content   json.RawMessage   `json:"content"`
	ToolCalls []mistralToolCall `json:"tool_calls"`
}

type mistralUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func (usage mistralUsage) normalized() ai.Usage {
	details := map[string]int{}
	if usage.CompletionTokensDetails.ReasoningTokens != 0 {
		details["reasoning_tokens"] = usage.CompletionTokensDetails.ReasoningTokens
	}
	if len(details) == 0 {
		details = nil
	}
	return ai.Usage{
		Requests: 1, InputTokens: usage.PromptTokens, OutputTokens: usage.CompletionTokens,
		CacheReadTokens: usage.PromptTokensDetails.CachedTokens,
		ReasoningTokens: usage.CompletionTokensDetails.ReasoningTokens, Details: details,
	}
}
