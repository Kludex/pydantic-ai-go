// Package cerebras implements ai.Model against Cerebras's OpenAI-compatible API.
package cerebras

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"os"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const defaultBaseURL = "https://api.cerebras.ai/v1"

// Settings combines portable settings with Cerebras reasoning controls.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// DisableReasoning disables reasoning on models that support disabling it.
	DisableReasoning *bool
	// ClearThinking controls whether GLM clears retained reasoning before generation.
	ClearThinking *bool
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if settings.DisableReasoning != nil && *settings.DisableReasoning {
		common.Thinking = &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}
	}
	if settings.ClearThinking == nil {
		return common, nil
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	if _, exists := extra["clear_thinking"]; exists {
		return ai.ModelSettings{}, fmt.Errorf("cerebras: extra body field %q conflicts with typed settings", "clear_thinking")
	}
	extra["clear_thinking"] = *settings.ClearThinking
	common.ExtraBody = extra
	return common, nil
}

// Model calls models served by Cerebras.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
	name  string
}

type config struct{ options []openai.Option }

// Option configures a Cerebras model.
type Option func(*config)

// WithAPIKey sets the API key. The default is CEREBRAS_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(key)) }
}

// WithBaseURL points the model at a Cerebras-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Cerebras request semantics.
func WithProvider(provider openai.ProviderConfig) Option {
	option := openai.WithProvider(provider)
	return func(config *config) { config.options = append(config.options, option) }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) {
		config.options = append(config.options, openai.WithDefaultSettings(settings))
	}
}

// NewProviderConfig returns reusable Cerebras endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	return openai.ProviderConfig{
		Name: "cerebras", BaseURL: defaultBaseURL, APIKey: os.Getenv("CEREBRAS_API_KEY"),
		Headers: http.Header{"X-Cerebras-3rd-Party-Integration": {"pydantic-ai"}},
	}
}

// NewModel creates a Cerebras model, such as gpt-oss-120b or zai-glm-4.7.
func NewModel(name string, options ...Option) *Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(NewProviderConfig()),
		openai.WithChatCompatibility(openai.ChatCompatibility{Reasoning: true}),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	model := openai.NewModel(name, openAIOptions...)
	return &Model{ModelWrapper: ai.WrapModel(model), model: model, name: strings.ToLower(name)}
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	messages = model.prepareMessages(messages)
	params, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	return model.model.Request(ctx, messages, params)
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	messages = model.prepareMessages(messages)
	params, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, messages, params)
}

func (model *Model) prepareParams(params ai.ModelRequestParams) (ai.ModelRequestParams, error) {
	settings := params.Settings.Clone()
	settings.LogitBias = nil
	thinking := settings.Thinking
	settings.Thinking = nil
	extra := maps.Clone(settings.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	if strings.HasPrefix(model.name, "zai") {
		if _, exists := extra["clear_thinking"]; !exists {
			extra["clear_thinking"] = false
		}
		if thinking != nil && thinking.Level == ai.ThinkingLevelDisabled {
			if _, exists := extra["reasoning_effort"]; exists {
				return ai.ModelRequestParams{}, fmt.Errorf(
					"cerebras: extra body field %q conflicts with portable thinking settings", "reasoning_effort",
				)
			}
			extra["reasoning_effort"] = "none"
		}
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	params.Settings = settings
	return params, nil
}

func (model *Model) prepareMessages(messages []ai.ModelMessage) []ai.ModelMessage {
	if !strings.HasPrefix(model.name, "zai") {
		return messages
	}
	prepared := slices.Clone(messages)
	for index, message := range messages {
		response, ok := message.(ai.ModelResponse)
		if !ok {
			continue
		}
		parts := make([]ai.ResponsePart, 0, len(response.Parts))
		var text []string
		for _, part := range response.Parts {
			switch part := part.(type) {
			case ai.TextPart:
				text = append(text, part.Content)
			case ai.ThinkingPart:
				if part.ProviderName == "" || part.ProviderName == model.ProviderName() {
					text = append(text, "<think>\n"+part.Content+"\n</think>")
				}
			default:
				parts = append(parts, part)
			}
		}
		if len(text) > 0 {
			parts = append([]ai.ResponsePart{ai.TextPart{Content: strings.Join(text, "\n\n")}}, parts...)
		}
		response.Parts = parts
		prepared[index] = response
	}
	return prepared
}
