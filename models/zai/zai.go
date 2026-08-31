// Package zai implements ai.Model against Z.AI's OpenAI-compatible Chat Completions API.
package zai

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const defaultBaseURL = "https://api.z.ai/api/paas/v4"

// Model calls Z.AI GLM models and preserves their reasoning across turns.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
}

type config struct {
	options []openai.Option
}

// Option configures a Z.AI model.
type Option func(*config)

// WithAPIKey sets the API key. The default is ZAI_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(key)) }
}

// WithBaseURL points the model at a Z.AI-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Z.AI request semantics.
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

// Settings combines portable settings with Z.AI-specific settings.
type Settings struct {
	Common        ai.ModelSettings
	ClearThinking *bool
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if settings.ClearThinking == nil {
		return common, nil
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	thinking := map[string]any{}
	if value, exists := extra["thinking"]; exists {
		object, ok := value.(map[string]any)
		if !ok {
			return ai.ModelSettings{}, fmt.Errorf("zai: extra body field %q must be an object", "thinking")
		}
		thinking = maps.Clone(object)
	}
	if _, exists := thinking["clear_thinking"]; exists {
		return ai.ModelSettings{}, fmt.Errorf(
			"zai: extra body field %q conflicts with typed settings", "thinking.clear_thinking",
		)
	}
	thinking["clear_thinking"] = *settings.ClearThinking
	extra["thinking"] = thinking
	common.ExtraBody = extra
	return common, nil
}

// NewModel creates a model for a Z.AI GLM model, such as glm-5.3-flash.
func NewModel(name string, options ...Option) *Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(openai.ProviderConfig{
			Name: "zai", BaseURL: defaultBaseURL, APIKey: os.Getenv("ZAI_API_KEY"),
		}),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			ReasoningContent: true,
			FinishReasons: map[string]ai.FinishReason{
				"sensitive":                     ai.FinishReasonContentFilter,
				"model_context_window_exceeded": ai.FinishReasonLength,
				"network_error":                 ai.FinishReasonError,
			},
		}),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	model := openai.NewModel(name, openAIOptions...)
	return &Model{ModelWrapper: ai.WrapModel(model), model: model}
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	prepared, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	return model.model.Request(ctx, messages, prepared)
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	prepared, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, messages, prepared)
}

func (model *Model) prepareParams(params ai.ModelRequestParams) (ai.ModelRequestParams, error) {
	settings := params.Settings.Clone()
	thinkingSettings := settings.Thinking
	settings.Thinking = nil
	extra := maps.Clone(settings.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	thinking := map[string]any{}
	if value, exists := extra["thinking"]; exists {
		object, ok := value.(map[string]any)
		if !ok {
			return ai.ModelRequestParams{}, fmt.Errorf("zai: extra body field %q must be an object", "thinking")
		}
		thinking = maps.Clone(object)
	}

	level := ai.ThinkingLevel("")
	if thinkingSettings != nil {
		level = thinkingSettings.Level
	}
	if level != "" {
		if _, exists := thinking["type"]; exists {
			return ai.ModelRequestParams{}, fmt.Errorf(
				"zai: extra body field %q conflicts with portable thinking settings", "thinking.type",
			)
		}
		if level == ai.ThinkingLevelDisabled {
			thinking["type"] = "disabled"
		} else {
			thinking["type"] = "enabled"
		}
	}

	name := strings.ToLower(model.Name())
	supportsThinking := strings.HasPrefix(name, "glm-5") || strings.HasPrefix(name, "glm-4.7") ||
		strings.HasPrefix(name, "glm-4.6") || strings.HasPrefix(name, "glm-4.5")
	if supportsThinking {
		if _, exists := thinking["clear_thinking"]; !exists {
			thinking["clear_thinking"] = false
		}
	}
	if len(thinking) > 0 {
		extra["thinking"] = thinking
	}

	supportsEffort := strings.HasPrefix(name, "glm-5.2") || strings.HasPrefix(name, "glm-5.3")
	if supportsEffort && level != "" && level != ai.ThinkingLevelDisabled && level != ai.ThinkingLevelEnabled {
		if _, exists := extra["reasoning_effort"]; exists {
			return ai.ModelRequestParams{}, fmt.Errorf(
				"zai: extra body field %q conflicts with portable thinking settings", "reasoning_effort",
			)
		}
		effort := string(level)
		if strings.HasPrefix(name, "glm-5.3") {
			effort = map[ai.ThinkingLevel]string{
				ai.ThinkingLevelMinimal: "low",
				ai.ThinkingLevelMedium:  "high",
				ai.ThinkingLevelXHigh:   "max",
			}[level]
			if effort == "" {
				effort = string(level)
			}
		}
		extra["reasoning_effort"] = effort
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	params.Settings = settings
	return params, nil
}
