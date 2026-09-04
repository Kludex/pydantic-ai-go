// Package groq implements ai.Model against Groq's Chat Completions API.
package groq

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

const defaultBaseURL = "https://api.groq.com/openai/v1"

// ReasoningFormat controls how Groq returns model reasoning.
type ReasoningFormat string

const (
	// ReasoningFormatHidden suppresses reasoning output.
	ReasoningFormatHidden ReasoningFormat = "hidden"
	// ReasoningFormatRaw returns reasoning inside model text.
	ReasoningFormatRaw ReasoningFormat = "raw"
	// ReasoningFormatParsed returns reasoning as a separate part.
	ReasoningFormatParsed ReasoningFormat = "parsed"
)

// ReasoningEffort controls Groq reasoning effort where supported.
type ReasoningEffort string

const (
	// ReasoningEffortNone disables reasoning where supported.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortDefault uses the model default.
	ReasoningEffortDefault ReasoningEffort = "default"
	// ReasoningEffortLow requests low effort.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium requests medium effort.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh requests high effort.
	ReasoningEffortHigh ReasoningEffort = "high"
)

// Settings combines portable settings with Groq reasoning options.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// ReasoningFormat controls reasoning representation.
	ReasoningFormat ReasoningFormat
	// ReasoningEffort controls model-specific reasoning effort.
	ReasoningEffort ReasoningEffort
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if err := validateReasoningFormat(settings.ReasoningFormat); err != nil {
		return ai.ModelSettings{}, err
	}
	if err := validateReasoningEffort(settings.ReasoningEffort); err != nil {
		return ai.ModelSettings{}, err
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	values := []struct {
		name  string
		value string
	}{
		{name: "reasoning_format", value: string(settings.ReasoningFormat)},
		{name: "reasoning_effort", value: string(settings.ReasoningEffort)},
	}
	for _, value := range values {
		if value.value == "" {
			continue
		}
		if _, exists := extra[value.name]; exists {
			return ai.ModelSettings{}, fmt.Errorf("groq: extra body field %q conflicts with typed settings", value.name)
		}
		extra[value.name] = value.value
	}
	if len(extra) == 0 {
		extra = nil
	}
	common.ExtraBody = extra
	return common, nil
}

// Model calls models served by Groq.
type Model struct {
	*ai.ModelWrapper
	name             string
	reasoningWarning func(ReasoningWarning)
}

type config struct {
	options          []openai.Option
	reasoningWarning func(ReasoningWarning)
}

// Option configures a Groq model.
type Option func(*config)

// WithAPIKey sets the API key. The default is GROQ_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(key)) }
}

// WithBaseURL points the model at a Groq-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Groq response semantics.
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

// ReasoningWarning describes a portable reasoning setting overridden by a model-family requirement.
type ReasoningWarning struct {
	// ModelName identifies the selected Groq model.
	ModelName string
	// Message explains which setting was overridden.
	Message string
}

// WithReasoningWarningHandler receives inspectable reasoning-setting warnings.
// The handler may be called concurrently when the model is shared by concurrent runs.
func WithReasoningWarningHandler(handler func(ReasoningWarning)) Option {
	return func(config *config) { config.reasoningWarning = handler }
}

// NewProviderConfig returns reusable Groq endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	baseURL := os.Getenv("GROQ_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return openai.ProviderConfig{Name: "groq", BaseURL: baseURL, APIKey: os.Getenv("GROQ_API_KEY")}
}

// NewModel creates a model for a Groq-hosted model.
func NewModel(name string, options ...Option) *Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(NewProviderConfig()),
		openai.WithChatCompatibility(openai.ChatCompatibility{Reasoning: true, ExecutedTools: true}),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	return &Model{
		ModelWrapper: ai.WrapModel(openai.NewModel(name, openAIOptions...)),
		name:         name, reasoningWarning: configuration.reasoningWarning,
	}
}

// SupportsNativeTool reports support for implicit web search on Groq compound models.
func (model *Model) SupportsNativeTool(tool ai.NativeTool) bool {
	if !isCompoundModel(model.name) {
		return false
	}
	switch value := tool.(type) {
	case ai.WebSearchTool:
		return true
	case *ai.WebSearchTool:
		return value != nil
	default:
		return false
	}
}

// Request maps Groq compound search settings before generation.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	prepared, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	return model.ModelWrapper.Request(ctx, messages, prepared)
}

// StreamRequest maps Groq compound search settings before streaming generation.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	prepared, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	return model.ModelWrapper.StreamRequest(ctx, messages, prepared)
}

func (model *Model) prepareParams(params ai.ModelRequestParams) (ai.ModelRequestParams, error) {
	params.Settings = model.prepareThinking(params.Settings)
	if len(params.NativeTools) == 0 {
		return params, nil
	}
	if !isCompoundModel(model.name) {
		return ai.ModelRequestParams{}, fmt.Errorf("groq: native web search requires a compound model")
	}
	settings := params.Settings.Clone()
	for _, nativeTool := range params.NativeTools {
		if nativeTool == nil {
			return ai.ModelRequestParams{}, fmt.Errorf("groq: native tool must not be nil")
		}
		var webSearch ai.WebSearchTool
		switch value := nativeTool.(type) {
		case ai.WebSearchTool:
			webSearch = value
		case *ai.WebSearchTool:
			if value == nil {
				return ai.ModelRequestParams{}, fmt.Errorf("groq: native web search must not be nil")
			}
			webSearch = *value
		default:
			return ai.ModelRequestParams{}, fmt.Errorf("groq: native tool %q is not supported", nativeTool.Kind())
		}
		if webSearch.SearchContextSize != "" || webSearch.UserLocation != nil || webSearch.MaxUses != 0 ||
			webSearch.ExternalWebAccess != nil {
			return ai.ModelRequestParams{}, fmt.Errorf("groq: compound web search only supports domain filters")
		}
		if _, exists := settings.ExtraBody["search_settings"]; exists {
			return ai.ModelRequestParams{}, fmt.Errorf("groq: extra body field %q conflicts with native web search", "search_settings")
		}
		if len(webSearch.AllowedDomains) > 0 || len(webSearch.BlockedDomains) > 0 {
			if settings.ExtraBody == nil {
				settings.ExtraBody = make(map[string]any)
			}
			search := map[string]any{}
			if len(webSearch.AllowedDomains) > 0 {
				search["include_domains"] = append([]string(nil), webSearch.AllowedDomains...)
			}
			if len(webSearch.BlockedDomains) > 0 {
				search["exclude_domains"] = append([]string(nil), webSearch.BlockedDomains...)
			}
			settings.ExtraBody["search_settings"] = search
		}
	}
	params.Settings = settings
	params.NativeTools = nil
	return params, nil
}

func (model *Model) prepareThinking(settings ai.ModelSettings) ai.ModelSettings {
	if settings.Thinking == nil {
		return settings
	}
	thinking := *settings.Thinking
	settings = settings.Clone()
	settings.Thinking = nil
	if !isReasoningModel(model.name) {
		return settings
	}
	if settings.ExtraBody == nil {
		settings.ExtraBody = make(map[string]any)
	}
	_, explicitFormat := settings.ExtraBody["reasoning_format"]
	_, explicitEffort := settings.ExtraBody["reasoning_effort"]
	if strings.HasPrefix(model.name, "qwen/qwen3") {
		if thinking.Level == ai.ThinkingLevelDisabled {
			if explicitEffort && model.reasoningWarning != nil {
				model.reasoningWarning(ReasoningWarning{
					ModelName: model.name,
					Message:   "disabled thinking overrides the configured reasoning effort with none",
				})
			}
			settings.ExtraBody["reasoning_effort"] = "none"
		} else if thinking.Level != "" && !explicitFormat {
			settings.ExtraBody["reasoning_format"] = "parsed"
		}
		return settings
	}
	if thinking.Level == ai.ThinkingLevelDisabled {
		if !explicitFormat {
			settings.ExtraBody["reasoning_format"] = "hidden"
		}
		return settings
	}
	if thinking.Level == "" {
		return settings
	}
	if !explicitFormat {
		settings.ExtraBody["reasoning_format"] = "parsed"
	}
	if strings.HasPrefix(model.name, "openai/gpt-oss") && !explicitEffort {
		settings.ExtraBody["reasoning_effort"] = groqReasoningEffort(thinking.Level)
	}
	return settings
}

func groqReasoningEffort(level ai.ThinkingLevel) string {
	switch level {
	case ai.ThinkingLevelMinimal, ai.ThinkingLevelLow:
		return "low"
	case ai.ThinkingLevelHigh, ai.ThinkingLevelXHigh:
		return "high"
	default:
		return "medium"
	}
}

func isReasoningModel(name string) bool {
	for _, prefix := range []string{
		"openai/gpt-oss", "qwen/qwen3", "qwen-qwq", "deepseek-r1", "llama-4-maverick",
	} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func isCompoundModel(name string) bool {
	return strings.HasPrefix(name, "compound-") || strings.HasPrefix(name, "groq/compound")
}

func validateReasoningFormat(format ReasoningFormat) error {
	switch format {
	case "", ReasoningFormatHidden, ReasoningFormatRaw, ReasoningFormatParsed:
		return nil
	default:
		return fmt.Errorf("groq: invalid reasoning format %q", format)
	}
}

func validateReasoningEffort(effort ReasoningEffort) error {
	switch effort {
	case "", ReasoningEffortNone, ReasoningEffortDefault, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh:
		return nil
	default:
		return fmt.Errorf("groq: invalid reasoning effort %q", effort)
	}
}
