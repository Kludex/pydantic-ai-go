// Package openrouter implements ai.Model against OpenRouter's Chat Completions API.
package openrouter

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"os"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// Model calls models routed through OpenRouter.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
}

type config struct {
	options        []openai.Option
	appURL         string
	appTitle       string
	attributionSet bool
}

// Option configures an OpenRouter model.
type Option func(*config)

// WithAPIKey sets the API key. The default is OPENROUTER_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(key)) }
}

// WithBaseURL points the model at an OpenRouter-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining OpenRouter request semantics.
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

// WithAppAttribution identifies the application to OpenRouter rankings and analytics.
func WithAppAttribution(url string, title string) Option {
	return func(config *config) {
		config.appURL = url
		config.appTitle = title
		config.attributionSet = true
	}
}

// NewProviderConfig returns reusable OpenRouter endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	headers := http.Header{}
	if appURL := os.Getenv("OPENROUTER_APP_URL"); appURL != "" {
		headers.Set("HTTP-Referer", appURL)
	}
	if appTitle := os.Getenv("OPENROUTER_APP_TITLE"); appTitle != "" {
		headers.Set("X-Title", appTitle)
	}
	return openai.ProviderConfig{
		Name: "openrouter", BaseURL: defaultBaseURL, APIKey: os.Getenv("OPENROUTER_API_KEY"), Headers: headers,
	}
}

// NewModel creates a model. OpenRouter names use the "provider/model" form.
func NewModel(name string, options ...Option) *Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	provider := NewProviderConfig()
	if configuration.attributionSet {
		provider.Headers = http.Header{}
		if configuration.appURL != "" {
			provider.Headers.Set("HTTP-Referer", configuration.appURL)
		}
		if configuration.appTitle != "" {
			provider.Headers.Set("X-Title", configuration.appTitle)
		}
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(provider),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			Reasoning: true, ReasoningDetails: true, LegacyMaxTokens: true, ExtendedMetadata: true,
			VideoInput: true, FileURLInput: true,
			NativeToolFunc: openRouterNativeTool,
			FinishReasons:  map[string]ai.FinishReason{"error": ai.FinishReasonError},
		}),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	model := openai.NewModel(name, openAIOptions...)
	return &Model{ModelWrapper: ai.WrapModel(model), model: model}
}

// SupportsNativeTool reports native tools rendered by OpenRouter.
func (*Model) SupportsNativeTool(tool ai.NativeTool) bool {
	if err := ai.ValidateNativeTools([]ai.NativeTool{tool}); err != nil {
		return false
	}
	switch tool.CloneNativeTool().(type) {
	case ai.WebSearchTool, ai.AdvisorTool:
		return true
	default:
		return false
	}
}

// PromptCacheRetention reports the longest downstream prompt-cache lifetime.
func (model *Model) PromptCacheRetention(settings ai.ModelSettings) (time.Duration, bool) {
	_, cache, err := extractCacheSettings(settings)
	if err != nil {
		return 0, false
	}
	provider, _, found := strings.Cut(strings.TrimPrefix(model.Name(), "~"), "/")
	if !found {
		return 0, false
	}
	if provider != "anthropic" {
		return 0, false
	}
	values := []CacheTTL{cache[cacheInstructionsKey], cache[cacheMessagesKey], cache[cacheToolsKey]}
	for _, value := range values {
		if value == CacheTTL1Hour {
			return time.Hour, true
		}
	}
	for _, value := range values {
		if value == CacheTTL5Minutes {
			return 5 * time.Minute, true
		}
	}
	return 0, false
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	ctx, prepared, err := model.prepareParams(ctx, params)
	if err != nil {
		return nil, err
	}
	return model.model.Request(ctx, prepareMessages(messages), prepared)
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	ctx, prepared, err := model.prepareParams(ctx, params)
	if err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, prepareMessages(messages), prepared)
}

func (model *Model) prepareParams(
	ctx context.Context, params ai.ModelRequestParams,
) (context.Context, ai.ModelRequestParams, error) {
	params, err := ai.ResolveNativeToolPreferences(model, params)
	if err != nil {
		return ctx, ai.ModelRequestParams{}, err
	}
	name := strings.TrimPrefix(model.Name(), "~")
	provider, routedModel, found := strings.Cut(name, "/")
	if !found || provider == "" || routedModel == "" {
		return ctx, ai.ModelRequestParams{}, fmt.Errorf(
			"openrouter: model name %q must use the provider/model form", model.Name(),
		)
	}
	settings, err := prepareSettings(params.Settings)
	if err != nil {
		return ctx, ai.ModelRequestParams{}, err
	}
	settings, cacheSettings, err := extractCacheSettings(settings)
	if err != nil {
		return ctx, ai.ModelRequestParams{}, err
	}
	params = transformSchemas(params, provider)
	thinkingActive := false
	if reasoning, exists := settings.ExtraBody["reasoning"]; exists {
		switch reasoning := reasoning.(type) {
		case Reasoning:
			thinkingActive = reasoning.IsEnabled()
		case *Reasoning:
			thinkingActive = reasoning != nil && reasoning.IsEnabled()
		case map[string]any:
			enabled, hasEnabled := reasoning["enabled"].(bool)
			effort, _ := reasoning["effort"].(string)
			thinkingActive = len(reasoning) > 0 && (!hasEnabled || enabled) && effort != string(ReasoningEffortNone)
		default:
			thinkingActive = reasoning != nil
		}
	}
	if provider == "anthropic" && thinkingActive {
		if choice, exists := settings.ExtraBody["tool_choice"]; exists {
			switch choice := choice.(type) {
			case string:
				if choice == "required" {
					return ctx, ai.ModelRequestParams{}, fmt.Errorf(
						"openrouter: tool choice %q cannot be forced with Anthropic thinking; use auto or disable thinking",
						choice,
					)
				}
			case []string, []any:
				return ctx, ai.ModelRequestParams{}, fmt.Errorf(
					"openrouter: specific tools cannot be forced with Anthropic thinking; use auto or disable thinking",
				)
			}
		}
		if params.OutputTool != nil && !params.AllowText {
			params.AllowText = true
			if _, explicit := settings.ExtraBody["tool_choice"]; !explicit {
				settings.ExtraBody = maps.Clone(settings.ExtraBody)
				settings.ExtraBody["tool_choice"] = "auto"
			}
		}
	}
	cache := openai.ChatPromptCache{}
	switch provider {
	case "anthropic":
		cache = openai.ChatPromptCache{
			InstructionsTTL: string(cacheSettings[cacheInstructionsKey]),
			MessagesTTL:     string(cacheSettings[cacheMessagesKey]),
			ToolsTTL:        string(cacheSettings[cacheToolsKey]),
			IncludeTTL:      true, SupportsDynamicInstructions: true,
			ExplicitMarkerStyle: openai.ChatPromptCacheMarkerControl, MaxPoints: 4,
		}
	case "google":
		cache = openai.ChatPromptCache{
			InstructionsTTL:     string(cacheSettings[cacheInstructionsKey]),
			MessagesTTL:         string(cacheSettings[cacheMessagesKey]),
			ExplicitMarkerStyle: openai.ChatPromptCacheMarkerControl,
		}
	case "openai":
		if strings.HasPrefix(strings.ToLower(routedModel), "gpt-5.6") {
			cache.ExplicitMarkerStyle = openai.ChatPromptCacheMarkerBreakpoint
		}
	}
	params.Settings = settings
	return openai.WithChatPromptCache(ctx, cache), params, nil
}
