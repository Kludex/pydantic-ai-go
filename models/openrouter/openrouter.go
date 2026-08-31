// Package openrouter implements ai.Model against OpenRouter's Chat Completions API.
package openrouter

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
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

// NewModel creates a model. OpenRouter names use the "provider/model" form.
func NewModel(name string, options ...Option) *Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	headers := http.Header{}
	appURL := os.Getenv("OPENROUTER_APP_URL")
	if configuration.attributionSet {
		appURL = configuration.appURL
	}
	if appURL != "" {
		headers.Set("HTTP-Referer", appURL)
	}
	appTitle := os.Getenv("OPENROUTER_APP_TITLE")
	if configuration.attributionSet {
		appTitle = configuration.appTitle
	}
	if appTitle != "" {
		headers.Set("X-Title", appTitle)
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(openai.ProviderConfig{
			Name: "openrouter", BaseURL: defaultBaseURL, APIKey: os.Getenv("OPENROUTER_API_KEY"), Headers: headers,
		}),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			Reasoning: true, ReasoningDetails: true, LegacyMaxTokens: true, ExtendedMetadata: true,
			NativeToolFunc: openRouterNativeTool,
			FinishReasons:  map[string]ai.FinishReason{"error": ai.FinishReasonError},
		}),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	model := openai.NewModel(name, openAIOptions...)
	return &Model{ModelWrapper: ai.WrapModel(model), model: model}
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
	return model.model.Request(ctx, messages, prepared)
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	ctx, prepared, err := model.prepareParams(ctx, params)
	if err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, messages, prepared)
}

func (model *Model) prepareParams(
	ctx context.Context, params ai.ModelRequestParams,
) (context.Context, ai.ModelRequestParams, error) {
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
