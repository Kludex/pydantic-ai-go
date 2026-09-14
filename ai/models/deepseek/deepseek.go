// Package deepseek implements generation models against the DeepSeek API.
package deepseek

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

const defaultBaseURL = "https://api.deepseek.com"

type config struct {
	apiKey          string
	baseURL         string
	httpClient      *http.Client
	provider        *openai.ProviderConfig
	defaultSettings ai.ModelSettings
}

// Option configures a DeepSeek model.
type Option func(*config)

// WithAPIKey sets the API key. The default is DEEPSEEK_API_KEY.
func WithAPIKey(apiKey string) Option { return func(config *config) { config.apiKey = apiKey } }

// WithBaseURL points the model at a DeepSeek-compatible endpoint.
func WithBaseURL(baseURL string) Option { return func(config *config) { config.baseURL = baseURL } }

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithProvider configures a gateway while retaining DeepSeek model behavior.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	return func(config *config) { config.provider = &provider }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = settings }
}

// NewProviderConfig returns reusable DeepSeek endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	return openai.ProviderConfig{
		Name: "deepseek", BaseURL: defaultBaseURL, APIKey: os.Getenv("DEEPSEEK_API_KEY"),
	}
}

// Model calls DeepSeek's Chat Completions API.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
	name  string
}

// NewModel creates a DeepSeek Chat Completions model.
func NewModel(name string, options ...Option) *Model {
	configuration := resolveConfig(options)
	delegate := openai.NewModel(name,
		openai.WithProvider(configuration.providerConfig()),
		openai.WithChatCompatibility(deepSeekCompatibility(name)),
		openai.WithDeferredToolSupport(false),
		openai.WithDefaultSettings(configuration.defaultSettings),
	)
	return &Model{ModelWrapper: ai.WrapModel(delegate), model: delegate, name: strings.ToLower(name)}
}

// ModelProfile selects tool output because DeepSeek Chat does not accept JSON Schema output.
func (model *Model) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, ContextWindow: model.model.ContextWindow()}
}

// Request sends one DeepSeek Chat Completions request.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	params, err := model.prepare(params, false)
	if err != nil {
		return nil, err
	}
	return model.model.Request(ctx, messages, params)
}

// StreamRequest streams one DeepSeek Chat Completions request.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	params, err := model.prepare(params, false)
	if err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, messages, params)
}

func (model *Model) prepare(params ai.ModelRequestParams, nativeOutput bool) (ai.ModelRequestParams, error) {
	if !nativeOutput && params.OutputMode == ai.OutputModeNative && params.OutputSchema != nil {
		return ai.ModelRequestParams{}, fmt.Errorf(
			"deepseek: Chat Completions does not support native structured output; use tool output or NewResponsesModel",
		)
	}
	params.Settings = prepareThinking(model.name, params.Settings)
	return params, nil
}

// ResponsesModel calls DeepSeek's Responses API.
type ResponsesModel struct {
	*ai.ModelWrapper
	model *openai.ResponsesModel
	name  string
}

// NewResponsesModel creates a DeepSeek Responses model.
func NewResponsesModel(name string, options ...Option) *ResponsesModel {
	configuration := resolveConfig(options)
	delegate := openai.NewResponsesModel(name,
		openai.WithProvider(configuration.providerConfig()),
		openai.WithChatCompatibility(deepSeekCompatibility(name)),
		openai.WithDeferredToolSupport(false),
		openai.WithResponsesPhaseSupport(false),
		openai.WithDefaultSettings(configuration.defaultSettings),
	)
	return &ResponsesModel{ModelWrapper: ai.WrapModel(delegate), model: delegate, name: strings.ToLower(name)}
}

// ModelProfile reports native JSON Schema output without OpenAI image output.
func (model *ResponsesModel) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, ContextWindow: model.model.ContextWindow()}
}

// SupportsNativeTool reports that DeepSeek Responses has no portable hosted tools.
func (*ResponsesModel) SupportsNativeTool(ai.NativeTool) bool { return false }

// Request sends one DeepSeek Responses request.
func (model *ResponsesModel) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	params.Settings = prepareThinking(model.name, params.Settings)
	return model.model.Request(ctx, prepareResponsesMessages(messages), params)
}

// StreamRequest streams one DeepSeek Responses request.
func (model *ResponsesModel) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	params.Settings = prepareThinking(model.name, params.Settings)
	return model.model.StreamRequest(ctx, prepareResponsesMessages(messages), params)
}

func prepareResponsesMessages(messages []ai.ModelMessage) []ai.ModelMessage {
	prepared := make([]ai.ModelMessage, len(messages))
	copy(prepared, messages)
	for index, message := range messages {
		response, ok := message.(ai.ModelResponse)
		if !ok {
			continue
		}
		unsettled := map[string]int{}
		skip := false
		for _, part := range response.Parts {
			switch part := part.(type) {
			case ai.ToolCallPart:
				unsettled[part.ToolCallID]++
			case ai.NativeToolCallPart, ai.NativeToolReturnPart, ai.CompactionPart:
				skip = true
			}
		}
		if skip || len(unsettled) == 0 {
			continue
		}
		for following := index + 1; following < len(messages); following++ {
			if _, nextResponse := messages[following].(ai.ModelResponse); nextResponse {
				break
			}
			request, ok := messages[following].(ai.ModelRequest)
			if !ok {
				continue
			}
			for _, part := range request.Parts {
				callID := ""
				switch part := part.(type) {
				case ai.ToolReturnPart:
					callID = part.ToolCallID
				case ai.RetryPromptPart:
					if part.ToolName != "" {
						callID = part.ToolCallID
					}
				}
				if count := unsettled[callID]; count == 1 {
					delete(unsettled, callID)
				} else if count > 1 {
					unsettled[callID] = count - 1
				}
			}
		}
		if len(unsettled) != 0 {
			continue
		}
		parts := make([]ai.ResponsePart, 0, len(response.Parts))
		for _, part := range response.Parts {
			if _, call := part.(ai.ToolCallPart); !call {
				parts = append(parts, part)
			}
		}
		for _, part := range response.Parts {
			if _, call := part.(ai.ToolCallPart); call {
				parts = append(parts, part)
			}
		}
		response.Parts = parts
		prepared[index] = response
	}
	return prepared
}

func resolveConfig(options []Option) config {
	configuration := config{
		apiKey: os.Getenv("DEEPSEEK_API_KEY"), baseURL: defaultBaseURL,
	}
	for _, option := range options {
		option(&configuration)
	}
	return configuration
}

func (configuration config) providerConfig() openai.ProviderConfig {
	provider := openai.ProviderConfig{
		Name: "deepseek", BaseURL: configuration.baseURL, APIKey: configuration.apiKey,
		HTTPClient: configuration.httpClient,
	}
	if configuration.provider != nil {
		provider = *configuration.provider
		provider.Name = "deepseek"
	}
	return provider
}

func deepSeekCompatibility(name string) openai.ChatCompatibility {
	name = strings.ToLower(name)
	isV4 := strings.HasPrefix(name, "deepseek-v4-")
	return openai.ChatCompatibility{
		ReasoningContent:                    true,
		DisableRequiredToolChoice:           name == "deepseek-reasoner",
		DisableForcedToolChoiceWithThinking: isV4,
		ReasoningEnabledByDefault:           isV4 || name == "deepseek-reasoner",
		ResponsesReasoningContent:           true,
	}
}

func prepareThinking(name string, settings ai.ModelSettings) ai.ModelSettings {
	if name == "deepseek-reasoner" && settings.Thinking != nil &&
		settings.Thinking.Level == ai.ThinkingLevelDisabled {
		settings = settings.Clone()
		settings.Thinking = nil
		return settings
	}
	if name != "deepseek-reasoner" && !strings.HasPrefix(name, "deepseek-v4-") {
		settings = settings.Clone()
		settings.Thinking = nil
	}
	return settings
}
