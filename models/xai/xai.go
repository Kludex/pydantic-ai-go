// Package xai implements ai.Model against xAI's Responses-compatible API.
package xai

import (
	"context"
	"iter"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const defaultBaseURL = "https://api.x.ai/v1"

// Model calls an xAI Grok model.
type Model struct {
	model          *openai.ResponsesModel
	apiKey         string
	baseURL        string
	httpClient     *http.Client
	headers        http.Header
	prepareRequest openai.RequestPreparationFunc
}

type config struct {
	options        []openai.Option
	apiKey         string
	baseURL        string
	httpClient     *http.Client
	headers        http.Header
	prepareRequest openai.RequestPreparationFunc
}

// Option configures an xAI model.
type Option func(*config)

// WithAPIKey sets the API key. The default is XAI_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) {
		config.apiKey = key
		config.options = append(config.options, openai.WithAPIKey(key))
	}
}

// WithBaseURL points the model at an xAI-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) {
		config.baseURL = baseURL
		config.options = append(config.options, openai.WithBaseURL(baseURL))
	}
}

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) {
		config.httpClient = client
		config.options = append(config.options, openai.WithHTTPClient(client))
	}
}

// WithProvider configures a gateway while retaining xAI request semantics.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Name = "xai"
	return func(config *config) {
		config.apiKey = provider.APIKey
		config.baseURL = provider.BaseURL
		config.httpClient = provider.HTTPClient
		config.headers = provider.Headers.Clone()
		config.prepareRequest = provider.PrepareRequest
		config.options = append(config.options, openai.WithProvider(provider))
	}
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) {
		config.options = append(config.options, openai.WithDefaultSettings(settings))
	}
}

// NewProviderConfig returns reusable xAI endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	return openai.ProviderConfig{
		Name: "xai", BaseURL: defaultBaseURL, APIKey: os.Getenv("XAI_API_KEY"),
	}
}

// NewModel creates an xAI model.
func NewModel(name string, options ...Option) *Model {
	provider := NewProviderConfig()
	configuration := config{
		apiKey: provider.APIKey, baseURL: provider.BaseURL, httpClient: provider.HTTPClient,
		headers: provider.Headers.Clone(), prepareRequest: provider.PrepareRequest,
	}
	for _, option := range options {
		option(&configuration)
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(NewProviderConfig()),
		openai.WithResponsesCodeExecutionOutputs(true),
		openai.WithResponsesFileSearchResults(true),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	httpClient := configuration.httpClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &Model{
		model: openai.NewResponsesModel(name, openAIOptions...), apiKey: configuration.apiKey,
		baseURL: configuration.baseURL, httpClient: httpClient, headers: configuration.headers.Clone(),
		prepareRequest: configuration.prepareRequest,
	}
}

// Name returns the configured model name.
func (model *Model) Name() string { return model.model.Name() }

// ProviderName returns the durable xAI provider identity.
func (model *Model) ProviderName() string { return model.model.ProviderName() }

// ProviderURL returns the configured provider endpoint.
func (model *Model) ProviderURL() string { return model.model.ProviderURL() }

// DefaultModelSettings returns detached request defaults.
func (model *Model) DefaultModelSettings() ai.ModelSettings {
	return model.model.DefaultModelSettings()
}

// ModelProfile reports xAI structured-output support.
func (*Model) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool}
}

// SupportsNativeTool reports native tools available to the selected Grok model.
func (model *Model) SupportsNativeTool(tool ai.NativeTool) bool {
	if !supportsNativeTools(model.Name()) || ai.ValidateNativeTools([]ai.NativeTool{tool}) != nil {
		return false
	}
	switch tool.CloneNativeTool().(type) {
	case ai.WebSearchTool, ai.CodeExecutionTool, ai.MCPServerTool, ai.XSearchTool, ai.FileSearchTool:
		return true
	default:
		return false
	}
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	messages, err := model.prepareMessages(ctx, messages)
	if err != nil {
		return nil, err
	}
	if err := validateMessages(messages); err != nil {
		return nil, err
	}
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
	messages, err := model.prepareMessages(ctx, messages)
	if err != nil {
		return nil, err
	}
	if err := validateMessages(messages); err != nil {
		return nil, err
	}
	prepared, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, messages, prepared)
}

func supportsNativeTools(name string) bool {
	return strings.HasPrefix(name, "grok-4") || strings.Contains(name, "code") || strings.Contains(name, "build") ||
		name == "grok-latest" || name == "grok-3"
}
