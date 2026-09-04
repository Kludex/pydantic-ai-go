// Package ollama implements ai.Model against Ollama's OpenAI-compatible Chat Completions API.
package ollama

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const defaultBaseURL = "http://localhost:11434/v1"

// Model calls a local Ollama server or Ollama Cloud.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
	cloud bool
}

type config struct {
	options []openai.Option
	baseURL string
}

// Option configures an Ollama model.
type Option func(*config)

// WithAPIKey sets the API key. The default is OLLAMA_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(key)) }
}

// WithBaseURL points the model at an Ollama OpenAI-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	if baseURL == "" {
		panic("ollama: base URL must not be empty")
	}
	baseURL = normalizeBaseURL(baseURL)
	return func(config *config) {
		config.baseURL = baseURL
		config.options = append(config.options, openai.WithBaseURL(baseURL))
	}
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Ollama request semantics.
func WithProvider(provider openai.ProviderConfig) Option {
	if provider.BaseURL == "" {
		panic("ollama: provider base URL must not be empty")
	}
	provider.BaseURL = normalizeBaseURL(provider.BaseURL)
	option := openai.WithProvider(provider)
	return func(config *config) {
		config.baseURL = provider.BaseURL
		config.options = append(config.options, option)
	}
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) {
		config.options = append(config.options, openai.WithDefaultSettings(settings))
	}
}

// NewProviderConfig returns reusable local Ollama endpoint configuration.
func NewProviderConfig() openai.ProviderConfig {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = os.Getenv("OLLAMA_HOST")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return openai.ProviderConfig{
		Name: "ollama", BaseURL: normalizeBaseURL(baseURL), APIKey: os.Getenv("OLLAMA_API_KEY"),
	}
}

// NewModel creates an Ollama model, such as qwen3 or llama3.2.
func NewModel(name string, options ...Option) *Model {
	provider := NewProviderConfig()
	configuration := config{baseURL: provider.BaseURL}
	for _, option := range options {
		option(&configuration)
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(provider),
		openai.WithStrictToolSupport(false),
		openai.WithDeferredToolSupport(false),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			Reasoning:            true,
			LegacyMaxTokens:      true,
			DisableDocumentInput: true,
		}),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	model := openai.NewModel(name, openAIOptions...)
	return &Model{
		ModelWrapper: ai.WrapModel(model),
		model:        model,
		cloud:        routesToCloud(configuration.baseURL, name),
	}
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	if err := model.validateRequest(messages, params); err != nil {
		return nil, err
	}
	return model.model.Request(ctx, messages, params)
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	if err := model.validateRequest(messages, params); err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, messages, params)
}

func (model *Model) validateRequest(messages []ai.ModelMessage, params ai.ModelRequestParams) error {
	if model.cloud && params.OutputMode == ai.OutputModeNative && len(params.OutputSchema) > 0 {
		return fmt.Errorf("ollama: native structured output is not enforced by Ollama Cloud; use tool or prompted output")
	}
	for _, message := range messages {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			switch part := part.(type) {
			case ai.UserPromptPart:
				if err := validateUserContent(part.Contents); err != nil {
					return err
				}
			case ai.ToolReturnPart:
				switch content := part.Content.(type) {
				case ai.ToolReturn:
					if err := validateUserContent(content.Content); err != nil {
						return err
					}
				case *ai.ToolReturn:
					if content != nil {
						if err := validateUserContent(content.Content); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

func validateUserContent(contents []ai.UserContent) error {
	for _, content := range contents {
		switch content := content.(type) {
		case ai.AudioURL:
			return fmt.Errorf("ollama: audio URL input is not supported")
		case ai.DocumentURL:
			return fmt.Errorf("ollama: document input is not supported")
		case ai.VideoURL:
			return fmt.Errorf("ollama: video input is not supported")
		case ai.UploadedFile:
			return fmt.Errorf("ollama: uploaded-file input is not supported")
		case ai.BinaryContent:
			if !strings.HasPrefix(strings.ToLower(content.MediaType), "image/") {
				return fmt.Errorf("ollama: binary media type %q is not supported", content.MediaType)
			}
		}
	}
	return nil
}

func normalizeBaseURL(baseURL string) string {
	baseURL = strings.TrimRight(baseURL, "/")
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL += "/v1"
	}
	return baseURL
}

func routesToCloud(baseURL, modelName string) bool {
	parsed, err := url.Parse(baseURL)
	if err == nil {
		hostname := parsed.Hostname()
		if hostname == "ollama.com" || strings.HasSuffix(hostname, ".ollama.com") {
			return true
		}
	}
	return strings.HasSuffix(modelName, "-cloud")
}
