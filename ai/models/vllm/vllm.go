// Package vllm implements ai.Model against a vLLM OpenAI-compatible server.
package vllm

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

// Model calls a local or remote vLLM server.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
	name  string
}

type config struct {
	baseURL         string
	apiKey          string
	httpClient      *http.Client
	provider        *openai.ProviderConfig
	defaultSettings ai.ModelSettings
}

// Option configures a vLLM model.
type Option func(*config)

// WithBaseURL sets the vLLM OpenAI-compatible API root.
func WithBaseURL(baseURL string) Option { return func(config *config) { config.baseURL = baseURL } }

// WithAPIKey sets the optional vLLM bearer token.
func WithAPIKey(apiKey string) Option { return func(config *config) { config.apiKey = apiKey } }

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithProvider configures a gateway while retaining vLLM behavior.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	return func(config *config) { config.provider = &provider }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = settings }
}

// NewProviderConfig returns reusable vLLM endpoint configuration.
func NewProviderConfig() (openai.ProviderConfig, error) {
	baseURL := os.Getenv("VLLM_BASE_URL")
	if baseURL == "" {
		return openai.ProviderConfig{}, fmt.Errorf("vllm: set VLLM_BASE_URL or use WithBaseURL")
	}
	return openai.ProviderConfig{Name: "vllm", BaseURL: baseURL, APIKey: os.Getenv("VLLM_API_KEY")}, nil
}

// NewModel creates a model served by vLLM.
func NewModel(name string, options ...Option) (*Model, error) {
	configuration := config{baseURL: os.Getenv("VLLM_BASE_URL"), apiKey: os.Getenv("VLLM_API_KEY")}
	for _, option := range options {
		option(&configuration)
	}
	provider := openai.ProviderConfig{
		Name: "vllm", BaseURL: configuration.baseURL, APIKey: configuration.apiKey,
		HTTPClient: configuration.httpClient,
	}
	if configuration.provider != nil {
		provider = *configuration.provider
		provider.Name = "vllm"
	}
	if provider.BaseURL == "" {
		return nil, fmt.Errorf("vllm: set VLLM_BASE_URL or use WithBaseURL")
	}
	lowerName := strings.ToLower(name)
	bareName := lowerName
	if index := strings.LastIndexByte(bareName, '/'); index >= 0 {
		bareName = bareName[index+1:]
	}
	compatibility := openai.ChatCompatibility{
		ReasoningFallback: true, DisableDocumentInput: true,
		DisableRequiredToolChoice:           strings.HasPrefix(bareName, "gpt-oss"),
		DisableForcedToolChoiceWithThinking: strings.HasPrefix(bareName, "deepseek-v4-"),
		ReasoningEnabledByDefault:           strings.HasPrefix(bareName, "deepseek-v4-"),
	}
	delegate := openai.NewModel(name,
		openai.WithProvider(provider),
		openai.WithChatCompatibility(compatibility),
		openai.WithDeferredToolSupport(false),
		openai.WithDefaultSettings(configuration.defaultSettings),
	)
	return &Model{ModelWrapper: ai.WrapModel(delegate), model: delegate, name: lowerName}, nil
}

// ModelProfile reports native structured output with schema instructions.
func (*Model) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, NativeOutputRequiresPrompt: true}
}

// Request sends one vLLM Chat Completions request.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	messages, params = model.prepare(messages, params)
	return model.model.Request(ctx, messages, params)
}

// StreamRequest streams one vLLM Chat Completions request.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	messages, params = model.prepare(messages, params)
	return model.model.StreamRequest(ctx, messages, params)
}

func (model *Model) prepare(messages []ai.ModelMessage, params ai.ModelRequestParams) ([]ai.ModelMessage, ai.ModelRequestParams) {
	if !vllmSupportsThinking(model.name) {
		params.Settings = params.Settings.Clone()
		params.Settings.Thinking = nil
	}
	messages = slices.Clone(messages)
	instructions := params.Instructions
	for index, message := range messages {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			break
		}
		kept := make([]ai.RequestPart, 0, len(request.Parts))
		for _, part := range request.Parts {
			if system, ok := part.(ai.SystemPromptPart); ok {
				if instructions != "" {
					instructions += "\n\n"
				}
				instructions += system.Content
				continue
			}
			kept = append(kept, part)
		}
		request.Parts = kept
		messages[index] = request
	}
	params.Instructions = instructions
	return messages, params
}

func vllmSupportsThinking(name string) bool {
	bare := name
	if index := strings.LastIndexByte(bare, '/'); index >= 0 {
		bare = bare[index+1:]
	}
	qwen3 := strings.HasPrefix(bare, "qwen3") && !strings.HasPrefix(bare, "qwen3-coder") &&
		!strings.HasPrefix(bare, "qwen3.8") && !strings.Contains(bare, "-instruct")
	return qwen3 || strings.HasPrefix(bare, "gemma-4") || strings.HasPrefix(bare, "deepseek-r1") ||
		strings.HasPrefix(bare, "deepseek-v4-") || strings.HasPrefix(bare, "magistral") ||
		strings.Contains(bare, "command-a-reasoning") || strings.HasPrefix(bare, "glm-4.7")
}
