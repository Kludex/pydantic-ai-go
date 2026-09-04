// Package huggingface implements ai.Model against Hugging Face Inference Providers.
package huggingface

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const defaultBaseURL = "https://router.huggingface.co/v1"

// Model calls a model through Hugging Face Inference Providers.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
}

type config struct{ options []openai.Option }

// Option configures a Hugging Face model.
type Option func(*config)

// WithToken sets the Hugging Face token. The default is HF_TOKEN.
func WithToken(token string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(token)) }
}

// WithBaseURL points the model at an OpenAI-compatible inference endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithInferenceProvider routes through a named Hugging Face inference provider.
func WithInferenceProvider(name string) Option {
	if name == "" {
		panic("huggingface: inference provider name must not be empty")
	}
	return WithBaseURL("https://router.huggingface.co/" + name + "/v1")
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Hugging Face request semantics.
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

// NewProviderConfig returns reusable Hugging Face endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	return openai.ProviderConfig{
		Name: "huggingface", BaseURL: defaultBaseURL, APIKey: os.Getenv("HF_TOKEN"),
	}
}

// NewModel creates a Hugging Face model, such as Qwen/Qwen3-32B.
func NewModel(name string, options ...Option) *Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(NewProviderConfig()),
		openai.WithStrictToolSupport(false),
		openai.WithDeferredToolSupport(false),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			LegacyMaxTokens:      true,
			DisableDocumentInput: true,
			FinishReasons: map[string]ai.FinishReason{
				"eos_token":     ai.FinishReasonStop,
				"stop_sequence": ai.FinishReasonStop,
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
	if err := validateRequest(messages, params); err != nil {
		return nil, err
	}
	return model.model.Request(ctx, model.prepareMessages(messages), params)
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	if err := validateRequest(messages, params); err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, model.prepareMessages(messages), params)
}

func validateRequest(messages []ai.ModelMessage, params ai.ModelRequestParams) error {
	if params.OutputMode == ai.OutputModeNative && len(params.OutputSchema) > 0 {
		return fmt.Errorf("huggingface: native structured output is not supported; use tool or prompted output")
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
			return fmt.Errorf("huggingface: audio URL input is not supported")
		case ai.DocumentURL:
			return fmt.Errorf("huggingface: document input is not supported")
		case ai.VideoURL:
			return fmt.Errorf("huggingface: video input is not supported")
		case ai.UploadedFile:
			return fmt.Errorf("huggingface: uploaded-file input is not supported")
		case ai.BinaryContent:
			if !strings.HasPrefix(strings.ToLower(content.MediaType), "image/") {
				return fmt.Errorf("huggingface: binary media type %q is not supported", content.MediaType)
			}
		}
	}
	return nil
}

func (model *Model) prepareMessages(messages []ai.ModelMessage) []ai.ModelMessage {
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
