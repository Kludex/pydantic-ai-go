// Package meta implements ai.Model against Meta's Model API for Muse models.
package meta

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

const defaultBaseURL = "https://api.meta.ai/v1"

type config struct{ options []openai.Option }

// Option configures a Meta model.
type Option func(*config)

// WithAPIKey sets the API key. The default is META_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(key)) }
}

// WithBaseURL points the model at a Meta-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Meta model behavior.
func WithProvider(provider openai.ProviderConfig) Option {
	option := openai.WithProvider(provider)
	return func(config *config) { config.options = append(config.options, option) }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.options = append(config.options, openai.WithDefaultSettings(settings)) }
}

// NewProviderConfig returns reusable Meta endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	return openai.ProviderConfig{Name: "meta", BaseURL: defaultBaseURL, APIKey: os.Getenv("META_API_KEY")}
}

type delegate interface {
	ai.StreamingModel
	SupportsNativeTool(ai.NativeTool) bool
}

// Model calls a Meta model and applies Muse reasoning, tool-choice, and native-tool behavior.
type Model struct {
	*ai.ModelWrapper
	model delegate
	muse  bool
}

// NewModel creates a Meta model on the Responses API, such as muse-spark-1.3.
// Responses replays Muse's commentary phase and encrypted reasoning, which keeps long tool-calling runs on track.
func NewModel(name string, options ...Option) *Model {
	muse := isMuse(name)
	return newModel(name, muse, openai.NewResponsesModel(name, openAIOptions(options,
		openai.WithResponsesPhaseSupport(muse),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			DisableRequiredToolChoice: true, ResponsesEncryptedReasoning: muse, SamplingWithReasoning: true,
		}),
	)...))
}

// NewChatModel creates a Meta model on the Chat Completions API. Chat Completions returns no reasoning parts.
func NewChatModel(name string, options ...Option) *Model {
	muse := isMuse(name)
	return newModel(name, muse, openai.NewModel(name, openAIOptions(options,
		openai.WithChatWebSearchSupport(muse),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			DisableRequiredToolChoice: true, SamplingWithReasoning: true,
		}),
	)...))
}

func newModel(name string, muse bool, model delegate) *Model {
	return &Model{ModelWrapper: ai.WrapModel(model), model: model, muse: muse}
}

func openAIOptions(options []Option, apiOptions ...openai.Option) []openai.Option {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	apiOptions = append([]openai.Option{
		openai.WithProvider(NewProviderConfig()), openai.WithDeferredToolSupport(false),
	}, apiOptions...)
	return append(apiOptions, configuration.options...)
}

// ModelProfile reports native JSON Schema output without image output.
func (model *Model) ModelProfile() ai.ModelProfile {
	profile := model.ModelWrapper.ModelProfile()
	profile.SupportsImageOutput = false
	return profile
}

// SupportsNativeTool reports native tools available to the model. Muse models support web search only.
func (model *Model) SupportsNativeTool(tool ai.NativeTool) bool {
	if !model.model.SupportsNativeTool(tool) {
		return false
	}
	_, webSearch := tool.CloneNativeTool().(ai.WebSearchTool)
	return webSearch || !model.muse
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	params, err := model.prepare(params)
	if err != nil {
		return nil, err
	}
	return model.model.Request(ctx, messages, params)
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	params, err := model.prepare(params)
	if err != nil {
		return nil, err
	}
	return model.model.StreamRequest(ctx, messages, params)
}

func (model *Model) prepare(params ai.ModelRequestParams) (ai.ModelRequestParams, error) {
	params, err := ai.ResolveNativeToolPreferences(model, params)
	if err != nil {
		return ai.ModelRequestParams{}, err
	}
	nativeTools := make([]ai.NativeTool, 0, len(params.NativeTools))
	for _, tool := range params.NativeTools {
		if model.SupportsNativeTool(tool) {
			nativeTools = append(nativeTools, tool)
		} else if !tool.IsOptional() {
			return ai.ModelRequestParams{}, fmt.Errorf(
				"meta: native tool %q is not supported by model %q", tool.Kind(), model.Name(),
			)
		}
	}
	params.NativeTools = nativeTools
	// Muse always reasons and rejects effort none; other Meta models do not reason.
	if thinking := params.Settings.Thinking; thinking != nil && (!model.muse || thinking.Level == ai.ThinkingLevelDisabled) {
		params.Settings = params.Settings.Clone()
		params.Settings.Thinking = nil
	}
	return params, nil
}

func isMuse(name string) bool {
	name = strings.ToLower(name)
	return strings.HasPrefix(name, "muse-spark-") || strings.HasPrefix(name, "muse-wonder-")
}
