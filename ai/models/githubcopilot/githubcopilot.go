// Package githubcopilot implements ai.Model against GitHub Copilot's Chat Completions API.
package githubcopilot

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

const defaultBaseURL = "https://api.githubcopilot.com"

// Model calls a model exposed by a GitHub Copilot subscription.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
	name  string
}

type config struct {
	apiKey          string
	baseURL         string
	httpClient      *http.Client
	provider        *openai.ProviderConfig
	defaultSettings ai.ModelSettings
}

// Option configures a GitHub Copilot model.
type Option func(*config)

// WithAPIKey sets the Copilot bearer token.
func WithAPIKey(apiKey string) Option { return func(config *config) { config.apiKey = apiKey } }

// WithBaseURL sets a Copilot Enterprise, GitHub Enterprise Server, or proxy endpoint.
func WithBaseURL(baseURL string) Option { return func(config *config) { config.baseURL = baseURL } }

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithProvider configures a gateway while retaining Copilot model behavior.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	return func(config *config) { config.provider = &provider }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = settings }
}

// NewProviderConfig returns reusable GitHub Copilot endpoint configuration.
func NewProviderConfig() (openai.ProviderConfig, error) {
	apiKey := firstEnvironment("GITHUB_COPILOT_API_KEY", "GITHUB_COPILOT_API_TOKEN", "COPILOT_GITHUB_TOKEN")
	if apiKey == "" {
		return openai.ProviderConfig{}, fmt.Errorf(
			"githubcopilot: set GITHUB_COPILOT_API_KEY or use WithAPIKey",
		)
	}
	baseURL := firstEnvironment("GITHUB_COPILOT_BASE_URL", "COPILOT_API_URL", "GITHUB_COPILOT_API_BASE")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return openai.ProviderConfig{
		Name: "github-copilot", BaseURL: baseURL, APIKey: apiKey, Headers: copilotHeaders(),
	}, nil
}

// NewModel creates a GitHub Copilot Chat Completions model.
func NewModel(name string, options ...Option) (*Model, error) {
	configuration := config{
		apiKey:  firstEnvironment("GITHUB_COPILOT_API_KEY", "GITHUB_COPILOT_API_TOKEN", "COPILOT_GITHUB_TOKEN"),
		baseURL: firstEnvironment("GITHUB_COPILOT_BASE_URL", "COPILOT_API_URL", "GITHUB_COPILOT_API_BASE"),
	}
	if configuration.baseURL == "" {
		configuration.baseURL = defaultBaseURL
	}
	for _, option := range options {
		option(&configuration)
	}
	provider := openai.ProviderConfig{
		Name: "github-copilot", BaseURL: configuration.baseURL, APIKey: configuration.apiKey,
		HTTPClient: configuration.httpClient, Headers: copilotHeaders(),
	}
	if configuration.provider != nil {
		provider = *configuration.provider
		provider.Name = "github-copilot"
		headers := copilotHeaders()
		for name, values := range provider.Headers {
			headers[name] = append([]string(nil), values...)
		}
		provider.Headers = headers
	}
	if provider.APIKey == "" && provider.PrepareRequest == nil {
		return nil, fmt.Errorf("githubcopilot: set GITHUB_COPILOT_API_KEY or use WithAPIKey")
	}
	bareName := strings.TrimPrefix(strings.ToLower(name), "copilot/")
	profile := copilotFamilyProfile(bareName)
	compatibility := openai.ChatCompatibility{
		DisableDocumentInput:      true,
		ReasoningText:             profile.reasoningText,
		ReasoningEnabledByDefault: profile.reasoningEnabledByDefault,
	}
	delegate := openai.NewModel(name,
		openai.WithProvider(provider),
		openai.WithChatCompatibility(compatibility),
		openai.WithDeferredToolSupport(false),
		openai.WithDefaultSettings(configuration.defaultSettings),
	)
	return &Model{ModelWrapper: ai.WrapModel(delegate), model: delegate, name: bareName}, nil
}

// Request sends one GitHub Copilot request.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	params = model.prepareParams(params)
	return model.model.Request(ctx, messages, params)
}

// StreamRequest streams one GitHub Copilot request.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	params = model.prepareParams(params)
	return model.model.StreamRequest(ctx, messages, params)
}

func (model *Model) prepareParams(params ai.ModelRequestParams) ai.ModelRequestParams {
	profile := copilotFamilyProfile(model.name)
	settings := params.Settings.Clone()
	if !profile.supportsThinking || profile.thinkingAlwaysEnabled && settings.Thinking != nil &&
		settings.Thinking.Level == ai.ThinkingLevelDisabled {
		settings.Thinking = nil
	}
	reasoningActiveByDefault := profile.reasoningEnabledByDefault &&
		(settings.Thinking == nil || settings.Thinking.Level != ai.ThinkingLevelDisabled)
	if profile.disallowsSampling || reasoningActiveByDefault {
		settings.Temperature = nil
		settings.TopP = nil
	}
	params.Settings = settings
	return params
}

type familyProfile struct {
	supportsThinking          bool
	reasoningEnabledByDefault bool
	thinkingAlwaysEnabled     bool
	reasoningText             bool
	disallowsSampling         bool
}

func copilotFamilyProfile(name string) familyProfile {
	name = strings.TrimPrefix(strings.ToLower(name), "copilot/")
	profile := familyProfile{}
	switch {
	case strings.HasPrefix(name, "claude-"):
		profile.supportsThinking = true
		profile.reasoningText = true
		profile.disallowsSampling = copilotAnthropicDisallowsSampling(name)
	case strings.HasPrefix(name, "gpt-"), strings.HasPrefix(name, "o1"), strings.HasPrefix(name, "o3"),
		strings.HasPrefix(name, "o4"), strings.HasPrefix(name, "mai-"), strings.HasPrefix(name, "oswe"),
		strings.HasPrefix(name, "raptor"), strings.HasPrefix(name, "exec-agent-"):
		profile.supportsThinking, profile.reasoningEnabledByDefault,
			profile.thinkingAlwaysEnabled = copilotOpenAIThinking(name)
	case strings.HasPrefix(name, "gemini-"):
		profile.supportsThinking = strings.Contains(name, "gemini-2.5") || strings.Contains(name, "gemini-3")
		profile.thinkingAlwaysEnabled = profile.supportsThinking && strings.Contains(name, "pro") &&
			!strings.Contains(name, "flash")
		profile.reasoningEnabledByDefault = profile.thinkingAlwaysEnabled
		profile.reasoningText = true
	case strings.HasPrefix(name, "grok-"):
		profile.supportsThinking, profile.thinkingAlwaysEnabled = copilotGrokThinking(name)
		profile.reasoningEnabledByDefault = profile.thinkingAlwaysEnabled
	case strings.HasPrefix(name, "kimi-"):
		profile.supportsThinking = strings.HasPrefix(name, "kimi-k2.5") || strings.HasPrefix(name, "kimi-k2.6") ||
			strings.HasPrefix(name, "kimi-k2.7") || strings.HasPrefix(name, "kimi-k2-thinking") ||
			strings.HasPrefix(name, "kimi-k3") || strings.HasPrefix(name, "kimi-thinking")
	}
	return profile
}

func copilotOpenAIThinking(name string) (bool, bool, bool) {
	for _, prefix := range []string{"gpt-6-astra", "gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.3-chat", "gpt-5.2-pro",
		"gpt-5.2-chat", "gpt-5.1-codex", "gpt-5.1-chat"} {
		if strings.HasPrefix(name, prefix) {
			return true, true, true
		}
	}
	if strings.HasPrefix(name, "gpt-5-chat") {
		return false, false, false
	}
	if strings.HasPrefix(name, "gpt-5.6") || strings.HasPrefix(name, "gpt-5.5") {
		return true, true, false
	}
	if strings.HasPrefix(name, "gpt-5.4") || strings.HasPrefix(name, "gpt-5.3") ||
		strings.HasPrefix(name, "gpt-5.2") || strings.HasPrefix(name, "gpt-5.1") {
		return true, false, false
	}
	if strings.HasPrefix(name, "gpt-5") || strings.HasPrefix(name, "o1") || strings.HasPrefix(name, "o3") ||
		strings.HasPrefix(name, "o4") || strings.HasPrefix(name, "oswe") {
		return true, true, true
	}
	return false, false, false
}

func copilotGrokThinking(name string) (bool, bool) {
	switch name {
	case "grok-4.3", "grok-4.3-latest", "grok-latest", "grok-4-0709", "grok-4-1-fast-reasoning",
		"grok-4-1-fast-non-reasoning", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-3":
		return true, false
	case "grok-4.5", "grok-4.5-latest", "grok-4.6", "grok-build-latest":
		return true, true
	}
	return strings.HasPrefix(name, "grok-3-mini"), strings.HasPrefix(name, "grok-3-mini")
}

func copilotAnthropicDisallowsSampling(name string) bool {
	for _, prefix := range []string{
		"claude-opus-4.7", "claude-opus-4.8", "claude-opus-5", "claude-sonnet-5",
		"claude-fable-5", "claude-mythos-5",
	} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func copilotHeaders() http.Header {
	return http.Header{
		"Editor-Version":         {"vscode/1.95.0"},
		"Copilot-Integration-Id": {"vscode-chat"},
		"Editor-Plugin-Version":  {"copilot-chat/0.26.7"},
		"Openai-Intent":          {"conversation-panel"},
		"X-Github-Api-Version":   {"2025-04-01"},
	}
}

func firstEnvironment(names ...string) string {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return value
		}
	}
	return ""
}
