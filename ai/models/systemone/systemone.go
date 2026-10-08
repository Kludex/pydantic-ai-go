// Package systemone implements typed decision models served over /v1/systemone.
package systemone

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
)

// RequestPreparationFunc prepares a request before per-request headers are applied.
type RequestPreparationFunc func(*http.Request) error

// ProviderConfig configures a System One endpoint and its caller-owned client.
type ProviderConfig struct {
	Name           string
	BaseURL        string
	APIKey         string
	HTTPClient     *http.Client
	Headers        http.Header
	PrepareRequest RequestPreparationFunc
}

// Clone returns detached provider configuration.
func (config ProviderConfig) Clone() ProviderConfig {
	config.Headers = config.Headers.Clone()
	return config
}

// NewProviderConfig reads SYSTEM_ONE_BASE_URL and the optional SYSTEM_ONE_API_KEY.
func NewProviderConfig() (ProviderConfig, error) {
	baseURL := strings.TrimRight(os.Getenv("SYSTEM_ONE_BASE_URL"), "/")
	if baseURL == "" {
		return ProviderConfig{}, fmt.Errorf("systemone: set SYSTEM_ONE_BASE_URL or pass WithBaseURL")
	}
	return ProviderConfig{Name: "system-one", BaseURL: baseURL, APIKey: os.Getenv("SYSTEM_ONE_API_KEY")}, nil
}

// Profile sets model-specific limits. Zero leaves a limit to the server.
type Profile struct {
	MaxChoiceOptions int
	MaxScoreLevels   int
	ContextWindow    int
	// RequiresInstructions defaults to true. False permits questions described only by their options.
	RequiresInstructions *bool
}

// Model answers typed questions about text, rather than generating text.
type Model struct {
	name            string
	provider        ProviderConfig
	profile         Profile
	defaultSettings ai.ModelSettings
}

// Option configures a System One model.
type Option func(*Model)

// WithBaseURL sets the API origin, optionally ending in /v1.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.provider.BaseURL = baseURL }
}

// WithAPIKey sets an optional bearer token, overriding SYSTEM_ONE_API_KEY.
func WithAPIKey(key string) Option {
	return func(model *Model) { model.provider.APIKey = key }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.provider.HTTPClient = client }
}

// WithProvider replaces environment configuration with a compatible endpoint.
func WithProvider(provider ProviderConfig) Option {
	provider = provider.Clone()
	return func(model *Model) { model.provider = provider.Clone() }
}

// WithProfile sets the served model's choice, rubric, and context limits.
func WithProfile(profile Profile) Option {
	if profile.RequiresInstructions != nil {
		required := *profile.RequiresInstructions
		profile.RequiresInstructions = &required
	}
	return func(model *Model) { model.profile = profile }
}

// WithDefaultSettings sets defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.defaultSettings = settings.Clone() }
}

// NewModel creates a decision model such as clm-latest, laya, or nimble.
func NewModel(name string, options ...Option) (*Model, error) {
	model := &Model{name: name, provider: ProviderConfig{
		Name: "system-one", BaseURL: os.Getenv("SYSTEM_ONE_BASE_URL"), APIKey: os.Getenv("SYSTEM_ONE_API_KEY"),
	}}
	for _, option := range options {
		option(model)
	}
	model.provider.BaseURL = strings.TrimRight(model.provider.BaseURL, "/")
	if model.provider.BaseURL == "" {
		return nil, fmt.Errorf("systemone: set SYSTEM_ONE_BASE_URL or pass WithBaseURL")
	}
	parsed, err := url.Parse(model.provider.BaseURL)
	if err != nil || parsed.Host == "" || parsed.Scheme != "http" && parsed.Scheme != "https" ||
		parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("systemone: base URL must be an HTTP(S) endpoint without a query or fragment")
	}
	if name == "" || model.provider.Name == "" {
		return nil, fmt.Errorf("systemone: model and provider names must not be empty")
	}
	if model.profile.MaxChoiceOptions < 0 || model.profile.MaxScoreLevels < 0 || model.profile.ContextWindow < 0 {
		return nil, fmt.Errorf("systemone: profile limits must be non-negative")
	}
	if model.provider.HTTPClient == nil {
		model.provider.HTTPClient = http.DefaultClient
	}
	return model, nil
}

// Name returns the requested model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the provider identifier.
func (model *Model) ProviderName() string { return model.provider.Name }

// ProviderURL returns the configured endpoint.
func (model *Model) ProviderURL() string { return model.provider.BaseURL }

// ContextWindow returns the configured token limit, or zero when unknown.
func (model *Model) ContextWindow() int { return model.profile.ContextWindow }

// DefaultModelSettings returns detached request defaults.
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }

// ModelProfile reports tool output without text or native JSON output.
func (model *Model) ModelProfile() ai.ModelProfile {
	unsupported := false
	return ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, SupportsTextOutput: &unsupported, ContextWindow: model.ContextWindow(),
	}
}

// Request maps typed output and tools to questions and reconstructs the selected route.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	settings, thresholds, err := extractSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	params.Settings = settings
	if settings.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, settings.RequestTimeout)
		defer cancel()
	}
	requiresInstructions := model.profile.RequiresInstructions == nil || *model.profile.RequiresInstructions
	return decision.Request(ctx, messages, params, decision.Config{
		ModelName: model.name, ProviderName: model.ProviderName(), ProviderURL: model.ProviderURL(),
		ToolCallID:           "system-one-1",
		RequiresInstructions: requiresInstructions,
		BooleanThreshold:     thresholds.BooleanThreshold, RouteThreshold: thresholds.RouteThreshold,
		MaxChoiceOptions: model.profile.MaxChoiceOptions, MaxScoreLevels: model.profile.MaxScoreLevels, Call: model.call,
		HandOff: func(route decision.Route) error {
			if route.Unfillable {
				return &UnfillableRoute{
					ModelName: model.name, ToolName: route.ToolName, Route: route.Name, Probability: route.Probability,
				}
			}
			return &UnsureRoute{
				ModelName: model.name, Route: route.Name, Probability: route.Probability,
				Probabilities: route.Probabilities, Threshold: route.Threshold,
			}
		},
	})
}
