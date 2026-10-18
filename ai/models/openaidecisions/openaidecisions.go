// Package openaidecisions implements typed decision models over OpenAI's /v1/decisions endpoint.
package openaidecisions

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"os"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

const settingsKey = "openai_decisions_settings"

// Settings combines portable settings with decision thresholds.
type Settings struct {
	Common ai.ModelSettings
	// BooleanThreshold controls the probability required for true. It defaults to 0.5.
	BooleanThreshold *float64
	// RouteThreshold hands uncertain routes to a fallback model. It defaults to zero.
	RouteThreshold *float64
}

// Build returns detached model settings.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if _, exists := common.ExtraBody[settingsKey]; exists {
		return ai.ModelSettings{}, fmt.Errorf("openai decisions: extra body field %q is reserved", settingsKey)
	}
	limits := thresholds{boolean: .5}
	if settings.BooleanThreshold != nil {
		limits.boolean = *settings.BooleanThreshold
	}
	if settings.RouteThreshold != nil {
		limits.route = *settings.RouteThreshold
	}
	if !probability(limits.boolean) || !probability(limits.route) {
		return ai.ModelSettings{}, fmt.Errorf("openai decisions: thresholds must be between 0 and 1")
	}
	if common.ExtraBody == nil {
		common.ExtraBody = map[string]any{}
	}
	common.ExtraBody[settingsKey] = limits
	return common, nil
}

type thresholds struct{ boolean, route float64 }

// Model calls OpenAI's Decisions API.
type Model struct {
	name            string
	provider        openai.ProviderConfig
	defaultSettings ai.ModelSettings
}

// Option configures a Decisions model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is OPENAI_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.provider.APIKey = key } }

// WithBaseURL sets the OpenAI-compatible API base URL, including /v1. The default is OPENAI_BASE_URL.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.provider.BaseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.provider.HTTPClient = client }
}

// WithProvider replaces endpoint, authentication, headers, and request preparation.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.BaseURL = strings.TrimRight(provider.BaseURL, "/")
	provider.Headers = provider.Headers.Clone()
	provider.Query = maps.Clone(provider.Query)
	for key, values := range provider.Query {
		provider.Query[key] = slices.Clone(values)
	}
	return func(model *Model) { model.provider = provider }
}

// WithDefaultSettings sets defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.defaultSettings = settings }
}

// NewModel creates an OpenAI Decisions model, such as gpt-6-luna.
func NewModel(name string, options ...Option) *Model {
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	model := &Model{name: name, provider: openai.ProviderConfig{
		BaseURL: strings.TrimRight(baseURL, "/"), APIKey: os.Getenv("OPENAI_API_KEY"),
	}}
	for _, option := range options {
		option(model)
	}
	if model.provider.Name == "" {
		model.provider.Name = "openai-decisions"
	}
	if model.provider.HTTPClient == nil {
		model.provider.HTTPClient = http.DefaultClient
	}
	return model
}

// Name returns the OpenAI model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the provider name persisted in responses.
func (model *Model) ProviderName() string { return model.provider.Name }

// ProviderURL returns the API base URL.
func (model *Model) ProviderURL() string { return model.provider.BaseURL }

// DefaultModelSettings returns detached default settings.
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }

// ModelProfile reports tool output without text generation.
func (*Model) ModelProfile() ai.ModelProfile {
	unsupported := false
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, SupportsTextOutput: &unsupported}
}

// Request maps output schemas and tools to Decisions questions.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	settings := params.Settings.Clone()
	limits := thresholds{boolean: .5}
	if value, exists := settings.ExtraBody[settingsKey]; exists {
		var ok bool
		if limits, ok = value.(thresholds); !ok {
			return nil, fmt.Errorf("openai decisions: settings must be built with Settings.Build")
		}
		delete(settings.ExtraBody, settingsKey)
	}
	params.Settings = settings
	if settings.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, settings.RequestTimeout)
		defer cancel()
	}
	return decision.Request(ctx, messages, params, decision.Config{
		ModelName: model.name, ProviderName: model.ProviderName(), ProviderURL: model.ProviderURL(),
		ToolCallID: "openai-decisions-1", BooleanThreshold: limits.boolean, RouteThreshold: limits.route,
		MaxChoiceOptions: 255, MaxScoreLevels: 10, MaxQuestions: 200, MaxImages: 128, SupportsImages: true,
		Call: model.call,
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
