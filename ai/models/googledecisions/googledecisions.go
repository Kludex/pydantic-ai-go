// Package googledecisions implements typed decision models over the Gemini API's decisions endpoint.
package googledecisions

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
)

const settingsKey = "google_decisions_settings"

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
		return ai.ModelSettings{}, fmt.Errorf("google decisions: extra body field %q is reserved", settingsKey)
	}
	limits := thresholds{boolean: .5}
	if settings.BooleanThreshold != nil {
		limits.boolean = *settings.BooleanThreshold
	}
	if settings.RouteThreshold != nil {
		limits.route = *settings.RouteThreshold
	}
	if !probability(limits.boolean) || !probability(limits.route) {
		return ai.ModelSettings{}, fmt.Errorf("google decisions: thresholds must be between 0 and 1")
	}
	if common.ExtraBody == nil {
		common.ExtraBody = map[string]any{}
	}
	common.ExtraBody[settingsKey] = limits
	return common, nil
}

type thresholds struct{ boolean, route float64 }

// Model calls the Gemini API's Decisions endpoint.
type Model struct {
	name            string
	provider        google.ProviderConfig
	defaultSettings ai.ModelSettings
}

// Option configures a Decisions model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is GOOGLE_API_KEY, then GEMINI_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.provider.APIKey = key } }

// WithBaseURL sets the Gemini API base URL, including the API version.
// The default is https://generativelanguage.googleapis.com/v1beta.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.provider.BaseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.provider.HTTPClient = client }
}

// WithProvider reuses a Gemini API provider configuration. The endpoint is not on Vertex AI.
func WithProvider(provider google.ProviderConfig) Option {
	if provider.Transport == google.TransportVertexAI {
		panic("google decisions: the decisions endpoint is only on the Gemini API")
	}
	provider.BaseURL = strings.TrimRight(provider.BaseURL, "/")
	return func(model *Model) {
		if provider.Name != "" {
			model.provider.Name = provider.Name
		}
		if provider.BaseURL != "" {
			model.provider.BaseURL = provider.BaseURL
		}
		if provider.HTTPClient != nil {
			model.provider.HTTPClient = provider.HTTPClient
		}
		model.provider.APIKey = provider.APIKey
		model.provider.PrepareRequest = provider.PrepareRequest
	}
}

// WithDefaultSettings sets defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.defaultSettings = settings }
}

// NewModel creates a Google Decisions model, such as gemini-decision-1.
func NewModel(name string, options ...Option) *Model {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	model := &Model{name: name, provider: google.ProviderConfig{
		Name: "google-decisions", BaseURL: "https://generativelanguage.googleapis.com/v1beta", APIKey: apiKey,
	}}
	for _, option := range options {
		option(model)
	}
	if model.provider.HTTPClient == nil {
		model.provider.HTTPClient = http.DefaultClient
	}
	return model
}

// Name returns the Gemini decision model name.
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
			return nil, fmt.Errorf("google decisions: settings must be built with Settings.Build")
		}
		delete(settings.ExtraBody, settingsKey)
	}
	for key := range settings.ExtraBody {
		if strings.EqualFold(strings.ReplaceAll(key, "_", ""), "questions") {
			return nil, fmt.Errorf("google decisions: extra body cannot set questions, which the model keys itself")
		}
	}
	params.Settings = settings
	if settings.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, settings.RequestTimeout)
		defer cancel()
	}
	return decision.Request(ctx, messages, params, decision.Config{
		ModelName: model.name, ProviderName: model.ProviderName(), ProviderURL: model.ProviderURL(),
		ToolCallID: "google-decisions-1", BooleanThreshold: limits.boolean, RouteThreshold: limits.route,
		MaxChoiceOptions: 255, MaxScoreLevels: 255, MaxQuestions: 1000, SupportsImages: true,
		RequiresInstructions: true, Call: model.call,
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

func probability(value float64) bool {
	return !math.IsNaN(value) && value >= 0 && value <= 1
}
