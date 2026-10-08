// Package typesafe implements TypeSafe Jev typed-decision models.
package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/contextwindow"
	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
)

const defaultBaseURL = "https://api.typesafe.ai"
const settingsKey = "typesafe_settings"

// Settings combines portable settings with Jev decision thresholds.
type Settings struct {
	Common ai.ModelSettings
	// BooleanThreshold controls the probability required for a true answer.
	BooleanThreshold *float64
	// RouteThreshold hands a route below this probability to a fallback model.
	RouteThreshold *float64
	// ToolCallThreshold is retained for source compatibility.
	//
	// Deprecated: ignored; use RouteThreshold.
	ToolCallThreshold *float64
}

// Build returns detached settings accepted by agent runs.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	for name, value := range map[string]*float64{
		"boolean threshold": settings.BooleanThreshold, "route threshold": settings.RouteThreshold,
	} {
		if value != nil && (*value < 0 || *value > 1) {
			return ai.ModelSettings{}, fmt.Errorf("typesafe: %s must be between 0 and 1", name)
		}
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	if _, exists := extra[settingsKey]; exists {
		return ai.ModelSettings{}, fmt.Errorf("typesafe: extra body field %q is reserved", settingsKey)
	}
	extra[settingsKey] = typedSettings{
		BooleanThreshold: settings.BooleanThreshold, RouteThreshold: settings.RouteThreshold,
	}
	common.ExtraBody = extra
	return common, nil
}

type typedSettings struct {
	BooleanThreshold *float64
	RouteThreshold   *float64
}

// RequestPreparationFunc prepares a request after provider headers and before per-request headers.
type RequestPreparationFunc func(*http.Request) error

// ProviderConfig configures TypeSafe API access.
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

// NewProviderConfig returns TypeSafe environment configuration.
func NewProviderConfig() ProviderConfig {
	baseURL := os.Getenv("TYPESAFE_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return ProviderConfig{Name: "typesafe", BaseURL: strings.TrimRight(baseURL, "/"), APIKey: os.Getenv("TYPESAFE_API_KEY")}
}

// Model asks Jev typed questions derived from tool schemas.
type Model struct {
	name             string
	providerName     string
	baseURL          string
	apiKey           string
	httpClient       *http.Client
	headers          http.Header
	prepareRequest   RequestPreparationFunc
	defaultSettings  ai.ModelSettings
	maxChoiceOptions int
	maxScoreLevels   int
}

// Option configures a TypeSafe model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is TYPESAFE_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.apiKey = key } }

// WithBaseURL sets the TypeSafe API origin.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.baseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.httpClient = client }
}

// WithProvider configures a compatible endpoint.
func WithProvider(provider ProviderConfig) Option {
	if provider.Name == "" || provider.BaseURL == "" {
		panic("typesafe: provider name and base URL must not be empty")
	}
	provider = provider.Clone()
	return func(model *Model) {
		model.providerName, model.baseURL, model.apiKey = provider.Name, strings.TrimRight(provider.BaseURL, "/"), provider.APIKey
		model.headers, model.prepareRequest = provider.Headers.Clone(), provider.PrepareRequest
		if provider.HTTPClient != nil {
			model.httpClient = provider.HTTPClient
		}
	}
}

// WithDecisionLimits overrides Jev's default choice and rubric limits.
// Zero disables the corresponding limit.
func WithDecisionLimits(maxChoiceOptions, maxScoreLevels int) Option {
	if maxChoiceOptions < 0 || maxScoreLevels < 0 {
		panic("typesafe: decision limits must not be negative")
	}
	return func(model *Model) {
		model.maxChoiceOptions = maxChoiceOptions
		model.maxScoreLevels = maxScoreLevels
	}
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.defaultSettings = settings.Clone() }
}

// NewModel creates a Jev model, such as jev-latest.
func NewModel(name string, options ...Option) *Model {
	provider := NewProviderConfig()
	model := &Model{
		name: name, providerName: provider.Name, baseURL: provider.BaseURL, apiKey: provider.APIKey,
		httpClient: http.DefaultClient, maxChoiceOptions: 255, maxScoreLevels: 10,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

func (model *Model) Name() string         { return model.name }
func (model *Model) ProviderName() string { return model.providerName }
func (model *Model) ProviderURL() string  { return model.baseURL }
func (model *Model) ContextWindow() int {
	return contextwindow.Lookup(model.name, "typesafe", model.baseURL)
}
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }

// ModelProfile reports that Jev produces tool output but cannot generate text.
func (model *Model) ModelProfile() ai.ModelProfile {
	unsupported := false
	return ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, SupportsTextOutput: &unsupported, ContextWindow: model.ContextWindow(),
	}
}

// DecisionHandOff is a route Jev could not safely complete. Fallback models can match this interface.
type DecisionHandOff interface {
	error
	IsModelAPIError() bool
	DecisionRoute() (route string, probability float64)
}

// ToolCallProposed reports a selected route whose fields Jev cannot express.
//
// Deprecated: use UnfillableRoute.
type ToolCallProposed struct {
	ModelName   string
	ToolName    string
	Route       string
	Probability float64
}

func (err *ToolCallProposed) Error() string {
	route := err.Route
	if route == "" {
		route = err.ToolName
	}
	return fmt.Sprintf("Jev picked %q (probability %.2f) but cannot fill it", route, err.Probability)
}
func (*ToolCallProposed) IsModelAPIError() bool { return true }
func (err *ToolCallProposed) DecisionRoute() (string, float64) {
	if err.Route != "" {
		return err.Route, err.Probability
	}
	return err.ToolName, err.Probability
}

// UnfillableRoute reports a selected route whose fields Jev cannot express.
type UnfillableRoute = ToolCallProposed

// UnsureRoute reports a selected route below the configured route threshold.
type UnsureRoute struct {
	ModelName     string
	Route         string
	Probability   float64
	Probabilities map[string]float64
	Threshold     float64
}

func (err *UnsureRoute) Error() string {
	return fmt.Sprintf(
		"Jev picked %q with probability %.2f, below route threshold %.2f", err.Route, err.Probability, err.Threshold,
	)
}
func (*UnsureRoute) IsModelAPIError() bool { return true }
func (err *UnsureRoute) DecisionRoute() (string, float64) {
	return err.Route, err.Probability
}

// APIError reports a non-successful TypeSafe response.
type APIError struct {
	StatusCode int
	Body       string
	Headers    http.Header
}

func (err *APIError) Error() string {
	return fmt.Sprintf("TypeSafe API returned status %d: %s", err.StatusCode, err.Body)
}
func (*APIError) IsModelAPIError() bool { return true }

type question = decision.Question
type responseBody = decision.Response

func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	settings, typed, err := extractSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	params.Settings = settings
	return decision.Request(ctx, messages, params, decision.Config{
		ModelName: model.name, ProviderName: model.providerName, ProviderURL: model.baseURL,
		ToolCallID:       "typesafe-1",
		BooleanThreshold: typed.booleanThreshold(), RouteThreshold: typed.routeThreshold(),
		MaxChoiceOptions: model.maxChoiceOptions, MaxScoreLevels: model.maxScoreLevels, Call: model.call,
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

func extractSettings(settings ai.ModelSettings) (ai.ModelSettings, typedSettings, error) {
	settings = settings.Clone()
	typed := typedSettings{}
	if value, ok := settings.ExtraBody[settingsKey]; ok {
		var valid bool
		typed, valid = value.(typedSettings)
		if !valid {
			return settings, typed, fmt.Errorf("typesafe: settings must be built with Settings.Build")
		}
		delete(settings.ExtraBody, settingsKey)
	}
	for _, threshold := range []*float64{typed.BooleanThreshold, typed.RouteThreshold} {
		if threshold != nil && (*threshold < 0 || *threshold > 1) {
			return settings, typed, fmt.Errorf("typesafe: thresholds must be between 0 and 1")
		}
	}
	return settings, typed, nil
}
func (settings typedSettings) booleanThreshold() float64 {
	if settings.BooleanThreshold != nil {
		return *settings.BooleanThreshold
	}
	return .5
}
func (settings typedSettings) routeThreshold() float64 {
	if settings.RouteThreshold != nil {
		return *settings.RouteThreshold
	}
	return 0
}

func (model *Model) call(
	ctx context.Context, state any, questions map[string]question, settings ai.ModelSettings,
) (responseBody, error) {
	body, err := json.Marshal(map[string]any{"state": state, "questions": questions, "model": model.name})
	if err != nil {
		return responseBody{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, model.baseURL+"/v1/systemone", bytes.NewReader(body))
	if err != nil {
		return responseBody{}, err
	}
	request.Header = model.headers.Clone()
	if request.Header == nil {
		request.Header = make(http.Header)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return responseBody{}, err
		}
	}
	for key, value := range settings.ExtraHeaders {
		request.Header.Set(key, value)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return responseBody{}, ai.NewModelTransportError(ctx, model, "request", err)
	}
	defer func() { _ = response.Body.Close() }()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return responseBody{}, ai.NewModelTransportError(ctx, model, "read response", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return responseBody{}, &APIError{response.StatusCode, string(data), response.Header.Clone()}
	}
	var decoded responseBody
	if err := json.Unmarshal(data, &decoded); err != nil {
		return responseBody{}, fmt.Errorf("typesafe: decode response: %w", err)
	}
	if decoded.Model == "" {
		return responseBody{}, errors.New("typesafe: response omitted model")
	}
	return decoded, nil
}
