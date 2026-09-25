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
	"slices"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/contextwindow"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
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
	name            string
	providerName    string
	baseURL         string
	apiKey          string
	httpClient      *http.Client
	headers         http.Header
	prepareRequest  RequestPreparationFunc
	defaultSettings ai.ModelSettings
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

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.defaultSettings = settings.Clone() }
}

// NewModel creates a Jev model, such as jev-latest.
func NewModel(name string, options ...Option) *Model {
	provider := NewProviderConfig()
	model := &Model{name: name, providerName: provider.Name, baseURL: provider.BaseURL, apiKey: provider.APIKey, httpClient: http.DefaultClient}
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

type question struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions,omitempty"`
	Criteria     any    `json:"criteria,omitempty"`
}

type requestBody struct {
	State     any                 `json:"state"`
	Questions map[string]question `json:"questions"`
	Model     string              `json:"model"`
}

type answer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Score         *float64           `json:"score"`
}

type responseBody struct {
	Answers map[string]answer `json:"answers"`
	Model   string            `json:"model"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func (model *Model) Request(ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	if params.AllowText && params.OutputTool == nil {
		return nil, fmt.Errorf("typesafe: text output is not supported")
	}
	settings, typed, err := extractSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	state, err := messageState(messages)
	if err != nil {
		return nil, err
	}
	type routeShape struct {
		tool      *ai.ToolDefinition
		fields    map[string]map[string]any
		defaults  map[string]bool
		contexts  map[string][]string
		questions map[string]question
		scoped    map[string]string
		included  bool
		err       error
	}
	available := toolsLeft(messages, params.Tools)
	routes := make(map[string]routeShape, len(available)+1)
	labels := make([]string, 0, len(available)+1)
	addRoute := func(label string, tool *ai.ToolDefinition, purpose string) {
		for routes[label].tool != nil {
			label += " (output)"
		}
		fields, defaults, contexts, shapeErr := schemaFields(tool.Schema)
		var questions map[string]question
		if shapeErr == nil {
			questions, shapeErr = buildQuestions(fields, contexts, purpose, params.Instructions, label)
		}
		routes[label] = routeShape{
			tool: tool, fields: fields, defaults: defaults, contexts: contexts, questions: questions, err: shapeErr,
		}
		labels = append(labels, label)
	}
	if params.OutputTool != nil {
		label := outputRouteLabel(*params.OutputTool)
		for slices.ContainsFunc(available, func(tool ai.ToolDefinition) bool { return tool.Name == label }) {
			label += " (output)"
		}
		addRoute(label, params.OutputTool, outputPurpose(*params.OutputTool, nil, params.Instructions))
	}
	for index := range available {
		addRoute(available[index].Name, &available[index], available[index].Description)
	}
	if len(routes) == 0 {
		return nil, fmt.Errorf("typesafe: request has no typed question")
	}
	selectedLabel := labels[0]
	probability := 1.0
	var routeDetails map[string]any
	questions := map[string]question{}
	if len(routes) == 1 {
		shape := routes[selectedLabel]
		if shape.err != nil {
			return nil, shape.err
		}
		shape.questions, shape.err = buildQuestions(
			shape.fields, shape.contexts, shape.tool.Description, params.Instructions, "",
		)
		if shape.err != nil {
			return nil, shape.err
		}
		routes[selectedLabel] = shape
		maps.Copy(questions, shape.questions)
	} else {
		criteria := make(map[string]any, len(routes))
		questionBytes := 0
		smallestQuestionBytes := 0
		allFillable := true
		for _, label := range labels {
			shape := routes[label]
			if shape.err != nil {
				allFillable = false
				continue
			}
			encoded, _ := json.Marshal(shape.questions)
			questionBytes += len(encoded)
			if smallestQuestionBytes == 0 || len(encoded) < smallestQuestionBytes {
				smallestQuestionBytes = len(encoded)
			}
		}
		stateJSON, _ := json.Marshal(state)
		unpickedBytes := questionBytes
		if allFillable {
			unpickedBytes -= smallestQuestionBytes
		}
		speculateAll := unpickedBytes/4 <= 260+len(stateJSON)/6 && len(stateJSON)/6+questionBytes/4 <= 16_000
		for _, label := range labels {
			shape := routes[label]
			criteria[label] = outputPurpose(*shape.tool, shape.fields, params.Instructions)
			if shape.tool != params.OutputTool {
				criteria[label] = shape.tool.Description
			}
			if shape.err == nil && (speculateAll || shape.tool == params.OutputTool) {
				shape.included = true
				shape.scoped = make(map[string]string, len(shape.questions))
				for name, value := range shape.questions {
					scoped := freeQuestionName(questions, label+"."+name)
					shape.scoped[name] = scoped
					questions[scoped] = value
				}
				routes[label] = shape
			}
		}
		questions["route"] = question{
			Type: "choice", Instructions: map[string]any{
				"question": "Which of these does this call for?", "background": params.Instructions,
			}, Criteria: criteria,
		}
	}
	if len(questions) == 0 {
		return nil, fmt.Errorf("typesafe: request has no typed question")
	}
	response, err := model.call(ctx, state, questions, settings)
	if err != nil {
		return nil, err
	}
	if len(routes) > 1 {
		choice := response.Answers["route"]
		selectedLabel = choice.Choice
		_, ok := routes[selectedLabel]
		probability, answered := choice.Probabilities[selectedLabel]
		if !ok || !answered || probability < 0 || probability > 1 {
			return nil, fmt.Errorf("typesafe: invalid route answer %q", selectedLabel)
		}
		if probability < typed.routeThreshold() {
			return nil, &UnsureRoute{
				ModelName: model.name, Route: selectedLabel, Probability: probability,
				Probabilities: maps.Clone(choice.Probabilities), Threshold: typed.routeThreshold(),
			}
		}
		routeDetails = map[string]any{
			"choice": selectedLabel, "probabilities": maps.Clone(choice.Probabilities), "offered": labels,
		}
	}
	selected := routes[selectedLabel]
	if selected.err != nil {
		return nil, &UnfillableRoute{
			ModelName: model.name, ToolName: selected.tool.Name, Route: selectedLabel, Probability: probability,
		}
	}
	usage := ai.Usage{Requests: 1, InputTokens: response.Usage.InputTokens, OutputTokens: response.Usage.OutputTokens}
	answers := response.Answers
	if len(routes) > 1 && selected.included {
		unscoped := make(map[string]answer, len(selected.questions))
		for name := range selected.questions {
			unscoped[name] = answers[selected.scoped[name]]
		}
		answers = unscoped
	} else if len(routes) > 1 && len(selected.questions) > 0 {
		response, err = model.call(ctx, state, selected.questions, settings)
		if err != nil {
			return nil, fmt.Errorf("typesafe: selected %q but failed to fill its fields: %w", selectedLabel, err)
		}
		usage.Requests++
		usage.InputTokens += response.Usage.InputTokens
		usage.OutputTokens += response.Usage.OutputTokens
		answers = response.Answers
	}
	arguments, details, err := decodeAnswers(
		answers, selected.fields, selected.questions, typed.booleanThreshold(), selected.defaults,
	)
	if err != nil {
		return nil, err
	}
	if routeDetails != nil {
		details["route"] = routeDetails
	}
	if usage.Requests > 1 {
		details["requests"] = usage.Requests
	}
	var output any = arguments
	if selected.tool.Schema["type"] != "object" && len(arguments) == 1 {
		if scalar, exists := arguments["response"]; exists {
			output = scalar
		}
	}
	encoded, _ := json.Marshal(output)
	return &ai.ModelResponse{
		Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: selected.tool.Name, ToolCallID: "typesafe-1", Args: encoded, ProviderName: model.providerName,
		}},
		Usage:     usage,
		ModelName: response.Model, ProviderName: model.providerName, ProviderURL: model.baseURL,
		ProviderDetails: details, FinishReason: ai.FinishReasonToolCall,
		State: ai.ModelResponseStateComplete, Timestamp: time.Now().UTC(),
	}, nil
}

func toolsLeft(messages []ai.ModelMessage, tools []ai.ToolDefinition) []ai.ToolDefinition {
	returned := map[string]bool{}
	for _, message := range messages {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			switch part := part.(type) {
			case ai.UserPromptPart:
				clear(returned)
			case ai.ToolReturnPart:
				returned[part.ToolName] = true
			}
		}
	}
	available := make([]ai.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if !returned[tool.Name] {
			available = append(available, tool)
		}
	}
	return available
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
) (result responseBody, err error) {
	tracer := trace.SpanFromContext(ctx).TracerProvider().Tracer("github.com/Kludex/pydantic-ai-go/ai")
	ctx, span := tracer.Start(ctx, "decide "+model.name, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(
		attribute.String("gen_ai.operation.name", "decide"),
		attribute.String("gen_ai.request.model", model.name),
		attribute.String("gen_ai.provider.name", model.providerName),
		attribute.Int("pydantic_ai.decision.question_count", len(questions)),
	))
	defer func() {
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		} else {
			span.SetAttributes(
				attribute.String("gen_ai.response.model", result.Model),
				attribute.Int("gen_ai.usage.input_tokens", result.Usage.InputTokens),
				attribute.Int("gen_ai.usage.output_tokens", result.Usage.OutputTokens),
			)
		}
		span.End()
	}()
	body, err := json.Marshal(requestBody{State: state, Questions: questions, Model: model.name})
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

func outputRouteLabel(tool ai.ToolDefinition) string {
	if title, _ := tool.Schema["title"].(string); title != "" {
		return title
	}
	if tool.Name != "" && tool.Name != "final_result" {
		return strings.TrimPrefix(tool.Name, "final_result_")
	}
	return "output"
}

func outputPurpose(tool ai.ToolDefinition, fields map[string]map[string]any, instructions string) string {
	if tool.Description != "" && tool.Description != "The final result of the run." {
		return tool.Description
	}
	if field := fields["response"]; field != nil {
		if description, _ := field["description"].(string); description != "" {
			return description
		}
	}
	return firstNonEmpty(instructions, tool.Description)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return "Structured output"
}
func freeQuestionName(questions map[string]question, name string) string {
	for {
		if _, exists := questions[name]; !exists {
			return name
		}
		name += "_"
	}
}
