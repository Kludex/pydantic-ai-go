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
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

const defaultBaseURL = "https://api.typesafe.ai"
const settingsKey = "typesafe_settings"

// Settings combines portable settings with Jev decision thresholds.
type Settings struct {
	Common            ai.ModelSettings
	BooleanThreshold  *float64
	ToolCallThreshold *float64
}

// Build returns detached settings accepted by agent runs.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	for name, value := range map[string]*float64{
		"boolean threshold": settings.BooleanThreshold, "tool-call threshold": settings.ToolCallThreshold,
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
	extra[settingsKey] = typedSettings{BooleanThreshold: settings.BooleanThreshold, ToolCallThreshold: settings.ToolCallThreshold}
	common.ExtraBody = extra
	return common, nil
}

type typedSettings struct {
	BooleanThreshold  *float64
	ToolCallThreshold *float64
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

func (model *Model) Name() string                           { return model.name }
func (model *Model) ProviderName() string                   { return model.providerName }
func (model *Model) ProviderURL() string                    { return model.baseURL }
func (model *Model) ContextWindow() int                     { return 64_000 }
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }

// ModelProfile reports that Jev produces tool output but cannot generate text.
func (model *Model) ModelProfile() ai.ModelProfile {
	unsupported := false
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, SupportsTextOutput: &unsupported, ContextWindow: 64_000}
}

// ToolCallProposed reports a selected route whose arguments Jev cannot express.
type ToolCallProposed struct {
	ModelName   string
	ToolName    string
	Probability float64
}

func (err *ToolCallProposed) Error() string {
	return fmt.Sprintf("Jev proposed calling %q (probability %.2f) but cannot fill its arguments", err.ToolName, err.Probability)
}
func (*ToolCallProposed) IsModelAPIError() bool { return true }

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
	instructions := params.Instructions
	var outputQuestions map[string]question
	var outputFields map[string]map[string]any
	if params.OutputTool != nil {
		outputFields, err = schemaFields(params.OutputTool.Schema)
		if err == nil {
			outputQuestions, err = buildQuestions(outputFields, params.OutputTool.Description, instructions, "")
		}
		if err != nil && len(params.Tools) == 0 {
			return nil, err
		}
	}
	questions := maps.Clone(outputQuestions)
	if questions == nil {
		questions = map[string]question{}
	}
	available := toolsLeft(messages, params.Tools)
	routeKey := ""
	if params.OutputTool != nil && len(available) > 0 {
		criteria := map[string]any{params.OutputTool.Name: outputPurpose(*params.OutputTool, outputFields, instructions)}
		for _, tool := range available {
			criteria[tool.Name] = tool.Description
		}
		routeKey = freeQuestionName(questions, "tool")
		questions[routeKey] = question{Type: "choice", Instructions: "Which of these does this call for?", Criteria: criteria}
	}
	if len(questions) == 0 {
		return nil, fmt.Errorf("typesafe: request has no typed question")
	}
	response, err := model.call(ctx, state, questions, settings)
	if err != nil {
		return nil, err
	}
	usage := ai.Usage{Requests: 1, InputTokens: response.Usage.InputTokens, OutputTokens: response.Usage.OutputTokens}
	selected := params.OutputTool
	toolProbability := 1.0
	var routeDetails map[string]any
	if routeKey != "" {
		choice := response.Answers[routeKey]
		var ok bool
		toolProbability, ok = choice.Probabilities[choice.Choice]
		if !ok || toolProbability < 0 || toolProbability > 1 {
			return nil, fmt.Errorf("typesafe: invalid route answer %q", choice.Choice)
		}
		offered := make([]string, len(available))
		matched := params.OutputTool != nil && choice.Choice == params.OutputTool.Name
		for index := range available {
			offered[index] = available[index].Name
			if available[index].Name == choice.Choice {
				matched = true
				if toolProbability >= typed.toolThreshold() {
					selected = &available[index]
				}
			}
		}
		if !matched {
			return nil, fmt.Errorf("typesafe: Jev selected unavailable route %q", choice.Choice)
		}
		routeDetails = map[string]any{"choice": choice.Choice, "probabilities": choice.Probabilities, "offered": offered}
	}
	if selected == nil {
		return nil, fmt.Errorf("typesafe: Jev did not select an available output")
	}
	fields, fieldErr := schemaFields(selected.Schema)
	selectedQuestions, questionErr := buildQuestions(fields, selected.Description, instructions, selected.Name)
	if fieldErr != nil || questionErr != nil {
		return nil, &ToolCallProposed{ModelName: model.name, ToolName: selected.Name, Probability: toolProbability}
	}
	answers := response.Answers
	if params.OutputTool == nil || selected.Name != params.OutputTool.Name {
		if len(selectedQuestions) == 0 {
			answers = map[string]answer{}
		} else {
			response, err = model.call(ctx, state, selectedQuestions, settings)
			if err != nil {
				return nil, fmt.Errorf("typesafe: selected %q but failed to fill arguments: %w", selected.Name, err)
			}
			usage.Requests++
			usage.InputTokens += response.Usage.InputTokens
			usage.OutputTokens += response.Usage.OutputTokens
			answers = response.Answers
		}
		outputFields, outputQuestions = fields, selectedQuestions
	}
	arguments, details, err := decodeAnswers(answers, outputFields, outputQuestions, typed.booleanThreshold())
	if err != nil {
		return nil, err
	}
	if routeDetails != nil {
		details["tool"] = routeDetails
	}
	if usage.Requests > 1 {
		details["requests"] = usage.Requests
	}
	var output any = arguments
	if selected.Schema["type"] != "object" && len(arguments) == 1 {
		if scalar, exists := arguments["response"]; exists {
			output = scalar
		}
	}
	encoded, _ := json.Marshal(output)
	return &ai.ModelResponse{
		Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: selected.Name, ToolCallID: "typesafe-1", Args: encoded, ProviderName: model.providerName}},
		Usage: usage, ModelName: response.Model, ProviderName: model.providerName, ProviderURL: model.baseURL,
		ProviderDetails: details, FinishReason: ai.FinishReasonToolCall, State: ai.ModelResponseStateComplete, Timestamp: time.Now().UTC(),
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
	for _, threshold := range []*float64{typed.BooleanThreshold, typed.ToolCallThreshold} {
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
func (settings typedSettings) toolThreshold() float64 {
	if settings.ToolCallThreshold != nil {
		return *settings.ToolCallThreshold
	}
	return .6
}

func (model *Model) call(ctx context.Context, state any, questions map[string]question, settings ai.ModelSettings) (responseBody, error) {
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
