// Package google implements ai.Model against the Google Gemini API.
package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	jsonschema "github.com/Kludex/pydantic-ai-go/internal/schema"
)

// Model calls the Gemini generateContent API. Create one with NewModel.
type Model struct {
	name              string
	apiKey            string
	baseURL           string
	httpClient        *http.Client
	strictToolSupport bool
	defaultSettings   ai.ModelSettings
}

// Option configures a Model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is the GEMINI_API_KEY environment variable.
func WithAPIKey(key string) Option { return func(m *Model) { m.apiKey = key } }

// WithBaseURL points the model at a different endpoint. The default is
// https://generativelanguage.googleapis.com/v1beta.
func WithBaseURL(url string) Option { return func(m *Model) { m.baseURL = url } }

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(c *http.Client) Option { return func(m *Model) { m.httpClient = c } }

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	return func(m *Model) { m.defaultSettings = settings.Clone() }
}

// WithStrictToolSupport overrides whether the model supports Gemini's
// VALIDATED function-calling mode. Use it for aliases and compatible proxies.
func WithStrictToolSupport(enabled bool) Option {
	return func(m *Model) { m.strictToolSupport = enabled }
}

// NewModel creates a Model for the named Gemini model, e.g. "gemini-2.5-flash".
func NewModel(name string, opts ...Option) *Model {
	m := &Model{
		name:              name,
		apiKey:            os.Getenv("GEMINI_API_KEY"),
		baseURL:           "https://generativelanguage.googleapis.com/v1beta",
		httpClient:        http.DefaultClient,
		strictToolSupport: supportsStrictTools(name),
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.name }

// DefaultModelSettings returns this model's request defaults.
func (m *Model) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

// Request implements ai.Model.
func (m *Model) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	payload, err := m.buildPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("google: marshal request: %w", err)
	}
	url := fmt.Sprintf("%s/models/%s:generateContent", m.baseURL, m.name)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", m.apiKey)
	setExtraHeaders(req, params.Settings.ExtraHeaders)

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("google: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("google: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	response, err := parseResponse(data)
	if response != nil {
		response.ProviderName = "google"
		response.ProviderURL = m.baseURL
		if serviceTier := resp.Header.Get("x-gemini-service-tier"); serviceTier != "" {
			if response.ProviderDetails == nil {
				response.ProviderDetails = map[string]any{}
			}
			response.ProviderDetails["service_tier"] = strings.ToLower(serviceTier)
		}
	}
	return response, err
}

// APIError is a non-200 response from the Gemini API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("google: API returned status %d: %s", e.StatusCode, e.Body)
}

// IsModelAPIError marks provider API responses as eligible for default model fallback.
func (*APIError) IsModelAPIError() bool { return true }

type generateRequest struct {
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	Contents          []content         `json:"contents"`
	Tools             []toolsParam      `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FileData         *fileData         `json:"fileData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

func convertUserPrompt(p ai.UserPromptPart) ([]part, error) {
	if len(p.Contents) == 0 {
		return []part{{Text: p.Content}}, nil
	}
	parts := make([]part, 0, len(p.Contents))
	for _, c := range p.Contents {
		switch item := c.(type) {
		case ai.TextContent:
			parts = append(parts, part{Text: item.Text})
		case ai.BinaryContent:
			parts = append(parts, part{InlineData: &inlineData{
				MimeType: item.MediaType,
				Data:     base64.StdEncoding.EncodeToString(item.Data),
			}})
		case ai.ImageURL:
			parts = append(parts, part{FileData: &fileData{FileURI: item.URL}})
		default:
			return nil, fmt.Errorf("google: unsupported user content type %T", c)
		}
	}
	return parts, nil
}

type functionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type functionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type toolsParam struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionDeclaration struct {
	Name                 string         `json:"name"`
	Description          string         `json:"description,omitempty"`
	ParametersJSONSchema map[string]any `json:"parametersJsonSchema,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig struct {
		Mode string `json:"mode"`
	} `json:"functionCallingConfig"`
}

type generationConfig struct {
	MaxOutputTokens  int             `json:"maxOutputTokens,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"topP,omitempty"`
	StopSequences    []string        `json:"stopSequences,omitempty"`
	ResponseMimeType string          `json:"responseMimeType,omitempty"`
	ResponseSchema   map[string]any  `json:"responseJsonSchema,omitempty"`
	ThinkingConfig   *thinkingConfig `json:"thinkingConfig,omitempty"`
	PresencePenalty  *float64        `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequencyPenalty,omitempty"`
	ResponseLogprobs *bool           `json:"responseLogprobs,omitempty"`
	Logprobs         *int            `json:"logprobs,omitempty"`
	ServiceTier      string          `json:"serviceTier,omitempty"`
}

type thinkingConfig struct {
	IncludeThoughts *bool  `json:"includeThoughts,omitempty"`
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
}

func (m *Model) buildPayload(msgs []ai.ModelMessage, params ai.ModelRequestParams) (*generateRequest, error) {
	req := &generateRequest{}
	if params.Instructions != "" {
		req.SystemInstruction = &content{Parts: []part{{Text: params.Instructions}}}
	}
	settings := params.Settings
	thinking, err := googleThinking(m.name, settings.Thinking)
	if err != nil {
		return nil, err
	}
	serviceTier, err := googleServiceTier(settings.ServiceTier)
	if err != nil {
		return nil, err
	}
	if settings.MaxTokens != 0 || settings.Temperature != nil || settings.TopP != nil ||
		settings.PresencePenalty != nil || settings.FrequencyPenalty != nil || settings.Logprobs != nil ||
		settings.TopLogprobs != nil || serviceTier != "" || len(settings.StopSequences) > 0 || thinking != nil {
		req.GenerationConfig = &generationConfig{
			MaxOutputTokens:  settings.MaxTokens,
			Temperature:      settings.Temperature,
			TopP:             settings.TopP,
			StopSequences:    settings.StopSequences,
			ThinkingConfig:   thinking,
			PresencePenalty:  settings.PresencePenalty,
			FrequencyPenalty: settings.FrequencyPenalty,
			ResponseLogprobs: settings.Logprobs,
			Logprobs:         settings.TopLogprobs,
			ServiceTier:      serviceTier,
		}
	}
	for _, msg := range msgs {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		req.Contents = append(req.Contents, converted...)
	}
	declarations := make([]functionDeclaration, 0, len(params.Tools)+1)
	strictDisabled := false
	for _, tool := range params.Tools {
		declarations = append(declarations, convertTool(tool))
		strictDisabled = strictDisabled || tool.Strict != nil && !*tool.Strict
	}
	if params.OutputTool != nil {
		declarations = append(declarations, convertTool(*params.OutputTool))
		strictDisabled = strictDisabled || params.OutputTool.Strict != nil && !*params.OutputTool.Strict
		if !params.AllowText {
			tc := &toolConfig{}
			tc.FunctionCallingConfig.Mode = "ANY"
			req.ToolConfig = tc
		}
	}
	if req.ToolConfig == nil && len(declarations) > 0 {
		tc := &toolConfig{}
		tc.FunctionCallingConfig.Mode = "AUTO"
		if m.strictToolSupport && !strictDisabled {
			tc.FunctionCallingConfig.Mode = "VALIDATED"
		}
		req.ToolConfig = tc
	}
	if params.OutputSchema != nil && params.OutputMode != ai.OutputModePrompted {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &generationConfig{}
		}
		req.GenerationConfig.ResponseMimeType = "application/json"
		req.GenerationConfig.ResponseSchema = transformSchema(params.OutputSchema)
	}
	if len(declarations) > 0 {
		req.Tools = []toolsParam{{FunctionDeclarations: declarations}}
	}
	return req, nil
}

func googleServiceTier(tier ai.ServiceTier) (string, error) {
	switch tier {
	case "", ai.ServiceTierAuto:
		return "", nil
	case ai.ServiceTierDefault:
		return "standard", nil
	case ai.ServiceTierFlex, ai.ServiceTierPriority:
		return string(tier), nil
	default:
		return "", fmt.Errorf("google: invalid service tier %q", tier)
	}
}

func googleThinking(modelName string, settings *ai.ThinkingSettings) (*thinkingConfig, error) {
	if settings == nil || settings.Level == "" && settings.TokenBudget == nil && settings.IncludeThoughts == nil {
		return nil, nil
	}
	config := &thinkingConfig{}
	gemini3 := strings.Contains(strings.ToLower(modelName), "gemini-3")
	if settings.Level == ai.ThinkingLevelDisabled {
		if settings.IncludeThoughts != nil {
			include := *settings.IncludeThoughts
			config.IncludeThoughts = &include
		}
		if gemini3 {
			config.ThinkingLevel = "MINIMAL"
		} else {
			budget := 0
			config.ThinkingBudget = &budget
		}
		return config, nil
	}
	include := true
	if settings.IncludeThoughts != nil {
		include = *settings.IncludeThoughts
	}
	config.IncludeThoughts = &include
	if settings.TokenBudget != nil {
		budget := *settings.TokenBudget
		config.ThinkingBudget = &budget
		return config, nil
	}
	if settings.Level == ai.ThinkingLevelEnabled || settings.Level == "" {
		return config, nil
	}
	if gemini3 {
		levels := map[ai.ThinkingLevel]string{
			ai.ThinkingLevelMinimal: "MINIMAL", ai.ThinkingLevelLow: "LOW",
			ai.ThinkingLevelMedium: "MEDIUM", ai.ThinkingLevelHigh: "HIGH", ai.ThinkingLevelXHigh: "HIGH",
		}
		level, ok := levels[settings.Level]
		if !ok {
			return nil, fmt.Errorf("google: invalid thinking level %q", settings.Level)
		}
		config.ThinkingLevel = level
		return config, nil
	}
	budgets := map[ai.ThinkingLevel]int{
		ai.ThinkingLevelMinimal: 128, ai.ThinkingLevelLow: 2048, ai.ThinkingLevelMedium: 8192,
		ai.ThinkingLevelHigh: 24576, ai.ThinkingLevelXHigh: 24576,
	}
	budget, ok := budgets[settings.Level]
	if !ok {
		return nil, fmt.Errorf("google: invalid thinking level %q", settings.Level)
	}
	config.ThinkingBudget = &budget
	return config, nil
}

func convertMessage(msg ai.ModelMessage) ([]content, error) {
	switch m := msg.(type) {
	case ai.ModelRequest:
		return convertRequest(m)
	case ai.ModelResponse:
		return convertResponse(m)
	default:
		return nil, fmt.Errorf("google: unknown message type %T", msg)
	}
}

func convertRequest(m ai.ModelRequest) ([]content, error) {
	var parts []part
	for _, p := range m.Parts {
		switch rp := p.(type) {
		case ai.SystemPromptPart:
			parts = append(parts, part{Text: rp.Content})
		case ai.UserPromptPart:
			converted, err := convertUserPrompt(rp)
			if err != nil {
				return nil, err
			}
			parts = append(parts, converted...)
		case ai.ToolReturnPart:
			key := "result"
			if rp.Outcome == ai.ToolReturnOutcomeFailed || rp.Outcome == ai.ToolReturnOutcomeInterrupted {
				key = "error"
			}
			parts = append(parts, part{FunctionResponse: &functionResponse{
				ID: rp.ToolCallID, Name: rp.ToolName, Response: map[string]any{key: rp.Content},
			}})
		case ai.ToolAvailabilityDeltaPart:
		case ai.RetryPromptPart:
			response := rp.ModelResponse()
			if rp.ToolName != "" {
				parts = append(parts, part{FunctionResponse: &functionResponse{
					ID:       rp.ToolCallID,
					Name:     rp.ToolName,
					Response: map[string]any{"error": response},
				}})
			} else {
				parts = append(parts, part{Text: response})
			}
		default:
			return nil, fmt.Errorf("google: unknown request part type %T", p)
		}
	}
	return []content{{Role: "user", Parts: parts}}, nil
}

func convertResponse(m ai.ModelResponse) ([]content, error) {
	var parts []part
	for _, p := range m.Parts {
		switch rp := p.(type) {
		case ai.TextPart:
			parts = append(parts, part{Text: rp.Content, ThoughtSignature: googleThoughtSignature(
				rp.ProviderName, rp.ProviderDetails,
			)})
		case ai.ThinkingPart:
			parts = append(parts, part{
				Text: rp.Content, Thought: true,
				ThoughtSignature: googleThoughtSignature(rp.ProviderName, rp.ProviderDetails),
			})
		case ai.ToolCallPart:
			var args map[string]any
			if len(rp.Args) > 0 {
				if err := json.Unmarshal(rp.Args, &args); err != nil {
					return nil, fmt.Errorf("google: tool call args: %w", err)
				}
			}
			parts = append(parts, part{
				FunctionCall:     &functionCall{ID: rp.ToolCallID, Name: rp.ToolName, Args: args},
				ThoughtSignature: googleThoughtSignature(rp.ProviderName, rp.ProviderDetails),
			})
		}
	}
	return []content{{Role: "model", Parts: parts}}, nil
}

func googlePartMetadata(signature string) (string, map[string]any) {
	if signature == "" {
		return "", nil
	}
	return "google", map[string]any{"thought_signature": signature}
}

func googleThoughtSignature(providerName string, details map[string]any) string {
	if providerName != "" && providerName != "google" {
		return ""
	}
	signature, _ := details["thought_signature"].(string)
	return signature
}

func convertTool(def ai.ToolDefinition) functionDeclaration {
	return functionDeclaration{
		Name:                 def.Name,
		Description:          def.Description,
		ParametersJSONSchema: transformSchema(def.Schema),
	}
}

func supportsStrictTools(name string) bool {
	return (strings.Contains(name, "gemini-2.5") || strings.Contains(name, "gemini-3")) &&
		!strings.Contains(name, "image")
}

func transformSchema(source map[string]any) map[string]any {
	return jsonschema.Transform(source, func(schema map[string]any) {
		for _, key := range []string{"$schema", "discriminator", "examples", "title", "exclusiveMinimum", "exclusiveMaximum"} {
			delete(schema, key)
		}
		if value, ok := schema["const"]; ok {
			delete(schema, "const")
			schema["enum"] = []any{value}
			if _, ok := schema["type"]; !ok {
				switch value.(type) {
				case string:
					schema["type"] = "string"
				case bool:
					schema["type"] = "boolean"
				case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
					schema["type"] = "integer"
				case float32, float64:
					schema["type"] = "number"
				}
			}
		}
		if schema["type"] == "string" {
			if format, ok := schema["format"].(string); ok {
				delete(schema, "format")
				if description, ok := schema["description"].(string); ok && description != "" {
					schema["description"] = fmt.Sprintf("%s (format: %s)", description, format)
				} else {
					schema["description"] = "Format: " + format
				}
			}
		}
	})
}

type generateResponse struct {
	ResponseID   string `json:"responseId"`
	ModelVersion string `json:"modelVersion"`
	Candidates   []struct {
		Content struct {
			Parts []part `json:"parts"`
		} `json:"content"`
		FinishReason   string         `json:"finishReason"`
		LogprobsResult map[string]any `json:"logprobsResult"`
		AvgLogprobs    *float64       `json:"avgLogprobs"`
	} `json:"candidates"`
	UsageMetadata googleUsage `json:"usageMetadata"`
}

type googleUsage struct {
	PromptTokenCount           int           `json:"promptTokenCount"`
	CandidatesTokenCount       int           `json:"candidatesTokenCount"`
	CachedContentTokenCount    int           `json:"cachedContentTokenCount"`
	ThoughtsTokenCount         int           `json:"thoughtsTokenCount"`
	ToolUsePromptTokenCount    int           `json:"toolUsePromptTokenCount"`
	PromptTokensDetails        []tokenDetail `json:"promptTokensDetails"`
	CacheTokensDetails         []tokenDetail `json:"cacheTokensDetails"`
	CandidatesTokensDetails    []tokenDetail `json:"candidatesTokensDetails"`
	ToolUsePromptTokensDetails []tokenDetail `json:"toolUsePromptTokensDetails"`
}

type tokenDetail struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

func (u googleUsage) usage() ai.Usage {
	usage := ai.Usage{
		Requests: 1, InputTokens: u.PromptTokenCount,
		OutputTokens:    u.CandidatesTokenCount + u.ThoughtsTokenCount,
		CacheReadTokens: u.CachedContentTokenCount, ReasoningTokens: u.ThoughtsTokenCount,
		Details: map[string]int{},
	}
	if u.CachedContentTokenCount != 0 {
		usage.Details["cached_content_tokens"] = u.CachedContentTokenCount
	}
	if u.ThoughtsTokenCount != 0 {
		usage.Details["thoughts_tokens"] = u.ThoughtsTokenCount
	}
	if u.ToolUsePromptTokenCount != 0 {
		usage.Details["tool_use_prompt_tokens"] = u.ToolUsePromptTokenCount
	}
	for _, detail := range u.PromptTokensDetails {
		if detail.Modality != "" && detail.TokenCount != 0 {
			usage.Details[strings.ToLower(detail.Modality)+"_prompt_tokens"] = detail.TokenCount
		}
		if detail.Modality == "AUDIO" {
			usage.InputAudioTokens += detail.TokenCount
		}
	}
	for _, detail := range u.CacheTokensDetails {
		if detail.Modality != "" && detail.TokenCount != 0 {
			usage.Details[strings.ToLower(detail.Modality)+"_cache_tokens"] = detail.TokenCount
		}
		if detail.Modality == "AUDIO" {
			usage.CacheAudioReadTokens += detail.TokenCount
		}
	}
	for _, detail := range u.CandidatesTokensDetails {
		if detail.Modality != "" && detail.TokenCount != 0 {
			usage.Details[strings.ToLower(detail.Modality)+"_candidates_tokens"] = detail.TokenCount
		}
		if detail.Modality == "AUDIO" {
			usage.OutputAudioTokens += detail.TokenCount
		}
	}
	for _, detail := range u.ToolUsePromptTokensDetails {
		if detail.Modality != "" && detail.TokenCount != 0 {
			usage.Details[strings.ToLower(detail.Modality)+"_tool_use_prompt_tokens"] = detail.TokenCount
		}
	}
	return usage
}

func googleFinishReason(reason string) ai.FinishReason {
	return map[string]ai.FinishReason{
		"STOP": ai.FinishReasonStop, "MAX_TOKENS": ai.FinishReasonLength,
		"SAFETY": ai.FinishReasonContentFilter, "RECITATION": ai.FinishReasonContentFilter,
		"BLOCKLIST": ai.FinishReasonContentFilter, "PROHIBITED_CONTENT": ai.FinishReasonContentFilter,
		"SPII": ai.FinishReasonContentFilter, "IMAGE_SAFETY": ai.FinishReasonContentFilter,
		"IMAGE_PROHIBITED_CONTENT": ai.FinishReasonContentFilter, "MODEL_ARMOR": ai.FinishReasonContentFilter,
		"LANGUAGE": ai.FinishReasonError, "MALFORMED_FUNCTION_CALL": ai.FinishReasonError,
		"UNEXPECTED_TOOL_CALL": ai.FinishReasonError, "NO_IMAGE": ai.FinishReasonError,
	}[reason]
}

func parseResponse(data []byte) (*ai.ModelResponse, error) {
	var gr generateResponse
	if err := json.Unmarshal(data, &gr); err != nil {
		return nil, fmt.Errorf("google: parse response: %w", err)
	}
	if len(gr.Candidates) == 0 {
		return nil, fmt.Errorf("google: response has no candidates")
	}
	providerDetails := map[string]any{}
	if gr.Candidates[0].FinishReason != "" {
		providerDetails["finish_reason"] = gr.Candidates[0].FinishReason
	}
	if gr.Candidates[0].LogprobsResult != nil {
		providerDetails["logprobs"] = gr.Candidates[0].LogprobsResult
	}
	if gr.Candidates[0].AvgLogprobs != nil {
		providerDetails["avg_logprobs"] = *gr.Candidates[0].AvgLogprobs
	}
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	resp := &ai.ModelResponse{
		ModelName: gr.ModelVersion, Usage: gr.UsageMetadata.usage(),
		ProviderDetails: providerDetails, ProviderResponseID: gr.ResponseID,
		FinishReason: googleFinishReason(gr.Candidates[0].FinishReason), State: ai.ModelResponseStateComplete,
	}
	for _, p := range gr.Candidates[0].Content.Parts {
		providerName, providerDetails := googlePartMetadata(p.ThoughtSignature)
		switch {
		case p.FunctionCall != nil:
			// args came from parsed JSON, so re-marshalling cannot fail
			args, _ := json.Marshal(p.FunctionCall.Args)
			resp.Parts = append(resp.Parts, ai.ToolCallPart{
				ToolName: p.FunctionCall.Name, Args: args, ToolCallID: p.FunctionCall.ID,
				ProviderName: providerName, ProviderDetails: providerDetails,
			})
		case p.Thought:
			resp.Parts = append(resp.Parts, ai.ThinkingPart{
				Content: p.Text, ProviderName: providerName, ProviderDetails: providerDetails,
			})
		case p.Text != "" || providerDetails != nil:
			resp.Parts = append(resp.Parts, ai.TextPart{
				Content: p.Text, ProviderName: providerName, ProviderDetails: providerDetails,
			})
		}
	}
	return resp, nil
}
