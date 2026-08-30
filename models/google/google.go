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

	ai "github.com/Kludex/pydantic-ai-go"
)

// Model calls the Gemini generateContent API. Create one with NewModel.
type Model struct {
	name       string
	apiKey     string
	baseURL    string
	httpClient *http.Client
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

// NewModel creates a Model for the named Gemini model, e.g. "gemini-2.5-flash".
func NewModel(name string, opts ...Option) *Model {
	m := &Model{
		name:       name,
		apiKey:     os.Getenv("GEMINI_API_KEY"),
		baseURL:    "https://generativelanguage.googleapis.com/v1beta",
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.name }

// Request implements ai.Model.
func (m *Model) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	payload, err := buildPayload(msgs, params)
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
	return parseResponse(data)
}

// APIError is a non-200 response from the Gemini API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("google: API returned status %d: %s", e.StatusCode, e.Body)
}

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
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type functionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type toolsParam struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig struct {
		Mode string `json:"mode"`
	} `json:"functionCallingConfig"`
}

type generationConfig struct {
	MaxOutputTokens  int            `json:"maxOutputTokens,omitempty"`
	Temperature      *float64       `json:"temperature,omitempty"`
	TopP             *float64       `json:"topP,omitempty"`
	StopSequences    []string       `json:"stopSequences,omitempty"`
	ResponseMimeType string         `json:"responseMimeType,omitempty"`
	ResponseSchema   map[string]any `json:"responseSchema,omitempty"`
}

func buildPayload(msgs []ai.ModelMessage, params ai.ModelRequestParams) (*generateRequest, error) {
	req := &generateRequest{}
	if params.Instructions != "" {
		req.SystemInstruction = &content{Parts: []part{{Text: params.Instructions}}}
	}
	settings := params.Settings
	if settings.MaxTokens != 0 || settings.Temperature != nil || settings.TopP != nil || len(settings.StopSequences) > 0 {
		req.GenerationConfig = &generationConfig{
			MaxOutputTokens: settings.MaxTokens,
			Temperature:     settings.Temperature,
			TopP:            settings.TopP,
			StopSequences:   settings.StopSequences,
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
	strictEnabled, strictDisabled := false, false
	for _, tool := range params.Tools {
		declarations = append(declarations, convertTool(tool))
		strictEnabled, strictDisabled = collectStrict(tool, strictEnabled, strictDisabled)
	}
	if params.OutputTool != nil {
		declarations = append(declarations, convertTool(*params.OutputTool))
		strictEnabled, strictDisabled = collectStrict(*params.OutputTool, strictEnabled, strictDisabled)
		if !params.AllowText {
			tc := &toolConfig{}
			tc.FunctionCallingConfig.Mode = "ANY"
			req.ToolConfig = tc
		}
	}
	if req.ToolConfig == nil && (strictEnabled || strictDisabled) {
		tc := &toolConfig{}
		if strictDisabled {
			tc.FunctionCallingConfig.Mode = "AUTO"
		} else {
			tc.FunctionCallingConfig.Mode = "VALIDATED"
		}
		req.ToolConfig = tc
	}
	if params.OutputSchema != nil {
		if req.GenerationConfig == nil {
			req.GenerationConfig = &generationConfig{}
		}
		req.GenerationConfig.ResponseMimeType = "application/json"
		req.GenerationConfig.ResponseSchema = sanitizeSchema(params.OutputSchema)
	}
	if len(declarations) > 0 {
		req.Tools = []toolsParam{{FunctionDeclarations: declarations}}
	}
	return req, nil
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
			parts = append(parts, part{FunctionResponse: &functionResponse{
				Name:     rp.ToolName,
				Response: map[string]any{"result": rp.Content},
			}})
		case ai.RetryPromptPart:
			if rp.ToolName != "" {
				parts = append(parts, part{FunctionResponse: &functionResponse{
					Name:     rp.ToolName,
					Response: map[string]any{"error": rp.Content},
				}})
			} else {
				parts = append(parts, part{Text: rp.Content})
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
			parts = append(parts, part{Text: rp.Content})
		case ai.ToolCallPart:
			var args map[string]any
			if len(rp.Args) > 0 {
				if err := json.Unmarshal(rp.Args, &args); err != nil {
					return nil, fmt.Errorf("google: tool call args: %w", err)
				}
			}
			parts = append(parts, part{FunctionCall: &functionCall{Name: rp.ToolName, Args: args}})
		}
	}
	return []content{{Role: "model", Parts: parts}}, nil
}

func collectStrict(def ai.ToolDefinition, enabled, disabled bool) (bool, bool) {
	if def.Strict == nil {
		return enabled, disabled
	}
	if *def.Strict {
		return true, disabled
	}
	return enabled, true
}

func convertTool(def ai.ToolDefinition) functionDeclaration {
	return functionDeclaration{
		Name:        def.Name,
		Description: def.Description,
		Parameters:  sanitizeSchema(def.Schema),
	}
}

// sanitizeSchema drops JSON Schema fields Gemini rejects.
func sanitizeSchema(schema map[string]any) map[string]any {
	out := make(map[string]any, len(schema))
	for key, value := range schema {
		if key == "additionalProperties" {
			continue
		}
		if nested, ok := value.(map[string]any); ok {
			out[key] = sanitizeSchema(nested)
			continue
		}
		out[key] = value
	}
	if properties, ok := out["properties"].(map[string]any); ok {
		for name, prop := range properties {
			if nested, ok := prop.(map[string]any); ok {
				properties[name] = sanitizeSchema(nested)
			}
		}
	}
	return out
}

type generateResponse struct {
	ModelVersion string `json:"modelVersion"`
	Candidates   []struct {
		Content struct {
			Parts []part `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
	} `json:"usageMetadata"`
}

func parseResponse(data []byte) (*ai.ModelResponse, error) {
	var gr generateResponse
	if err := json.Unmarshal(data, &gr); err != nil {
		return nil, fmt.Errorf("google: parse response: %w", err)
	}
	if len(gr.Candidates) == 0 {
		return nil, fmt.Errorf("google: response has no candidates")
	}
	resp := &ai.ModelResponse{
		ModelName: gr.ModelVersion,
		Usage: ai.Usage{
			Requests:     1,
			InputTokens:  gr.UsageMetadata.PromptTokenCount,
			OutputTokens: gr.UsageMetadata.CandidatesTokenCount,
		},
	}
	for _, p := range gr.Candidates[0].Content.Parts {
		switch {
		case p.FunctionCall != nil:
			// args came from parsed JSON, so re-marshalling cannot fail
			args, _ := json.Marshal(p.FunctionCall.Args)
			resp.Parts = append(resp.Parts, ai.ToolCallPart{ToolName: p.FunctionCall.Name, Args: args})
		case p.Thought:
			resp.Parts = append(resp.Parts, ai.ThinkingPart{Content: p.Text})
		case p.Text != "":
			resp.Parts = append(resp.Parts, ai.TextPart{Content: p.Text})
		}
	}
	return resp, nil
}
