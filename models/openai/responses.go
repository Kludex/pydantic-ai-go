package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// ResponsesModel calls the OpenAI Responses API, the successor to Chat
// Completions. Create one with NewResponsesModel.
type ResponsesModel struct {
	name                   string
	providerName           string
	apiKey                 string
	baseURL                string
	httpClient             *http.Client
	providerHeaders        http.Header
	providerQuery          url.Values
	prepareRequest         RequestPreparationFunc
	strictToolSupport      bool
	deferredToolSupport    bool
	defaultSettings        ai.ModelSettings
	background             *bool
	backgroundPollInterval time.Duration
	phaseSupport           *bool
	codeExecutionOutputs   bool
}

// NewResponsesModel creates a ResponsesModel for the named OpenAI model.
// It accepts the same options as NewModel.
func NewResponsesModel(name string, opts ...Option) *ResponsesModel {
	m := NewModel(name, opts...)
	var phaseSupport *bool
	if m.responsesPhaseSupport != nil {
		enabled := *m.responsesPhaseSupport
		phaseSupport = &enabled
	}
	return &ResponsesModel{
		name: m.name, providerName: m.providerName, apiKey: m.apiKey,
		baseURL: m.baseURL, httpClient: m.httpClient,
		providerHeaders: m.providerHeaders.Clone(), providerQuery: cloneURLValues(m.providerQuery),
		prepareRequest: m.prepareRequest, strictToolSupport: m.strictToolSupport,
		deferredToolSupport: m.deferredToolSupport, defaultSettings: m.defaultSettings,
		background: m.background, backgroundPollInterval: m.backgroundPollInterval,
		phaseSupport:         phaseSupport,
		codeExecutionOutputs: m.responsesCodeExecutionOutputs,
	}
}

// Name returns the model name.
func (m *ResponsesModel) Name() string { return m.name }

// ProviderName returns the durable provider identity.
func (m *ResponsesModel) ProviderName() string { return m.providerName }

// ProviderURL returns the configured provider API URL.
func (m *ResponsesModel) ProviderURL() string { return m.baseURL }

// DefaultModelSettings returns this model's request defaults.
func (m *ResponsesModel) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

// Request implements ai.Model.
func (m *ResponsesModel) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	if responseID, ok := suspendedResponsesID(msgs, m.providerName); ok {
		return m.retrieveResponse(ctx, responseID, params.Settings.ExtraHeaders)
	}
	payload, err := m.buildResponsesPayload(msgs, params, true)
	if err != nil {
		return nil, err
	}
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := m.configureRequest(req, params.Settings.ExtraHeaders); err != nil {
		return nil, err
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	response, err := parseResponsesResponse(data)
	if response != nil {
		setResponsesProvider(response, m.providerName, m.baseURL)
	}
	return response, err
}

// CountTokens counts input tokens through the Responses input_tokens endpoint.
func (m *ResponsesModel) CountTokens(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (ai.Usage, error) {
	if len(messages) == 0 {
		return ai.Usage{}, fmt.Errorf("openai: cannot count tokens without messages")
	}
	payload, err := m.buildResponsesPayload(messages, params, true)
	if err != nil {
		return ai.Usage{}, err
	}
	countPayload := struct {
		Model             string              `json:"model"`
		Instructions      string              `json:"instructions,omitempty"`
		Input             []responsesInput    `json:"input"`
		Tools             []responsesTool     `json:"tools,omitempty"`
		ToolChoice        any                 `json:"tool_choice,omitempty"`
		ParallelToolCalls *bool               `json:"parallel_tool_calls,omitempty"`
		Reasoning         *responsesReasoning `json:"reasoning,omitempty"`
	}{
		Model: payload.Model, Instructions: payload.Instructions, Input: payload.Input,
		Tools: payload.Tools, ToolChoice: payload.ToolChoice, ParallelToolCalls: payload.ParallelToolCalls,
		Reasoning: payload.Reasoning,
	}
	body, err := marshalRequest(countPayload, params.Settings.ExtraBody)
	if err != nil {
		return ai.Usage{}, fmt.Errorf("openai: marshal token count request: %w", err)
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, m.baseURL+"/responses/input_tokens", bytes.NewReader(body),
	)
	if err != nil {
		return ai.Usage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := m.configureRequest(req, params.Settings.ExtraHeaders); err != nil {
		return ai.Usage{}, err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return ai.Usage{}, fmt.Errorf("openai: token count request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.Usage{}, fmt.Errorf("openai: read token count response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.Usage{}, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	var counted struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(data, &counted); err != nil {
		return ai.Usage{}, fmt.Errorf("openai: decode token count response: %w", err)
	}
	return ai.Usage{InputTokens: counted.InputTokens}, nil
}

// CompactMessages calls the stateless Responses compaction endpoint.
func (m *ResponsesModel) CompactMessages(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	payload, err := m.buildResponsesPayload(messages, params, false)
	if err != nil {
		return nil, err
	}
	body, _ := json.Marshal(struct {
		Model        string           `json:"model"`
		Instructions string           `json:"instructions,omitempty"`
		Input        []responsesInput `json:"input"`
	}{Model: payload.Model, Instructions: payload.Instructions, Input: payload.Input}) // The closed payload is JSON-safe.
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/responses/compact", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := m.configureRequest(req, params.Settings.ExtraHeaders); err != nil {
		return nil, err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: compact request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read compaction response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	compacted, err := parseResponsesResponse(data)
	if err != nil {
		return nil, err
	}
	if len(compacted.Parts) == 0 {
		return nil, fmt.Errorf("openai: compaction response contained no output")
	}
	part, ok := compacted.Parts[len(compacted.Parts)-1].(ai.CompactionPart)
	if !ok {
		return nil, fmt.Errorf("openai: last compaction response item has type %T", compacted.Parts[len(compacted.Parts)-1])
	}
	part.ProviderDetails[ai.StandingPromptPlantedKey] = true
	compacted.Parts[len(compacted.Parts)-1] = part
	setResponsesProvider(compacted, m.providerName, m.baseURL)
	return compacted, nil
}

// ContinuationDelay implements ai.ModelContinuationDelayer.
func (m *ResponsesModel) ContinuationDelay(response ai.ModelResponse) time.Duration {
	if response.State == ai.ModelResponseStateSuspended && providerBool(response.ProviderDetails, "background") {
		return m.backgroundPollInterval
	}
	return 0
}

// CancelSuspendedResponse implements ai.SuspendedResponseCanceler.
func (m *ResponsesModel) CancelSuspendedResponse(ctx context.Context, response ai.ModelResponse) error {
	if response.ProviderName != m.providerName || response.ProviderResponseID == "" ||
		!providerBool(response.ProviderDetails, "background") {
		return nil
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		m.baseURL+"/responses/"+url.PathEscape(response.ProviderResponseID)+"/cancel", nil,
	)
	if err != nil {
		return err
	}
	if err := m.configureRequest(req, nil); err != nil {
		return err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("openai: cancel background response: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("openai: read cancel response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	return nil
}

func (m *ResponsesModel) retrieveResponse(
	ctx context.Context, responseID string, headers map[string]string,
) (*ai.ModelResponse, error) {
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, m.baseURL+"/responses/"+url.PathEscape(responseID), nil,
	)
	if err != nil {
		return nil, err
	}
	if err := m.configureRequest(req, headers); err != nil {
		return nil, err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: retrieve background response: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	response, err := parseResponsesResponse(data)
	if response != nil {
		setResponsesProvider(response, m.providerName, m.baseURL)
	}
	return response, err
}

func suspendedResponsesID(messages []ai.ModelMessage, providerNames ...string) (string, bool) {
	providerName := "openai"
	if len(providerNames) > 0 {
		providerName = providerNames[0]
	}
	if len(messages) == 0 {
		return "", false
	}
	response, ok := messages[len(messages)-1].(ai.ModelResponse)
	if !ok || response.State != ai.ModelResponseStateSuspended || response.ProviderName != providerName ||
		response.ProviderResponseID == "" || !providerBool(response.ProviderDetails, "background") {
		return "", false
	}
	return response.ProviderResponseID, true
}

func providerBool(details map[string]any, key string) bool {
	value, _ := details[key].(bool)
	return value
}

func setResponsesProvider(response *ai.ModelResponse, providerName, providerURL string) {
	response.ProviderName = providerName
	response.ProviderURL = providerURL
	for index, responsePart := range response.Parts {
		switch part := responsePart.(type) {
		case ai.TextPart:
			part.ProviderName = providerName
			response.Parts[index] = part
		case ai.ThinkingPart:
			part.ProviderName = providerName
			response.Parts[index] = part
		case ai.FilePart:
			part.ProviderName = providerName
			response.Parts[index] = part
		case ai.ToolCallPart:
			part.ProviderName = providerName
			response.Parts[index] = part
		case ai.NativeToolCallPart:
			part.ProviderName = providerName
			response.Parts[index] = part
		case ai.NativeToolReturnPart:
			part.ProviderName = providerName
			response.Parts[index] = part
		case ai.CompactionPart:
			part.ProviderName = providerName
			response.Parts[index] = part
		}
	}
}

type responsesRequest struct {
	Model             string              `json:"model"`
	Instructions      string              `json:"instructions,omitempty"`
	Input             []responsesInput    `json:"input"`
	Tools             []responsesTool     `json:"tools,omitempty"`
	ToolChoice        any                 `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool               `json:"parallel_tool_calls,omitempty"`
	MaxTokens         int                 `json:"max_output_tokens,omitempty"`
	Temperature       *float64            `json:"temperature,omitempty"`
	TopP              *float64            `json:"top_p,omitempty"`
	Stream            bool                `json:"stream,omitempty"`
	Background        *bool               `json:"background,omitempty"`
	Reasoning         *responsesReasoning `json:"reasoning,omitempty"`
	Text              *responsesText      `json:"text,omitempty"`
	TopLogprobs       *int                `json:"top_logprobs,omitempty"`
	Include           []string            `json:"include,omitempty"`
	ServiceTier       ai.ServiceTier      `json:"service_tier,omitempty"`
}

type responsesReasoning struct {
	Effort string `json:"effort,omitempty"`
}

type responsesText struct {
	Format responsesTextFormat `json:"format"`
}

type responsesTextFormat struct {
	Type   string         `json:"type"`
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict *bool          `json:"strict,omitempty"`
}

type responsesInput struct {
	// message
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"`
	// function_call and function_call_output items
	Type             string          `json:"type,omitempty"`
	ID               string          `json:"id,omitempty"`
	CallID           any             `json:"call_id,omitempty"`
	Name             string          `json:"name,omitempty"`
	Arguments        any             `json:"arguments,omitempty"`
	Action           json.RawMessage `json:"action,omitempty"`
	Namespace        string          `json:"namespace,omitempty"`
	Output           string          `json:"output,omitempty"`
	Execution        string          `json:"execution,omitempty"`
	Status           string          `json:"status,omitempty"`
	Phase            string          `json:"phase,omitempty"`
	Tools            []responsesTool `json:"tools,omitempty"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`
	ContainerID      string          `json:"container_id,omitempty"`
	Code             string          `json:"code,omitempty"`
	Outputs          any             `json:"outputs,omitempty"`
}

type responsesInputContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

type responsesTool struct {
	Type              string                          `json:"type"`
	Name              string                          `json:"name,omitempty"`
	Description       string                          `json:"description,omitempty"`
	Parameters        map[string]any                  `json:"parameters,omitempty"`
	Strict            *bool                           `json:"strict,omitempty"`
	DeferLoading      bool                            `json:"defer_loading,omitempty"`
	Execution         string                          `json:"execution,omitempty"`
	SearchContextSize ai.WebSearchContextSize         `json:"search_context_size,omitempty"`
	UserLocation      *responsesWebSearchLocation     `json:"user_location,omitempty"`
	Filters           *responsesWebSearchFilters      `json:"filters,omitempty"`
	ExternalWebAccess *bool                           `json:"external_web_access,omitempty"`
	Container         *responsesCodeContainer         `json:"container,omitempty"`
	Action            ai.ImageGenerationAction        `json:"action,omitempty"`
	Background        ai.ImageGenerationBackground    `json:"background,omitempty"`
	InputFidelity     ai.ImageGenerationInputFidelity `json:"input_fidelity,omitempty"`
	Moderation        ai.ImageGenerationModeration    `json:"moderation,omitempty"`
	Model             string                          `json:"model,omitempty"`
	OutputCompression *int                            `json:"output_compression,omitempty"`
	OutputFormat      ai.ImageGenerationOutputFormat  `json:"output_format,omitempty"`
	PartialImages     int                             `json:"partial_images"`
	Quality           ai.ImageGenerationQuality       `json:"quality,omitempty"`
	Size              ai.ImageGenerationSize          `json:"size,omitempty"`
}

type responsesCodeContainer struct {
	Type    string   `json:"type"`
	FileIDs []string `json:"file_ids,omitempty"`
}

type responsesWebSearchLocation struct {
	Type     string `json:"type"`
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

type responsesWebSearchFilters struct {
	AllowedDomains []string `json:"allowed_domains"`
}

func prepareResponsesNativeTool(nativeTool ai.NativeTool, providerName string) (responsesTool, bool, error) {
	var webSearch ai.WebSearchTool
	switch tool := nativeTool.(type) {
	case ai.WebSearchTool:
		webSearch = tool
	case *ai.WebSearchTool:
		webSearch = *tool
	case ai.CodeExecutionTool:
		return responsesCodeExecutionTool(tool, providerName), true, nil
	case *ai.CodeExecutionTool:
		return responsesCodeExecutionTool(*tool, providerName), true, nil
	case ai.ImageGenerationTool:
		prepared, err := responsesImageGenerationTool(tool)
		return prepared, true, err
	case *ai.ImageGenerationTool:
		prepared, err := responsesImageGenerationTool(*tool)
		return prepared, true, err
	default:
		if nativeTool.IsOptional() {
			return responsesTool{}, false, nil
		}
		return responsesTool{}, false, fmt.Errorf("openai: Responses does not support native tool %q", nativeTool.Kind())
	}
	contextSize := webSearch.SearchContextSize
	if contextSize == "" {
		contextSize = ai.WebSearchContextMedium
	}
	tool := responsesTool{Type: "web_search", SearchContextSize: contextSize}
	if webSearch.UserLocation != nil {
		tool.UserLocation = &responsesWebSearchLocation{
			Type: "approximate", City: webSearch.UserLocation.City, Country: webSearch.UserLocation.Country,
			Region: webSearch.UserLocation.Region, Timezone: webSearch.UserLocation.Timezone,
		}
	}
	if len(webSearch.AllowedDomains) > 0 {
		tool.Filters = &responsesWebSearchFilters{AllowedDomains: slices.Clone(webSearch.AllowedDomains)}
	}
	if webSearch.ExternalWebAccess != nil {
		external := *webSearch.ExternalWebAccess
		tool.ExternalWebAccess = &external
	}
	return tool, true, nil
}

func responsesCodeExecutionTool(tool ai.CodeExecutionTool, providerName string) responsesTool {
	container := &responsesCodeContainer{Type: "auto"}
	for _, file := range tool.Files {
		if file.ProviderName == providerName {
			container.FileIDs = append(container.FileIDs, file.FileID)
		}
	}
	return responsesTool{Type: "code_interpreter", Container: container}
}

func responsesImageGenerationTool(tool ai.ImageGenerationTool) (responsesTool, error) {
	action := tool.Action
	if action == "" {
		action = ai.ImageGenerationActionAuto
	}
	background := tool.Background
	if background == "" {
		background = ai.ImageGenerationBackgroundAuto
	}
	moderation := tool.Moderation
	if moderation == "" {
		moderation = ai.ImageGenerationModerationAuto
	}
	compression := 100
	if tool.OutputCompression != nil {
		compression = *tool.OutputCompression
	}
	outputFormat := tool.OutputFormat
	if outputFormat == "" {
		outputFormat = ai.ImageGenerationOutputPNG
	}
	quality := tool.Quality
	if quality == "" {
		quality = ai.ImageGenerationQualityAuto
	}
	size, err := responsesImageGenerationSize(tool.Size, tool.AspectRatio)
	if err != nil {
		return responsesTool{}, err
	}
	return responsesTool{
		Type: "image_generation", Action: action, Background: background, InputFidelity: tool.InputFidelity,
		Moderation: moderation, Model: tool.Model, OutputCompression: &compression, OutputFormat: outputFormat,
		PartialImages: tool.PartialImages, Quality: quality, Size: size,
	}, nil
}

func responsesImageGenerationSize(size ai.ImageGenerationSize, aspectRatio ai.ImageAspectRatio) (ai.ImageGenerationSize, error) {
	if aspectRatio == "" {
		if size == "" {
			return ai.ImageGenerationSizeAuto, nil
		}
		switch size {
		case ai.ImageGenerationSizeAuto, ai.ImageGenerationSize1024x1024,
			ai.ImageGenerationSize1024x1536, ai.ImageGenerationSize1536x1024:
			return size, nil
		default:
			return "", fmt.Errorf("openai: unsupported image generation size %q", size)
		}
	}
	mapped := map[ai.ImageAspectRatio]ai.ImageGenerationSize{
		ai.ImageAspectRatio1x1: ai.ImageGenerationSize1024x1024,
		ai.ImageAspectRatio2x3: ai.ImageGenerationSize1024x1536,
		ai.ImageAspectRatio3x2: ai.ImageGenerationSize1536x1024,
	}[aspectRatio]
	if mapped == "" {
		return "", fmt.Errorf("openai: unsupported image generation aspect ratio %q", aspectRatio)
	}
	if size != "" && size != ai.ImageGenerationSizeAuto && size != mapped {
		return "", fmt.Errorf("openai: image generation aspect ratio %q conflicts with size %q", aspectRatio, size)
	}
	return mapped, nil
}

func (m *ResponsesModel) buildResponsesPayload(
	msgs []ai.ModelMessage, params ai.ModelRequestParams, nativeDeferred bool,
) (*responsesRequest, error) {
	if err := ai.ValidateNativeTools(params.NativeTools); err != nil {
		return nil, fmt.Errorf("openai: native tools: %w", err)
	}
	reasoningEffort, err := openAIThinkingEffort(params.Settings.Thinking)
	if err != nil {
		return nil, err
	}
	serviceTier, err := openAIServiceTier(params.Settings.ServiceTier)
	if err != nil {
		return nil, err
	}
	req := &responsesRequest{
		Model:        m.name,
		Instructions: params.Instructions,
		MaxTokens:    params.Settings.MaxTokens,
		Temperature:  params.Settings.Temperature,
		TopP:         params.Settings.TopP,
		Background:   m.background,
		TopLogprobs:  params.Settings.TopLogprobs,
		ServiceTier:  serviceTier,
	}
	if params.Settings.Logprobs != nil && *params.Settings.Logprobs {
		req.Include = append(req.Include, "message.output_text.logprobs")
	}
	if reasoningEffort != "" {
		req.Reasoning = &responsesReasoning{Effort: reasoningEffort}
	}
	if openAIReasoningActive(reasoningEffort) {
		req.Temperature = nil
		req.TopP = nil
	}
	var searchTool *ai.ToolDefinition
	for _, tool := range params.Tools {
		if tool.ToolKind == ai.ToolPartKindToolSearch && tool.Name == ai.ToolSearchName {
			definition := tool
			searchTool = &definition
			break
		}
	}
	if searchTool != nil &&
		(searchTool.ToolSearchStrategy == ai.ToolSearchStrategyBM25 ||
			searchTool.ToolSearchStrategy == ai.ToolSearchStrategyRegex) {
		return nil, fmt.Errorf(
			"openai: tool search strategy %q is not supported; use automatic, keywords, or custom search",
			searchTool.ToolSearchStrategy,
		)
	}
	activeToolSearch := searchTool != nil && nativeDeferred && m.deferredToolSupport && len(params.DeferredTools) > 0
	clientToolSearch := activeToolSearch &&
		(searchTool.ToolSearchStrategy == ai.ToolSearchStrategyKeywords ||
			searchTool.ToolSearchStrategy == ai.ToolSearchStrategyCustom)
	serverToolSearch := activeToolSearch && !clientToolSearch
	for _, nativeTool := range params.NativeTools {
		tool, include, err := prepareResponsesNativeTool(nativeTool, m.providerName)
		if err != nil {
			return nil, err
		}
		if include {
			req.Tools = append(req.Tools, tool)
			if tool.Type == "code_interpreter" && m.codeExecutionOutputs {
				req.Include = append(req.Include, "code_interpreter_call.outputs")
			}
		}
	}
	deferred := make(map[string]ai.ToolDefinition, len(params.DeferredTools))
	if activeToolSearch {
		for _, tool := range params.DeferredTools {
			deferred[tool.Name] = tool
		}
	}
	converter := responsesMessageConverter{
		providerName:     m.providerName,
		clientToolSearch: activeToolSearch,
		serverToolSearch: serverToolSearch,
		deferred:         deferred,
		rendered:         make(map[string]struct{}),
		strictSupport:    m.strictToolSupport,
		phaseSupport:     responsesPhaseSupported(m.name, m.phaseSupport),
	}
	for _, msg := range trimOpenAICompactionMessages(msgs, m.providerName) {
		items, err := converter.convert(msg)
		if err != nil {
			return nil, err
		}
		req.Input = append(req.Input, items...)
	}
	for _, tool := range params.Tools {
		if activeToolSearch && (tool.Name == ai.ToolSearchName || tool.DeferLoading) {
			continue
		}
		converted, err := prepareResponsesFunctionTool(tool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
	}
	if activeToolSearch {
		for _, tool := range params.DeferredTools {
			converted, err := prepareResponsesFunctionTool(tool, m.strictToolSupport)
			if err != nil {
				return nil, err
			}
			converted.DeferLoading = true
			req.Tools = append(req.Tools, converted)
		}
	}
	if params.OutputTool != nil {
		converted, err := prepareResponsesFunctionTool(*params.OutputTool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
		if !params.AllowText {
			req.ToolChoice = "required"
		}
	}
	if clientToolSearch {
		schema, _, err := prepareOpenAITool(*searchTool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, responsesTool{
			Type: "tool_search", Description: searchTool.Description, Parameters: schema, Execution: "client",
		})
	} else if serverToolSearch {
		req.Tools = append(req.Tools, responsesTool{Type: "tool_search"})
	}
	if len(req.Tools) > 0 {
		req.ParallelToolCalls = params.Settings.ParallelToolCalls
	}
	if params.OutputSchema != nil && params.OutputMode != ai.OutputModePrompted {
		strict := true
		schema, _, err := prepareOpenAISchema(params.OutputSchema, &strict)
		if err != nil {
			return nil, fmt.Errorf("openai: output schema: %w", err)
		}
		var strictFlag *bool
		if m.strictToolSupport {
			strictFlag = &strict
		}
		req.Text = &responsesText{Format: responsesTextFormat{
			Type: "json_schema", Name: "final_result", Schema: schema, Strict: strictFlag,
		}}
	}
	return req, nil
}

type incompleteDetails struct {
	Reason string `json:"reason"`
}

type responsesResponse struct {
	ID                string             `json:"id"`
	Model             string             `json:"model"`
	CreatedAt         float64            `json:"created_at"`
	Status            string             `json:"status"`
	Background        bool               `json:"background"`
	ServiceTier       string             `json:"service_tier"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Error             *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Output []responsesOutputItem `json:"output"`
	Usage  responsesUsage        `json:"usage"`
}

type responsesOutputItem struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Content []struct {
		Type        string           `json:"type"`
		Text        string           `json:"text"`
		Refusal     string           `json:"refusal"`
		Logprobs    []map[string]any `json:"logprobs"`
		Annotations []map[string]any `json:"annotations"`
	} `json:"content"`
	CallID           *string         `json:"call_id"`
	Name             string          `json:"name"`
	Arguments        json.RawMessage `json:"arguments"`
	Action           json.RawMessage `json:"action"`
	Namespace        string          `json:"namespace"`
	Execution        string          `json:"execution"`
	Status           string          `json:"status"`
	Phase            string          `json:"phase"`
	Tools            []responsesTool `json:"tools"`
	EncryptedContent string          `json:"encrypted_content"`
	ContainerID      string          `json:"container_id"`
	Code             string          `json:"code"`
	Background       string          `json:"background"`
	Quality          string          `json:"quality"`
	Size             string          `json:"size"`
	RevisedPrompt    string          `json:"revised_prompt"`
	OutputFormat     string          `json:"output_format"`
	Result           string          `json:"result"`
	Outputs          []struct {
		Type string `json:"type"`
		Logs string `json:"logs"`
		URL  string `json:"url"`
	} `json:"outputs"`
	Summary []struct {
		Text string `json:"text"`
	} `json:"summary"`
}

type responsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func (u responsesUsage) usage() ai.Usage {
	return ai.Usage{
		Requests: 1, InputTokens: u.InputTokens, OutputTokens: u.OutputTokens,
		CacheReadTokens: u.InputTokensDetails.CachedTokens,
		ReasoningTokens: u.OutputTokensDetails.ReasoningTokens,
		Details:         map[string]int{"reasoning_tokens": u.OutputTokensDetails.ReasoningTokens},
	}
}

func responsesPhaseSupported(modelName string, override *bool) bool {
	if override != nil {
		return *override
	}
	return strings.HasPrefix(modelName, "gpt-5.3-codex") || strings.HasPrefix(modelName, "gpt-5.4") ||
		strings.HasPrefix(modelName, "gpt-5.5") || strings.HasPrefix(modelName, "gpt-5.6")
}

func openAIResponsesFinishReason(reason string) ai.FinishReason {
	return map[string]ai.FinishReason{
		"completed": ai.FinishReasonStop, "max_output_tokens": ai.FinishReasonLength,
		"content_filter": ai.FinishReasonContentFilter, "cancelled": ai.FinishReasonError,
		"failed": ai.FinishReasonError,
	}[reason]
}

func openAIResponsesState(status string, background bool) ai.ModelResponseState {
	switch status {
	case "queued", "in_progress":
		if background {
			return ai.ModelResponseStateSuspended
		}
		return ai.ModelResponseStateIncomplete
	default:
		return ai.ModelResponseStateComplete
	}
}

func responsesMetadata(
	status string, details *incompleteDetails, createdAt float64, background bool,
) (string, map[string]any, time.Time, ai.ModelResponseState) {
	rawFinishReason := status
	if details != nil {
		rawFinishReason = details.Reason
	}
	providerDetails := map[string]any{}
	if rawFinishReason != "" {
		providerDetails["finish_reason"] = rawFinishReason
	}
	var timestamp time.Time
	if createdAt != 0 {
		seconds, fraction := math.Modf(createdAt)
		timestamp = time.Unix(int64(seconds), int64(fraction*float64(time.Second))).UTC()
		providerDetails["timestamp"] = timestamp
	}
	if background {
		providerDetails["background"] = true
	}
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	return rawFinishReason, providerDetails, timestamp, openAIResponsesState(status, background)
}

func parseResponsesResponse(data []byte) (*ai.ModelResponse, error) {
	var rr responsesResponse
	if err := json.Unmarshal(data, &rr); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}
	return modelResponseFromResponses(rr)
}

func responsesCodeExecutionParts(
	item responsesOutputItem, timestamp time.Time,
) (ai.NativeToolCallPart, []ai.FilePart, ai.NativeToolReturnPart, error) {
	args, _ := json.Marshal(struct {
		ContainerID string `json:"container_id"`
		Code        string `json:"code"`
	}{ContainerID: item.ContainerID, Code: item.Code})
	content := map[string]any{"status": item.Status}
	logs := make([]string, 0)
	files := make([]ai.FilePart, 0)
	for _, output := range item.Outputs {
		switch output.Type {
		case "logs":
			logs = append(logs, output.Logs)
		case "image":
			binary, err := responsesDataURI(output.URL)
			if err != nil {
				return ai.NativeToolCallPart{}, nil, ai.NativeToolReturnPart{}, err
			}
			files = append(files, ai.FilePart{Content: binary, ID: item.ID, ProviderName: "openai"})
		default:
			return ai.NativeToolCallPart{}, nil, ai.NativeToolReturnPart{}, fmt.Errorf(
				"openai: unknown code interpreter output type %q", output.Type,
			)
		}
	}
	if len(logs) > 0 {
		content["logs"] = logs
	}
	return ai.NativeToolCallPart{
			ToolName: "code_execution", Args: args, ToolCallID: item.ID,
			ToolKind: ai.ToolPartKindCodeExecution, ID: item.ID, ProviderName: "openai",
		}, files, ai.NativeToolReturnPart{
			ToolName: "code_execution", ToolCallID: item.ID, ToolKind: ai.ToolPartKindCodeExecution,
			Content: content, Timestamp: timestamp, ProviderName: "openai",
		}, nil
}

func responsesImageGenerationParts(
	item responsesOutputItem, timestamp time.Time,
) (ai.NativeToolCallPart, *ai.FilePart, ai.NativeToolReturnPart, error) {
	content := map[string]any{"status": item.Status}
	if item.Background != "" {
		content["background"] = item.Background
	}
	if item.Quality != "" {
		content["quality"] = item.Quality
	}
	if item.Size != "" {
		content["size"] = item.Size
	}
	if item.RevisedPrompt != "" {
		content["revised_prompt"] = item.RevisedPrompt
	}
	var file *ai.FilePart
	if item.Result != "" {
		generated, err := responsesGeneratedImage(item.ID, item.Result, item.OutputFormat)
		if err != nil {
			return ai.NativeToolCallPart{}, nil, ai.NativeToolReturnPart{}, err
		}
		file = &generated
		content["status"] = "completed"
	}
	return ai.NativeToolCallPart{
			ToolName: "image_generation", ToolCallID: item.ID, ToolKind: ai.ToolPartKindImageGeneration,
			ID: item.ID, ProviderName: "openai",
		}, file, ai.NativeToolReturnPart{
			ToolName: "image_generation", ToolCallID: item.ID, ToolKind: ai.ToolPartKindImageGeneration,
			Content: content, Timestamp: timestamp, ProviderName: "openai",
		}, nil
}

func responsesGeneratedImage(itemID, encoded, outputFormat string) (ai.FilePart, error) {
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return ai.FilePart{}, fmt.Errorf("openai: decode generated image: %w", err)
	}
	if outputFormat == "" {
		outputFormat = "png"
	}
	return ai.FilePart{
		Content: ai.BinaryContent{Data: data, MediaType: "image/" + outputFormat},
		ID:      itemID, ProviderName: "openai",
	}, nil
}

func responsesDataURI(value string) (ai.BinaryContent, error) {
	header, encoded, ok := strings.Cut(value, ",")
	if !ok || !strings.HasPrefix(header, "data:") || !strings.HasSuffix(header, ";base64") {
		return ai.BinaryContent{}, fmt.Errorf("openai: invalid code interpreter image data URI")
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return ai.BinaryContent{}, fmt.Errorf("openai: decode code interpreter image: %w", err)
	}
	return ai.BinaryContent{Data: data, MediaType: strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")}, nil
}

func modelResponseFromResponses(rr responsesResponse) (*ai.ModelResponse, error) {
	rawFinishReason, providerDetails, timestamp, state := responsesMetadata(
		rr.Status, rr.IncompleteDetails, rr.CreatedAt, rr.Background,
	)
	if rr.ServiceTier != "" {
		if providerDetails == nil {
			providerDetails = map[string]any{}
		}
		providerDetails["service_tier"] = rr.ServiceTier
	}
	resp := &ai.ModelResponse{
		ModelName: rr.Model, Usage: rr.Usage.usage(), Timestamp: timestamp, ProviderDetails: providerDetails,
		ProviderResponseID: rr.ID, FinishReason: openAIResponsesFinishReason(rawFinishReason), State: state,
	}
	searchPairs, pairedOutputs := pairResponsesToolSearchItems(rr.Output)
	refusal := ""
	hasRefusal := false
	for itemIndex, item := range rr.Output {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				switch content.Type {
				case "refusal":
					hasRefusal = true
					refusal = content.Refusal
				case "output_text":
					var details map[string]any
					if len(content.Logprobs) > 0 {
						details = map[string]any{"logprobs": content.Logprobs}
					}
					if len(content.Annotations) > 0 {
						if details == nil {
							details = map[string]any{}
						}
						details["annotations"] = content.Annotations
					}
					if item.Phase != "" {
						if details == nil {
							details = map[string]any{}
						}
						details["phase"] = item.Phase
					}
					resp.Parts = append(resp.Parts, ai.TextPart{
						Content: content.Text, ID: item.ID, ProviderName: "openai", ProviderDetails: details,
					})
				}
			}
		case "compaction":
			if item.EncryptedContent != "" {
				resp.Parts = append(resp.Parts, ai.CompactionPart{
					ID: item.ID, ProviderName: "openai",
					ProviderDetails: map[string]any{"encrypted_content": item.EncryptedContent},
				})
				if resp.ProviderDetails == nil {
					resp.ProviderDetails = map[string]any{}
				}
				resp.ProviderDetails["compaction"] = true
			}
		case "function_call":
			arguments, err := normalizeResponsesArguments(item.Arguments)
			if err != nil {
				return nil, err
			}
			var providerDetails map[string]any
			if item.Namespace != "" {
				providerDetails = map[string]any{"namespace": item.Namespace}
			}
			resp.Parts = append(resp.Parts, ai.ToolCallPart{
				ToolName: item.Name, Args: arguments, ToolCallID: responsesCallID(item.CallID),
				ID: item.ID, ProviderName: "openai", ProviderDetails: providerDetails,
			})
		case "code_interpreter_call":
			call, files, returned, err := responsesCodeExecutionParts(item, timestamp)
			if err != nil {
				return nil, err
			}
			resp.Parts = append(resp.Parts, call)
			for _, file := range files {
				resp.Parts = append(resp.Parts, file)
			}
			resp.Parts = append(resp.Parts, returned)
		case "image_generation_call":
			call, file, returned, err := responsesImageGenerationParts(item, timestamp)
			if err != nil {
				return nil, err
			}
			resp.Parts = append(resp.Parts, call)
			if file != nil {
				resp.Parts = append(resp.Parts, *file)
			}
			resp.Parts = append(resp.Parts, returned)
		case "web_search_call":
			arguments := slices.Clone(item.Action)
			if len(arguments) == 0 || string(arguments) == "null" {
				arguments = json.RawMessage(`{}`)
			}
			resp.Parts = append(resp.Parts,
				ai.NativeToolCallPart{
					ToolName: "web_search", Args: arguments, ToolCallID: item.ID, ToolKind: ai.ToolPartKindWebSearch,
					ID: item.ID, ProviderName: "openai", ProviderDetails: map[string]any{"status": item.Status},
				},
				ai.NativeToolReturnPart{
					ToolName: "web_search", ToolCallID: item.ID, ToolKind: ai.ToolPartKindWebSearch,
					Content: map[string]any{"status": item.Status}, Timestamp: timestamp, ProviderName: "openai",
				},
			)
		case "tool_search_call":
			arguments, err := normalizeResponsesToolSearchArguments(item.Arguments, item.Execution)
			if err != nil {
				return nil, err
			}
			callID := responsesEffectiveCallID(item)
			if item.Execution == "client" {
				resp.Parts = append(resp.Parts, ai.ToolCallPart{
					ToolName: ai.ToolSearchName, Args: arguments, ToolCallID: callID,
					ToolKind: ai.ToolPartKindToolSearch, ID: item.ID, ProviderName: "openai",
					ProviderDetails: map[string]any{"execution": item.Execution, "status": item.Status},
				})
				continue
			}
			if item.Execution != "server" {
				continue
			}
			resp.Parts = append(resp.Parts, ai.NativeToolCallPart{
				ToolName: ai.ToolSearchName, Args: arguments, ToolCallID: callID,
				ToolKind: ai.ToolPartKindToolSearch, ID: item.ID, ProviderName: "openai",
				ProviderDetails: map[string]any{
					"call_id": responsesNullableCallID(item.CallID), "execution": item.Execution, "status": item.Status,
				},
			})
			if outputIndex, ok := searchPairs[itemIndex]; ok {
				resp.Parts = append(resp.Parts, responsesToolSearchReturn(rr.Output[outputIndex], callID, timestamp, "openai"))
			}
		case "tool_search_output":
			if item.Execution != "server" || pairedOutputs[itemIndex] {
				continue
			}
			resp.Parts = append(resp.Parts, responsesToolSearchReturn(item, responsesEffectiveCallID(item), timestamp, "openai"))
		case "reasoning":
			if len(item.Summary) == 0 && item.EncryptedContent != "" {
				resp.Parts = append(resp.Parts, ai.ThinkingPart{
					ID: item.ID, Signature: item.EncryptedContent, ProviderName: "openai",
				})
			}
			for index, s := range item.Summary {
				signature := ""
				if index == 0 {
					signature = item.EncryptedContent
				}
				resp.Parts = append(resp.Parts, ai.ThinkingPart{
					Content: s.Text, ID: item.ID, Signature: signature, ProviderName: "openai",
				})
			}
		}
	}
	if hasRefusal {
		resp.Parts = nil
		resp.FinishReason = ai.FinishReasonContentFilter
		if resp.ProviderDetails == nil {
			resp.ProviderDetails = map[string]any{}
		}
		delete(resp.ProviderDetails, "finish_reason")
		resp.ProviderDetails["refusal"] = refusal
	}
	return resp, nil
}

func pairResponsesToolSearchItems(items []responsesOutputItem) (map[int]int, map[int]bool) {
	pairs := make(map[int]int)
	pairedOutputs := make(map[int]bool)
	outputsByCallID := make(map[string][]int)
	var nullCalls, nullOutputs []int
	for index, item := range items {
		if item.Execution != "server" {
			continue
		}
		switch item.Type {
		case "tool_search_call":
			if callID := responsesCallID(item.CallID); callID == "" {
				nullCalls = append(nullCalls, index)
			}
		case "tool_search_output":
			if callID := responsesCallID(item.CallID); callID != "" {
				outputsByCallID[callID] = append(outputsByCallID[callID], index)
			} else {
				nullOutputs = append(nullOutputs, index)
			}
		}
	}
	for index, item := range items {
		if item.Type != "tool_search_call" || item.Execution != "server" {
			continue
		}
		callID := responsesCallID(item.CallID)
		if len(outputsByCallID[callID]) == 0 || callID == "" {
			continue
		}
		outputIndex := outputsByCallID[callID][0]
		outputsByCallID[callID] = outputsByCallID[callID][1:]
		pairs[index] = outputIndex
		pairedOutputs[outputIndex] = true
	}
	if len(nullCalls) == 1 && len(nullOutputs) == 1 {
		pairs[nullCalls[0]] = nullOutputs[0]
		pairedOutputs[nullOutputs[0]] = true
	}
	return pairs, pairedOutputs
}

func responsesToolSearchReturn(
	item responsesOutputItem, callID string, timestamp time.Time, providerName string,
) ai.NativeToolReturnPart {
	matches := make([]ai.ToolSearchMatch, 0, len(item.Tools))
	for _, tool := range item.Tools {
		if tool.Type == "function" && tool.Name != "" {
			matches = append(matches, ai.ToolSearchMatch{Name: tool.Name})
		}
	}
	return ai.NativeToolReturnPart{
		ToolName: ai.ToolSearchName, ToolCallID: callID, ToolKind: ai.ToolPartKindToolSearch,
		Content: ai.ToolSearchResult{DiscoveredTools: matches}, Timestamp: timestamp, ProviderName: providerName,
		ProviderDetails: map[string]any{
			"id": item.ID, "call_id": responsesNullableCallID(item.CallID),
			"execution": item.Execution, "status": item.Status,
		},
	}
}

func responsesCallID(callID *string) string {
	if callID == nil {
		return ""
	}
	return *callID
}

func responsesNullableCallID(callID *string) any {
	if callID == nil {
		return nil
	}
	return *callID
}

func responsesEffectiveCallID(item responsesOutputItem) string {
	if callID := responsesCallID(item.CallID); callID != "" {
		return callID
	}
	return item.ID
}

func normalizeResponsesToolSearchArguments(raw json.RawMessage, execution string) (json.RawMessage, error) {
	arguments, err := normalizeResponsesArguments(raw)
	if err != nil || execution != "server" {
		return arguments, err
	}
	var value map[string]any
	if json.Unmarshal(arguments, &value) != nil {
		return arguments, nil
	}
	queries, ok := value["paths"]
	if !ok {
		queries, ok = value["queries"]
	}
	if !ok {
		queries = []string{}
	}
	return json.Marshal(map[string]any{"queries": queries})
}

func normalizeResponsesArguments(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage(`{}`), nil
	}
	if raw[0] != '"' {
		return append(json.RawMessage(nil), raw...), nil
	}
	var arguments string
	_ = json.Unmarshal(raw, &arguments)
	if !json.Valid([]byte(arguments)) {
		return nil, fmt.Errorf("openai: invalid response arguments %q", arguments)
	}
	return json.RawMessage(arguments), nil
}

// SupportsToolSearchStrategy reports the required named strategies available
// through OpenAI Responses. OpenAI exposes hosted search without a named algorithm.
func (m *ResponsesModel) SupportsToolSearchStrategy(strategy ai.ToolSearchStrategy) bool {
	return strategy != ai.ToolSearchStrategyBM25 && strategy != ai.ToolSearchStrategyRegex
}

// NativeToolSearchProvider identifies histories this model can replay natively.
func (m *ResponsesModel) NativeToolSearchProvider() string {
	if !m.deferredToolSupport {
		return ""
	}
	return m.providerName
}

var (
	_ ai.Model                        = (*ResponsesModel)(nil)
	_ ai.ToolSearchStrategyModel      = (*ResponsesModel)(nil)
	_ ai.NativeToolSearchHistoryModel = (*ResponsesModel)(nil)
	_ ai.ModelContinuationDelayer     = (*ResponsesModel)(nil)
	_ ai.SuspendedResponseCanceler    = (*ResponsesModel)(nil)
)
