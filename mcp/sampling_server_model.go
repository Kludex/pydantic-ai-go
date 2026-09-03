//nolint:staticcheck // MCP sampling remains supported during its protocol deprecation window.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

const samplingPreferencesKey = "mcp_model_preferences"

// ModelPreferences describes advisory client-side model selection priorities.
type ModelPreferences = mcpsdk.ModelPreferences

// SamplingSettings adds MCP model preferences to provider-neutral settings.
type SamplingSettings struct {
	// Common contains provider-neutral settings.
	Common ai.ModelSettings
	// ModelPreferences guides client-side model selection.
	ModelPreferences *ModelPreferences
}

// Build returns detached settings for SamplingModel.
func (settings SamplingSettings) Build() (ai.ModelSettings, error) {
	built := settings.Common.Clone()
	if settings.ModelPreferences == nil {
		return built, nil
	}
	if _, exists := built.ExtraBody[samplingPreferencesKey]; exists {
		return ai.ModelSettings{}, fmt.Errorf("ai/mcp: extra body field %q conflicts with ModelPreferences", samplingPreferencesKey)
	}
	if built.ExtraBody == nil {
		built.ExtraBody = map[string]any{}
	}
	built.ExtraBody[samplingPreferencesKey] = cloneProtocol(settings.ModelPreferences)
	return built, nil
}

// SamplingSession sends basic and tool-enabled model requests to an MCP client.
type SamplingSession interface {
	// CreateMessage sends a basic sampling request.
	CreateMessage(context.Context, *mcpsdk.CreateMessageParams) (*mcpsdk.CreateMessageResult, error)
	// CreateMessageWithTools sends a sampling request with callable tools.
	CreateMessageWithTools(
		context.Context, *mcpsdk.CreateMessageWithToolsParams,
	) (*mcpsdk.CreateMessageWithToolsResult, error)
}

// SamplingModelOption configures a SamplingModel.
type SamplingModelOption func(*SamplingModel)

// WithSamplingDefaultMaxTokens sets the required MCP fallback when MaxTokens is zero.
func WithSamplingDefaultMaxTokens(maxTokens int) SamplingModelOption {
	if maxTokens <= 0 {
		panic("ai/mcp: sampling default max tokens must be positive")
	}
	return func(model *SamplingModel) { model.defaultMaxTokens = maxTokens }
}

// SamplingModel calls the model exposed by an MCP client to one server session.
type SamplingModel struct {
	session          SamplingSession
	defaultMaxTokens int
}

// NewSamplingModel creates a model backed by client sampling on session.
func NewSamplingModel(session SamplingSession, options ...SamplingModelOption) *SamplingModel {
	if samplingSessionIsNil(session) {
		panic("ai/mcp: sampling server session must not be nil")
	}
	model := &SamplingModel{session: session, defaultMaxTokens: 16_384}
	for _, option := range options {
		if option == nil {
			panic("ai/mcp: sampling model option must not be nil")
		}
		option(model)
	}
	return model
}

// Name returns the stable placeholder used before the MCP client selects a model.
func (*SamplingModel) Name() string { return "mcp-sampling" }

// ProviderName identifies the MCP sampling transport.
func (*SamplingModel) ProviderName() string { return "mcp" }

// ProviderURL is empty because an MCP session does not expose a stable model endpoint.
func (*SamplingModel) ProviderURL() string { return "" }

// Request sends one sampling request to the connected MCP client.
func (model *SamplingModel) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	if model == nil || model.session == nil {
		return nil, errors.New("ai/mcp: sampling model is not connected")
	}
	maxTokens := params.Settings.MaxTokens
	if maxTokens < 0 {
		return nil, fmt.Errorf("ai/mcp: sampling max tokens must not be negative, got %d", maxTokens)
	}
	if maxTokens == 0 {
		maxTokens = model.defaultMaxTokens
	}
	preferences, err := samplingModelPreferences(params.Settings)
	if err != nil {
		return nil, err
	}
	if samplingNeedsTools(messages, params) {
		request, err := samplingRequestWithTools(messages, params, maxTokens, preferences)
		if err != nil {
			return nil, err
		}
		result, err := model.session.CreateMessageWithTools(ctx, request)
		if err != nil {
			return nil, ai.NewModelTransportError(ctx, model, "sampling request with tools", err)
		}
		if result == nil {
			return nil, &ai.UnexpectedModelBehaviorError{Message: "MCP tool-enabled sampling returned no result"}
		}
		return samplingModelResponse(result.Role, result.Content, result.Model, result.StopReason, result.Meta)
	}
	request, err := samplingRequest(messages, params, maxTokens, preferences)
	if err != nil {
		return nil, err
	}
	result, err := model.session.CreateMessage(ctx, request)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "sampling request", err)
	}
	if result == nil {
		return nil, &ai.UnexpectedModelBehaviorError{Message: "MCP sampling returned no result"}
	}
	return samplingModelResponse(result.Role, []mcpsdk.Content{result.Content}, result.Model, result.StopReason, result.Meta)
}

func samplingSessionIsNil(session SamplingSession) bool {
	if session == nil {
		return true
	}
	value := reflect.ValueOf(session)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func samplingModelPreferences(settings ai.ModelSettings) (*ModelPreferences, error) {
	value, ok := settings.ExtraBody[samplingPreferencesKey]
	if !ok {
		return nil, nil
	}
	preferences, ok := value.(*ModelPreferences)
	if !ok || preferences == nil {
		return nil, fmt.Errorf("ai/mcp: %s must be a non-nil *mcp.ModelPreferences", samplingPreferencesKey)
	}
	return cloneProtocol(preferences), nil
}

func samplingRequest(
	messages []ai.ModelMessage, params ai.ModelRequestParams, maxTokens int, preferences *ModelPreferences,
) (*mcpsdk.CreateMessageParams, error) {
	systemPrompt, samplingMessages, err := samplingModelMessages(messages, params.Instructions)
	if err != nil {
		return nil, err
	}
	request := &mcpsdk.CreateMessageParams{
		SystemPrompt: systemPrompt, Messages: samplingMessages, MaxTokens: int64(maxTokens),
		ModelPreferences: preferences, StopSequences: append([]string(nil), params.Settings.StopSequences...),
	}
	if params.Settings.Temperature != nil {
		request.Temperature = *params.Settings.Temperature
	}
	return request, nil
}

func samplingRequestWithTools(
	messages []ai.ModelMessage, params ai.ModelRequestParams, maxTokens int, preferences *ModelPreferences,
) (*mcpsdk.CreateMessageWithToolsParams, error) {
	systemPrompt, samplingMessages, err := samplingModelMessagesV2(messages, params.Instructions)
	if err != nil {
		return nil, err
	}
	request := &mcpsdk.CreateMessageWithToolsParams{
		SystemPrompt: systemPrompt, Messages: samplingMessages, MaxTokens: int64(maxTokens),
		ModelPreferences: preferences, StopSequences: append([]string(nil), params.Settings.StopSequences...),
		ToolChoice: &mcpsdk.ToolChoice{Mode: "auto"},
	}
	if params.Settings.Temperature != nil {
		request.Temperature = *params.Settings.Temperature
	}
	definitions := append([]ai.ToolDefinition(nil), params.Tools...)
	if params.OutputTool != nil {
		definitions = append(definitions, *params.OutputTool)
		if !params.AllowText {
			request.ToolChoice.Mode = "required"
		}
	}
	for _, definition := range definitions {
		tool := &mcpsdk.Tool{
			Name: definition.Name, Description: definition.Description, InputSchema: cloneJSON(definition.Schema),
		}
		if definition.IncludeReturnSchema != nil && *definition.IncludeReturnSchema {
			tool.OutputSchema = cloneJSON(definition.ReturnSchema)
		}
		request.Tools = append(request.Tools, tool)
	}
	return request, nil
}

func samplingNeedsTools(messages []ai.ModelMessage, params ai.ModelRequestParams) bool {
	if len(params.Tools) > 0 || params.OutputTool != nil {
		return true
	}
	for _, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				switch part.(type) {
				case ai.ToolReturnPart, ai.RetryPromptPart:
					return true
				}
			}
		case ai.ModelResponse:
			if len(message.ToolCalls()) > 0 {
				return true
			}
		}
	}
	return false
}

func samplingModelMessages(
	messages []ai.ModelMessage, instructions string,
) (string, []*mcpsdk.SamplingMessage, error) {
	systemPrompt, messagesV2, err := samplingModelMessagesV2(messages, instructions)
	if err != nil {
		return "", nil, err
	}
	var converted []*mcpsdk.SamplingMessage
	for _, message := range messagesV2 {
		for _, content := range message.Content {
			converted = append(converted, &mcpsdk.SamplingMessage{Role: message.Role, Content: content})
		}
	}
	return systemPrompt, converted, nil
}

func samplingModelMessagesV2(
	messages []ai.ModelMessage, instructions string,
) (string, []*mcpsdk.SamplingMessageV2, error) {
	var systemPrompts []string
	if instructions != "" {
		systemPrompts = append(systemPrompts, instructions)
	}
	var converted []*mcpsdk.SamplingMessageV2
	for _, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			var content []mcpsdk.Content
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.SystemPromptPart:
					systemPrompts = append(systemPrompts, part.Content)
				case ai.UserPromptPart:
					parts, err := samplingUserContents(part)
					if err != nil {
						return "", nil, err
					}
					content = append(content, parts...)
				case ai.ToolReturnPart:
					result, err := samplingToolResult(part)
					if err != nil {
						return "", nil, err
					}
					content = append(content, result)
				case ai.RetryPromptPart:
					if part.ToolCallID == "" {
						content = append(content, &mcpsdk.TextContent{Text: part.Content})
					} else {
						content = append(content, &mcpsdk.ToolResultContent{
							ToolUseID: part.ToolCallID, Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: part.Content}},
							IsError: true,
						})
					}
				case ai.ToolAvailabilityDeltaPart:
				default:
					return "", nil, fmt.Errorf("ai/mcp: unsupported sampling request part %T", part)
				}
			}
			if len(content) > 0 {
				converted = append(converted, &mcpsdk.SamplingMessageV2{Role: mcpsdk.Role("user"), Content: content})
			}
		case ai.ModelResponse:
			var content []mcpsdk.Content
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.TextPart:
					content = append(content, &mcpsdk.TextContent{Text: part.Content})
				case ai.ThinkingPart, ai.CompactionPart:
				case ai.FilePart:
					file, err := samplingBinaryContent(part.Content)
					if err != nil {
						return "", nil, err
					}
					content = append(content, file)
				case ai.ToolCallPart:
					var input map[string]any
					if err := json.Unmarshal(part.Args, &input); err != nil {
						return "", nil, fmt.Errorf("ai/mcp: decode tool call %q arguments: %w", part.ToolName, err)
					}
					content = append(content, &mcpsdk.ToolUseContent{ID: part.ToolCallID, Name: part.ToolName, Input: input})
				default:
					return "", nil, fmt.Errorf("ai/mcp: unsupported sampling response part %T", part)
				}
			}
			if len(content) > 0 {
				converted = append(converted, &mcpsdk.SamplingMessageV2{Role: mcpsdk.Role("assistant"), Content: content})
			}
		}
	}
	return strings.Join(systemPrompts, "\n\n"), converted, nil
}

func samplingUserContents(part ai.UserPromptPart) ([]mcpsdk.Content, error) {
	if part.Content != "" {
		return []mcpsdk.Content{&mcpsdk.TextContent{Text: part.Content}}, nil
	}
	content := make([]mcpsdk.Content, 0, len(part.Contents))
	for _, item := range part.Contents {
		switch item := item.(type) {
		case ai.TextContent:
			content = append(content, &mcpsdk.TextContent{Text: item.Text})
		case ai.BinaryContent:
			mapped, err := samplingBinaryContent(item)
			if err != nil {
				return nil, err
			}
			content = append(content, mapped)
		case ai.CachePoint:
		default:
			return nil, fmt.Errorf("ai/mcp: unsupported sampling user content %T", item)
		}
	}
	return content, nil
}

func samplingBinaryContent(content ai.BinaryContent) (mcpsdk.Content, error) {
	data := append([]byte(nil), content.Data...)
	switch {
	case strings.HasPrefix(content.MediaType, "image/"):
		return &mcpsdk.ImageContent{Data: data, MIMEType: content.MediaType}, nil
	case strings.HasPrefix(content.MediaType, "audio/"):
		return &mcpsdk.AudioContent{Data: data, MIMEType: content.MediaType}, nil
	default:
		return nil, fmt.Errorf("ai/mcp: unsupported sampling binary media type %q", content.MediaType)
	}
}

func samplingToolResult(part ai.ToolReturnPart) (*mcpsdk.ToolResultContent, error) {
	result := &mcpsdk.ToolResultContent{
		ToolUseID: part.ToolCallID, IsError: part.Outcome != "" && part.Outcome != ai.ToolReturnOutcomeSuccess,
	}
	switch value := part.Content.(type) {
	case string:
		result.Content = []mcpsdk.Content{&mcpsdk.TextContent{Text: value}}
	case ai.BinaryContent:
		content, err := samplingBinaryContent(value)
		if err != nil {
			return nil, err
		}
		result.Content = []mcpsdk.Content{content}
	default:
		structured, err := cloneJSONValue(value)
		if err != nil {
			return nil, fmt.Errorf("ai/mcp: encode tool result %q: %w", part.ToolCallID, err)
		}
		result.StructuredContent = structured
	}
	return result, nil
}

func samplingModelResponse(
	role mcpsdk.Role, content []mcpsdk.Content, modelName, stopReason string, metadata mcpsdk.Meta,
) (*ai.ModelResponse, error) {
	if role != mcpsdk.Role("assistant") {
		return nil, &ai.UnexpectedModelBehaviorError{
			Message: fmt.Sprintf("MCP sampling returned role %q instead of assistant", role),
		}
	}
	response := &ai.ModelResponse{
		ModelName: modelName, ProviderName: "mcp", FinishReason: samplingFinishReason(stopReason),
		ProviderDetails: map[string]any{"mcp_stop_reason": stopReason, "mcp_meta": cloneJSON(metadata)},
	}
	for _, item := range content {
		switch item := item.(type) {
		case *mcpsdk.TextContent:
			response.Parts = append(response.Parts, ai.TextPart{Content: item.Text, ProviderName: "mcp"})
		case *mcpsdk.ImageContent:
			response.Parts = append(response.Parts, ai.FilePart{
				Content:      ai.BinaryContent{Data: append([]byte(nil), item.Data...), MediaType: item.MIMEType},
				ProviderName: "mcp",
			})
		case *mcpsdk.AudioContent:
			response.Parts = append(response.Parts, ai.FilePart{
				Content:      ai.BinaryContent{Data: append([]byte(nil), item.Data...), MediaType: item.MIMEType},
				ProviderName: "mcp",
			})
		case *mcpsdk.ToolUseContent:
			arguments, err := json.Marshal(item.Input)
			if err != nil {
				return nil, fmt.Errorf("ai/mcp: encode sampled tool call %q arguments: %w", item.Name, err)
			}
			response.Parts = append(response.Parts, ai.ToolCallPart{
				ToolName: item.Name, ToolCallID: item.ID, Args: arguments, ProviderName: "mcp",
			})
		default:
			return nil, &ai.UnexpectedModelBehaviorError{
				Message: fmt.Sprintf("MCP sampling returned unsupported content %T", item),
			}
		}
	}
	return response, nil
}

func samplingFinishReason(reason string) ai.FinishReason {
	switch reason {
	case "endTurn", "stopSequence":
		return ai.FinishReasonStop
	case "maxTokens":
		return ai.FinishReasonLength
	case "toolUse":
		return ai.FinishReasonToolCall
	default:
		return ""
	}
}

var _ ai.Model = (*SamplingModel)(nil)
var _ ai.ModelProviderIdentity = (*SamplingModel)(nil)
