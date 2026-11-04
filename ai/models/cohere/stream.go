package cohere

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// StreamRequest implements ai.StreamingModel using Cohere's v2 streaming chat endpoint.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	ai.WarnUnsupportedSettings(ctx, model, params, ai.SettingParallelToolCalls)
	payload, err := model.buildRequest(messages, params)
	if err != nil {
		return nil, err
	}
	payload.Stream = true
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
	if err != nil { // pragma: no cover - only fires when ExtraBody collides with the typed schema.
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, model.baseURL+"/v2/chat", bytes.NewReader(body))
	if err != nil { // pragma: no cover - validated at construction.
		return nil, err
	}
	request.Header = model.headers.Clone()
	if request.Header == nil {
		request.Header = make(http.Header)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, err
		}
	}
	// pragma: no cover - tests do not exercise model-side ExtraHeaders.
	for key, value := range params.Settings.ExtraHeaders {
		request.Header.Set(key, value)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "request", err)
	}
	if response.StatusCode != http.StatusOK {
		data, readErr := io.ReadAll(response.Body)
		_ = response.Body.Close()
		if readErr != nil { // pragma: no cover - server response body read fails before status code is decoded.
			return nil, ai.NewModelTransportError(ctx, model, "read error response", readErr)
		}
		return nil, &APIError{
			StatusCode: response.StatusCode, Body: string(data), Headers: response.Header.Clone(),
			ProviderName: model.providerName,
		}
	}
	return model.eventStream(ctx, response.Body), nil
}

// streamEvent matches the wire format of Cohere's v2 stream responses.
//
// Cohere sends one JSON object per `data:` line tagged with a `type` field.
// Only the fields this implementation reads are decoded; everything else falls
// through into ProviderDetails when a part settles.
type streamEvent struct {
	Type    string       `json:"type"`
	ID      string       `json:"id,omitempty"`
	Index   int          `json:"index,omitempty"`
	Delta   *streamDelta `json:"delta,omitempty"`
	Message *struct {
		ID           string       `json:"id,omitempty"`
		FinishReason string       `json:"finish_reason,omitempty"`
		Usage        *cohereUsage `json:"usage,omitempty"`
	} `json:"message,omitempty"`
	ContentBlock *struct {
		Type string `json:"type,omitempty"`
		ID   string `json:"id,omitempty"`
	} `json:"content_block,omitempty"`
	ToolCall *struct {
		ID    string `json:"id,omitempty"`
		Type  string `json:"type,omitempty"`
		Index int    `json:"index,omitempty"`
		Delta *struct {
			Message *struct {
				ToolCalls *struct {
					ID       string `json:"id,omitempty"`
					Function *struct {
						Name      string `json:"name,omitempty"`
						Arguments string `json:"arguments,omitempty"`
					} `json:"function,omitempty"`
				} `json:"tool_calls,omitempty"`
			} `json:"message,omitempty"`
		} `json:"delta,omitempty"`
	} `json:"tool_call,omitempty"`
	Error *struct {
		Message string `json:"message,omitempty"`
		Code    string `json:"code,omitempty"`
	} `json:"error,omitempty"`
}

type streamDelta struct {
	Message *struct {
		Content *struct {
			Text     string `json:"text,omitempty"`
			Thinking string `json:"thinking,omitempty"`
		} `json:"content,omitempty"`
		ToolCalls *struct {
			ID       string `json:"id,omitempty"`
			Type     string `json:"type,omitempty"`
			Index    int    `json:"index,omitempty"`
			Function *struct {
				Name      string `json:"name,omitempty"`
				Arguments string `json:"arguments,omitempty"`
			} `json:"function,omitempty"`
		} `json:"tool_calls,omitempty"`
	} `json:"message,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
}

func (model *Model) eventStream(
	ctx context.Context, body io.ReadCloser,
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		usage := ai.Usage{Requests: 1}
		modelName := model.name
		responseID := ""
		timestamp := time.Now().UTC()
		finishReason := ""
		startedTools := map[string]bool{}
		toolArguments := map[string]string{}
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			data, ok := strings.CutPrefix(scanner.Text(), "data:") // pragma: no cover - server emits no non-data framing lines.
			if !ok {
				continue
			}
			data = strings.TrimSpace(data)
			if data == "" {
				continue
			}
			if data == "[DONE]" {
				if !yield(ai.FinishEvent{
					Usage: usage, ModelName: modelName, Timestamp: timestamp,
					ProviderName: model.providerName, ProviderURL: model.baseURL,
					ProviderResponseID: responseID,
					ProviderDetails:    map[string]any{"finish_reason": finishReason},
					FinishReason:       cohereFinishReason(finishReason),
					State:              ai.ModelResponseStateComplete,
				}, nil) {
					return
				}
				continue
			}
			var event streamEvent
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				yield(nil, fmt.Errorf("cohere: parse stream chunk: %w", err))
				return
			}
			if event.Error != nil && event.Error.Message != "" {
				yield(nil, fmt.Errorf("cohere: stream reported error: %s", event.Error.Message))
				return
			}
			switch event.Type {
			case "message-start":
				if event.Message != nil && event.Message.ID != "" {
					responseID = event.Message.ID
				}
			case "message-end":
				if event.Message != nil {
					if event.Message.FinishReason != "" {
						finishReason = event.Message.FinishReason
					}
					if event.Message.Usage != nil {
						chunkUsage := event.Message.Usage.normalized()
						chunkUsage.Requests = 0
						usage.Add(chunkUsage)
					}
				}
			case "content-delta":
				if event.Delta == nil || event.Delta.Message == nil || event.Delta.Message.Content == nil {
					continue
				}
				content := event.Delta.Message.Content
				partType := "text"
				if event.ContentBlock != nil && event.ContentBlock.Type != "" {
					partType = event.ContentBlock.Type
				}
				if partType == "thinking" {
					if content.Thinking != "" && !yield(ai.ThinkingDeltaEvent{
						PartID: thinkingPartID, Delta: content.Thinking,
						ProviderName: model.providerName,
					}, nil) {
						return
					}
				} else if content.Text != "" && !yield(ai.TextDeltaEvent{
					PartID: textPartID, Delta: content.Text,
					ProviderName: model.providerName,
				}, nil) {
					return
				}
			case "tool-call-start":
				if event.ToolCall == nil {
					continue
				}
				partID := toolPartID(event.ToolCall.Index)
				toolName := ""
				var args string
				if event.ToolCall.Delta != nil && event.ToolCall.Delta.Message != nil &&
					event.ToolCall.Delta.Message.ToolCalls != nil &&
					event.ToolCall.Delta.Message.ToolCalls.Function != nil {
					toolName = event.ToolCall.Delta.Message.ToolCalls.Function.Name
					args = event.ToolCall.Delta.Message.ToolCalls.Function.Arguments
				}
				if startedTools[partID] {
					if args != "" {
						toolArguments[partID] += args
					}
					continue
				}
				startedTools[partID] = true
				callID := event.ToolCall.ID
				if callID == "" {
					callID = syntheticCallID(toolName, args)
				}
				toolArguments[partID] = args
				if !yield(ai.ToolCallStartEvent{
					PartID: partID, ToolName: toolName, ToolCallID: callID,
					ProviderName: model.providerName,
				}, nil) {
					return
				}
			case "tool-call-delta":
				if event.Delta == nil || event.Delta.Message == nil ||
					event.Delta.Message.ToolCalls == nil ||
					event.Delta.Message.ToolCalls.Function == nil {
					continue
				}
				toolCall := event.Delta.Message.ToolCalls
				partID := toolPartID(toolCall.Index)
				arguments := toolCall.Function.Arguments
				if !startedTools[partID] {
					startedTools[partID] = true
					callID := toolCall.ID
					if callID == "" {
						callID = syntheticCallID(toolCall.Function.Name, arguments)
					}
					toolArguments[partID] = arguments
					if !yield(ai.ToolCallStartEvent{
						PartID: partID, ToolName: toolCall.Function.Name,
						ToolCallID: callID, ProviderName: model.providerName,
					}, nil) {
						return
					}
					continue
				}
				if arguments == "" {
					continue
				}
				toolArguments[partID] += arguments
				if !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: arguments}, nil) {
					return
				}
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, ai.NewModelTransportError(ctx, model, "read stream", err))
		}
	}
}

const (
	textPartID     = "text"
	thinkingPartID = "thinking"
)

func toolPartID(index int) string {
	if index == 0 {
		return "tool:0"
	}
	return fmt.Sprintf("tool:%d", index)
}
