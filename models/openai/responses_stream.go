package openai

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

	ai "github.com/Kludex/pydantic-ai-go"
)

// StreamRequest implements ai.StreamingModel using the Responses SSE API.
func (m *ResponsesModel) StreamRequest(
	ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	payload, err := m.buildResponsesPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	payload.Stream = true
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		data, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("openai: read error response: %w", err)
		}
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	return m.responsesEventStream(resp.Body), nil
}

type responsesStreamEvent struct {
	Type         string `json:"type"`
	Delta        string `json:"delta"`
	ItemID       string `json:"item_id"`
	OutputIndex  int    `json:"output_index"`
	ContentIndex int    `json:"content_index"`
	SummaryIndex int    `json:"summary_index"`
	Item         struct {
		ID        string `json:"id"`
		Type      string `json:"type"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`
	Part struct {
		Text string `json:"text"`
	} `json:"part"`
	Response struct {
		Model  string `json:"model"`
		Status string `json:"status"`
		Error  *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	} `json:"response"`
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (m *ResponsesModel) responsesEventStream(body io.ReadCloser) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			data, ok := strings.CutPrefix(scanner.Text(), "data:")
			if !ok {
				continue
			}
			data = strings.TrimSpace(data)
			if data == "[DONE]" {
				continue
			}
			var event responsesStreamEvent
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				yield(nil, fmt.Errorf("openai: parse Responses stream event: %w", err))
				return
			}
			switch event.Type {
			case "response.output_text.delta":
				partID := fmt.Sprintf("output:%d:content:%d:text", event.OutputIndex, event.ContentIndex)
				if !yield(ai.TextDeltaEvent{PartID: partID, Delta: event.Delta}, nil) {
					return
				}
			case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
				partID := responsesThinkingPartID(event)
				if !yield(ai.ThinkingDeltaEvent{PartID: partID, Delta: event.Delta}, nil) {
					return
				}
			case "response.reasoning_summary_part.added":
				partID := responsesThinkingPartID(event)
				if event.Part.Text != "" && !yield(ai.ThinkingDeltaEvent{PartID: partID, Delta: event.Part.Text}, nil) {
					return
				}
			case "response.output_item.added":
				if event.Item.Type == "function_call" {
					partID := responsesToolPartID(event)
					if !yield(ai.ToolCallStartEvent{
						PartID: partID, ToolName: event.Item.Name, ToolCallID: event.Item.CallID,
					}, nil) {
						return
					}
					if event.Item.Arguments != "" && !yield(ai.ToolCallDeltaEvent{
						PartID: partID, ArgsDelta: event.Item.Arguments,
					}, nil) {
						return
					}
				}
			case "response.function_call_arguments.delta":
				if !yield(ai.ToolCallDeltaEvent{PartID: responsesToolPartID(event), ArgsDelta: event.Delta}, nil) {
					return
				}
			case "response.completed":
				modelName := event.Response.Model
				if modelName == "" {
					modelName = m.name
				}
				usage := ai.Usage{
					Requests: 1, InputTokens: event.Response.Usage.InputTokens,
					OutputTokens: event.Response.Usage.OutputTokens,
				}
				yield(ai.FinishEvent{Usage: usage, ModelName: modelName}, nil)
				return
			case "response.failed", "response.incomplete":
				message := event.Response.Status
				if event.Response.Error != nil {
					message = event.Response.Error.Code + ": " + event.Response.Error.Message
				}
				yield(nil, fmt.Errorf("openai: Responses stream %s: %s", event.Type, message))
				return
			case "error":
				yield(nil, fmt.Errorf("openai: Responses stream error %s: %s", event.Error.Code, event.Error.Message))
				return
			case "response.created", "response.in_progress", "response.queued",
				"response.output_item.done", "response.content_part.added", "response.content_part.done",
				"response.output_text.done", "response.function_call_arguments.done",
				"response.reasoning_summary_part.done", "response.reasoning_summary_text.done",
				"response.reasoning_text.done":
			default:
				yield(nil, fmt.Errorf("openai: unknown Responses stream event type %q", event.Type))
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("openai: read Responses stream: %w", err))
			return
		}
		yield(nil, fmt.Errorf("openai: Responses stream ended without response.completed"))
	}
}

func responsesToolPartID(event responsesStreamEvent) string {
	if event.ItemID != "" {
		return "item:" + event.ItemID
	}
	if event.Item.ID != "" {
		return "item:" + event.Item.ID
	}
	return fmt.Sprintf("output:%d:tool", event.OutputIndex)
}

func responsesThinkingPartID(event responsesStreamEvent) string {
	if event.ItemID != "" {
		return fmt.Sprintf("item:%s:thinking:%d", event.ItemID, event.SummaryIndex)
	}
	return fmt.Sprintf("output:%d:thinking:%d", event.OutputIndex, event.SummaryIndex)
}

var _ ai.StreamingModel = (*ResponsesModel)(nil)
