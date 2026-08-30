package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strconv"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

// StreamRequest implements ai.StreamingModel using Anthropic server-sent events.
func (m *Model) StreamRequest(
	ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	payload, err := m.buildPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	payload.Stream = true
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", m.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		data, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("anthropic: read error response: %w", err)
		}
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	return m.eventStream(resp.Body), nil
}

type streamEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message struct {
		Model string      `json:"model"`
		Usage streamUsage `json:"usage"`
	} `json:"message"`
	ContentBlock struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		Thinking string          `json:"thinking"`
		ID       string          `json:"id"`
		Name     string          `json:"name"`
		Input    json.RawMessage `json:"input"`
	} `json:"content_block"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
	Usage streamUsage `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type streamUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func (m *Model) eventStream(body io.ReadCloser) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		usage := ai.Usage{Requests: 1}
		modelName := m.name
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			data, ok := strings.CutPrefix(scanner.Text(), "data:")
			if !ok {
				continue
			}
			data = strings.TrimSpace(data)
			var event streamEvent
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				yield(nil, fmt.Errorf("anthropic: parse stream event: %w", err))
				return
			}
			switch event.Type {
			case "message_start":
				if event.Message.Model != "" {
					modelName = event.Message.Model
				}
				usage.InputTokens = event.Message.Usage.InputTokens
				usage.OutputTokens = event.Message.Usage.OutputTokens
			case "content_block_start":
				if !m.emitContentBlockStart(yield, event) {
					return
				}
			case "content_block_delta":
				if !emitContentBlockDelta(yield, event) {
					return
				}
			case "message_delta":
				usage.OutputTokens = event.Usage.OutputTokens
			case "message_stop":
				yield(ai.FinishEvent{Usage: usage, ModelName: modelName}, nil)
				return
			case "error":
				yield(nil, fmt.Errorf("anthropic: stream error %s: %s", event.Error.Type, event.Error.Message))
				return
			case "content_block_stop", "ping":
			default:
				yield(nil, fmt.Errorf("anthropic: unknown stream event type %q", event.Type))
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("anthropic: read stream: %w", err))
			return
		}
		yield(nil, fmt.Errorf("anthropic: stream ended without message_stop"))
	}
}

func (m *Model) emitContentBlockStart(yield func(ai.ModelStreamEvent, error) bool, event streamEvent) bool {
	partID := strconv.Itoa(event.Index)
	switch event.ContentBlock.Type {
	case "text":
		return event.ContentBlock.Text == "" || yield(ai.TextDeltaEvent{PartID: partID, Delta: event.ContentBlock.Text}, nil)
	case "thinking":
		return event.ContentBlock.Thinking == "" || yield(ai.ThinkingDeltaEvent{
			PartID: partID, Delta: event.ContentBlock.Thinking,
		}, nil)
	case "tool_use":
		if !yield(ai.ToolCallStartEvent{
			PartID: partID, ToolName: event.ContentBlock.Name, ToolCallID: event.ContentBlock.ID,
		}, nil) {
			return false
		}
		input := strings.TrimSpace(string(event.ContentBlock.Input))
		return input == "" || input == "{}" || yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: input}, nil)
	default:
		return yield(nil, fmt.Errorf("anthropic: unsupported content block type %q", event.ContentBlock.Type))
	}
}

func emitContentBlockDelta(yield func(ai.ModelStreamEvent, error) bool, event streamEvent) bool {
	partID := strconv.Itoa(event.Index)
	switch event.Delta.Type {
	case "text_delta":
		return yield(ai.TextDeltaEvent{PartID: partID, Delta: event.Delta.Text}, nil)
	case "thinking_delta":
		return yield(ai.ThinkingDeltaEvent{PartID: partID, Delta: event.Delta.Thinking}, nil)
	case "input_json_delta":
		return yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: event.Delta.PartialJSON}, nil)
	case "signature_delta", "citations_delta":
		return true
	default:
		return yield(nil, fmt.Errorf("anthropic: unsupported content block delta type %q", event.Delta.Type))
	}
}
