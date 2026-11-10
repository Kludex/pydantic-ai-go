package mistral

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

// StreamRequest implements ai.StreamingModel using server-sent events.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	payload, err := model.buildRequest(ctx, messages, params, true)
	if err != nil {
		return nil, err
	}
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
	if err != nil {
		return nil, err
	}
	request, err := model.newRequest(ctx, body, params.Settings.ExtraHeaders, "text/event-stream")
	if err != nil {
		return nil, err
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "request", err)
	}
	if response.StatusCode != http.StatusOK {
		data, readErr := io.ReadAll(response.Body)
		_ = response.Body.Close()
		if readErr != nil {
			return nil, ai.NewModelTransportError(ctx, model, "read error response", readErr)
		}
		return nil, &APIError{
			StatusCode: response.StatusCode, Body: string(data), Headers: response.Header.Clone(),
			ProviderName: model.providerName,
		}
	}
	return model.eventStream(ctx, response.Body), nil
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
		providerTimestamp := false
		finishReason := ""
		toolCalls := map[int]*streamedToolCall{}
		var order []int
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			data, ok := strings.CutPrefix(scanner.Text(), "data:")
			if !ok {
				continue
			}
			data = strings.TrimSpace(data)
			if data == "[DONE]" {
				for _, index := range order {
					state := toolCalls[index]
					partID := fmt.Sprintf("tool:%d", index)
					if !state.started {
						if state.id == "" {
							state.id = generatedToolCallID(state.name, json.RawMessage(state.arguments))
						}
						if !yield(ai.ToolCallStartEvent{
							PartID: partID, ToolName: state.name, ToolCallID: state.id, ProviderName: model.providerName,
						}, nil) {
							return
						}
						if state.arguments != "" && !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: state.arguments}, nil) {
							return
						}
					}
					if state.arguments == "" && !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: "{}"}, nil) {
						return
					}
				}
				details := map[string]any{"finish_reason": finishReason}
				if providerTimestamp {
					details["timestamp"] = timestamp
				}
				yield(ai.FinishEvent{
					Usage: usage, ModelName: modelName, Timestamp: timestamp,
					ProviderName: model.providerName, ProviderURL: model.baseURL,
					ProviderResponseID: responseID, ProviderDetails: details,
					FinishReason: mistralFinishReason(finishReason), State: ai.ModelResponseStateComplete,
				}, nil)
				return
			}
			var chunk chatResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				yield(nil, ai.NewModelTransportError(ctx, model, "parse stream chunk", err))
				return
			}
			if chunk.ID != "" {
				responseID = chunk.ID
			}
			if chunk.Model != "" {
				modelName = chunk.Model
			}
			if chunk.Created != 0 {
				timestamp = time.Unix(chunk.Created, 0).UTC()
				providerTimestamp = true
			}
			chunkUsage := chunk.Usage.normalized()
			chunkUsage.Requests = 0
			usage.Add(chunkUsage)
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
			text, thinking, err := parseContent(choice.Delta.Content)
			if err != nil {
				yield(nil, fmt.Errorf("mistral: decode stream content: %w", err))
				return
			}
			for _, thought := range thinking {
				if !yield(ai.ThinkingDeltaEvent{
					PartID: "thinking", Delta: thought, ProviderName: model.providerName,
				}, nil) {
					return
				}
			}
			if text != "" && !yield(ai.TextDeltaEvent{
				PartID: "text", Delta: text, ProviderName: model.providerName,
			}, nil) {
				return
			}
			for position, call := range choice.Delta.ToolCalls {
				index := position
				if call.Index != nil {
					index = *call.Index
				}
				if call.Type != "" && call.Type != "function" {
					yield(nil, fmt.Errorf("mistral: unsupported stream tool call type %q", call.Type))
					return
				}
				state := toolCalls[index]
				if state == nil {
					state = &streamedToolCall{}
					toolCalls[index] = state
					order = append(order, index)
				}
				if state.name == "" {
					state.name = call.Function.Name
				}
				if state.id == "" && call.ID != "null" {
					state.id = call.ID
				}
				// Mistral sends whole snapshots or fragments; a snapshot repeats what came before.
				fragment := ""
				if raw := bytes.TrimSpace(call.Function.Arguments); len(raw) != 0 && !bytes.Equal(raw, []byte("null")) {
					fragment = string(normalizeArguments(raw))
				}
				fragment = strings.TrimPrefix(fragment, state.arguments)
				state.arguments += fragment
				if !state.started && state.name == "" {
					continue
				}
				if !state.started {
					state.started = true
					if state.id == "" {
						state.id = generatedToolCallID(state.name, json.RawMessage(state.arguments))
					}
					if !yield(ai.ToolCallStartEvent{
						PartID: fmt.Sprintf("tool:%d", index), ToolName: state.name, ToolCallID: state.id,
						ProviderName: model.providerName,
					}, nil) {
						return
					}
					fragment = state.arguments
				}
				if fragment != "" && !yield(ai.ToolCallDeltaEvent{PartID: fmt.Sprintf("tool:%d", index), ArgsDelta: fragment}, nil) {
					return
				}
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, ai.NewModelTransportError(ctx, model, "read stream", err))
			return
		}
		yield(nil, fmt.Errorf("mistral: stream ended without [DONE]"))
	}
}

type streamedToolCall struct {
	name, id, arguments string
	started             bool
}
