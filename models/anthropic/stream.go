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
	"slices"
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
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	setExtraHeaders(req, params.Settings.ExtraHeaders)
	m.setRequestHeaders(req, payload, true)

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
		ID    string         `json:"id"`
		Model string         `json:"model"`
		Usage anthropicUsage `json:"usage"`
	} `json:"message"`
	ContentBlock responseContentBlock `json:"content_block"`
	Delta        struct {
		Type        string `json:"type"`
		StopReason  string `json:"stop_reason"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		PartialJSON string `json:"partial_json"`
		Signature   string `json:"signature"`
	} `json:"delta"`
	Usage anthropicUsage `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func (m *Model) eventStream(body io.ReadCloser) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		usage := ai.Usage{Requests: 1}
		modelName := m.name
		responseID := ""
		stopReason := ""
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		searchCalls := make(map[int]responseContentBlock)
		searchArgs := make(map[int]*strings.Builder)
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
				if event.Message.ID != "" {
					responseID = event.Message.ID
				}
				if event.Message.Model != "" {
					modelName = event.Message.Model
				}
				usage = event.Message.Usage.usage()
			case "content_block_start":
				_, toolSearch := anthropicToolSearchStrategy(event.ContentBlock.Name)
				if event.ContentBlock.Type == "server_tool_use" &&
					(event.ContentBlock.Name == "web_search" || event.ContentBlock.Name == "web_fetch" || toolSearch) {
					searchCalls[event.Index] = event.ContentBlock
					searchArgs[event.Index] = &strings.Builder{}
					continue
				}
				if event.ContentBlock.Type == "web_search_tool_result" ||
					event.ContentBlock.Type == "web_fetch_tool_result" {
					toolName := "web_search"
					kind := ai.ToolPartKindWebSearch
					if event.ContentBlock.Type == "web_fetch_tool_result" {
						toolName = "web_fetch"
						kind = ai.ToolPartKindWebFetch
					}
					part := parseAnthropicWebResult(event.ContentBlock, toolName, kind)
					if !yield(ai.NativeToolReturnEvent{PartID: strconv.Itoa(event.Index), Part: part}, nil) {
						return
					}
					continue
				}
				if event.ContentBlock.Type == "tool_search_tool_result" {
					part, err := parseAnthropicToolSearchResult(event.ContentBlock)
					if err != nil {
						yield(nil, err)
						return
					}
					if !yield(ai.NativeToolReturnEvent{PartID: strconv.Itoa(event.Index), Part: part}, nil) {
						return
					}
					continue
				}
				if !m.emitContentBlockStart(yield, event) {
					return
				}
			case "content_block_delta":
				if arguments, ok := searchArgs[event.Index]; ok && event.Delta.Type == "input_json_delta" {
					arguments.WriteString(event.Delta.PartialJSON)
					continue
				}
				if !emitContentBlockDelta(yield, event) {
					return
				}
			case "message_delta":
				usage.OutputTokens = event.Usage.OutputTokens
				if event.Delta.StopReason != "" {
					stopReason = event.Delta.StopReason
				}
			case "message_stop":
				providerDetails := map[string]any{}
				if stopReason != "" {
					providerDetails["finish_reason"] = stopReason
				}
				if len(providerDetails) == 0 {
					providerDetails = nil
				}
				state := ai.ModelResponseStateComplete
				if stopReason == "pause_turn" {
					state = ai.ModelResponseStateSuspended
				}
				yield(ai.FinishEvent{
					Usage: usage, ModelName: modelName, ProviderName: "anthropic", ProviderURL: m.baseURL,
					ProviderDetails: providerDetails, ProviderResponseID: responseID,
					FinishReason: anthropicFinishReason(stopReason), State: state,
				}, nil)
				return
			case "error":
				yield(nil, fmt.Errorf("anthropic: stream error %s: %s", event.Error.Type, event.Error.Message))
				return
			case "content_block_stop":
				block, ok := searchCalls[event.Index]
				if !ok {
					continue
				}
				raw := block.Input
				if arguments := searchArgs[event.Index].String(); arguments != "" {
					raw = json.RawMessage(arguments)
				}
				strategy, _ := anthropicToolSearchStrategy(block.Name)
				toolName := ai.ToolSearchName
				toolKind := ai.ToolPartKindToolSearch
				var details map[string]any
				var args json.RawMessage
				if block.Name == "web_search" || block.Name == "web_fetch" {
					toolName = block.Name
					toolKind = ai.ToolPartKindWebSearch
					if block.Name == "web_fetch" {
						toolKind = ai.ToolPartKindWebFetch
					}
					args = slices.Clone(raw)
					if len(args) == 0 || string(args) == "null" {
						args = json.RawMessage(`{}`)
					}
				} else {
					var err error
					args, err = normalizeAnthropicToolSearchArguments(raw, strategy)
					if err != nil {
						yield(nil, err)
						return
					}
					details = map[string]any{"strategy": strategy}
				}
				if callerType, _ := block.Caller["type"].(string); callerType != "" && callerType != "direct" {
					if details == nil {
						details = map[string]any{}
					}
					details["anthropic_caller"] = block.Caller
				}
				partID := strconv.Itoa(event.Index)
				if !yield(ai.ToolCallStartEvent{
					PartID: partID, ToolName: toolName, ToolCallID: block.ID,
					ToolKind: toolKind, ProviderName: "anthropic",
					ProviderDetails: details, Native: true,
				}, nil) {
					return
				}
				if !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(args)}, nil) {
					return
				}
				delete(searchCalls, event.Index)
				delete(searchArgs, event.Index)
			case "ping":
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
	case "compaction":
		var details map[string]any
		if event.ContentBlock.EncryptedContent != "" {
			details = map[string]any{"encrypted_content": event.ContentBlock.EncryptedContent}
		}
		return yield(ai.CompactionEvent{
			PartID: partID, Content: rawJSONString(event.ContentBlock.Content),
			ProviderName: "anthropic", ProviderDetails: details,
		}, nil)
	case "thinking":
		return event.ContentBlock.Thinking == "" && event.ContentBlock.Signature == "" ||
			yield(ai.ThinkingDeltaEvent{
				PartID: partID, Delta: event.ContentBlock.Thinking,
				SignatureDelta: event.ContentBlock.Signature, ProviderName: "anthropic",
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
	case "signature_delta":
		return yield(ai.ThinkingDeltaEvent{
			PartID: partID, SignatureDelta: event.Delta.Signature, ProviderName: "anthropic",
		}, nil)
	case "citations_delta":
		return true
	default:
		return yield(nil, fmt.Errorf("anthropic: unsupported content block delta type %q", event.Delta.Type))
	}
}
