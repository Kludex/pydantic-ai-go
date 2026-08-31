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
	"net/url"
	"strconv"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

// StreamRequest implements ai.StreamingModel using the Responses SSE API.
func (m *ResponsesModel) StreamRequest(
	ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	if responseID, ok := suspendedResponsesID(msgs); ok {
		sequence, hasSequence := suspendedResponsesSequence(msgs)
		if !hasSequence {
			response, err := m.retrieveResponse(ctx, responseID, params.Settings.ExtraHeaders)
			if err != nil {
				return nil, err
			}
			return staticResponsesEventStream(response), nil
		}
		return m.retrieveResponseStream(ctx, responseID, sequence, params.Settings.ExtraHeaders)
	}
	payload, err := m.buildResponsesPayload(msgs, params, true)
	if err != nil {
		return nil, err
	}
	payload.Stream = true
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
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
	setExtraHeaders(req, params.Settings.ExtraHeaders)

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
	return m.responsesEventStream(resp.Body, nil), nil
}

func (m *ResponsesModel) retrieveResponseStream(
	ctx context.Context, responseID string, sequence int, headers map[string]string,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	query := url.Values{"stream": {"true"}, "starting_after": {strconv.Itoa(sequence)}}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet,
		m.baseURL+"/responses/"+url.PathEscape(responseID)+"?"+query.Encode(), nil,
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	req.Header.Set("Accept", "text/event-stream")
	setExtraHeaders(req, headers)
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: retrieve background response stream: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		data, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("openai: read error response: %w", err)
		}
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	return m.responsesEventStream(resp.Body, &ai.ResponseMetadataEvent{
		ModelName: m.name, ProviderName: "openai", ProviderURL: m.baseURL,
		ProviderResponseID: responseID,
		ProviderDetails:    map[string]any{"background": true, "sequence_number": sequence},
		State:              ai.ModelResponseStateSuspended,
	}), nil
}

func suspendedResponsesSequence(messages []ai.ModelMessage) (int, bool) {
	response := messages[len(messages)-1].(ai.ModelResponse)
	switch sequence := response.ProviderDetails["sequence_number"].(type) {
	case int:
		return sequence, true
	case float64:
		return int(sequence), true
	default:
		return 0, false
	}
}

type responsesStreamEvent struct {
	Type           string `json:"type"`
	SequenceNumber *int   `json:"sequence_number"`
	Delta          string `json:"delta"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	ContentIndex   int    `json:"content_index"`
	SummaryIndex   int    `json:"summary_index"`
	Item           struct {
		ID               string          `json:"id"`
		Type             string          `json:"type"`
		CallID           string          `json:"call_id"`
		Name             string          `json:"name"`
		Arguments        json.RawMessage `json:"arguments"`
		Execution        string          `json:"execution"`
		Namespace        string          `json:"namespace"`
		EncryptedContent string          `json:"encrypted_content"`
	} `json:"item"`
	Part struct {
		Text string `json:"text"`
	} `json:"part"`
	Response responsesResponse `json:"response"`
	Error    struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (m *ResponsesModel) responsesEventStream(
	body io.ReadCloser, seed *ai.ResponseMetadataEvent,
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		var latest *responsesStreamEvent
		var lastSequence *int
		emittedParts := false
		if seed != nil {
			if sequence, ok := seed.ProviderDetails["sequence_number"].(int); ok {
				lastSequence = &sequence
			}
			latest = &responsesStreamEvent{Response: responsesResponse{
				ID: seed.ProviderResponseID, Model: seed.ModelName, Status: "in_progress",
				Background: providerBool(seed.ProviderDetails, "background"),
			}}
			if !yield(*seed, nil) {
				return
			}
		}
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
			if event.SequenceNumber != nil {
				sequence := *event.SequenceNumber
				lastSequence = &sequence
			}
			if event.Response.ID != "" {
				snapshot := event
				latest = &snapshot
			}
			var metadataResponse *responsesResponse
			if event.Response.ID != "" {
				metadataResponse = &event.Response
			} else if event.SequenceNumber != nil && latest != nil {
				metadataResponse = &latest.Response
			}
			if metadataResponse != nil && event.Type != "response.completed" &&
				event.Type != "response.failed" && event.Type != "response.incomplete" {
				rawFinishReason, providerDetails, timestamp, state := responsesMetadata(
					metadataResponse.Status, metadataResponse.IncompleteDetails,
					metadataResponse.CreatedAt, metadataResponse.Background,
				)
				if lastSequence != nil {
					if providerDetails == nil {
						providerDetails = map[string]any{}
					}
					providerDetails["sequence_number"] = *lastSequence
				}
				modelName := metadataResponse.Model
				if modelName == "" {
					modelName = m.name
				}
				if !yield(ai.ResponseMetadataEvent{
					Usage: metadataResponse.Usage.usage(), ModelName: modelName, Timestamp: timestamp,
					ProviderName: "openai", ProviderURL: m.baseURL, ProviderDetails: providerDetails,
					ProviderResponseID: metadataResponse.ID,
					FinishReason:       openAIResponsesFinishReason(rawFinishReason), State: state,
				}, nil) {
					return
				}
			}
			switch event.Type {
			case "response.output_text.delta":
				emittedParts = true
				partID := fmt.Sprintf("output:%d:content:%d:text", event.OutputIndex, event.ContentIndex)
				providerName := ""
				if event.ItemID != "" {
					providerName = "openai"
				}
				if !yield(ai.TextDeltaEvent{
					PartID: partID, Delta: event.Delta, ID: event.ItemID, ProviderName: providerName,
				}, nil) {
					return
				}
			case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
				emittedParts = true
				partID := responsesThinkingPartID(event)
				if !yield(ai.ThinkingDeltaEvent{
					PartID: partID, Delta: event.Delta, ID: event.ItemID, ProviderName: "openai",
				}, nil) {
					return
				}
			case "response.reasoning_summary_part.added":
				partID := responsesThinkingPartID(event)
				if event.Part.Text != "" {
					emittedParts = true
				}
				if event.Part.Text != "" && !yield(ai.ThinkingDeltaEvent{
					PartID: partID, Delta: event.Part.Text, ID: event.ItemID, ProviderName: "openai",
				}, nil) {
					return
				}
			case "response.output_item.added":
				switch event.Item.Type {
				case "function_call":
					emittedParts = true
					partID := responsesToolPartID(event)
					var providerDetails map[string]any
					if event.Item.Namespace != "" {
						providerDetails = map[string]any{"namespace": event.Item.Namespace}
					}
					if !yield(ai.ToolCallStartEvent{
						PartID: partID, ToolName: event.Item.Name, ToolCallID: event.Item.CallID,
						ID: event.Item.ID, ProviderName: "openai", ProviderDetails: providerDetails,
					}, nil) {
						return
					}
					if len(event.Item.Arguments) > 0 && string(event.Item.Arguments) != `""` {
						arguments, err := normalizeResponsesArguments(event.Item.Arguments)
						if err != nil {
							yield(nil, err)
							return
						}
						if !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(arguments)}, nil) {
							return
						}
					}
				case "tool_search_call":
					emittedParts = true
					if event.Item.Execution != "client" {
						yield(nil, fmt.Errorf("openai: server-executed tool search is not supported yet"))
						return
					}
					if !yield(ai.ToolCallStartEvent{
						PartID: responsesToolPartID(event), ToolName: ai.ToolSearchName,
						ToolKind: ai.ToolPartKindToolSearch, ID: event.Item.ID,
						ProviderName: "openai", ProviderDetails: map[string]any{"execution": "client"},
					}, nil) {
						return
					}
				case "reasoning":
					if event.Item.EncryptedContent != "" {
						emittedParts = true
					}
					if event.Item.EncryptedContent != "" && !yield(ai.ThinkingDeltaEvent{
						PartID: responsesThinkingPartID(event), ID: event.Item.ID,
						SignatureDelta: event.Item.EncryptedContent, ProviderName: "openai",
					}, nil) {
						return
					}
				}
			case "response.function_call_arguments.delta":
				if !yield(ai.ToolCallDeltaEvent{PartID: responsesToolPartID(event), ArgsDelta: event.Delta}, nil) {
					return
				}
			case "response.output_item.done":
				if event.Item.Type == "tool_search_call" && event.Item.Execution == "client" {
					arguments, err := normalizeResponsesArguments(event.Item.Arguments)
					if err != nil {
						yield(nil, err)
						return
					}
					callID := event.Item.CallID
					if callID == "" {
						callID = event.Item.ID
					}
					if !yield(ai.ToolCallDeltaEvent{
						PartID: responsesToolPartID(event), ToolCallID: callID, ArgsDelta: string(arguments),
					}, nil) {
						return
					}
				}
			case "response.completed":
				var snapshotParts []ai.ResponsePart
				if len(event.Response.Output) > 0 {
					response, err := modelResponseFromResponses(event.Response)
					if err != nil {
						yield(nil, err)
						return
					}
					snapshotParts = response.Parts
					if !emittedParts && !yieldStaticResponsesParts(response, yield) {
						return
					}
				}
				modelName := event.Response.Model
				if modelName == "" {
					modelName = m.name
				}
				rawFinishReason, providerDetails, timestamp, state := responsesMetadata(
					event.Response.Status, event.Response.IncompleteDetails,
					event.Response.CreatedAt, event.Response.Background,
				)
				if lastSequence != nil {
					if providerDetails == nil {
						providerDetails = map[string]any{}
					}
					providerDetails["sequence_number"] = *lastSequence
				}
				yield(ai.FinishEvent{
					Parts: snapshotParts, Usage: event.Response.Usage.usage(), ModelName: modelName, Timestamp: timestamp,
					ProviderName: "openai", ProviderURL: m.baseURL, ProviderDetails: providerDetails,
					ProviderResponseID: event.Response.ID,
					FinishReason:       openAIResponsesFinishReason(rawFinishReason), State: state,
				}, nil)
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
				"response.content_part.added", "response.content_part.done",
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
		if latest != nil {
			var snapshotParts []ai.ResponsePart
			if len(latest.Response.Output) > 0 {
				response, err := modelResponseFromResponses(latest.Response)
				if err != nil {
					yield(nil, err)
					return
				}
				snapshotParts = response.Parts
				if !emittedParts && !yieldStaticResponsesParts(response, yield) {
					return
				}
			}
			rawFinishReason, providerDetails, timestamp, state := responsesMetadata(
				latest.Response.Status, latest.Response.IncompleteDetails,
				latest.Response.CreatedAt, latest.Response.Background,
			)
			if state == ai.ModelResponseStateSuspended {
				if lastSequence != nil {
					providerDetails["sequence_number"] = *lastSequence
				}
				modelName := latest.Response.Model
				if modelName == "" {
					modelName = m.name
				}
				yield(ai.FinishEvent{
					Parts: snapshotParts, Usage: latest.Response.Usage.usage(), ModelName: modelName, Timestamp: timestamp,
					ProviderName: "openai", ProviderURL: m.baseURL, ProviderDetails: providerDetails,
					ProviderResponseID: latest.Response.ID,
					FinishReason:       openAIResponsesFinishReason(rawFinishReason), State: state,
				}, nil)
				return
			}
		}
		yield(nil, fmt.Errorf("openai: Responses stream ended without response.completed"))
	}
}

func staticResponsesEventStream(response *ai.ModelResponse) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		if !yieldStaticResponsesParts(response, yield) {
			return
		}
		yield(ai.FinishEvent{
			Usage: response.Usage, ModelName: response.ModelName, Timestamp: response.Timestamp,
			ProviderName: response.ProviderName, ProviderURL: response.ProviderURL,
			ProviderDetails: response.ProviderDetails, ProviderResponseID: response.ProviderResponseID,
			FinishReason: response.FinishReason, State: response.State,
		}, nil)
	}
}

func yieldStaticResponsesParts(
	response *ai.ModelResponse, yield func(ai.ModelStreamEvent, error) bool,
) bool {
	for index, part := range response.Parts {
		partID := "static:" + strconv.Itoa(index)
		switch part := part.(type) {
		case ai.TextPart:
			if !yield(ai.TextDeltaEvent{
				PartID: partID, Delta: part.Content, ID: part.ID,
				ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
			}, nil) {
				return false
			}
		case ai.ThinkingPart:
			if !yield(ai.ThinkingDeltaEvent{
				PartID: partID, Delta: part.Content, ID: part.ID, SignatureDelta: part.Signature,
				ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
			}, nil) {
				return false
			}
		case ai.ToolCallPart:
			if !yield(ai.ToolCallStartEvent{
				PartID: partID, ToolName: part.ToolName, ToolCallID: part.ToolCallID,
				ToolKind: part.ToolKind, ID: part.ID,
				ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
			}, nil) {
				return false
			}
			if !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(part.Args)}, nil) {
				return false
			}
		}
	}
	return true
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
