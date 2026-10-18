package google

import (
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

// StreamRequest implements ai.StreamingModel using Gemini server-sent events.
func (m *Model) StreamRequest(
	ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	payload, err := m.buildPayload(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	payload.ModelArmorConfig = nil
	body, serviceTier, err := m.openStream(ctx, payload, params.Settings)
	if err != nil {
		return nil, err
	}
	return m.eventStream(
		m.continuedStream(ctx, body, payload, params.Settings), serviceTier, hasGoogleFileSearch(params.NativeTools),
	), nil
}

func (m *Model) openStream(
	ctx context.Context, payload *generateRequest, settings ai.ModelSettings,
) (io.ReadCloser, string, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, "", fmt.Errorf("google: marshal request: %w", err)
	}
	endpoint := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", m.baseURL, m.name)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	if err := m.prepareHTTPRequest(req, settings); err != nil {
		return nil, "", err
	}
	req.Header.Set("Accept", "text/event-stream")

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, "", ai.NewModelTransportError(ctx, m, "request", err)
	}
	if resp.StatusCode != http.StatusOK {
		data, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, "", ai.NewModelTransportError(ctx, m, "read error response", err)
		}
		return nil, "", &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	return resp.Body, resp.Header.Get("x-gemini-service-tier"), nil
}

func (m *Model) eventStream(
	chunks iter.Seq2[googleStreamChunk, error], serviceTier string, fileSearchEnabled bool,
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		usage := ai.Usage{Requests: 1}
		var hopUsage ai.Usage
		hop := 0
		initialInputTokens := 0
		partPrefix := ""
		modelName := m.name
		responseID := ""
		finishReason := ""
		trafficType := ""
		responseTimestamp := time.Now().UTC()
		webSearchEmitted := false
		var logprobs map[string]any
		var avgLogprobs *float64
		blockReason := ""
		blockReasonMessage := ""
		var safetyRatings []map[string]any
		var groundingMetadata map[string]any
		var urlContextMetadata map[string]any
		webFetchEmitted := false
		fileSearchEmitted := false
		explicitNativeTools := false
		lastCodeCallID := ""
		lastFileSearchCallID := ""
		var pendingFileSearchReturns []ai.NativeToolReturnPart
		codeCallIndex := 0
		fileSearchIndex := 0
		fileIndex := 0
		received := false
		mediaCallID := ""
		mediaPendingPartID := ""
		fillMediaCall := func(id string) bool {
			partID := mediaPendingPartID
			mediaPendingPartID = ""
			return partID == "" || yield(ai.ToolCallDeltaEvent{PartID: partID, ToolCallID: id}, nil)
		}
		for streamed, err := range chunks {
			if err != nil {
				yield(nil, err)
				return
			}
			chunk := streamed.response
			if streamed.hop != hop {
				if hop == 0 {
					initialInputTokens = usage.InputTokens
				}
				hop, hopUsage = streamed.hop, usage.Clone()
				partPrefix = fmt.Sprintf("%d:", hop)
			}
			received = true
			if chunk.ResponseID != "" {
				responseID = chunk.ResponseID
			}
			if chunk.ModelVersion != "" {
				modelName = chunk.ModelVersion
			}
			chunkUsage := chunk.usage(modelName)
			if chunk.UsageMetadata.hasTokens() {
				usage = hopUsage.Clone()
				usage.Add(chunkUsage)
				usage.Requests = 1
			} else if count, ok := chunkUsage.Details["web_search_requests"]; ok {
				if usage.Details == nil {
					usage.Details = map[string]int{}
				}
				usage.Details["web_search_requests"] = count
				usage.Details["web_searches"] = chunkUsage.Details["web_searches"]
			}
			if chunk.UsageMetadata.TrafficType != "" {
				trafficType = chunk.UsageMetadata.TrafficType
			}
			if len(chunk.Candidates) == 0 {
				if chunk.PromptFeedback.BlockReason != "" {
					blockReason = chunk.PromptFeedback.BlockReason
					blockReasonMessage = chunk.PromptFeedback.BlockReasonMessage
					safetyRatings = chunk.PromptFeedback.SafetyRatings
				}
				continue
			}
			if chunk.Candidates[0].FinishReason != "" && blockReason == "" {
				finishReason = chunk.Candidates[0].FinishReason
			}
			if chunk.Candidates[0].SafetyRatings != nil {
				safetyRatings = chunk.Candidates[0].SafetyRatings
			}
			if chunk.Candidates[0].LogprobsResult != nil {
				logprobs = chunk.Candidates[0].LogprobsResult
			}
			if chunk.Candidates[0].AvgLogprobs != nil {
				avgLogprobs = chunk.Candidates[0].AvgLogprobs
			}
			if chunk.Candidates[0].GroundingMetadata != nil {
				groundingMetadata = chunk.Candidates[0].GroundingMetadata
			}
			if chunk.Candidates[0].URLContextMetadata != nil {
				urlContextMetadata = chunk.Candidates[0].URLContextMetadata
			}
			for _, candidatePart := range chunk.Candidates[0].Content.Parts {
				if googleNativeInvocation(candidatePart) {
					explicitNativeTools = true
					break
				}
			}
			if !explicitNativeTools && !webSearchEmitted {
				call, returned := googleWebSearchParts(
					chunk.Candidates[0].GroundingMetadata, responseID, m.providerName, responseTimestamp,
				)
				if call != nil {
					if !emitGoogleNativeTool(yield, call, returned) {
						return
					}
					webSearchEmitted = true
				}
			}
			if !explicitNativeTools && !webFetchEmitted {
				call, returned := googleWebFetchParts(
					chunk.Candidates[0].URLContextMetadata, responseID, m.providerName, responseTimestamp,
				)
				if call != nil {
					if !emitGoogleNativeTool(yield, call, returned) {
						return
					}
					webFetchEmitted = true
				}
			}
			for index, part := range chunk.Candidates[0].Content.Parts {
				switch {
				case part.ToolCall != nil:
					name, kind, ok := googleNativeToolIdentity(part.ToolCall.ToolType, part.ToolCall.Args)
					if !ok {
						yield(nil, fmt.Errorf("google: unknown native tool type %q", part.ToolCall.ToolType))
						return
					}
					callID := googleNativeCallID(
						part.ToolCall.ID, responseID, googleToolType(name, part.ToolCall.ToolType), fileSearchIndex,
					)
					fileSearchIndex++
					args := part.ToolCall.Args
					if args == nil {
						args = map[string]any{}
					}
					encoded, _ := json.Marshal(args)
					_, providerDetails := googlePartMetadata(part.ThoughtSignature, m.providerName)
					if name == googleMediaProcessingToolName {
						if !fillMediaCall(mediaCallID) {
							return
						}
						mediaCallID = callID
						partID := fmt.Sprintf("media_processing:%d", fileSearchIndex)
						if part.ToolCall.ID == "" {
							mediaPendingPartID = partID
						}
						if !yield(ai.ToolCallStartEvent{
							PartID: partID, ToolName: name, ToolCallID: part.ToolCall.ID,
							ProviderName: m.providerName, ProviderDetails: providerDetails, Native: true,
						}, nil) || !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(encoded)}, nil) {
							return
						}
						continue
					}
					call := ai.NativeToolCallPart{
						ToolName: name, Args: encoded, ToolCallID: callID, ToolKind: kind,
						ProviderName: m.providerName, ProviderDetails: providerDetails,
					}
					if !emitGoogleNativeCall(yield, &call) {
						return
					}
					if kind == ai.ToolPartKindFileSearch {
						lastFileSearchCallID = callID
					}
				case part.ToolResponse != nil:
					name, kind, ok := googleNativeToolIdentity(part.ToolResponse.ToolType, part.ToolResponse.Response)
					if !ok {
						yield(nil, fmt.Errorf("google: unknown native tool type %q", part.ToolResponse.ToolType))
						return
					}
					callID := part.ToolResponse.ID
					switch {
					case callID == "" && kind == ai.ToolPartKindFileSearch && lastFileSearchCallID != "":
						callID = lastFileSearchCallID
					case name == googleMediaProcessingToolName:
						if callID == "" {
							callID = mediaCallID
						}
						callID = googleNativeCallID(callID, responseID, "MEDIA_PROCESSING", fileSearchIndex)
						mediaCallID = ""
						if !fillMediaCall(callID) {
							return
						}
					default:
						callID = googleNativeCallID(callID, responseID, part.ToolResponse.ToolType, fileSearchIndex)
					}
					_, providerDetails := googlePartMetadata(part.ThoughtSignature, m.providerName)
					returned := ai.NativeToolReturnPart{
						ToolName: name, ToolCallID: callID, ToolKind: kind, Content: part.ToolResponse.Response,
						Timestamp: responseTimestamp, ProviderName: m.providerName, ProviderDetails: providerDetails,
					}
					if kind == ai.ToolPartKindFileSearch && returned.Content == nil {
						pendingFileSearchReturns = append(pendingFileSearchReturns, returned)
						continue
					}
					if !yield(ai.NativeToolReturnEvent{PartID: "return:" + callID, Part: returned}, nil) {
						return
					}
					if kind == ai.ToolPartKindFileSearch {
						fileSearchEmitted = true
					}
				case part.InlineData != nil:
					if part.Thought {
						continue
					}
					providerName, providerDetails := googlePartMetadata(part.ThoughtSignature, m.providerName)
					file, err := googleInlineFilePart(*part.InlineData, providerName, providerDetails)
					if err != nil {
						yield(nil, err)
						return
					}
					partID := fmt.Sprintf("file:%d", fileIndex)
					fileIndex++
					if !yield(ai.FileEvent{PartID: partID, Part: file}, nil) {
						return
					}
				case part.ExecutableCode != nil:
					if fileSearchEnabled {
						if query, ok := googleFileSearchQuery(part.ExecutableCode.Code); ok {
							lastFileSearchCallID = fmt.Sprintf("%s:file_search:%d", responseID, fileSearchIndex)
							if responseID == "" {
								lastFileSearchCallID = fmt.Sprintf("file_search:%d", fileSearchIndex)
							}
							fileSearchIndex++
							args, _ := json.Marshal(map[string]any{"query": query})
							call := ai.NativeToolCallPart{
								ToolName: "file_search", Args: args, ToolCallID: lastFileSearchCallID,
								ToolKind: ai.ToolPartKindFileSearch, ProviderName: m.providerName,
							}
							if !emitGoogleNativeCall(yield, &call) {
								return
							}
							continue
						}
					}
					lastCodeCallID = fmt.Sprintf("%s:code_execution:%d", responseID, codeCallIndex)
					if responseID == "" {
						lastCodeCallID = fmt.Sprintf("code_execution:%d", codeCallIndex)
					}
					codeCallIndex++
					args, _ := json.Marshal(map[string]any{
						"code": part.ExecutableCode.Code, "language": part.ExecutableCode.Language,
					})
					partID := "code-execution:" + lastCodeCallID
					if !yield(ai.ToolCallStartEvent{
						PartID: partID, ToolName: "code_execution", ToolCallID: lastCodeCallID,
						ToolKind: ai.ToolPartKindCodeExecution, ProviderName: m.providerName, Native: true,
					}, nil) || !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(args)}, nil) {
						return
					}
				case part.CodeExecutionResult != nil:
					if lastCodeCallID == "" {
						lastCodeCallID = fmt.Sprintf("%s:code_execution:%d", responseID, codeCallIndex)
						if responseID == "" {
							lastCodeCallID = fmt.Sprintf("code_execution:%d", codeCallIndex)
						}
						codeCallIndex++
					}
					returned := ai.NativeToolReturnPart{
						ToolName: "code_execution", ToolCallID: lastCodeCallID,
						ToolKind: ai.ToolPartKindCodeExecution,
						Content: map[string]any{
							"outcome": part.CodeExecutionResult.Outcome, "output": part.CodeExecutionResult.Output,
						},
						Timestamp: responseTimestamp, ProviderName: m.providerName,
					}
					if !yield(ai.NativeToolReturnEvent{PartID: "return:" + lastCodeCallID, Part: returned}, nil) {
						return
					}
					lastCodeCallID = ""
				case part.signatureOnly:
					// Vertex AI reports agentic media-processing steps as signature-only parts that it rejects on replay.
				default:
					if !emitPart(yield, part, partPrefix, index, m.providerName) {
						return
					}
				}
			}
			contexts := googleFileSearchContexts(chunk.Candidates[0].GroundingMetadata)
			switch {
			case len(pendingFileSearchReturns) > 0 && len(contexts) > 0:
				for index := range pendingFileSearchReturns {
					pendingFileSearchReturns[index].Content = contexts
					if !yield(ai.NativeToolReturnEvent{
						PartID: "return:" + pendingFileSearchReturns[index].ToolCallID,
						Part:   pendingFileSearchReturns[index],
					}, nil) {
						return
					}
				}
				pendingFileSearchReturns = nil
				fileSearchEmitted = true
			case lastFileSearchCallID != "" && len(contexts) > 0 && !fileSearchEmitted:
				returned := ai.NativeToolReturnPart{
					ToolName: "file_search", ToolCallID: lastFileSearchCallID,
					ToolKind: ai.ToolPartKindFileSearch, Content: contexts,
					Timestamp: responseTimestamp, ProviderName: m.providerName,
				}
				if !yield(ai.NativeToolReturnEvent{
					PartID: "return:" + lastFileSearchCallID, Part: returned,
				}, nil) {
					return
				}
				fileSearchEmitted = true
			case fileSearchEnabled && !explicitNativeTools && !fileSearchEmitted && len(contexts) > 0:
				call, returned := googleFileSearchParts(
					chunk.Candidates[0].GroundingMetadata, responseID, m.providerName, responseTimestamp,
				)
				if call != nil && !emitGoogleNativeTool(yield, call, returned) {
					return
				}
				fileSearchEmitted = call != nil
			}
		}
		if !fillMediaCall(mediaCallID) {
			return
		}
		for _, pending := range pendingFileSearchReturns {
			if !yield(ai.NativeToolReturnEvent{
				PartID: "return:" + pending.ToolCallID, Part: pending,
			}, nil) {
				return
			}
		}
		if !received {
			yield(nil, fmt.Errorf("google: stream ended without a response"))
			return
		}
		providerDetails := map[string]any{}
		normalizedFinishReason := googleFinishReason(finishReason)
		if blockReason != "" {
			providerDetails["block_reason"] = blockReason
			if blockReasonMessage != "" {
				providerDetails["block_reason_message"] = blockReasonMessage
			}
			normalizedFinishReason = ai.FinishReasonContentFilter
		} else if finishReason != "" {
			providerDetails["finish_reason"] = finishReason
		}
		if safetyRatings != nil {
			providerDetails["safety_ratings"] = safetyRatings
		}
		if logprobs != nil {
			providerDetails["logprobs"] = logprobs
		}
		if avgLogprobs != nil {
			providerDetails["avg_logprobs"] = *avgLogprobs
		}
		if serviceTier != "" {
			providerDetails["service_tier"] = strings.ToLower(serviceTier)
		}
		if trafficType != "" {
			providerDetails["traffic_type"] = trafficType
		}
		if groundingMetadata != nil {
			providerDetails["grounding_metadata"] = groundingMetadata
		}
		if urlContextMetadata != nil {
			providerDetails["url_context_metadata"] = urlContextMetadata
		}
		if len(providerDetails) == 0 {
			providerDetails = nil
		}
		var merged ai.ModelResponse
		if hop > 0 {
			ai.MarkInitialInputTokens(&merged, initialInputTokens)
		}
		yield(ai.FinishEvent{
			Usage: usage, ModelName: modelName, ProviderName: m.providerName, ProviderURL: m.baseURL,
			ProviderDetails: providerDetails, ProviderResponseID: responseID,
			FinishReason: normalizedFinishReason, State: ai.ModelResponseStateComplete, Metadata: merged.Metadata,
		}, nil)
	}
}

func emitGoogleNativeTool(
	yield func(ai.ModelStreamEvent, error) bool,
	call *ai.NativeToolCallPart,
	returned *ai.NativeToolReturnPart,
) bool {
	return emitGoogleNativeCall(yield, call) &&
		yield(ai.NativeToolReturnEvent{PartID: "return:" + call.ToolCallID, Part: *returned}, nil)
}

func emitGoogleNativeCall(
	yield func(ai.ModelStreamEvent, error) bool, call *ai.NativeToolCallPart,
) bool {
	partID := string(call.ToolKind) + ":" + call.ToolCallID
	return yield(ai.ToolCallStartEvent{
		PartID: partID, ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		ToolKind: call.ToolKind, ProviderName: call.ProviderName,
		ProviderDetails: call.ProviderDetails, Native: true,
	}, nil) && yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(call.Args)}, nil)
}

func emitPart(
	yield func(ai.ModelStreamEvent, error) bool, part part, partPrefix string, index int, modelProviderName string,
) bool {
	providerName, providerDetails := googlePartMetadata(part.ThoughtSignature, modelProviderName)
	switch {
	case part.FunctionCall != nil:
		partID := fmt.Sprintf("%stool:%d", partPrefix, index)
		if !yield(ai.ToolCallStartEvent{
			PartID: partID, ToolName: part.FunctionCall.Name, ToolCallID: part.FunctionCall.ID,
			ProviderName: providerName, ProviderDetails: providerDetails,
		}, nil) {
			return false
		}
		args, _ := json.Marshal(part.FunctionCall.Args)
		return yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(args)}, nil)
	case part.Thought:
		return part.Text == "" && providerDetails == nil || yield(ai.ThinkingDeltaEvent{
			PartID: fmt.Sprintf("%sthinking:%d", partPrefix, index), Delta: part.Text,
			ProviderName: providerName, ProviderDetails: providerDetails,
		}, nil)
	case part.Text != "" || providerDetails != nil:
		return yield(ai.TextDeltaEvent{
			PartID: fmt.Sprintf("%stext:%d", partPrefix, index), Delta: part.Text,
			ProviderName: providerName, ProviderDetails: providerDetails,
		}, nil)
	case part.FileData != nil:
		return yield(nil, fmt.Errorf("google: streamed file-data output is not supported"))
	default:
		return true
	}
}
