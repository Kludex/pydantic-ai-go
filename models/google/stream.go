package google

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

	ai "github.com/Kludex/pydantic-ai-go"
)

// StreamRequest implements ai.StreamingModel using Gemini server-sent events.
func (m *Model) StreamRequest(
	ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	payload, err := m.buildPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("google: marshal request: %w", err)
	}
	endpoint := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", m.baseURL, m.name)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if err := m.prepareHTTPRequest(req, params.Settings); err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/event-stream")

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("google: request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		data, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("google: read error response: %w", err)
		}
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	return m.eventStream(resp.Body, resp.Header.Get("x-gemini-service-tier")), nil
}

func (m *Model) eventStream(body io.ReadCloser, serviceTier string) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		usage := ai.Usage{Requests: 1}
		modelName := m.name
		responseID := ""
		finishReason := ""
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
		lastCodeCallID := ""
		codeCallIndex := 0
		fileIndex := 0
		received := false
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			data, ok := strings.CutPrefix(scanner.Text(), "data:")
			if !ok {
				continue
			}
			var chunk generateResponse
			if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &chunk); err != nil {
				yield(nil, fmt.Errorf("google: parse stream chunk: %w", err))
				return
			}
			received = true
			if chunk.ResponseID != "" {
				responseID = chunk.ResponseID
			}
			if chunk.ModelVersion != "" {
				modelName = chunk.ModelVersion
			}
			if chunk.UsageMetadata.PromptTokenCount != 0 || chunk.UsageMetadata.CandidatesTokenCount != 0 {
				usage = chunk.UsageMetadata.usage()
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
			if !webSearchEmitted {
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
			if !webFetchEmitted {
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
				default:
					if !emitPart(yield, part, index, m.providerName) {
						return
					}
				}
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("google: read stream: %w", err))
			return
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
		if groundingMetadata != nil {
			providerDetails["grounding_metadata"] = groundingMetadata
		}
		if urlContextMetadata != nil {
			providerDetails["url_context_metadata"] = urlContextMetadata
		}
		if len(providerDetails) == 0 {
			providerDetails = nil
		}
		yield(ai.FinishEvent{
			Usage: usage, ModelName: modelName, ProviderName: m.providerName, ProviderURL: m.baseURL,
			ProviderDetails: providerDetails, ProviderResponseID: responseID,
			FinishReason: normalizedFinishReason, State: ai.ModelResponseStateComplete,
		}, nil)
	}
}

func emitGoogleNativeTool(
	yield func(ai.ModelStreamEvent, error) bool,
	call *ai.NativeToolCallPart,
	returned *ai.NativeToolReturnPart,
) bool {
	partID := string(call.ToolKind) + ":" + call.ToolCallID
	return yield(ai.ToolCallStartEvent{
		PartID: partID, ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		ToolKind: call.ToolKind, ProviderName: call.ProviderName, Native: true,
	}, nil) && yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(call.Args)}, nil) &&
		yield(ai.NativeToolReturnEvent{PartID: "return:" + call.ToolCallID, Part: *returned}, nil)
}

func emitPart(yield func(ai.ModelStreamEvent, error) bool, part part, index int, modelProviderName string) bool {
	providerName, providerDetails := googlePartMetadata(part.ThoughtSignature, modelProviderName)
	switch {
	case part.FunctionCall != nil:
		partID := fmt.Sprintf("tool:%d", index)
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
			PartID: fmt.Sprintf("thinking:%d", index), Delta: part.Text,
			ProviderName: providerName, ProviderDetails: providerDetails,
		}, nil)
	case part.Text != "" || providerDetails != nil:
		return yield(ai.TextDeltaEvent{
			PartID: fmt.Sprintf("text:%d", index), Delta: part.Text,
			ProviderName: providerName, ProviderDetails: providerDetails,
		}, nil)
	case part.FileData != nil:
		return yield(nil, fmt.Errorf("google: streamed file-data output is not supported"))
	default:
		return true
	}
}
