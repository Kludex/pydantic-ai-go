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
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// StreamRequest implements ai.StreamingModel using server-sent events.
func (m *Model) StreamRequest(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	payload, err := m.buildPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	payload.Stream = true
	payload.StreamOptions = &streamOptions{IncludeUsage: true}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/chat/completions", bytes.NewReader(body))
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
	return m.eventStream(resp.Body), nil
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatChunk struct {
	ID                string `json:"id"`
	Model             string `json:"model"`
	Created           int64  `json:"created"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
		Logprobs     *struct {
			Content []map[string]any `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage *chatUsage `json:"usage"`
}

func (m *Model) eventStream(body io.ReadCloser) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		var usage ai.Usage
		modelName := m.name
		responseID := ""
		created := int64(0)
		finishReason := ""
		providerDetails := map[string]any{}
		startedTools := map[int]bool{}

		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			data, ok := strings.CutPrefix(line, "data: ")
			if !ok {
				continue
			}
			if data == "[DONE]" {
				var timestamp time.Time
				if created != 0 {
					timestamp = time.Unix(created, 0).UTC()
					providerDetails["timestamp"] = timestamp
				}
				if finishReason != "" {
					providerDetails["finish_reason"] = finishReason
				}
				if len(providerDetails) == 0 {
					providerDetails = nil
				}
				yield(ai.FinishEvent{
					Usage: usage, ModelName: modelName, Timestamp: timestamp,
					ProviderName: "openai", ProviderURL: m.baseURL, ProviderDetails: providerDetails,
					ProviderResponseID: responseID, FinishReason: openAIChatFinishReason(finishReason),
					State: ai.ModelResponseStateComplete,
				}, nil)
				return
			}
			var chunk chatChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				yield(nil, fmt.Errorf("openai: parse stream chunk: %w", err))
				return
			}
			if chunk.ID != "" {
				responseID = chunk.ID
			}
			if chunk.Model != "" {
				modelName = chunk.Model
			}
			if chunk.Created != 0 {
				created = chunk.Created
			}
			if chunk.ServiceTier != "" {
				providerDetails["service_tier"] = chunk.ServiceTier
			}
			if chunk.SystemFingerprint != "" {
				providerDetails["system_fingerprint"] = chunk.SystemFingerprint
			}
			if chunk.Usage != nil {
				usage = chunk.Usage.usage()
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			if chunk.Choices[0].FinishReason != "" {
				finishReason = chunk.Choices[0].FinishReason
			}
			if chunk.Choices[0].Logprobs != nil {
				logprobs, _ := providerDetails["logprobs"].([]map[string]any)
				providerDetails["logprobs"] = append(logprobs, chunk.Choices[0].Logprobs.Content...)
			}
			delta := chunk.Choices[0].Delta
			if delta.Content != "" {
				if !yield(ai.TextDeltaEvent{PartID: "text", Delta: delta.Content}, nil) {
					return
				}
			}
			for _, call := range delta.ToolCalls {
				partID := fmt.Sprintf("tool:%d", call.Index)
				if !startedTools[call.Index] {
					startedTools[call.Index] = true
					if !yield(ai.ToolCallStartEvent{
						PartID: partID, ToolName: call.Function.Name, ToolCallID: call.ID,
					}, nil) {
						return
					}
				}
				if call.Function.Arguments != "" {
					if !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: call.Function.Arguments}, nil) {
						return
					}
				}
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("openai: read stream: %w", err))
			return
		}
		yield(nil, fmt.Errorf("openai: stream ended without [DONE]"))
	}
}
