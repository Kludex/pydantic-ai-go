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
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", m.baseURL, m.name)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", m.apiKey)
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
	return m.eventStream(resp.Body), nil
}

func (m *Model) eventStream(body io.ReadCloser) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = body.Close() }()
		usage := ai.Usage{Requests: 1}
		modelName := m.name
		responseID := ""
		finishReason := ""
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
				continue
			}
			if chunk.Candidates[0].FinishReason != "" {
				finishReason = chunk.Candidates[0].FinishReason
			}
			for index, part := range chunk.Candidates[0].Content.Parts {
				if !emitPart(yield, part, index) {
					return
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
		if finishReason != "" {
			providerDetails["finish_reason"] = finishReason
		}
		if len(providerDetails) == 0 {
			providerDetails = nil
		}
		yield(ai.FinishEvent{
			Usage: usage, ModelName: modelName, ProviderName: "google", ProviderURL: m.baseURL,
			ProviderDetails: providerDetails, ProviderResponseID: responseID,
			FinishReason: googleFinishReason(finishReason), State: ai.ModelResponseStateComplete,
		}, nil)
	}
}

func emitPart(yield func(ai.ModelStreamEvent, error) bool, part part, index int) bool {
	switch {
	case part.FunctionCall != nil:
		partID := fmt.Sprintf("tool:%d", index)
		if !yield(ai.ToolCallStartEvent{
			PartID: partID, ToolName: part.FunctionCall.Name, ToolCallID: part.FunctionCall.ID,
		}, nil) {
			return false
		}
		args, _ := json.Marshal(part.FunctionCall.Args)
		return yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(args)}, nil)
	case part.Thought:
		return part.Text == "" || yield(ai.ThinkingDeltaEvent{
			PartID: fmt.Sprintf("thinking:%d", index), Delta: part.Text,
		}, nil)
	case part.Text != "":
		return yield(ai.TextDeltaEvent{PartID: fmt.Sprintf("text:%d", index), Delta: part.Text}, nil)
	case part.InlineData != nil || part.FileData != nil:
		return yield(nil, fmt.Errorf("google: streamed binary output is not supported"))
	default:
		return true
	}
}
