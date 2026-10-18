package google

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"maps"
	"strconv"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (r generateResponse) continuationToken() string {
	if len(r.Candidates) == 0 {
		return ""
	}
	candidate := r.Candidates[0]
	if candidate.FinishReason != "" && candidate.FinishReason != "CONTINUATION" {
		return ""
	}
	return candidate.ContinuationToken
}

// continueGeneration follows the continuation tokens of a long decode and merges every request's
// response into one, the way google-genai does.
func (m *Model) continueGeneration(
	ctx context.Context, payload *generateRequest, settings ai.ModelSettings, data []byte, token string,
) ([]byte, string, error) {
	merged, _ := decodeGoogleJSON(data) // Request already decoded data.
	serviceTier := ""
	for token != "" {
		payload.ContinuationToken = token
		data, tier, err := m.generateContent(ctx, payload, settings)
		if err != nil {
			return nil, "", err
		}
		next, err := decodeGoogleJSON(data)
		if err != nil {
			return nil, "", ai.NewModelTransportError(ctx, m, "parse response", err)
		}
		var hop generateResponse
		_ = json.Unmarshal(data, &hop)
		token = hop.continuationToken()
		merged = mergeGoogleResponses(merged, next)
		serviceTier = tier
	}
	data, _ = json.Marshal(merged)
	return data, serviceTier, nil
}

func decodeGoogleJSON(data []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value map[string]any
	return value, decoder.Decode(&value)
}

func mergeGoogleResponses(prev, curr map[string]any) map[string]any {
	merged := mergeGoogleJSON("", prev, curr).(map[string]any)
	prevCandidates, _ := prev["candidates"].([]any)
	currCandidates, _ := curr["candidates"].([]any)
	candidates := make([]any, max(len(prevCandidates), len(currCandidates)))
	for index := range candidates {
		switch {
		case index >= len(currCandidates):
			candidates[index] = mergeGoogleCandidate(prevCandidates[index], map[string]any{})
		case index >= len(prevCandidates):
			candidates[index] = currCandidates[index]
		default:
			candidates[index] = mergeGoogleCandidate(prevCandidates[index], currCandidates[index])
		}
	}
	merged["candidates"] = candidates
	return merged
}

func mergeGoogleCandidate(prevValue, currValue any) any {
	prev, _ := prevValue.(map[string]any)
	curr, _ := currValue.(map[string]any)
	shiftGoogleGroundingSupports(prev, curr)
	prev = maps.Clone(prev)
	for _, terminal := range []string{"continuationToken", "finishReason", "finishMessage"} {
		delete(prev, terminal)
	}
	return mergeGoogleJSON("", prev, curr)
}

func shiftGoogleGroundingSupports(prev, curr map[string]any) {
	metadata, _ := curr["groundingMetadata"].(map[string]any)
	supports, _ := metadata["groundingSupports"].([]any)
	content, _ := prev["content"].(map[string]any)
	parts, _ := content["parts"].([]any)
	prevMetadata, _ := prev["groundingMetadata"].(map[string]any)
	chunks, _ := prevMetadata["groundingChunks"].([]any)
	for _, value := range supports {
		support, _ := value.(map[string]any)
		if indices, ok := support["groundingChunkIndices"].([]any); ok && len(chunks) > 0 {
			for index, chunkIndex := range indices {
				indices[index] = addGoogleNumbers(chunkIndex, json.Number(strconv.Itoa(len(chunks))))
			}
		}
		if segment, ok := support["segment"].(map[string]any); ok && len(parts) > 0 {
			partIndex, ok := segment["partIndex"]
			if !ok {
				partIndex = json.Number("0")
			}
			segment["partIndex"] = addGoogleNumbers(partIndex, json.Number(strconv.Itoa(len(parts))))
		}
	}
}

func mergeGoogleJSON(key string, prev, curr any) any {
	switch prevValue := prev.(type) {
	case map[string]any:
		currValue, ok := curr.(map[string]any)
		if !ok {
			break
		}
		merged := make(map[string]any, len(prevValue)+len(currValue))
		maps.Copy(merged, prevValue)
		for name, value := range currValue {
			if existing, ok := merged[name]; ok {
				value = mergeGoogleJSON(name, existing, value)
			}
			merged[name] = value
		}
		return merged
	case []any:
		currValue, ok := curr.([]any)
		if !ok {
			break
		}
		switch {
		case strings.HasSuffix(key, "TokensDetails"):
			return mergeGoogleListBy("modality", prevValue, currValue, true)
		case key == "safetyRatings":
			return mergeGoogleListBy("category", prevValue, currValue, false)
		}
		return append(append([]any{}, prevValue...), currValue...)
	case json.Number:
		if _, ok := curr.(json.Number); ok && (strings.HasSuffix(key, "Count") || strings.HasSuffix(key, "Sum")) {
			return addGoogleNumbers(prevValue, curr)
		}
	}
	if curr == nil {
		return prev
	}
	return curr
}

// mergeGoogleListBy keeps one entry per key value in first-seen order, summing token counts or
// keeping the latest entry.
func mergeGoogleListBy(key string, prev, curr []any, sum bool) []any {
	var merged []any
	positions := map[any]int{}
	for _, value := range append(append([]any{}, prev...), curr...) {
		entry, _ := value.(map[string]any)
		position, seen := positions[entry[key]]
		if !seen {
			positions[entry[key]] = len(merged)
			merged = append(merged, entry)
			continue
		}
		if sum {
			entry = map[string]any{key: entry[key], "tokenCount": addGoogleNumbers(
				merged[position].(map[string]any)["tokenCount"], entry["tokenCount"],
			)}
		}
		merged[position] = entry
	}
	return merged
}

func addGoogleNumbers(a, b any) json.Number {
	left, _ := a.(json.Number)
	right, _ := b.(json.Number)
	if x, err := left.Int64(); err == nil {
		if y, err := right.Int64(); err == nil {
			return json.Number(strconv.FormatInt(x+y, 10))
		}
	}
	x, _ := left.Float64()
	y, _ := right.Float64()
	return json.Number(strconv.FormatFloat(x+y, 'g', -1, 64))
}

type googleStreamChunk struct {
	response generateResponse
	hop      int
}

// continuedStream yields the chunks of a streamed long decode across its continuation requests,
// numbering each request as a hop. Like google-genai, it resumes a dropped stream from its latest
// continuation token, or retries a request that dropped before any output from the token it started with.
func (m *Model) continuedStream(
	ctx context.Context, body io.ReadCloser, payload *generateRequest, settings ai.ModelSettings,
) iter.Seq2[googleStreamChunk, error] {
	return func(yield func(googleStreamChunk, error) bool) {
		resumeToken, reconnects := "", 0
		var lastErr error
		for hop := 0; ; hop++ {
			state := streamHop{start: resumeToken}
			var err error
			if hop > 0 {
				payload.ContinuationToken = resumeToken
				body, _, err = m.openStream(ctx, payload, settings)
			}
			if err == nil {
				scanner := bufio.NewScanner(body)
				scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
				for scanner.Scan() {
					data, ok := strings.CutPrefix(scanner.Text(), "data:")
					if !ok {
						continue
					}
					var chunk generateResponse
					if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &chunk); err != nil {
						_ = body.Close()
						yield(googleStreamChunk{}, ai.NewModelTransportError(ctx, m, "parse stream chunk", err))
						return
					}
					state.observe(chunk)
					if !yield(googleStreamChunk{response: chunk, hop: hop}, nil) {
						_ = body.Close()
						return
					}
				}
				_ = body.Close()
				if scanErr := scanner.Err(); scanErr != nil {
					err = ai.NewModelTransportError(ctx, m, "read stream", scanErr)
				}
			}
			var apiErr *APIError
			if errors.As(err, &apiErr) && !state.received {
				yield(googleStreamChunk{}, err)
				return
			}
			if err != nil {
				lastErr = err
			} else if state.content || state.token != "" && state.token != state.start {
				lastErr = nil
			}
			resumeToken, reconnects = state.next(reconnects)
			if resumeToken != "" {
				if state.uncheckpointed {
					yield(googleStreamChunk{}, &APIError{Body: "the stream was interrupted after output that no " +
						"continuation token covers, and resuming it would repeat that output"})
					return
				}
				continue
			}
			if err == nil && state.finishReason == "" && !state.promptFeedback {
				err = lastErr
			}
			if err != nil {
				yield(googleStreamChunk{}, err)
			}
			return
		}
	}
}

type streamHop struct {
	start          string
	token          string
	finishReason   string
	promptFeedback bool
	content        bool
	received       bool
	uncheckpointed bool
}

func (hop *streamHop) observe(chunk generateResponse) {
	hop.received = true
	if chunk.PromptFeedback.BlockReason != "" {
		hop.promptFeedback = true
	}
	if len(chunk.Candidates) == 0 {
		return
	}
	candidate := chunk.Candidates[0]
	if candidate.FinishReason != "" {
		hop.finishReason = candidate.FinishReason
	}
	switch {
	case candidate.ContinuationToken != "":
		hop.token, hop.uncheckpointed = candidate.ContinuationToken, false
	case len(candidate.Content.Parts) > 0:
		hop.uncheckpointed = true
	}
	if candidate.ContinuationToken == "" && candidate.FinishReason != "" && candidate.FinishReason != "CONTINUATION" {
		hop.token = ""
	}
	if len(candidate.Content.Parts) > 0 {
		hop.content = true
	}
}

// next returns the token to resume from, or "" to stop, and the count of consecutive requests
// that dropped without progress, which stops retrying at three.
func (hop *streamHop) next(reconnects int) (string, int) {
	newToken := hop.token != "" && hop.token != hop.start
	if hop.finishReason != "" || hop.promptFeedback {
		if hop.finishReason == "CONTINUATION" && !hop.promptFeedback && newToken {
			return hop.token, 0
		}
		return "", 0
	}
	if newToken {
		return hop.token, 0
	}
	if hop.content || reconnects >= 3 {
		return "", reconnects
	}
	return hop.start, reconnects + 1
}
