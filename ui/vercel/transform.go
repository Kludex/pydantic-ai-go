package vercel

import (
	"encoding/json"
	"fmt"
	"iter"
	"strconv"
	"sync/atomic"

	ai "github.com/Kludex/pydantic-ai-go"
)

var generatedID atomic.Uint64

// TransformStream converts agent events to Vercel AI SDK UI v5 chunks.
func TransformStream(stream ai.EventStream, serverMessageID string) iter.Seq2[Chunk, error] {
	return TransformStreamWithConfig(stream, StreamConfig{SDKVersion: 5, ServerMessageID: serverMessageID})
}

// TransformStreamWithConfig converts agent events for AI SDK UI v5, v6, or v7.
func TransformStreamWithConfig(stream ai.EventStream, config StreamConfig) iter.Seq2[Chunk, error] {
	if config.SDKVersion == 0 {
		config.SDKVersion = 5
	}
	if config.ServerMessageID == "" {
		config.ServerMessageID = nextID("message")
	}
	return func(yield func(Chunk, error) bool) {
		if config.SDKVersion < 5 || config.SDKVersion > 7 {
			err := fmt.Errorf("vercel: SDK version must be 5, 6, or 7")
			yield(Chunk{Type: ChunkError, ErrorText: err.Error()}, err)
			return
		}
		if !yield(Chunk{Type: ChunkStart, MessageID: config.ServerMessageID}, nil) {
			return
		}
		state := transformState{sdkVersion: config.SDKVersion, partIDs: map[string]string{}, toolIDs: map[string]string{}}
		for event, eventErr := range stream {
			if eventErr != nil {
				if !state.finishStep(yield) {
					return
				}
				yield(Chunk{Type: ChunkError, ErrorText: eventErr.Error()}, eventErr)
				return
			}
			if !state.transform(yield, event) {
				return
			}
		}
		if !state.finishStep(yield) {
			return
		}
		if !yield(Chunk{Type: ChunkFinish, FinishReason: state.finishReason}, nil) {
			return
		}
		yield(Chunk{Type: ChunkDone}, nil)
	}
}

type transformState struct {
	sdkVersion   int
	step         bool
	finishReason string
	partIDs      map[string]string
	toolIDs      map[string]string
}

func (state *transformState) transform(yield func(Chunk, error) bool, event ai.StreamEvent) bool {
	switch value := event.(type) {
	case ai.PartStartEvent:
		if !state.startStep(yield) {
			return false
		}
		id := value.PartID
		if id == "" {
			id = nextID("part")
		}
		state.partIDs[value.PartID] = id
		switch part := value.Part.(type) {
		case ai.TextPart:
			if !yield(Chunk{Type: ChunkTextStart, ID: id}, nil) {
				return false
			}
			if part.Content != "" && !yield(Chunk{Type: ChunkTextDelta, ID: id, Delta: part.Content}, nil) {
				return false
			}
		case ai.ThinkingPart:
			if !yield(Chunk{Type: ChunkReasoningStart, ID: id}, nil) {
				return false
			}
			if part.Content != "" && !yield(Chunk{Type: ChunkReasoningDelta, ID: id, Delta: part.Content}, nil) {
				return false
			}
		case ai.FilePart:
			return yield(fileChunk(part), nil)
		case ai.CompactionPart:
			return yield(Chunk{Type: ChunkDataCompaction, Data: compactionData(part)}, nil)
		case ai.ToolCallPart:
			if !state.startTool(yield, value.PartID, part.ToolCallID, part.ToolName, string(part.Args), false) {
				return false
			}
		case ai.NativeToolCallPart:
			if !state.startTool(yield, value.PartID, part.ToolCallID, part.ToolName, string(part.Args), true) {
				return false
			}
		case ai.NativeToolReturnPart:
			return state.toolOutput(yield, part.ToolCallID, part.Content, part.Outcome, true)
		}
	case ai.PartDeltaEvent:
		id := state.partIDs[value.PartID]
		switch delta := value.Delta.(type) {
		case ai.TextPartDelta:
			if delta.ContentDelta != "" {
				return yield(Chunk{Type: ChunkTextDelta, ID: id, Delta: delta.ContentDelta}, nil)
			}
		case ai.ThinkingPartDelta:
			if delta.ContentDelta != "" {
				return yield(Chunk{Type: ChunkReasoningDelta, ID: id, Delta: delta.ContentDelta}, nil)
			}
		case ai.ToolCallPartDelta:
			return state.toolDelta(yield, value.PartID, delta.ToolCallID, delta.ArgsDelta)
		case ai.NativeToolCallPartDelta:
			converted := ai.ToolCallPartDelta(delta)
			return state.toolDelta(yield, value.PartID, converted.ToolCallID, converted.ArgsDelta)
		case ai.FilePartDelta:
			return yield(fileChunk(delta.Part), nil)
		}
	case ai.PartEndEvent:
		id := state.partIDs[value.PartID]
		switch value.Part.(type) {
		case ai.TextPart:
			return yield(Chunk{Type: ChunkTextEnd, ID: id}, nil)
		case ai.ThinkingPart:
			return yield(Chunk{Type: ChunkReasoningEnd, ID: id}, nil)
		}
	case ai.FunctionToolCallEvent:
		return state.toolAvailable(yield, value.Part, false)
	case ai.OutputToolCallEvent:
		return state.toolAvailable(yield, value.Part, false)
	case ai.FunctionToolResultEvent:
		return state.requestResult(yield, value.Part, false)
	case ai.OutputToolResultEvent:
		return state.requestResult(yield, value.Part, false)
	case ai.ToolAvailabilityDeltaEvent:
		return yield(Chunk{Type: ChunkDataToolAvailability, Data: toolAvailabilityData(value.Part)}, nil)
	case ai.DeferredToolRequestsEvent:
		if state.sdkVersion >= 6 {
			for _, call := range value.Requests.Approvals {
				if !yield(Chunk{
					Type: ChunkToolApprovalRequest, ApprovalID: call.ToolCallID, ToolCallID: call.ToolCallID,
				}, nil) {
					return false
				}
			}
		}
	case ai.FinishEvent:
		state.finishReason = vercelFinishReason(value.FinishReason)
		return state.finishStep(yield)
	}
	return true
}

func (state *transformState) startStep(yield func(Chunk, error) bool) bool {
	if state.step {
		return true
	}
	state.step = yield(Chunk{Type: ChunkStartStep}, nil)
	return state.step
}

func (state *transformState) finishStep(yield func(Chunk, error) bool) bool {
	if !state.step {
		return true
	}
	state.step = false
	return yield(Chunk{Type: ChunkFinishStep}, nil)
}

func (state *transformState) startTool(
	yield func(Chunk, error) bool, partID string, toolCallID string, name string, args string, native bool,
) bool {
	state.toolIDs[partID] = toolCallID
	providerExecuted := native
	if !yield(Chunk{
		Type: ChunkToolInputStart, ToolCallID: toolCallID, ToolName: name, ProviderExecuted: &providerExecuted,
	}, nil) {
		return false
	}
	if args != "" {
		return yield(Chunk{Type: ChunkToolInputDelta, ToolCallID: toolCallID, InputTextDelta: args}, nil)
	}
	return true
}

func (state *transformState) toolDelta(
	yield func(Chunk, error) bool, partID string, toolCallID string, delta string,
) bool {
	if toolCallID == "" {
		toolCallID = state.toolIDs[partID]
	}
	if delta == "" {
		return true
	}
	return yield(Chunk{Type: ChunkToolInputDelta, ToolCallID: toolCallID, InputTextDelta: delta}, nil)
}

func (state *transformState) toolAvailable(
	yield func(Chunk, error) bool, part ai.ToolCallPart, native bool,
) bool {
	var input any
	if len(part.Args) > 0 {
		if err := json.Unmarshal(part.Args, &input); err != nil {
			return yield(Chunk{
				Type: ChunkError, ErrorText: fmt.Sprintf("vercel: decode tool input: %v", err),
			}, err)
		}
	}
	providerExecuted := native
	return yield(Chunk{
		Type: ChunkToolInputAvailable, ToolCallID: part.ToolCallID, ToolName: part.ToolName,
		Input: input, ProviderExecuted: &providerExecuted,
	}, nil)
}

func (state *transformState) requestResult(yield func(Chunk, error) bool, part ai.RequestPart, native bool) bool {
	switch value := part.(type) {
	case ai.ToolReturnPart:
		return state.toolOutput(yield, value.ToolCallID, value.Content, value.Outcome, native)
	case ai.RetryPromptPart:
		return state.toolOutput(yield, value.ToolCallID, value.ModelResponse(), ai.ToolReturnOutcomeFailed, native)
	default:
		err := fmt.Errorf("vercel: unsupported tool result part %T", part)
		return yield(Chunk{Type: ChunkError, ErrorText: err.Error()}, err)
	}
}

func (state *transformState) toolOutput(
	yield func(Chunk, error) bool, toolCallID string, output any, outcome ai.ToolReturnOutcome, native bool,
) bool {
	providerExecuted := native
	if outcome == ai.ToolReturnOutcomeFailed || outcome == ai.ToolReturnOutcomeDenied ||
		outcome == ai.ToolReturnOutcomeInterrupted {
		return yield(Chunk{
			Type: ChunkToolOutputError, ToolCallID: toolCallID, ErrorText: fmt.Sprint(output),
			ProviderExecuted: &providerExecuted,
		}, nil)
	}
	return yield(Chunk{
		Type: ChunkToolOutputAvailable, ToolCallID: toolCallID, Output: output,
		ProviderExecuted: &providerExecuted,
	}, nil)
}

func vercelFinishReason(reason ai.FinishReason) string {
	switch reason {
	case ai.FinishReasonStop:
		return "stop"
	case ai.FinishReasonLength:
		return "length"
	case ai.FinishReasonContentFilter:
		return "content-filter"
	case ai.FinishReasonToolCall:
		return "tool-calls"
	case ai.FinishReasonError:
		return "error"
	default:
		return "other"
	}
}

func nextID(kind string) string {
	return "vercel-" + kind + "-" + strconv.FormatUint(generatedID.Add(1), 10)
}
