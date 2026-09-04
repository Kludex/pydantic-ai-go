package vercel

import (
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"slices"
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
		state := transformState{
			sdkVersion: config.SDKVersion, partIDs: map[string]string{}, toolIDs: map[string]string{},
			streamedCalls: map[string]ai.ToolCallPart{}, streamedNative: map[string]bool{},
			invalidatedCalls: map[string]ai.ToolCallPart{},
		}
		for event, eventErr := range stream {
			if eventErr != nil {
				if !state.flushToolInputs(yield) || !state.finishStep(yield) {
					return
				}
				if errors.Is(eventErr, ai.ErrRunCancelled) {
					if yield(Chunk{Type: ChunkAbort, Reason: "The agent run was cancelled."}, nil) {
						yield(Chunk{Type: ChunkDone}, nil)
					}
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
		messageMetadata := state.messageMetadata
		if len(state.externalCallIDs) > 0 {
			messageMetadata = withExternalCallIDs(messageMetadata, state.externalCallIDs)
		}
		if messageMetadata != nil && !yield(Chunk{
			Type: ChunkMessageMetadata, MessageMetadata: messageMetadata,
		}, nil) {
			return
		}
		if !yield(Chunk{Type: ChunkFinish, FinishReason: state.finishReason}, nil) {
			return
		}
		yield(Chunk{Type: ChunkDone}, nil)
	}
}

type transformState struct {
	sdkVersion        int
	step              bool
	finishReason      string
	messageMetadata   map[string]any
	externalCallIDs   []string
	partIDs           map[string]string
	toolIDs           map[string]string
	streamedCalls     map[string]ai.ToolCallPart
	streamedCallOrder []string
	streamedNative    map[string]bool
	invalidatedCalls  map[string]ai.ToolCallPart
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
			metadata := dumpPartMetadata(part.ID, "", part.ProviderName, part.ProviderDetails, "", nil)
			if !yield(Chunk{Type: ChunkTextStart, ID: id, ProviderMetadata: metadata}, nil) {
				return false
			}
			if part.Content != "" && !yield(Chunk{
				Type: ChunkTextDelta, ID: id, Delta: part.Content, ProviderMetadata: metadata,
			}, nil) {
				return false
			}
		case ai.ThinkingPart:
			metadata := dumpPartMetadata(
				part.ID, part.Signature, part.ProviderName, part.ProviderDetails, "", nil,
			)
			if !yield(Chunk{Type: ChunkReasoningStart, ID: id, ProviderMetadata: metadata}, nil) {
				return false
			}
			if part.Content != "" && !yield(Chunk{
				Type: ChunkReasoningDelta, ID: id, Delta: part.Content, ProviderMetadata: metadata,
			}, nil) {
				return false
			}
		case ai.FilePart:
			return yield(fileChunk(part), nil)
		case ai.CompactionPart:
			return yield(Chunk{Type: ChunkDataCompaction, Data: compactionData(part)}, nil)
		case ai.ToolCallPart:
			state.rememberStreamedCall(part, false)
			metadata := dumpPartMetadata(part.ID, "", part.ProviderName, part.ProviderDetails, part.ToolKind, nil)
			if !state.startTool(yield, value.PartID, part.ToolCallID, part.ToolName, string(part.Args), false, metadata) {
				return false
			}
		case ai.NativeToolCallPart:
			state.rememberStreamedCall(ai.ToolCallPart(part), true)
			metadata := dumpPartMetadata(part.ID, "", part.ProviderName, part.ProviderDetails, part.ToolKind, nil)
			if !state.startTool(yield, value.PartID, part.ToolCallID, part.ToolName, string(part.Args), true, metadata) {
				return false
			}
		case ai.NativeToolReturnPart:
			metadata := dumpPartMetadata("", "", part.ProviderName, part.ProviderDetails, part.ToolKind, nil)
			return state.toolOutput(yield, part.ToolCallID, part.Content, part.Outcome, true, metadata)
		}
	case ai.PartDeltaEvent:
		id := state.partIDs[value.PartID]
		switch delta := value.Delta.(type) {
		case ai.TextPartDelta:
			if delta.ContentDelta != "" {
				metadata := dumpPartMetadata("", "", delta.ProviderName, delta.ProviderDetails, "", nil)
				return yield(Chunk{
					Type: ChunkTextDelta, ID: id, Delta: delta.ContentDelta, ProviderMetadata: metadata,
				}, nil)
			}
		case ai.ThinkingPartDelta:
			if delta.ContentDelta != "" {
				metadata := dumpPartMetadata(
					"", delta.SignatureDelta, delta.ProviderName, delta.ProviderDetails, "", nil,
				)
				return yield(Chunk{
					Type: ChunkReasoningDelta, ID: id, Delta: delta.ContentDelta, ProviderMetadata: metadata,
				}, nil)
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
		switch part := value.Part.(type) {
		case ai.TextPart:
			metadata := dumpPartMetadata(part.ID, "", part.ProviderName, part.ProviderDetails, "", nil)
			return yield(Chunk{Type: ChunkTextEnd, ID: id, ProviderMetadata: metadata}, nil)
		case ai.ThinkingPart:
			metadata := dumpPartMetadata(
				part.ID, part.Signature, part.ProviderName, part.ProviderDetails, "", nil,
			)
			return yield(Chunk{Type: ChunkReasoningEnd, ID: id, ProviderMetadata: metadata}, nil)
		case ai.ToolCallPart:
			state.rememberStreamedCall(part, false)
		case ai.NativeToolCallPart:
			delete(state.streamedCalls, part.ToolCallID)
			delete(state.streamedNative, part.ToolCallID)
			return state.toolAvailable(yield, ai.ToolCallPart(part), true)
		}
	case ai.FunctionToolCallEvent:
		return state.functionToolCall(yield, value.Part, value.ArgsValid)
	case ai.OutputToolCallEvent:
		return state.functionToolCall(yield, value.Part, value.ArgsValid)
	case ai.FunctionToolResultEvent:
		if !state.requestResult(yield, value.Part, false) {
			return false
		}
		return state.resultChunks(yield, value.Part)
	case ai.OutputToolResultEvent:
		if !state.requestResult(yield, value.Part, false) {
			return false
		}
		return state.resultChunks(yield, value.Part)
	case ai.ToolAvailabilityDeltaEvent:
		return yield(Chunk{Type: ChunkDataToolAvailability, Data: toolAvailabilityData(value.Part)}, nil)
	case ai.DeferredToolRequestsEvent:
		for _, call := range value.Requests.Calls {
			if !slices.Contains(state.externalCallIDs, call.ToolCallID) {
				state.externalCallIDs = append(state.externalCallIDs, call.ToolCallID)
			}
		}
		if state.sdkVersion >= 6 {
			for _, call := range value.Requests.Approvals {
				if !yield(Chunk{
					Type: ChunkToolApprovalRequest, ApprovalID: call.ToolCallID, ToolCallID: call.ToolCallID,
				}, nil) {
					return false
				}
			}
		}
	case ai.DeferredToolResultsEvent:
		for callID := range value.Results.Calls {
			state.externalCallIDs = slices.DeleteFunc(state.externalCallIDs, func(pendingID string) bool {
				return pendingID == callID
			})
		}
	case ai.FinishEvent:
		state.finishReason = vercelFinishReason(value.FinishReason)
		state.messageMetadata = dumpMessageMetadata(value.Metadata, value.Timestamp)
		return state.finishStep(yield)
	}
	return true
}

func (state *transformState) resultChunks(yield func(Chunk, error) bool, part ai.RequestPart) bool {
	result, ok := part.(ai.ToolReturnPart)
	if !ok {
		return true
	}
	chunks, err := toolResultChunks(result.Metadata)
	if err != nil {
		yield(Chunk{Type: ChunkError, ErrorText: err.Error()}, err)
		return false
	}
	for _, chunk := range chunks {
		if !yield(chunk, nil) {
			return false
		}
	}
	return true
}

func (state *transformState) rememberStreamedCall(part ai.ToolCallPart, native bool) {
	if _, ok := state.streamedCalls[part.ToolCallID]; !ok {
		state.streamedCallOrder = append(state.streamedCallOrder, part.ToolCallID)
	}
	state.streamedCalls[part.ToolCallID] = part
	state.streamedNative[part.ToolCallID] = native
}

func (state *transformState) flushToolInputs(yield func(Chunk, error) bool) bool {
	for _, toolCallID := range state.streamedCallOrder {
		part, ok := state.streamedCalls[toolCallID]
		if !ok {
			continue
		}
		delete(state.streamedCalls, toolCallID)
		native := state.streamedNative[toolCallID]
		delete(state.streamedNative, toolCallID)
		if !state.toolAvailable(yield, part, native) {
			return false
		}
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
	yield func(Chunk, error) bool,
	partID string,
	toolCallID string,
	name string,
	args string,
	native bool,
	providerMetadata map[string]any,
) bool {
	state.toolIDs[partID] = toolCallID
	providerExecuted := native
	startMetadata := providerMetadata
	if state.sdkVersion < 6 {
		startMetadata = nil
	}
	if !yield(Chunk{
		Type: ChunkToolInputStart, ToolCallID: toolCallID, ToolName: name, ProviderExecuted: &providerExecuted,
		ProviderMetadata: startMetadata,
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

func (state *transformState) functionToolCall(
	yield func(Chunk, error) bool, part ai.ToolCallPart, argsValid *bool,
) bool {
	delete(state.streamedCalls, part.ToolCallID)
	delete(state.streamedNative, part.ToolCallID)
	if argsValid != nil && !*argsValid && state.sdkVersion >= 6 {
		state.invalidatedCalls[part.ToolCallID] = part
		return true
	}
	return state.toolAvailable(yield, part, false)
}

func (state *transformState) toolAvailable(
	yield func(Chunk, error) bool, part ai.ToolCallPart, native bool,
) bool {
	input := toolInput(part.Args)
	providerExecuted := native
	metadata := dumpPartMetadata(part.ID, "", part.ProviderName, part.ProviderDetails, part.ToolKind, nil)
	return yield(Chunk{
		Type: ChunkToolInputAvailable, ToolCallID: part.ToolCallID, ToolName: part.ToolName,
		Input: input, ProviderExecuted: &providerExecuted, ProviderMetadata: metadata,
	}, nil)
}

func (state *transformState) requestResult(yield func(Chunk, error) bool, part ai.RequestPart, native bool) bool {
	toolCallID := ""
	switch value := part.(type) {
	case ai.ToolReturnPart:
		toolCallID = value.ToolCallID
	case ai.RetryPromptPart:
		toolCallID = value.ToolCallID
	default:
		err := fmt.Errorf("vercel: unsupported tool result part %T", part)
		return yield(Chunk{Type: ChunkError, ErrorText: err.Error()}, err)
	}
	invalidated, invalid := state.invalidatedCalls[toolCallID]
	delete(state.invalidatedCalls, toolCallID)
	streamed, pending := state.streamedCalls[toolCallID]
	delete(state.streamedCalls, toolCallID)
	delete(state.streamedNative, toolCallID)
	if pending && !invalid && !state.toolAvailable(yield, streamed, native) {
		return false
	}
	if result, ok := part.(ai.ToolReturnPart); ok && result.Outcome == ai.ToolReturnOutcomeDenied &&
		state.sdkVersion >= 6 {
		return yield(Chunk{Type: ChunkToolOutputDenied, ToolCallID: toolCallID}, nil)
	}
	if invalid {
		input := toolInput(invalidated.Args)
		errorText := ""
		if result, ok := part.(ai.ToolReturnPart); ok {
			errorText = fmt.Sprint(result.Content)
		} else {
			errorText = part.(ai.RetryPromptPart).ModelResponse()
		}
		metadata := dumpPartMetadata(
			invalidated.ID, "", invalidated.ProviderName, invalidated.ProviderDetails, invalidated.ToolKind, nil,
		)
		return yield(Chunk{
			Type: ChunkToolInputError, ToolCallID: toolCallID, ToolName: invalidated.ToolName,
			Input: input, ErrorText: errorText, ProviderMetadata: metadata,
		}, nil)
	}
	if value, ok := part.(ai.ToolReturnPart); ok {
		return state.toolOutput(yield, value.ToolCallID, value.Content, value.Outcome, native, nil)
	}
	value := part.(ai.RetryPromptPart)
	return state.toolOutput(
		yield, value.ToolCallID, value.ModelResponse(), ai.ToolReturnOutcomeFailed, native, nil,
	)
}

func toolInput(args json.RawMessage) any {
	if len(args) == 0 {
		return nil
	}
	var input any
	if err := json.Unmarshal(args, &input); err != nil {
		return map[string]any{"INVALID_JSON": string(args)}
	}
	return input
}

func (state *transformState) toolOutput(
	yield func(Chunk, error) bool,
	toolCallID string,
	output any,
	outcome ai.ToolReturnOutcome,
	native bool,
	providerMetadata map[string]any,
) bool {
	providerExecuted := native
	if outcome == ai.ToolReturnOutcomeDenied {
		if state.sdkVersion >= 6 {
			return yield(Chunk{Type: ChunkToolOutputDenied, ToolCallID: toolCallID}, nil)
		}
		return yield(Chunk{
			Type: ChunkToolOutputAvailable, ToolCallID: toolCallID, Output: output,
			ProviderExecuted: &providerExecuted, ProviderMetadata: providerMetadata,
		}, nil)
	}
	if outcome == ai.ToolReturnOutcomeFailed {
		return yield(Chunk{
			Type: ChunkToolOutputError, ToolCallID: toolCallID, ErrorText: fmt.Sprint(output),
			ProviderExecuted: &providerExecuted, ProviderMetadata: providerMetadata,
		}, nil)
	}
	return yield(Chunk{
		Type: ChunkToolOutputAvailable, ToolCallID: toolCallID, Output: output,
		ProviderExecuted: &providerExecuted, ProviderMetadata: providerMetadata,
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
