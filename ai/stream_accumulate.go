package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
)

type accumulatedPart struct {
	index           int
	id              string
	kind            ResponsePartKind
	text            string
	responseID      string
	signature       string
	toolName        string
	toolCallID      string
	toolKind        ToolPartKind
	toolArgs        string
	providerName    string
	providerDetails map[string]any
	speech          *SpeechPart
	complete        ResponsePart
}

func (p *accumulatedPart) responsePart() ResponsePart {
	if p.complete != nil {
		return cloneResponsePart(p.complete)
	}
	if p.kind == ResponsePartKindSpeech {
		return cloneSpeechPart(*p.speech)
	}
	if p.kind == ResponsePartKindText {
		return TextPart{
			Content: p.text, ID: p.responseID,
			ProviderName: p.providerName, ProviderDetails: cloneSchemaMap(p.providerDetails),
		}
	}
	if p.kind == ResponsePartKindThinking {
		return ThinkingPart{
			Content: p.text, ID: p.responseID, Signature: p.signature,
			ProviderName: p.providerName, ProviderDetails: cloneSchemaMap(p.providerDetails),
		}
	}
	if p.kind == ResponsePartKindCompaction {
		return CompactionPart{
			Content: p.text, ID: p.responseID,
			ProviderName: p.providerName, ProviderDetails: cloneSchemaMap(p.providerDetails),
		}
	}
	if p.kind == ResponsePartKindNativeToolCall {
		return NativeToolCallPart{
			ToolName: p.toolName, Args: json.RawMessage(p.toolArgs), ToolCallID: p.toolCallID,
			ToolKind: p.toolKind, ID: p.responseID,
			ProviderName: p.providerName, ProviderDetails: cloneSchemaMap(p.providerDetails),
		}
	}
	return ToolCallPart{
		ToolName: p.toolName, Args: json.RawMessage(p.toolArgs), ToolCallID: p.toolCallID,
		ToolKind: p.toolKind, ID: p.responseID,
		ProviderName: p.providerName, ProviderDetails: cloneSchemaMap(p.providerDetails),
	}
}

// accumulate replays provider deltas into a ModelResponse and emits
// normalized lifecycle events. Parts with IDs are tracked independently, so
// deltas may be interleaved while response order follows first appearance.
func accumulate(
	events iter.Seq2[ModelStreamEvent, error], params ModelRequestParams, emit func(StreamEvent) bool,
	observe func(*ModelResponse),
) (*ModelResponse, error) {
	response := &ModelResponse{}
	parts := make([]*accumulatedPart, 0)
	partsByID := map[string]*accumulatedPart{}
	var current *accumulatedPart
	var lastStarted *accumulatedPart
	finalResultSent := false

	notify := func() {
		if observe != nil {
			observe(cloneModelResponse(response))
		}
	}
	emitEvent := func(event StreamEvent) error {
		notify()
		if emit != nil && !emit(event) {
			return context.Canceled
		}
		return nil
	}
	newPart := func(id string, kind ResponsePartKind) *accumulatedPart {
		part := &accumulatedPart{index: len(parts), id: id, kind: kind}
		parts = append(parts, part)
		if id != "" {
			partsByID[id] = part
		}
		current = part
		return part
	}
	partForDelta := func(id string, kind ResponsePartKind) (*accumulatedPart, bool, error) {
		if id != "" {
			if part, ok := partsByID[id]; ok {
				if part.kind != kind {
					return nil, false, &UnexpectedModelBehaviorError{
						Message: fmt.Sprintf("stream part %q changed from %s to %s", id, part.kind, kind),
					}
				}
				current = part
				return part, false, nil
			}
			return newPart(id, kind), true, nil
		}
		if current != nil && current.kind == kind {
			return current, false, nil
		}
		return newPart("", kind), true, nil
	}
	endLastPart := func(next ResponsePartKind) error {
		if lastStarted == nil {
			return nil
		}
		return emitEvent(PartEndEvent{
			Index: lastStarted.index, PartID: lastStarted.id,
			Part: lastStarted.responsePart(), NextPartKind: next,
		})
	}
	startPart := func(part *accumulatedPart) error {
		if err := endLastPart(part.kind); err != nil {
			return err
		}
		var previous ResponsePartKind
		if lastStarted != nil {
			previous = lastStarted.kind
		}
		lastStarted = part
		if err := emitEvent(PartStartEvent{
			Index: part.index, PartID: part.id, Part: part.responsePart(), PreviousPartKind: previous,
		}); err != nil {
			return err
		}
		if finalResultSent {
			return nil
		}
		switch value := part.responsePart().(type) {
		case TextPart:
			if params.AllowText {
				finalResultSent = true
				return emitEvent(FinalResultEvent{})
			}
		case SpeechPart:
			if params.AllowText && value.Content() != "" {
				finalResultSent = true
				return emitEvent(FinalResultEvent{})
			}
		case ToolCallPart:
			if params.OutputTool != nil && value.ToolName == params.OutputTool.Name {
				finalResultSent = true
				return emitEvent(FinalResultEvent{ToolName: value.ToolName, ToolCallID: value.ToolCallID})
			}
		}
		return nil
	}
	materialize := func() {
		response.Parts = nil
		for _, part := range parts {
			switch part.kind {
			case ResponsePartKindText, ResponsePartKindThinking:
				if part.text != "" || part.responseID != "" || part.signature != "" ||
					part.providerName != "" || len(part.providerDetails) > 0 {
					response.Parts = append(response.Parts, part.responsePart())
				}
			case ResponsePartKindSpeech, ResponsePartKindFile, ResponsePartKindCompaction, ResponsePartKindToolCall,
				ResponsePartKindNativeToolCall, ResponsePartKindNativeToolReturn:
				response.Parts = append(response.Parts, part.responsePart())
			}
		}
	}
	partialResponse := func() *ModelResponse {
		materialize()
		if len(response.Parts) == 0 && response.ProviderResponseID == "" && response.ProviderName == "" &&
			response.ModelName == "" && response.State == "" && response.Usage.IsZero() {
			return nil
		}
		if response.State == "" {
			response.State = ModelResponseStateIncomplete
		}
		return response
	}

	for event, err := range events {
		if err != nil {
			return partialResponse(), err
		}
		switch event := event.(type) {
		case ResponseMetadataEvent:
			response.Usage = event.Usage.Clone()
			response.ModelName = event.ModelName
			response.Timestamp = event.Timestamp
			response.ProviderName = event.ProviderName
			response.ProviderURL = event.ProviderURL
			response.ProviderDetails = cloneSchemaMap(event.ProviderDetails)
			response.Metadata = cloneSchemaMap(event.Metadata)
			response.ProviderResponseID = event.ProviderResponseID
			response.FinishReason = event.FinishReason
			response.State = event.State
			notify()
		case TextDeltaEvent:
			part, started, err := partForDelta(event.PartID, ResponsePartKindText)
			if err != nil {
				return nil, err
			}
			part.text += event.Delta
			if event.ID != "" {
				part.responseID = event.ID
			}
			if event.ProviderName != "" {
				part.providerName = event.ProviderName
			}
			part.providerDetails = mergeProviderDetails(part.providerDetails, event.ProviderDetails)
			if started {
				if err := startPart(part); err != nil {
					return partialResponse(), err
				}
			} else if err := emitEvent(PartDeltaEvent{
				Index: part.index, PartID: part.id, Delta: TextPartDelta{
					ContentDelta: event.Delta, ProviderName: event.ProviderName,
					ProviderDetails: cloneSchemaMap(event.ProviderDetails),
				},
			}); err != nil {
				return partialResponse(), err
			}
		case SpeechDeltaEvent:
			part, started, err := partForDelta(event.PartID, ResponsePartKindSpeech)
			if err != nil {
				return nil, err
			}
			if started {
				speech := cloneSpeechPart(event.Part)
				if speech.Speaker == "" {
					speech.Speaker = event.Delta.Speaker
				}
				part.speech = &speech
			}
			speech := applySpeechPartDelta(part.responsePart().(SpeechPart), event.Delta)
			part.speech = &speech
			if started {
				if err := startPart(part); err != nil {
					return partialResponse(), err
				}
			} else {
				delta := event.Delta
				delta.Transcript = clonePointer(delta.Transcript)
				delta.AudioChunk = append([]byte(nil), delta.AudioChunk...)
				if err := emitEvent(PartDeltaEvent{
					Index: part.index, PartID: part.id, Delta: delta,
				}); err != nil {
					return partialResponse(), err
				}
			}
		case ThinkingDeltaEvent:
			part, started, err := partForDelta(event.PartID, ResponsePartKindThinking)
			if err != nil {
				return nil, err
			}
			part.text += event.Delta
			if event.ID != "" {
				part.responseID = event.ID
			}
			if event.SignatureDelta != "" {
				part.signature = event.SignatureDelta
			}
			if event.ProviderName != "" {
				part.providerName = event.ProviderName
			}
			part.providerDetails = mergeProviderDetails(part.providerDetails, event.ProviderDetails)
			if started {
				if err := startPart(part); err != nil {
					return partialResponse(), err
				}
			} else if err := emitEvent(PartDeltaEvent{
				Index: part.index, PartID: part.id, Delta: ThinkingPartDelta{
					ContentDelta: event.Delta, SignatureDelta: event.SignatureDelta,
					ProviderName:    event.ProviderName,
					ProviderDetails: cloneSchemaMap(event.ProviderDetails),
				},
			}); err != nil {
				return partialResponse(), err
			}
		case CompactionEvent:
			if event.PartID != "" {
				if _, exists := partsByID[event.PartID]; exists {
					return nil, &UnexpectedModelBehaviorError{
						Message: fmt.Sprintf("duplicate compaction stream part %q", event.PartID),
					}
				}
			}
			part := newPart(event.PartID, ResponsePartKindCompaction)
			part.text = event.Content
			part.responseID = event.ID
			part.providerName = event.ProviderName
			part.providerDetails = cloneSchemaMap(event.ProviderDetails)
			if err := startPart(part); err != nil {
				return partialResponse(), err
			}
		case ToolCallStartEvent:
			if event.PartID != "" {
				if _, exists := partsByID[event.PartID]; exists {
					return nil, &UnexpectedModelBehaviorError{
						Message: fmt.Sprintf("duplicate tool call start for stream part %q", event.PartID),
					}
				}
			}
			kind := ResponsePartKindToolCall
			if event.Native {
				kind = ResponsePartKindNativeToolCall
			}
			part := newPart(event.PartID, kind)
			part.toolName = event.ToolName
			part.toolCallID = event.ToolCallID
			part.toolKind = event.ToolKind
			part.responseID = event.ID
			part.providerName = event.ProviderName
			part.providerDetails = cloneSchemaMap(event.ProviderDetails)
			if err := startPart(part); err != nil {
				return partialResponse(), err
			}
		case ToolCallDeltaEvent:
			var part *accumulatedPart
			if event.PartID != "" {
				part = partsByID[event.PartID]
			} else if current != nil &&
				(current.kind == ResponsePartKindToolCall || current.kind == ResponsePartKindNativeToolCall) {
				part = current
			}
			if part == nil ||
				(part.kind != ResponsePartKindToolCall && part.kind != ResponsePartKindNativeToolCall) {
				return nil, &UnexpectedModelBehaviorError{Message: "tool call delta before tool call start"}
			}
			current = part
			if event.ToolCallID != "" && part.toolCallID != "" && event.ToolCallID != part.toolCallID {
				return nil, &UnexpectedModelBehaviorError{Message: fmt.Sprintf(
					"tool call ID changed from %q to %q", part.toolCallID, event.ToolCallID,
				)}
			}
			if event.ToolCallID != "" {
				part.toolCallID = event.ToolCallID
			}
			part.toolArgs += event.ArgsDelta
			var delta ResponsePartDelta = ToolCallPartDelta{
				ArgsDelta: event.ArgsDelta, ToolCallID: event.ToolCallID,
			}
			if part.kind == ResponsePartKindNativeToolCall {
				delta = NativeToolCallPartDelta{
					ArgsDelta: event.ArgsDelta, ToolCallID: event.ToolCallID,
				}
			}
			if err := emitEvent(PartDeltaEvent{Index: part.index, PartID: part.id, Delta: delta}); err != nil {
				return partialResponse(), err
			}
		case FileEvent:
			if event.PartID != "" {
				if part, exists := partsByID[event.PartID]; exists {
					if part.kind != ResponsePartKindFile || !event.Replace {
						return nil, &UnexpectedModelBehaviorError{
							Message: fmt.Sprintf("duplicate file stream part %q", event.PartID),
						}
					}
					part.complete = cloneResponsePart(event.Part)
					current = part
					if err := emitEvent(PartDeltaEvent{
						Index: part.index, PartID: part.id, Delta: FilePartDelta{Part: event.Part},
					}); err != nil {
						return partialResponse(), err
					}
					continue
				}
			}
			part := newPart(event.PartID, ResponsePartKindFile)
			part.complete = cloneResponsePart(event.Part)
			if err := startPart(part); err != nil {
				return partialResponse(), err
			}
		case NativeToolReturnEvent:
			if event.PartID != "" {
				if _, exists := partsByID[event.PartID]; exists {
					return nil, &UnexpectedModelBehaviorError{
						Message: fmt.Sprintf("duplicate native tool return stream part %q", event.PartID),
					}
				}
			}
			part := newPart(event.PartID, ResponsePartKindNativeToolReturn)
			part.complete = cloneResponsePart(event.Part)
			if err := startPart(part); err != nil {
				return partialResponse(), err
			}
		case FinishEvent:
			if err := endLastPart(""); err != nil {
				return partialResponse(), err
			}
			materialize()
			if event.Parts != nil {
				snapshot := cloneModelResponse(&ModelResponse{Parts: event.Parts})
				response.Parts = snapshot.Parts
			}
			response.Usage = event.Usage
			response.ModelName = event.ModelName
			response.Timestamp = event.Timestamp
			response.ProviderName = event.ProviderName
			response.ProviderURL = event.ProviderURL
			response.ProviderDetails = cloneSchemaMap(event.ProviderDetails)
			response.Metadata = cloneSchemaMap(event.Metadata)
			response.ProviderResponseID = event.ProviderResponseID
			response.FinishReason = event.FinishReason
			response.State = event.State
			if err := emitEvent(event); err != nil {
				return response, err
			}
			return response, nil
		default:
			return nil, fmt.Errorf("ai: unknown model stream event type %T", event)
		}
	}
	return partialResponse(), &UnexpectedModelBehaviorError{Message: "stream ended without a finish event"}
}
