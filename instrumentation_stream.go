package ai

import "time"

type telemetryStreamAccumulator struct {
	response *ModelResponse
	parts    []*accumulatedPart
	byID     map[string]*accumulatedPart
	current  *accumulatedPart
}

func newTelemetryStreamAccumulator() *telemetryStreamAccumulator {
	return &telemetryStreamAccumulator{response: &ModelResponse{}, byID: map[string]*accumulatedPart{}}
}

func (accumulator *telemetryStreamAccumulator) part(id string, kind ResponsePartKind) *accumulatedPart {
	if id != "" {
		if part := accumulator.byID[id]; part != nil && part.kind == kind {
			accumulator.current = part
			return part
		}
	}
	if id == "" && accumulator.current != nil && accumulator.current.kind == kind {
		return accumulator.current
	}
	part := &accumulatedPart{index: len(accumulator.parts), id: id, kind: kind}
	accumulator.parts = append(accumulator.parts, part)
	if id != "" {
		accumulator.byID[id] = part
	}
	accumulator.current = part
	return part
}

func (accumulator *telemetryStreamAccumulator) observe(event ModelStreamEvent) {
	switch event := event.(type) {
	case ResponseMetadataEvent:
		accumulator.applyMetadata(event.Usage, event.ModelName, event.Timestamp, event.ProviderName,
			event.ProviderURL, event.ProviderDetails, event.Metadata, event.ProviderResponseID, event.FinishReason, event.State)
	case TextDeltaEvent:
		part := accumulator.part(event.PartID, ResponsePartKindText)
		part.text += event.Delta
		part.responseID = firstNonEmpty(event.ID, part.responseID)
		part.providerName = firstNonEmpty(event.ProviderName, part.providerName)
		part.providerDetails = mergeProviderDetails(part.providerDetails, event.ProviderDetails)
	case ThinkingDeltaEvent:
		part := accumulator.part(event.PartID, ResponsePartKindThinking)
		part.text += event.Delta
		part.signature = event.SignatureDelta
		part.responseID = firstNonEmpty(event.ID, part.responseID)
		part.providerName = firstNonEmpty(event.ProviderName, part.providerName)
		part.providerDetails = mergeProviderDetails(part.providerDetails, event.ProviderDetails)
	case CompactionEvent:
		part := accumulator.part(event.PartID, ResponsePartKindCompaction)
		part.text = event.Content
		part.responseID = event.ID
		part.providerName = event.ProviderName
		part.providerDetails = cloneSchemaMap(event.ProviderDetails)
	case ToolCallStartEvent:
		kind := ResponsePartKindToolCall
		if event.Native {
			kind = ResponsePartKindNativeToolCall
		}
		part := accumulator.part(event.PartID, kind)
		part.toolName = event.ToolName
		part.toolCallID = event.ToolCallID
		part.toolKind = event.ToolKind
		part.responseID = event.ID
		part.providerName = event.ProviderName
		part.providerDetails = cloneSchemaMap(event.ProviderDetails)
	case ToolCallDeltaEvent:
		part := accumulator.current
		if event.PartID != "" {
			part = accumulator.byID[event.PartID]
		}
		if part != nil {
			part.toolArgs += event.ArgsDelta
			part.toolCallID = firstNonEmpty(event.ToolCallID, part.toolCallID)
		}
	case NativeToolReturnEvent:
		part := accumulator.part(event.PartID, ResponsePartKindNativeToolReturn)
		part.complete = cloneResponsePart(event.Part)
	case FinishEvent:
		if event.Parts != nil {
			accumulator.response.Parts = cloneModelResponse(&ModelResponse{Parts: event.Parts}).Parts
		}
		accumulator.applyMetadata(event.Usage, event.ModelName, event.Timestamp, event.ProviderName,
			event.ProviderURL, event.ProviderDetails, event.Metadata, event.ProviderResponseID, event.FinishReason, event.State)
	}
}

func (accumulator *telemetryStreamAccumulator) applyMetadata(
	usage Usage, modelName string, timestamp time.Time, providerName, providerURL string,
	providerDetails, metadata map[string]any, responseID string, finishReason FinishReason, state ModelResponseState,
) {
	accumulator.response.Usage = usage.Clone()
	accumulator.response.ModelName = modelName
	accumulator.response.Timestamp = timestamp
	accumulator.response.ProviderName = providerName
	accumulator.response.ProviderURL = providerURL
	accumulator.response.ProviderDetails = cloneSchemaMap(providerDetails)
	accumulator.response.Metadata = cloneSchemaMap(metadata)
	accumulator.response.ProviderResponseID = responseID
	accumulator.response.FinishReason = finishReason
	accumulator.response.State = state
}

func (accumulator *telemetryStreamAccumulator) snapshot() *ModelResponse {
	if accumulator.response.Parts == nil {
		for _, part := range accumulator.parts {
			accumulator.response.Parts = append(accumulator.response.Parts, part.responsePart())
		}
	}
	if accumulator.response.State == "" {
		accumulator.response.State = ModelResponseStateIncomplete
	}
	return cloneModelResponse(accumulator.response)
}

func firstNonEmpty(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}
