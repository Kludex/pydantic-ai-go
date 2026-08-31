package ai

import (
	"context"
	"fmt"
	"slices"
	"time"
)

const (
	// MaxGenerationContinuations bounds fresh suspended-generation segments in one model turn.
	MaxGenerationContinuations = 10
	// MaxBackgroundPolls bounds repeated polls of one provider response ID.
	MaxBackgroundPolls = 1000
)

// ModelContinuationDelayer optionally delays reissuing a suspended model response.
// Providers that poll a server-side job use this to avoid busy polling.
type ModelContinuationDelayer interface {
	ContinuationDelay(response ModelResponse) time.Duration
}

// SuspendedResponseCanceler optionally cancels a server-side suspended response.
// The agent calls it best-effort when a continuation is abandoned.
type SuspendedResponseCanceler interface {
	CancelSuspendedResponse(ctx context.Context, response ModelResponse) error
}

type continuationMergeMode uint8

const (
	continuationAccumulate continuationMergeMode = iota
	continuationReplaceSameID
	continuationReplaceNew
)

func modelResponseMergeMode(existing, next *ModelResponse) continuationMergeMode {
	if hasReplacePreviousResponseMarker(next.Metadata) {
		return continuationReplaceNew
	}
	if existing.ProviderResponseID != "" && existing.ProviderResponseID == next.ProviderResponseID {
		return continuationReplaceSameID
	}
	if existing.ModelName != "" && next.ModelName != "" && existing.ModelName != next.ModelName {
		return continuationReplaceNew
	}
	return continuationAccumulate
}

func mergeModelResponses(existing, next *ModelResponse) (*ModelResponse, continuationMergeMode) {
	mode := modelResponseMergeMode(existing, next)
	merged := cloneModelResponse(next)
	switch mode {
	case continuationReplaceSameID:
	case continuationReplaceNew:
		merged.Usage = existing.Usage.Clone()
		merged.Usage.Add(next.Usage)
	default:
		merged.Parts = append(cloneModelResponse(existing).Parts, merged.Parts...)
		merged.Usage = existing.Usage.Clone()
		merged.Usage.Add(next.Usage)
		if merged.ProviderResponseID == "" {
			merged.ProviderResponseID = existing.ProviderResponseID
		}
	}
	merged.ProviderDetails = mergeProviderDetails(existing.ProviderDetails, merged.ProviderDetails)
	merged.Metadata = mergeProviderDetails(existing.Metadata, merged.Metadata)
	stripReplacePreviousResponseMarker(merged.Metadata)
	return merged, mode
}

func cloneModelResponse(response *ModelResponse) *ModelResponse {
	cloned := *response
	cloned.Parts = slices.Clone(response.Parts)
	for index, part := range cloned.Parts {
		switch part := part.(type) {
		case TextPart:
			part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
			cloned.Parts[index] = part
		case ThinkingPart:
			part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
			cloned.Parts[index] = part
		case FilePart:
			part.Content.Data = slices.Clone(part.Content.Data)
			part.Content.VendorMetadata = cloneSchemaMap(part.Content.VendorMetadata)
			part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
			cloned.Parts[index] = part
		case ToolCallPart:
			part.Args = slices.Clone(part.Args)
			part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
			cloned.Parts[index] = part
		case NativeToolCallPart:
			part.Args = slices.Clone(part.Args)
			part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
			cloned.Parts[index] = part
		case NativeToolReturnPart:
			part.Content = cloneSchemaValue(part.Content)
			part.Metadata = cloneSchemaMap(part.Metadata)
			part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
			cloned.Parts[index] = part
		case CompactionPart:
			part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
			cloned.Parts[index] = part
		}
	}
	cloned.Usage = response.Usage.Clone()
	cloned.ProviderDetails = cloneSchemaMap(response.ProviderDetails)
	cloned.Metadata = cloneSchemaMap(response.Metadata)
	return &cloned
}

func hasReplacePreviousResponseMarker(metadata map[string]any) bool {
	namespace, ok := metadata["__pydantic_ai__"].(map[string]any)
	if !ok {
		return false
	}
	replace, _ := namespace["replace_previous_response"].(bool)
	return replace
}

func stripReplacePreviousResponseMarker(metadata map[string]any) {
	namespace, ok := metadata["__pydantic_ai__"].(map[string]any)
	if !ok {
		return
	}
	delete(namespace, "replace_previous_response")
	if len(namespace) == 0 {
		delete(metadata, "__pydantic_ai__")
	}
}

func reindexContinuationEvent(event StreamEvent, offset int) StreamEvent {
	if offset == 0 {
		return event
	}
	switch event := event.(type) {
	case PartStartEvent:
		event.Index += offset
		return event
	case PartDeltaEvent:
		event.Index += offset
		return event
	case PartEndEvent:
		event.Index += offset
		return event
	default:
		return event
	}
}

func continuationLimitError(response *ModelResponse, mode continuationMergeMode, generationCount, pollCount *int) error {
	if mode == continuationReplaceSameID {
		(*pollCount)++
		if *pollCount > MaxBackgroundPolls {
			return &UnexpectedModelBehaviorError{Message: fmt.Sprintf(
				"model response for job %q remained suspended after polling the maximum of %d times",
				response.ProviderResponseID, MaxBackgroundPolls,
			)}
		}
		return nil
	}
	(*generationCount)++
	if *generationCount > MaxGenerationContinuations {
		return &UnexpectedModelBehaviorError{Message: fmt.Sprintf(
			"model response %q was suspended more than the maximum of %d times",
			response.ProviderResponseID, MaxGenerationContinuations,
		)}
	}
	return nil
}
