package ai

import (
	"context"
	"fmt"
)

// RaiseContentFilterError treats every content-filtered model response as a
// terminal error, including responses that contain partial text.
type RaiseContentFilterError struct{}

// Setup implements Capability.
func (RaiseContentFilterError) Setup(*CapabilityRegistry) error { return nil }

// AfterModelRequest returns a ContentFilterError for filtered responses.
func (RaiseContentFilterError) AfterModelRequest(
	_ context.Context, _ *RunInfo, _ ModelRequestContext, response *ModelResponse,
) (*ModelResponse, error) {
	if response == nil || response.FinishReason != FinishReasonContentFilter {
		return response, nil
	}
	return response, newContentFilterError(response)
}

func newContentFilterError(response *ModelResponse) *ContentFilterError {
	details := response.ProviderDetails
	message := "Content filter triggered."
	switch {
	case details["finish_reason"] != nil:
		message = fmt.Sprintf("Content filter triggered. Finish reason: '%v'", details["finish_reason"])
	case details["block_reason"] != nil:
		message = fmt.Sprintf("Content filter triggered. Block reason: '%v'", details["block_reason"])
	case details["refusal"] != nil:
		message = fmt.Sprintf("Content filter triggered. Refusal: %q", details["refusal"])
	}
	cloned := cloneModelResponse(response)
	body, _ := MarshalMessages([]ModelMessage{*cloned})
	return &ContentFilterError{Message: message, response: *cloned, body: body}
}
