package ai_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestEmptyContentFilterResponseFails(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			FinishReason: ai.FinishReasonContentFilter,
			ProviderDetails: map[string]any{
				"block_reason": "PROHIBITED_CONTENT",
			},
		}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	_, err := agent.Run(t.Context(), "filtered", struct{}{})
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) ||
		filtered.Error() != "ai: Content filter triggered. Block reason: 'PROHIBITED_CONTENT'" {
		t.Fatalf("unexpected content filter error: %v", err)
	}
	response := filtered.Response()
	if response.FinishReason != ai.FinishReasonContentFilter || response.RunID == "" || response.ConversationID == "" {
		t.Fatalf("filtered response was not retained: %+v", response)
	}
	response.ProviderDetails["block_reason"] = "mutated"
	if filtered.Response().ProviderDetails["block_reason"] != "PROHIBITED_CONTENT" {
		t.Fatal("filtered response was not detached")
	}
	body := filtered.Body()
	messages, decodeErr := ai.UnmarshalMessages(body)
	if decodeErr != nil || len(messages) != 1 {
		t.Fatalf("unexpected interoperable body: messages=%+v err=%v", messages, decodeErr)
	}
	body[0] = 'x'
	if filtered.Body()[0] != '[' {
		t.Fatal("filtered body was not detached")
	}
}

func TestRaiseContentFilterErrorCapability(t *testing.T) {
	for _, test := range []struct {
		name    string
		details map[string]any
		want    string
	}{
		{
			name: "raw finish reason", details: map[string]any{"finish_reason": "sensitive"},
			want: "Finish reason: 'sensitive'",
		},
		{
			name: "block reason", details: map[string]any{"block_reason": "SAFETY"},
			want: "Block reason: 'SAFETY'",
		},
		{
			name: "refusal", details: map[string]any{"refusal": "I cannot comply."},
			want: `Refusal: "I cannot comply."`,
		},
		{name: "generic", want: "Content filter triggered."},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{
					Parts:           []ai.ResponsePart{ai.TextPart{Content: "partial"}},
					FinishReason:    ai.FinishReasonContentFilter,
					ProviderDetails: test.details,
				}, nil
			})
			agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.RaiseContentFilterError{}))
			_, err := agent.Run(t.Context(), "filtered", struct{}{})
			var filtered *ai.ContentFilterError
			if !errors.As(err, &filtered) || !strings.Contains(filtered.Error(), test.want) {
				t.Fatalf("unexpected content filter error: %v", err)
			}
			if filtered.Response().Text() != "partial" || len(filtered.Body()) == 0 {
				t.Fatalf("partial response was not retained: %+v", filtered.Response())
			}
		})
	}
}

func TestRaiseContentFilterErrorNoop(t *testing.T) {
	capability := ai.RaiseContentFilterError{}
	if response, err := capability.AfterModelRequest(t.Context(), nil, ai.ModelRequestContext{}, nil); err != nil ||
		response != nil {
		t.Fatalf("nil response changed: response=%v err=%v", response, err)
	}
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, FinishReason: ai.FinishReasonStop,
		}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	result, err := agent.Run(t.Context(), "allowed", struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected allowed result: %+v err=%v", result, err)
	}
}

type filteredStreamingModel struct{}

func (filteredStreamingModel) Name() string { return "filtered-stream" }

func (filteredStreamingModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{}, nil
}

func (filteredStreamingModel) StreamRequest(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		if !yield(ai.TextDeltaEvent{PartID: "text", Delta: "partial"}, nil) {
			return
		}
		yield(ai.FinishEvent{
			FinishReason: ai.FinishReasonContentFilter,
			ProviderDetails: map[string]any{
				"finish_reason": "sensitive",
			},
			State: ai.ModelResponseStateComplete,
		}, nil)
	}, nil
}

func TestRaiseContentFilterErrorStreaming(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](
		filteredStreamingModel{}, ai.WithCapabilities(ai.RaiseContentFilterError{}),
	)
	stream := agent.RunStream(t.Context(), "filtered", struct{}{})
	var streamErr error
	for _, err := range stream.Events() {
		if err != nil {
			streamErr = err
		}
	}
	var filtered *ai.ContentFilterError
	if !errors.As(streamErr, &filtered) || filtered.Response().Text() != "partial" {
		t.Fatalf("unexpected streamed filter error: %v response=%+v", streamErr, filtered)
	}
}

func TestContentFilterBodySerializationFailure(t *testing.T) {
	response := &ai.ModelResponse{
		Parts:           []ai.ResponsePart{ai.TextPart{Content: "partial"}},
		FinishReason:    ai.FinishReasonContentFilter,
		ProviderDetails: map[string]any{"unsupported": make(chan struct{})},
	}
	_, err := (ai.RaiseContentFilterError{}).AfterModelRequest(
		t.Context(), nil, ai.ModelRequestContext{}, response,
	)
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) || filtered.Body() != nil || filtered.Response().Text() != "partial" {
		t.Fatalf("unexpected serialization failure fallback: %v", err)
	}
}
