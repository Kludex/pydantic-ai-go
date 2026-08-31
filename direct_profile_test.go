package ai_test

import (
	"context"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestDirectRequestsResolveAutomaticOutputProfiles(t *testing.T) {
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModePrompted || params.OutputSchema == nil ||
			!strings.Contains(params.OutputPrompt, "Direct schema:") {
			t.Fatalf("unexpected direct parameters: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"ok"}`}}}, nil
	})
	model := ai.NewProfiledModel(base, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModePrompted, PromptedOutputTemplate: "Direct schema:",
	})
	params := ai.ModelRequestParams{
		OutputMode: ai.OutputModeAuto, OutputSchema: map[string]any{"type": "object"},
	}
	if _, err := ai.RequestModel(t.Context(), model, nil, params); err != nil {
		t.Fatal(err)
	}
}

func TestRequestModelReportsInvalidAutomaticProfile(t *testing.T) {
	model := invalidProfileModel{Model: fakes.NewTestModel()}
	_, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{
		OutputMode: ai.OutputModeAuto, OutputSchema: map[string]any{"type": "object"},
	})
	if err == nil || !strings.Contains(err.Error(), "invalid default output mode") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDirectStreamReportsInvalidAutomaticProfile(t *testing.T) {
	model := invalidProfileModel{Model: fakes.NewTestModel()}
	stream := ai.StreamModel(t.Context(), model, nil, ai.ModelRequestParams{
		OutputMode: ai.OutputModeAuto, OutputSchema: map[string]any{"type": "object"},
	})
	var streamErr error
	for _, err := range stream.Events() {
		streamErr = err
	}
	if streamErr == nil || !strings.Contains(streamErr.Error(), "invalid default output mode") || !stream.Done() ||
		stream.Err() == nil {
		t.Fatalf("unexpected stream state: err=%v done=%v terminal=%v", streamErr, stream.Done(), stream.Err())
	}
}
