package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

type lifecycleModel struct {
	name      string
	request   func([]ai.ModelMessage) (*ai.ModelResponse, error)
	log       *[]string
	openErr   error
	closeErr  error
	nilCloser bool
}

func (m *lifecycleModel) Name() string { return m.name }

func (m *lifecycleModel) Request(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return m.request(messages)
}

func (m *lifecycleModel) OpenModel(context.Context) (ai.ModelCloseFunc, error) {
	*m.log = append(*m.log, "open:"+m.name)
	if m.openErr != nil {
		return nil, m.openErr
	}
	if m.nilCloser {
		return nil, nil
	}
	return func(ctx context.Context) error {
		if ctx.Err() != nil {
			return errors.New("close context was canceled")
		}
		*m.log = append(*m.log, "close:"+m.name)
		return m.closeErr
	}, nil
}

func TestSelectedModelsOpenOnceAndCloseInReverseOrder(t *testing.T) {
	var log []string
	aRequests := 0
	modelA := &lifecycleModel{name: "a", log: &log}
	modelB := &lifecycleModel{name: "b", log: &log}
	modelA.request = func([]ai.ModelMessage) (*ai.ModelResponse, error) {
		aRequests++
		if aRequests == 1 {
			return lifecycleToolCall(), nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	}
	modelB.request = func([]ai.ModelMessage) (*ai.ModelResponse, error) { return lifecycleToolCall(), nil }
	agent := ai.NewAgent[deps, string](modelA)
	agent.AddModelSelector(func(_ context.Context, selection ai.ModelSelectionContext[deps]) (ai.ModelSelection, error) {
		if selection.Step == 2 {
			return ai.ModelSelection{Model: modelB}, nil
		}
		return ai.ModelSelection{Model: modelA}, nil
	})
	ai.AddSimpleTool(agent, "next", func(context.Context, struct{}) (string, error) { return "next", nil })
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected lifecycle run result=%+v err=%v", result, err)
	}
	if !slices.Equal(log, []string{"open:a", "open:b", "close:b", "close:a"}) {
		t.Fatalf("unexpected model lifecycle order: %v", log)
	}
}

func TestModelOpenFailureClosesPreviouslySelectedModels(t *testing.T) {
	var log []string
	modelA := &lifecycleModel{name: "a", log: &log, request: func([]ai.ModelMessage) (*ai.ModelResponse, error) {
		return lifecycleToolCall(), nil
	}}
	modelB := &lifecycleModel{name: "b", log: &log, openErr: errors.New("cannot connect")}
	modelB.request = func([]ai.ModelMessage) (*ai.ModelResponse, error) { return nil, errors.New("unreachable") }
	agent := ai.NewAgent[deps, string](modelA)
	agent.AddModelSelector(func(_ context.Context, selection ai.ModelSelectionContext[deps]) (ai.ModelSelection, error) {
		if selection.Step == 1 {
			return ai.ModelSelection{Model: modelA}, nil
		}
		return ai.ModelSelection{Model: modelB}, nil
	})
	ai.AddSimpleTool(agent, "next", func(context.Context, struct{}) (string, error) { return "next", nil })
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "open model \"b\": cannot connect") ||
		!slices.Equal(log, []string{"open:a", "open:b", "close:a"}) {
		t.Fatalf("unexpected model open failure err=%v log=%v", err, log)
	}
}

func TestModelCloseErrorsJoin(t *testing.T) {
	var log []string
	aRequests := 0
	modelA := &lifecycleModel{
		name: "a", log: &log, closeErr: errors.New("close a"),
		request: func([]ai.ModelMessage) (*ai.ModelResponse, error) {
			aRequests++
			if aRequests == 1 {
				return lifecycleToolCall(), nil
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		},
	}
	modelB := &lifecycleModel{
		name: "b", log: &log, closeErr: errors.New("close b"),
		request: func([]ai.ModelMessage) (*ai.ModelResponse, error) { return lifecycleToolCall(), nil },
	}
	agent := ai.NewAgent[deps, string](modelA)
	agent.AddModelSelector(func(_ context.Context, selection ai.ModelSelectionContext[deps]) (ai.ModelSelection, error) {
		if selection.Step == 2 {
			return ai.ModelSelection{Model: modelB}, nil
		}
		return ai.ModelSelection{Model: modelA}, nil
	})
	ai.AddSimpleTool(agent, "next", func(context.Context, struct{}) (string, error) { return "next", nil })
	result, err := agent.Run(t.Context(), "go", deps{})
	if result != nil || err == nil || !strings.Contains(err.Error(), "close a") ||
		!strings.Contains(err.Error(), "close b") ||
		!slices.Equal(log, []string{"open:a", "open:b", "close:b", "close:a"}) {
		t.Fatalf("unexpected model close result=%+v err=%v log=%v", result, err, log)
	}
}

func TestModelNilCloserStillOpensOnce(t *testing.T) {
	var log []string
	requests := 0
	model := &lifecycleModel{
		name: "model", log: &log, nilCloser: true,
		request: func([]ai.ModelMessage) (*ai.ModelResponse, error) {
			requests++
			if requests == 1 {
				return lifecycleToolCall(), nil
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		},
	}
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "next", func(context.Context, struct{}) (string, error) { return "next", nil })
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || !slices.Equal(log, []string{"open:model"}) {
		t.Fatalf("nil closer model reopened: result=%+v err=%v log=%v", result, err, log)
	}
}

func TestModelsCloseBeforeToolsets(t *testing.T) {
	var log []string
	model := &lifecycleModel{
		name: "model", log: &log,
		request: func([]ai.ModelMessage) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		},
	}
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(&lifecycleToolset{log: &log})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected resource-order run result=%+v err=%v", result, err)
	}
	modelClose := slices.Index(log, "close:model")
	toolsetClose := slices.Index(log, "close")
	if modelClose < 0 || toolsetClose < 0 || modelClose > toolsetClose {
		t.Fatalf("resources closed out of acquisition order: %v", log)
	}
}

func TestModelClosesWhenStreamConsumerStops(t *testing.T) {
	var log []string
	model := &lifecycleModel{
		name: "stream", log: &log,
		request: func([]ai.ModelMessage) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		},
	}
	agent := ai.NewAgent[deps, string](model)
	stream := agent.RunStream(t.Context(), "go", deps{})
	for range stream.Events() {
		break
	}
	if stream.Result() != nil || !slices.Equal(log, []string{"open:stream", "close:stream"}) {
		t.Fatalf("stream lifecycle did not close: result=%+v log=%v", stream.Result(), log)
	}
}

func lifecycleToolCall() *ai.ModelResponse {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
		ToolName: "next", ToolCallID: "next", Args: json.RawMessage(`{}`),
	}}}
}
