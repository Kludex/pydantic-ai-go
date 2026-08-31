package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type lifecycleToolset struct {
	log      *[]string
	step     int
	closeErr error
}

func (t *lifecycleToolset) ToolsetID() string { return "remote" }

func (t *lifecycleToolset) ForRun(
	_ context.Context, rc *ai.RunContext[deps],
) (ai.Toolset[deps], error) {
	*t.log = append(*t.log, fmt.Sprintf("for-run:%d", rc.RunStep))
	return &lifecycleToolset{log: t.log, closeErr: t.closeErr}, nil
}

func (t *lifecycleToolset) OpenToolset(
	_ context.Context, rc *ai.RunContext[deps],
) (ai.Toolset[deps], ai.ToolsetCloseFunc, error) {
	*t.log = append(*t.log, fmt.Sprintf("open:%d", rc.RunStep))
	opened := *t
	return &opened, func(context.Context) error {
		*t.log = append(*t.log, "close")
		return t.closeErr
	}, nil
}

func (t *lifecycleToolset) ForRunStep(
	_ context.Context, rc *ai.RunContext[deps],
) (ai.Toolset[deps], error) {
	*t.log = append(*t.log, fmt.Sprintf("step:%d", rc.RunStep))
	next := *t
	next.step = rc.RunStep
	return &next, nil
}

func (t *lifecycleToolset) ToolsetInstructions(
	_ context.Context, _ *ai.RunContext[deps],
) ([]ai.InstructionPart, error) {
	return []ai.InstructionPart{{Content: fmt.Sprintf("remote step %d", t.step), Dynamic: true}}, nil
}

func (t *lifecycleToolset) Tools(
	_ context.Context, _ *ai.RunContext[deps],
) ([]ai.Tool[deps], error) {
	*t.log = append(*t.log, fmt.Sprintf("tools:%d", t.step))
	work := ai.NewTool("work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		*t.log = append(*t.log, fmt.Sprintf("call:%s:%d", rc.ToolName, rc.RunStep))
		return "worked", nil
	}, ai.WithDescription(fmt.Sprintf("step %d", t.step)))
	owned := ai.NewRawTool[deps](ai.ToolDefinition{
		Name: "owned", ToolsetID: "owner", Schema: map[string]any{"type": "object"},
	}, func(context.Context, json.RawMessage) (any, error) {
		return "owned", nil
	})
	return []ai.Tool[deps]{work, owned}, nil
}

func TestToolsetRunStepAndResourceLifecycle(t *testing.T) {
	var log []string
	remote := &lifecycleToolset{log: &log}
	local := ai.NewFunctionToolset(ai.NewSimpleTool[deps](
		"local", func(context.Context, struct{}) (string, error) { return "local", nil },
	))
	wrapped := ai.Toolset[deps](ai.PrefixToolset[deps](remote, "remote"))
	wrapped = ai.FilterToolset(wrapped, func(
		context.Context, *ai.RunContext[deps], ai.ToolDefinition,
	) (bool, error) {
		return true, nil
	})
	wrapped = ai.RenameToolset(wrapped, map[string]string{})
	wrapped = ai.PrepareToolset(wrapped, func(
		_ context.Context, _ *ai.RunContext[deps], definitions []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		return definitions, nil
	})
	wrapped = ai.WithToolsetMaxRetries(wrapped, 2)
	wrapped = ai.WithToolsetTimeout(wrapped, time.Second)
	wrapped = ai.DeferLoadingToolset(wrapped, "missing")
	wrapped = ai.WithToolSearch(wrapped, ai.ToolSearchConfig[deps]{})
	wrapped = ai.SetToolsetMetadata(wrapped, map[string]any{"remote": true})
	toolset := ai.CombineToolsets(wrapped, local)
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if params.Instructions != fmt.Sprintf("remote step %d", request) {
			t.Fatalf("unexpected toolset instructions on step %d: %q", request, params.Instructions)
		}
		var remoteDefinition ai.ToolDefinition
		var ownedDefinition ai.ToolDefinition
		for _, definition := range params.Tools {
			switch definition.Name {
			case "remote_work":
				remoteDefinition = definition
			case "remote_owned":
				ownedDefinition = definition
			}
		}
		if remoteDefinition.Description != fmt.Sprintf("step %d", request) ||
			remoteDefinition.ToolsetID != "remote" || ownedDefinition.ToolsetID != "owner" {
			t.Fatalf(
				"unexpected remote definitions on step %d: work=%+v owned=%+v",
				request, remoteDefinition, ownedDefinition,
			)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "remote_work", ToolCallID: "work", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithInstructions(""))
	agent.AddToolset(toolset)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || request != 2 {
		t.Fatalf("unexpected lifecycle result: %+v requests=%d", result, request)
	}
	want := []string{
		"for-run:0", "open:0", "step:1", "tools:1", "call:work:1", "step:2", "tools:2", "close",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected lifecycle order: got %v want %v", log, want)
	}
}

func TestToolsetCloseFailureFailsRunAndStream(t *testing.T) {
	closeErr := errors.New("cannot close remote")
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})

	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			var log []string
			agent := ai.NewAgent[deps, string](model)
			agent.AddToolset(&lifecycleToolset{log: &log, closeErr: closeErr})
			if !stream {
				result, err := agent.Run(t.Context(), "go", deps{})
				if result != nil || !errors.Is(err, closeErr) {
					t.Fatalf("unexpected close result=%+v err=%v", result, err)
				}
				return
			}
			var got error
			run := agent.RunStream(t.Context(), "go", deps{})
			for _, err := range run.Events() {
				if err != nil {
					got = err
				}
			}
			if run.Result() != nil || !errors.Is(got, closeErr) {
				t.Fatalf("unexpected streamed close result=%+v err=%v", run.Result(), got)
			}
		})
	}
}

type openingToolset struct {
	name    string
	log     *[]string
	openErr error
}

func (t openingToolset) Tools(context.Context, *ai.RunContext[deps]) ([]ai.Tool[deps], error) {
	return nil, nil
}

func (t openingToolset) OpenToolset(
	context.Context, *ai.RunContext[deps],
) (ai.Toolset[deps], ai.ToolsetCloseFunc, error) {
	*t.log = append(*t.log, "open:"+t.name)
	if t.openErr != nil {
		return nil, nil, t.openErr
	}
	return t, func(context.Context) error {
		*t.log = append(*t.log, "close:"+t.name)
		return nil
	}, nil
}

type failingToolset struct {
	phase     string
	err       error
	nilResult bool
}

func (t failingToolset) Tools(context.Context, *ai.RunContext[deps]) ([]ai.Tool[deps], error) {
	return nil, nil
}

func (t failingToolset) ForRun(
	context.Context, *ai.RunContext[deps],
) (ai.Toolset[deps], error) {
	if t.phase == "run" {
		if t.nilResult {
			return nil, nil
		}
		return nil, t.err
	}
	return t, nil
}

func (t failingToolset) ForRunStep(
	context.Context, *ai.RunContext[deps],
) (ai.Toolset[deps], error) {
	if t.phase == "step" {
		return nil, t.err
	}
	return t, nil
}

func TestToolsetReplacementFailures(t *testing.T) {
	sentinel := errors.New("replacement failed")
	for _, test := range []struct {
		name     string
		toolset  ai.Toolset[deps]
		contains string
	}{
		{
			name: "run error",
			toolset: ai.PrefixToolset(ai.CombineToolsets[deps](
				failingToolset{phase: "run", err: sentinel},
			), "nested"),
			contains: "toolset for run",
		},
		{
			name:     "run nil",
			toolset:  failingToolset{phase: "run", nilResult: true},
			contains: "toolset ForRun returned nil",
		},
		{
			name: "step error",
			toolset: ai.PrefixToolset(ai.CombineToolsets[deps](
				failingToolset{phase: "step", err: sentinel},
			), "nested"),
			contains: "toolset for run step",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			agent := ai.NewAgent[deps, string](model)
			agent.AddToolset(test.toolset)
			_, err := agent.Run(t.Context(), "go", deps{})
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected replacement error: %v", err)
			}
			if test.name != "run nil" && !errors.Is(err, sentinel) {
				t.Fatalf("replacement error lost cause: %v", err)
			}
		})
	}
}

func TestCombinedToolsetOpenRollsBackInReverse(t *testing.T) {
	var log []string
	openErr := errors.New("cannot open second")
	combined := ai.CombineToolsets[deps](
		openingToolset{name: "first", log: &log},
		openingToolset{name: "second", log: &log, openErr: openErr},
	)
	agent := ai.NewAgent[deps, string](nil)
	agent.AddToolset(ai.SetToolsetMetadata(combined, map[string]any{"source": "remote"}))
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, openErr) {
		t.Fatalf("unexpected open error: %v", err)
	}
	want := []string{"open:first", "open:second", "close:first"}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected rollback order: got %v want %v", log, want)
	}
}
