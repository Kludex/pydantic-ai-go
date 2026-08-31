package ai_test

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type reusableToolArgs struct {
	Value string `json:"value"`
}

func TestReusableToolForOneRun(t *testing.T) {
	called := 0
	lookup := ai.NewTool("lookup", func(
		_ context.Context, rc *ai.RunContext[deps], args reusableToolArgs,
	) (string, error) {
		called++
		return args.Value + rc.Deps.Location, nil
	}, ai.WithDescription("Look up a value"))

	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			if len(params.Tools) != 1 || params.Tools[0].Name != "lookup" {
				t.Fatalf("run tool was not exposed: %+v", params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "lookup", ToolCallID: "call", Args: []byte(`{"value":"item"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	result, err := agent.Run(t.Context(), "go", deps{Location: "!"}, ai.WithRunTools(lookup))
	if err != nil || result.Output != "done" || called != 1 {
		t.Fatalf("run tool failed: result=%+v called=%d err=%v", result, called, err)
	}
	toolReturn := result.Messages()[2].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if toolReturn.Content != "item!" {
		t.Fatalf("run tool received wrong dependencies: %+v", toolReturn)
	}

	withoutTool := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 0 {
			t.Fatalf("run tool leaked into another run: %+v", params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "clean"}}}, nil
	})
	cleanAgent := ai.NewAgent[deps, string](withoutTool)
	clean, err := cleanAgent.Run(t.Context(), "go", deps{})
	if err != nil || clean.Output != "clean" {
		t.Fatalf("run without tool failed: result=%+v err=%v", clean, err)
	}
}

func TestReusablePreparedToolAndDetachedDefinition(t *testing.T) {
	tool := ai.NewPreparedTool("lookup", func(
		_ context.Context, _ *ai.RunContext[deps], args reusableToolArgs,
	) (string, error) {
		return args.Value, nil
	}, func(
		_ context.Context, rc *ai.RunContext[deps], definition ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		definition.Description = rc.Deps.Location
		return &definition, nil
	}, ai.WithDescription("original"), ai.WithToolMetadata(map[string]any{"source": "reusable"}))
	definition := tool.Definition()
	definition.Description = "changed"
	definition.Metadata["source"] = "changed"

	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 1 || params.Tools[0].Description != "prepared" ||
			params.Tools[0].Metadata["source"] != "reusable" {
			t.Fatalf("unexpected prepared run tool: %+v", params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	if _, err := agent.Run(t.Context(), "go", deps{Location: "prepared"}, ai.WithRunTools(tool)); err != nil {
		t.Fatal(err)
	}
}

func TestAgentAcceptsReusableTool(t *testing.T) {
	tool := ai.NewTool("lookup", func(
		context.Context, *ai.RunContext[deps], reusableToolArgs,
	) (string, error) {
		return "value", nil
	})
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 1 || params.Tools[0].Name != "lookup" {
			t.Fatalf("reusable agent tool missing: %+v", params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddTool(tool)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestRunToolValidation(t *testing.T) {
	tool := ai.NewTool("lookup", func(
		context.Context, *ai.RunContext[deps], reusableToolArgs,
	) (string, error) {
		return "value", nil
	})
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddTool(tool)
	_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunTools(tool))
	if err == nil || !strings.Contains(err.Error(), `duplicate run tool name "lookup"`) {
		t.Fatalf("unexpected duplicate tool error: %v", err)
	}

	type otherDeps struct{}
	otherTool := ai.NewTool("other", func(
		context.Context, *ai.RunContext[otherDeps], reusableToolArgs,
	) (string, error) {
		return "value", nil
	})
	_, err = ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
		t.Context(), "go", deps{}, ai.WithRunTools(otherTool),
	)
	if err == nil || !strings.Contains(err.Error(), "run tool dependencies do not match agent") {
		t.Fatalf("unexpected dependency mismatch: %v", err)
	}
}

func TestReusableRunToolSupportsConcurrentRuns(t *testing.T) {
	var calls atomic.Int64
	tool := ai.NewTool("lookup", func(
		context.Context, *ai.RunContext[deps], reusableToolArgs,
	) (string, error) {
		calls.Add(1)
		return "value", nil
	})
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "lookup", ToolCallID: "call", Args: []byte(`{"value":"item"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	var wait sync.WaitGroup
	errors := make(chan error, 2)
	for range 2 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunTools(tool))
			errors <- err
		}()
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatal(err)
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("unexpected concurrent tool calls: %d", calls.Load())
	}
}
