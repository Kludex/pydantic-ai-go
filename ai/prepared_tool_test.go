package ai_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestPreparedToolCanBeModifiedOrOmittedPerRun(t *testing.T) {
	var exposed []int
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		exposed = append(exposed, len(params.Tools))
		if len(params.Tools) == 1 && params.Tools[0].Description != "available" {
			t.Fatalf("tool was not modified: %+v", params.Tools[0])
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[bool, string](model)
	ai.AddPreparedTool(
		agent,
		"work",
		func(context.Context, *ai.RunContext[bool], struct{}) (string, error) { return "done", nil },
		func(_ context.Context, rc *ai.RunContext[bool], tool ai.ToolDefinition) (*ai.ToolDefinition, error) {
			if !rc.Deps {
				return nil, nil
			}
			tool.Description = "available"
			return &tool, nil
		},
	)
	globalCalls := 0
	agent.AddToolsPrepareFunc(func(
		_ context.Context, _ *ai.RunContext[bool], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		globalCalls++
		if len(tools) == 1 && tools[0].Description != "available" {
			t.Fatalf("global preparation ran before per-tool preparation: %+v", tools[0])
		}
		return tools, nil
	})
	if _, err := agent.Run(t.Context(), "hidden", false); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(t.Context(), "visible", true); err != nil {
		t.Fatal(err)
	}
	if len(exposed) != 2 || exposed[0] != 0 || exposed[1] != 1 || globalCalls != 2 {
		t.Fatalf("unexpected preparation results %v after %d global calls", exposed, globalCalls)
	}
}

func TestPreparedToolReceivesFreshDefinitionEveryStep(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "call", `{}`)}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	prepareCalls := 0
	ai.AddPreparedTool(
		agent,
		"work",
		func(context.Context, *ai.RunContext[deps], struct{}) (string, error) { return "done", nil },
		func(_ context.Context, _ *ai.RunContext[deps], tool ai.ToolDefinition) (*ai.ToolDefinition, error) {
			prepareCalls++
			if tool.Description != "original" {
				t.Fatalf("prepared definition leaked into step %d: %+v", prepareCalls, tool)
			}
			tool.Description = "changed"
			return &tool, nil
		},
		ai.WithDescription("original"),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if prepareCalls != 2 {
		t.Fatalf("expected two prepare calls, got %d", prepareCalls)
	}
}

func TestPreparedToolErrorStopsBeforeModelRequest(t *testing.T) {
	boom := errors.New("boom")
	requested := false
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		requested = true
		return nil, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddPreparedTool(
		agent,
		"work",
		func(context.Context, *ai.RunContext[deps], struct{}) (string, error) { return "", nil },
		func(context.Context, *ai.RunContext[deps], ai.ToolDefinition) (*ai.ToolDefinition, error) {
			return nil, boom
		},
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, boom) {
		t.Fatalf("expected prepare error, got %v", err)
	}
	if requested {
		t.Fatal("model was called after preparation failed")
	}
}

func TestPreparedToolReadsItsOwnRetryCountUnderPrefixesAndRenames(t *testing.T) {
	type observed struct {
		name  string
		retry int
	}
	newTool := func(seen *[]observed) ai.Tool[deps] {
		return ai.NewPreparedTool(
			"work",
			func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
				if rc.Retry == 0 {
					return "", ai.Retryf("again")
				}
				return "done", nil
			},
			func(_ context.Context, rc *ai.RunContext[deps], tool ai.ToolDefinition) (*ai.ToolDefinition, error) {
				*seen = append(*seen, observed{rc.ToolName, rc.Retry})
				return &tool, nil
			},
		)
	}
	wrappers := map[string]struct {
		exposed string
		wrap    func(ai.Toolset[deps]) ai.Toolset[deps]
	}{
		"plain": {"work", func(toolset ai.Toolset[deps]) ai.Toolset[deps] { return toolset }},
		"prefixed": {"db_work", func(toolset ai.Toolset[deps]) ai.Toolset[deps] {
			return ai.PrefixToolset(toolset, "db")
		}},
		"renamed": {"job", func(toolset ai.Toolset[deps]) ai.Toolset[deps] {
			return ai.RenameToolset(toolset, map[string]string{"work": "job"})
		}},
		"prefixed renamed": {"db_job", func(toolset ai.Toolset[deps]) ai.Toolset[deps] {
			return ai.PrefixToolset(ai.RenameToolset(toolset, map[string]string{"work": "job"}), "db")
		}},
		"prefixed with other renames": {"db_work", func(toolset ai.Toolset[deps]) ai.Toolset[deps] {
			return ai.PrefixToolset(ai.RenameToolset(toolset, map[string]string{"other": "job"}), "db")
		}},
		"renamed prefixed": {"job", func(toolset ai.Toolset[deps]) ai.Toolset[deps] {
			return ai.RenameToolset(ai.PrefixToolset(toolset, "db"), map[string]string{"db_work": "job"})
		}},
		"filtered prefixed": {"db_work", func(toolset ai.Toolset[deps]) ai.Toolset[deps] {
			return ai.PrepareToolset(ai.FilterToolset(ai.PrefixToolset(toolset, "db"), func(
				context.Context, *ai.RunContext[deps], ai.ToolDefinition,
			) (bool, error) {
				return true, nil
			}), func(_ context.Context, _ *ai.RunContext[deps], tools []ai.ToolDefinition) ([]ai.ToolDefinition, error) {
				return tools, nil
			})
		}},
	}
	for name, wrapper := range wrappers {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				if len(messages) < 5 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall(wrapper.exposed, "call", `{}`)}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			var seen []observed
			agent := ai.NewAgent[deps, string](model)
			agent.AddToolset(wrapper.wrap(ai.NewFunctionToolset(newTool(&seen))))
			if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
				t.Fatal(err)
			}
			want := []observed{{"work", 0}, {"work", 1}, {"work", 1}}
			if !slices.Equal(seen, want) {
				t.Fatalf("prepare saw %v, want %v", seen, want)
			}
		})
	}
}

func TestPreparedAgentToolReadsItsOwnRetryCount(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) < 3 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "call", `{}`)}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	var retries []int
	ai.AddPreparedTool(
		agent,
		"work",
		func(context.Context, *ai.RunContext[deps], struct{}) (string, error) { return "", ai.Retryf("again") },
		func(_ context.Context, rc *ai.RunContext[deps], tool ai.ToolDefinition) (*ai.ToolDefinition, error) {
			retries = append(retries, rc.Retry)
			return &tool, nil
		},
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(retries, []int{0, 1}) {
		t.Fatalf("prepare saw retries %v, want [0 1]", retries)
	}
}
