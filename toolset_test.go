package ai_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type toolsetArgs struct {
	Value string `json:"value"`
}

type emptyNameToolset struct{}

func (emptyNameToolset) Tools(context.Context, *ai.RunContext[deps]) ([]ai.Tool[deps], error) {
	return []ai.Tool[deps]{{}}, nil
}

type instructedToolset struct {
	wrapped      ai.Toolset[deps]
	instructions []ai.InstructionPart
	toolsErr     error
	instructErr  error
}

func (t instructedToolset) Tools(
	ctx context.Context, rc *ai.RunContext[deps],
) ([]ai.Tool[deps], error) {
	if t.toolsErr != nil {
		return nil, t.toolsErr
	}
	return t.wrapped.Tools(ctx, rc)
}

func (t instructedToolset) ToolsetInstructions(
	context.Context, *ai.RunContext[deps],
) ([]ai.InstructionPart, error) {
	return t.instructions, t.instructErr
}

func TestComposableToolsets(t *testing.T) {
	readCalls := 0
	read := ai.NewTool("read", func(
		_ context.Context, rc *ai.RunContext[deps], args toolsetArgs,
	) (string, error) {
		readCalls++
		if rc.ToolName != "read" {
			t.Fatalf("wrapped tool received exposed name %q", rc.ToolName)
		}
		return args.Value + rc.Deps.Location, nil
	})
	write := ai.NewTool("write", func(
		context.Context, *ai.RunContext[deps], toolsetArgs,
	) (string, error) {
		t.Fatal("filtered tool executed")
		return "", nil
	})
	base := instructedToolset{
		wrapped:      ai.NewFunctionToolset(read, write),
		instructions: []ai.InstructionPart{{Content: "Use the database tools.", Dynamic: true}},
	}
	filtered := ai.FilterToolset[deps](base, func(
		_ context.Context, rc *ai.RunContext[deps], definition ai.ToolDefinition,
	) (bool, error) {
		return definition.Name == "read" || rc.Deps.Location == "allow-write", nil
	})
	prepared := ai.PrepareToolset[deps](filtered, func(
		_ context.Context, _ *ai.RunContext[deps], definitions []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		for index := range definitions {
			definitions[index].Description = "Prepared database operation"
		}
		return definitions, nil
	})
	renamed := ai.RenameToolset[deps](prepared, map[string]string{"read": "fetch"})
	prefixed := ai.PrefixToolset[deps](renamed, "db")
	toolset := ai.SetToolsetMetadata[deps](prefixed, map[string]any{"group": "database"})

	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if params.Instructions != "Use the database tools." || len(params.Tools) != 1 {
			t.Fatalf("unexpected toolset request: instructions=%q tools=%+v", params.Instructions, params.Tools)
		}
		definition := params.Tools[0]
		if definition.Name != "db_fetch" || definition.Description != "Prepared database operation" ||
			definition.Metadata["group"] != "database" {
			t.Fatalf("toolset wrappers were not applied: %+v", definition)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "db_fetch", ToolCallID: "call", Args: []byte(`{"value":"item"}`),
			}}}, nil
		}
		toolReturn := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
		if toolReturn.Content != "item!" {
			t.Fatalf("unexpected wrapped tool result: %+v", toolReturn)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(toolset)
	result, err := agent.Run(t.Context(), "go", deps{Location: "!"})
	if err != nil || result.Output != "done" || request != 2 || readCalls != 1 {
		t.Fatalf("composed toolset failed: result=%+v requests=%d calls=%d err=%v", result, request, readCalls, err)
	}
}

func TestCombinedAndPerRunToolsets(t *testing.T) {
	one := ai.NewTool("one", func(
		context.Context, *ai.RunContext[deps], toolsetArgs,
	) (string, error) {
		return "one", nil
	})
	two := ai.NewTool("two", func(
		context.Context, *ai.RunContext[deps], toolsetArgs,
	) (string, error) {
		return "two", nil
	})
	combined := ai.CombineToolsets[deps](
		instructedToolset{
			wrapped:      ai.NewFunctionToolset(one),
			instructions: []ai.InstructionPart{{Content: "One."}},
		},
		instructedToolset{
			wrapped:      ai.NewFunctionToolset(two),
			instructions: []ai.InstructionPart{{Content: "Two."}},
		},
	)
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 2 || params.Instructions != "One.\n\nTwo." {
			t.Fatalf("combined toolset missing: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunToolsets(combined)); err != nil {
		t.Fatal(err)
	}

	without := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 0 || params.Instructions != "" {
			t.Fatalf("run toolset leaked: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "clean"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](without).Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestFunctionToolsetPreparation(t *testing.T) {
	called := false
	kept := ai.NewPreparedTool("kept", func(
		_ context.Context, rc *ai.RunContext[deps], _ toolsetArgs,
	) (string, error) {
		called = true
		if rc.ToolName != "kept" {
			t.Fatalf("prepared rename leaked into tool context: %q", rc.ToolName)
		}
		return "ok", nil
	}, func(
		_ context.Context, _ *ai.RunContext[deps], definition ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		definition.Name = "renamed"
		definition.Description = "prepared"
		return &definition, nil
	})
	omitted := ai.NewPreparedTool("omitted", func(
		context.Context, *ai.RunContext[deps], toolsetArgs,
	) (string, error) {
		return "", nil
	}, func(
		context.Context, *ai.RunContext[deps], ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		return nil, nil
	})
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if len(params.Tools) != 1 || params.Tools[0].Name != "renamed" || params.Tools[0].Description != "prepared" {
			t.Fatalf("unexpected prepared function toolset: %+v", params.Tools)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "renamed", ToolCallID: "call", Args: []byte(`{"value":"x"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "go", deps{}, ai.WithRunToolsets(ai.NewFunctionToolset(kept, omitted)),
	); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("prepared function tool was not called")
	}
}

func TestToolsetRetryAndTimeoutDefaults(t *testing.T) {
	defaulted := ai.NewTool("defaulted", func(
		ctx context.Context, rc *ai.RunContext[deps], _ toolsetArgs,
	) (string, error) {
		if rc.MaxRetries != 3 {
			t.Fatalf("unexpected toolset retry default: %d", rc.MaxRetries)
		}
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) > 500*time.Millisecond {
			t.Fatalf("toolset timeout default missing: deadline=%v ok=%v", deadline, ok)
		}
		return "defaulted", nil
	})
	explicit := ai.NewTool("explicit", func(
		ctx context.Context, rc *ai.RunContext[deps], _ toolsetArgs,
	) (string, error) {
		if rc.MaxRetries != 7 {
			t.Fatalf("explicit retry budget was replaced: %d", rc.MaxRetries)
		}
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) < 500*time.Millisecond {
			t.Fatalf("explicit timeout was replaced: deadline=%v ok=%v", deadline, ok)
		}
		return "explicit", nil
	}, ai.WithToolMaxRetries(7), ai.WithToolTimeout(time.Second))
	toolset := ai.WithToolsetTimeout(
		ai.WithToolsetMaxRetries(ai.NewFunctionToolset(defaulted, explicit), 3),
		100*time.Millisecond,
	)
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "defaulted", ToolCallID: "one", Args: []byte(`{"value":"x"}`)},
				ai.ToolCallPart{ToolName: "explicit", ToolCallID: "two", Args: []byte(`{"value":"x"}`)},
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "go", deps{}, ai.WithRunToolsets(toolset),
	); err != nil {
		t.Fatal(err)
	}
}

func TestToolsetDefaultValidation(t *testing.T) {
	toolset := ai.NewFunctionToolset[deps]()
	for name, fn := range map[string]func(){
		"retries": func() { ai.WithToolsetMaxRetries(toolset, -1) },
		"timeout": func() { ai.WithToolsetTimeout(toolset, 0) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			fn()
		})
	}
}

func TestToolsetResolutionErrors(t *testing.T) {
	tool := ai.NewTool("same", func(
		context.Context, *ai.RunContext[deps], toolsetArgs,
	) (string, error) {
		return "", nil
	})
	sentinel := errors.New("toolset failed")
	tests := map[string]struct {
		toolset ai.Toolset[deps]
		want    string
	}{
		"listing": {
			toolset: instructedToolset{wrapped: ai.NewFunctionToolset[deps](), toolsErr: sentinel},
			want:    "toolset failed",
		},
		"empty name": {
			toolset: emptyNameToolset{},
			want:    "toolset returned an empty tool name",
		},
		"combined conflict": {
			toolset: ai.CombineToolsets(ai.NewFunctionToolset(tool), ai.NewFunctionToolset(tool)),
			want:    `duplicate toolset tool name "same"`,
		},
		"agent conflict": {
			toolset: ai.NewFunctionToolset(tool),
			want:    `duplicate tool name "same"`,
		},
		"rename conflict": {
			toolset: ai.RenameToolset(
				ai.NewFunctionToolset(tool, ai.NewTool("other", func(
					context.Context, *ai.RunContext[deps], toolsetArgs,
				) (string, error) {
					return "", nil
				},
				)),
				map[string]string{"other": "same"},
			),
			want: `renamed tool name "same" conflicts`,
		},
		"empty rename": {
			toolset: ai.RenameToolset(ai.NewFunctionToolset(tool), map[string]string{"same": ""}),
			want:    "renamed tool name must not be empty",
		},
		"prepare adds": {
			toolset: ai.PrepareToolset(ai.NewFunctionToolset(tool), func(
				context.Context, *ai.RunContext[deps], []ai.ToolDefinition,
			) ([]ai.ToolDefinition, error) {
				return []ai.ToolDefinition{{Name: "added"}}, nil
			}),
			want: `added or renamed tool "added"`,
		},
		"prepare duplicates": {
			toolset: ai.PrepareToolset(ai.NewFunctionToolset(tool), func(
				_ context.Context, _ *ai.RunContext[deps], definitions []ai.ToolDefinition,
			) ([]ai.ToolDefinition, error) {
				return append(definitions, definitions[0]), nil
			}),
			want: `duplicate tool "same"`,
		},
		"prepare error": {
			toolset: ai.PrepareToolset(ai.NewFunctionToolset(tool), func(
				context.Context, *ai.RunContext[deps], []ai.ToolDefinition,
			) ([]ai.ToolDefinition, error) {
				return nil, sentinel
			}),
			want: "toolset failed",
		},
		"filter error": {
			toolset: ai.FilterToolset(ai.NewFunctionToolset(tool), func(
				context.Context, *ai.RunContext[deps], ai.ToolDefinition,
			) (bool, error) {
				return false, sentinel
			}),
			want: "toolset failed",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, string](fakes.NewTestModel())
			if name == "agent conflict" {
				agent.AddTool(tool)
			}
			_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunToolsets(test.toolset))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v, want %q", err, test.want)
			}
		})
	}
}

func TestWrapperToolsetsPropagateListingErrors(t *testing.T) {
	sentinel := errors.New("listing failed")
	failed := instructedToolset{wrapped: ai.NewFunctionToolset[deps](), toolsErr: sentinel}
	wrappers := map[string]ai.Toolset[deps]{
		"combined": ai.CombineToolsets[deps](failed),
		"filtered": ai.FilterToolset[deps](failed, func(
			context.Context, *ai.RunContext[deps], ai.ToolDefinition,
		) (bool, error) {
			return true, nil
		}),
		"prefixed": ai.PrefixToolset[deps](failed, "prefix"),
		"renamed":  ai.RenameToolset[deps](failed, nil),
		"prepared": ai.PrepareToolset[deps](failed, func(
			context.Context, *ai.RunContext[deps], []ai.ToolDefinition,
		) ([]ai.ToolDefinition, error) {
			return nil, nil
		}),
		"metadata": ai.SetToolsetMetadata[deps](failed, nil),
		"defaults": ai.WithToolsetMaxRetries[deps](failed, 1),
	}
	for name, toolset := range wrappers {
		t.Run(name, func(t *testing.T) {
			_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
				t.Context(), "go", deps{}, ai.WithRunToolsets(toolset),
			)
			if !errors.Is(err, sentinel) {
				t.Fatalf("listing error was not propagated: %v", err)
			}
		})
	}
}

func TestCombinedToolsetPropagatesInstructionError(t *testing.T) {
	sentinel := errors.New("instructions failed")
	toolset := ai.CombineToolsets[deps](instructedToolset{
		wrapped: ai.NewFunctionToolset[deps](), instructErr: sentinel,
	})
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
		t.Context(), "go", deps{}, ai.WithRunToolsets(toolset),
	)
	if !errors.Is(err, sentinel) {
		t.Fatalf("instruction error was not propagated: %v", err)
	}
}

func TestToolsetPreparationAndInstructionErrors(t *testing.T) {
	sentinel := errors.New("failed")
	prepared := ai.NewPreparedTool("prepared", func(
		context.Context, *ai.RunContext[deps], toolsetArgs,
	) (string, error) {
		return "", nil
	}, func(
		context.Context, *ai.RunContext[deps], ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		return nil, sentinel
	})
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
		t.Context(), "go", deps{}, ai.WithRunToolsets(ai.NewFunctionToolset(prepared)),
	)
	if err == nil || !strings.Contains(err.Error(), `prepare tool "prepared"`) {
		t.Fatalf("unexpected prepared tool error: %v", err)
	}

	instructions := instructedToolset{
		wrapped:     ai.NewFunctionToolset[deps](),
		instructErr: sentinel,
	}
	_, err = ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
		t.Context(), "go", deps{}, ai.WithRunToolsets(ai.PrefixToolset[deps](instructions, "p")),
	)
	if err == nil || !strings.Contains(err.Error(), "ai: toolset instructions: failed") {
		t.Fatalf("unexpected instruction error: %v", err)
	}
}

func TestRunToolsetDependencyMismatch(t *testing.T) {
	type otherDeps struct{}
	toolset := ai.NewFunctionToolset(ai.NewTool("other", func(
		context.Context, *ai.RunContext[otherDeps], toolsetArgs,
	) (string, error) {
		return "", nil
	}))
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
		t.Context(), "go", deps{}, ai.WithRunToolsets(toolset),
	)
	if err == nil || !strings.Contains(err.Error(), "run toolset dependencies do not match agent") {
		t.Fatalf("unexpected dependency mismatch: %v", err)
	}
}

func ExamplePrefixToolset() {
	type exampleDeps struct{}
	lookup := ai.NewTool("lookup", func(
		context.Context, *ai.RunContext[exampleDeps], toolsetArgs,
	) (string, error) {
		return "value", nil
	})
	toolset := ai.PrefixToolset(ai.NewFunctionToolset(lookup), "catalog")
	definitions, _ := toolset.Tools(context.Background(), &ai.RunContext[exampleDeps]{})
	fmt.Println(definitions[0].Definition().Name)
	// Output: catalog_lookup
}
