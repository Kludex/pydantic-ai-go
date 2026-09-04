package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func toolDefinitionNames(definitions []ai.ToolDefinition) []string {
	names := make([]string, len(definitions))
	for index, definition := range definitions {
		names[index] = definition.Name
	}
	slices.Sort(names)
	return names
}

func TestDeferredResultRevealAppliesToImmediateContinuation(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		names := toolDefinitionNames(params.Tools)
		if request == 1 {
			if !slices.Equal(names, []string{"loader"}) {
				t.Fatalf("hidden tool was initially visible: %v", names)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "loader", ToolCallID: "loader", Args: []byte(`{}`),
			}}}, nil
		}
		if !slices.Equal(names, []string{"hidden", "loader"}) {
			t.Fatalf("deferred result reveal was not applied before continuation: %v", names)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddExternalTool[deps, string, struct{}, ai.ToolReturn](agent, "loader")
	ai.AddSimpleTool(agent, "hidden", func(context.Context, struct{}) (string, error) {
		return "hidden", nil
	}, ai.WithDeferredLoading())
	ai.AddSimpleTool(agent, "still_hidden", func(context.Context, struct{}) (string, error) {
		return "still hidden", nil
	}, ai.WithDeferredLoading())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{
			"loader": ai.ToolReturn{ReturnValue: "loaded", Tools: []string{"hidden"}},
		}}),
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected revealed continuation: result=%+v err=%v", result, err)
	}
}

func TestDeferredResultCanRepeatHistoryReveal(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "loader", ToolCallID: "loader", Args: []byte(`{}`),
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolAvailabilityDeltaPart{
			ToolsAdded: []string{"hidden"}, ToolCallID: "earlier-loader",
		}}},
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !slices.Equal(toolDefinitionNames(params.Tools), []string{"hidden", "loader"}) {
			t.Fatalf("history-revealed tool was not retained: %+v", params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddExternalTool[deps, string, struct{}, ai.ToolReturn](agent, "loader")
	ai.AddSimpleTool(agent, "hidden", func(context.Context, struct{}) (string, error) {
		return "hidden", nil
	}, ai.WithDeferredLoading())
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{
			"loader": ai.ToolReturn{ReturnValue: "loaded", Tools: []string{"hidden"}},
		}}),
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected repeated reveal result=%+v err=%v", result, err)
	}
}

func TestDeferredResultRevealCompilesNewlyVisibleSchema(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "loader", ToolCallID: "loader", Args: []byte(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddExternalTool[deps, string, struct{}, ai.ToolReturn](agent, "loader")
	agent.AddRawTool(ai.ToolDefinition{
		Name: "invalid", Schema: map[string]any{"type": "invalid"}, DeferLoading: true,
	}, func(context.Context, json.RawMessage) (any, error) {
		return nil, nil
	})
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{
			"loader": ai.ToolReturn{ReturnValue: "loaded", Tools: []string{"invalid"}},
		}}),
	)
	if err == nil || !strings.Contains(err.Error(), "compile schema") {
		t.Fatalf("unexpected newly visible schema error: %v", err)
	}
}

func TestDeferredToolRevealAndHistoryResume(t *testing.T) {
	request := 0
	secretCalls := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		names := toolDefinitionNames(params.Tools)
		if !slices.Equal(toolDefinitionNames(params.DeferredTools), []string{"secret"}) {
			t.Fatalf("deferred tool corpus missing from provider parameters: %+v", params.DeferredTools)
		}
		switch request {
		case 1:
			if !slices.Equal(names, []string{"loader", "public"}) {
				t.Fatalf("deferred tool was visible before reveal: %v", names)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "secret", ToolCallID: "early", Args: []byte(`{}`),
			}}}, nil
		case 2:
			if !slices.Equal(names, []string{"loader", "public"}) {
				t.Fatalf("hidden call changed visibility: %v", names)
			}
			retry := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
			if retry.ToolName != "secret" || retry.Content !=
				"Tool 'secret' is not available yet: search for it first, then call it again once you've seen its schema." {
				t.Fatalf("unexpected hidden-tool retry: %+v", retry)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "loader", ToolCallID: "reveal", Args: []byte(`{}`),
			}}}, nil
		case 3:
			if !slices.Equal(names, []string{"loader", "public", "secret"}) {
				t.Fatalf("revealed tool is unavailable: %v", names)
			}
			parts := messages[len(messages)-1].(ai.ModelRequest).Parts
			if len(parts) != 3 {
				t.Fatalf("unexpected reveal request: %+v", parts)
			}
			returned := parts[0].(ai.ToolReturnPart)
			delta := parts[1].(ai.ToolAvailabilityDeltaPart)
			content := parts[2].(ai.UserPromptPart)
			if returned.ToolCallID != "reveal" ||
				!slices.Equal(delta.ToolsAdded, []string{"secret"}) || delta.ToolCallID != "reveal" ||
				content.Contents[0].(ai.TextContent).Text != "Secret access enabled." {
				t.Fatalf("unexpected reveal history: %+v", parts)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "secret", ToolCallID: "secret", Args: []byte(`{}`),
			}}}, nil
		default:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
	})

	loader := ai.NewSimpleTool[deps]("loader", func(context.Context, struct{}) (ai.ToolReturn, error) {
		return ai.ToolReturn{
			ReturnValue: "loaded",
			Tools:       []string{"secret", "missing", "loader", "secret"},
			Content:     []ai.UserContent{ai.TextContent{Text: "Secret access enabled."}},
		}, nil
	})
	public := ai.NewSimpleTool[deps]("public", func(context.Context, struct{}) (string, error) {
		return "public", nil
	})
	secret := ai.NewTool("secret", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		if rc.Retry != 0 {
			t.Fatalf("free availability refusal consumed retry budget: %+v", rc)
		}
		secretCalls++
		return "secret", nil
	}, ai.WithDeferredLoading())
	agent := ai.NewAgent[deps, string](model)
	agent.AddTool(loader)
	agent.AddToolset(ai.DeferLoadingToolset(ai.NewFunctionToolset(public, secret), "secret"))

	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || secretCalls != 1 {
		t.Fatalf("unexpected deferred result=%+v secret calls=%d", result, secretCalls)
	}

	resumeModel := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !slices.Equal(toolDefinitionNames(params.Tools), []string{"loader", "public", "secret"}) ||
			!slices.Equal(toolDefinitionNames(params.DeferredTools), []string{"secret"}) {
			t.Fatalf("history did not retain reveal: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "resumed"}}}, nil
	})
	resumed := ai.NewAgent[deps, string](resumeModel)
	resumed.AddTool(loader)
	resumed.AddToolset(ai.DeferLoadingToolset(ai.NewFunctionToolset(public, secret), "secret"))
	resumedResult, err := resumed.Run(
		t.Context(), "continue", deps{},
		ai.WithMessageHistory(result.Messages()),
		ai.WithRunModelSelector(func(
			context.Context, ai.ModelSelectionContext[deps],
		) (ai.ModelSelection, error) {
			return ai.ModelSelection{Model: resumeModel}, nil
		}),
	)
	if err != nil || resumedResult.Output != "resumed" {
		t.Fatalf("resume failed: result=%+v err=%v", resumedResult, err)
	}
}

type deferredSourceToolset struct {
	id  string
	err error
}

func (t deferredSourceToolset) ToolsetID() string { return t.id }

func (t deferredSourceToolset) Tools(
	context.Context, *ai.RunContext[deps],
) ([]ai.Tool[deps], error) {
	if t.err != nil {
		return nil, t.err
	}
	return []ai.Tool[deps]{ai.NewSimpleTool[deps](
		"hidden", func(context.Context, struct{}) (string, error) { return "hidden", nil },
	)}, nil
}

func TestDeferredAvailabilityRefusalIsFreeOnce(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "hidden", ToolCallID: "hidden", Args: []byte(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithRetryLimits(ai.RetryLimits{}))
	ai.AddSimpleTool(agent, "hidden", func(context.Context, struct{}) (string, error) {
		t.Fatal("hidden tool ran")
		return "", nil
	}, ai.WithDeferredLoading())
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrMaxRetriesExceeded) || requests != 2 {
		t.Fatalf("availability refusal did not preserve one correction: requests=%d err=%v", requests, err)
	}
}

func TestDeferredToolsetDefaultsAndErrors(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 0 {
			t.Fatalf("all deferred tools should be hidden: %+v", params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(ai.DeferLoadingToolset[deps](deferredSourceToolset{}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}

	sentinel := errors.New("cannot list deferred tools")
	failed := ai.NewAgent[deps, string](model)
	failed.AddToolset(ai.DeferLoadingToolset[deps](deferredSourceToolset{err: sentinel}))
	if _, err := failed.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected deferred listing error: %v", err)
	}
}

func TestConcurrentToolRevealsAreDeduplicatedInModelOrder(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			if len(params.Tools) != 2 {
				t.Fatalf("targets were visible before reveal: %+v", params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "first", ToolCallID: "first", Args: []byte(`{}`)},
				ai.ToolCallPart{ToolName: "second", ToolCallID: "second", Args: []byte(`{}`)},
			}}, nil
		}
		if !slices.Equal(
			toolDefinitionNames(params.Tools),
			[]string{"alpha", "beta", "first", "gamma", "second"},
		) {
			t.Fatalf("concurrent reveals were not applied: %+v", params.Tools)
		}
		parts := messages[len(messages)-1].(ai.ModelRequest).Parts
		if len(parts) != 6 {
			t.Fatalf("unexpected concurrent reveal parts: %+v", parts)
		}
		first := parts[2].(ai.ToolAvailabilityDeltaPart)
		second := parts[3].(ai.ToolAvailabilityDeltaPart)
		if !slices.Equal(first.ToolsAdded, []string{"alpha", "beta"}) ||
			!slices.Equal(second.ToolsAdded, []string{"gamma"}) {
			t.Fatalf("reveals were not deduplicated in call order: %+v %+v", first, second)
		}
		firstContent := parts[4].(ai.UserPromptPart).Contents[0].(ai.TextContent)
		secondContent := parts[5].(ai.UserPromptPart).Contents[0].(ai.TextContent)
		if firstContent.Text != "first content" || secondContent.Text != "second content" {
			t.Fatalf("trailing content was not ordered after deltas: %+v", parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "first", func(context.Context, struct{}) (ai.ToolReturn, error) {
		return ai.ToolReturn{
			ReturnValue: "first", Tools: []string{"alpha", "beta"},
			Content: []ai.UserContent{ai.TextContent{Text: "first content"}},
		}, nil
	})
	ai.AddSimpleTool(agent, "second", func(context.Context, struct{}) (ai.ToolReturn, error) {
		return ai.ToolReturn{
			ReturnValue: "second", Tools: []string{"beta", "gamma"},
			Content: []ai.UserContent{ai.TextContent{Text: "second content"}},
		}, nil
	})
	for _, name := range []string{"alpha", "beta", "gamma"} {
		ai.AddSimpleTool(agent, name, func(context.Context, struct{}) (string, error) {
			return name, nil
		}, ai.WithDeferredLoading())
	}
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}
