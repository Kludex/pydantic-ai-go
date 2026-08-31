package ai_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func toolCallingModel(name string) ai.Model {
	return namedFunctionModel{name: name, request: func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "advance", ToolCallID: "call", Args: []byte(`{}`)}},
			Usage: ai.Usage{Requests: 1},
		}, nil
	}}
}

func TestModelLessAgentUsesSelectorsAndModelIDs(t *testing.T) {
	t.Run("agent selector", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](nil)
		agent.AddModelSelector(func(
			context.Context, ai.ModelSelectionContext[deps],
		) (ai.ModelSelection, error) {
			return ai.ModelSelection{Model: textModel("selected", "done", nil)}, nil
		})
		result, err := agent.Run(t.Context(), "go", deps{})
		if err != nil || result.Output != "done" {
			t.Fatalf("model-less selector run failed: result=%+v err=%v", result, err)
		}
	})

	t.Run("capability selector", func(t *testing.T) {
		capability := &selectionCapability{model: textModel("selected", "done", nil)}
		agent := ai.NewAgent[deps, string](nil, ai.WithCapabilities(capability))
		result, err := agent.Run(t.Context(), "go", deps{})
		if err != nil || result.Output != "done" {
			t.Fatalf("model-less capability run failed: result=%+v err=%v", result, err)
		}
	})

	t.Run("model ID", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](nil)
		agent.AddModelIDResolver(func(
			context.Context, ai.ModelResolutionContext[deps], string,
		) (ai.Model, error) {
			return textModel("resolved", "done", nil), nil
		})
		result, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelID("alias"))
		if err != nil || result.Output != "done" {
			t.Fatalf("model-less ID run failed: result=%+v err=%v", result, err)
		}
	})

	t.Run("missing selection", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](nil)
		if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrNoModel) {
			t.Fatalf("expected ErrNoModel, got %v", err)
		}
	})

	t.Run("stream", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](nil)
		agent.AddModelSelector(func(
			context.Context, ai.ModelSelectionContext[deps],
		) (ai.ModelSelection, error) {
			return ai.ModelSelection{Model: textModel("selected", "done", nil)}, nil
		})
		stream := agent.RunStream(t.Context(), "go", deps{})
		for _, err := range stream.Events() {
			if err != nil {
				t.Fatal(err)
			}
		}
		if stream.Result() == nil || stream.Result().Output != "done" {
			t.Fatalf("model-less stream failed: %+v", stream.Result())
		}
	})
}

func TestAgentSelectsModelBeforeEveryRequest(t *testing.T) {
	defaultModel := textModel("default", "wrong", nil)
	first := toolCallingModel("first")
	second := textModel("second", "done", nil)
	var contexts []ai.ModelSelectionContext[deps]
	agent := ai.NewAgent[deps, string](defaultModel)
	agent.AddModelSelector(func(
		_ context.Context, selection ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		contexts = append(contexts, selection)
		if selection.Step == 1 {
			return ai.ModelSelection{Model: first}, nil
		}
		selection.Messages[1].(ai.ModelResponse).Parts[0] = ai.TextPart{Content: "mutated copy"}
		return ai.ModelSelection{Model: second}, nil
	})
	agent.AddModelSettingsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (ai.ModelSettings, error) {
		if rc.Model.Name() != []string{"first", "second"}[rc.RunStep-1] {
			t.Fatalf("settings resolved before model selection: step=%d model=%s", rc.RunStep, rc.Model.Name())
		}
		return ai.ModelSettings{}, nil
	})
	ai.AddSimpleTool(agent, "advance", func(context.Context, struct{}) (string, error) { return "advanced", nil })

	result, err := agent.Run(t.Context(), "go", deps{Location: "tenant"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || len(contexts) != 2 {
		t.Fatalf("unexpected adaptive result=%+v contexts=%+v", result, contexts)
	}
	if contexts[0].Step != 1 || contexts[0].Model.Name() != defaultModel.Name() || contexts[0].ModelID != "" ||
		len(contexts[0].Messages) != 0 || contexts[0].Usage.Requests != 0 || contexts[0].Deps.Location != "tenant" {
		t.Fatalf("unexpected first selection context: %+v", contexts[0])
	}
	if contexts[1].Step != 2 || contexts[1].Model.Name() != "first" || len(contexts[1].Messages) != 2 ||
		contexts[1].Usage.Requests != 1 {
		t.Fatalf("unexpected second selection context: %+v", contexts[1])
	}
	response := result.Messages()[1].(ai.ModelResponse)
	if response.Parts[0].(ai.ToolCallPart).ToolName != "advance" || response.ModelName != "first" ||
		result.Messages()[3].(ai.ModelResponse).ModelName != "second" {
		t.Fatalf("selection context mutated history or model names: %+v", result.Messages())
	}
}

type selectionCapability struct {
	model       ai.Model
	selectErr   error
	resolveID   string
	resolved    ai.Model
	resolveErr  error
	selectCalls int
	resolutions int
}

func (*selectionCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *selectionCapability) SelectModel(
	_ context.Context, _ *ai.RunInfo, selection ai.ModelSelectionInfo,
) (ai.ModelSelection, error) {
	c.selectCalls++
	if c.selectErr != nil {
		return ai.ModelSelection{}, c.selectErr
	}
	if selection.Step < 1 {
		return ai.ModelSelection{}, errors.New("invalid selection context")
	}
	if c.model == nil {
		return ai.ModelSelection{}, nil
	}
	return ai.ModelSelection{Model: c.model}, nil
}

func (c *selectionCapability) ResolveModelID(
	_ context.Context, _ *ai.RunInfo, modelID string,
) (ai.Model, error) {
	c.resolutions++
	if c.resolveErr != nil {
		return nil, c.resolveErr
	}
	if modelID == c.resolveID {
		return c.resolved, nil
	}
	return nil, nil
}

func TestCapabilitySelectionOverridesAgentAndRunModelSkipsSelectors(t *testing.T) {
	agentSelected := textModel("agent", "agent", nil)
	capabilitySelected := textModel("capability", "capability", nil)
	capability := &selectionCapability{model: capabilitySelected}
	agent := ai.NewAgent[deps, string](textModel("default", "default", nil), ai.WithCapabilities(capability))
	agent.AddModelSelector(func(
		context.Context, ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		return ai.ModelSelection{Model: agentSelected}, nil
	})

	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "capability" || capability.selectCalls != 1 {
		t.Fatalf("capability did not override agent selection: result=%+v err=%v", result, err)
	}
	explicit := textModel("explicit", "explicit", nil)
	result, err = agent.Run(t.Context(), "go", deps{}, ai.WithRunModel(explicit))
	if err != nil || result.Output != "explicit" || capability.selectCalls != 1 {
		t.Fatalf("explicit model did not skip selectors: result=%+v err=%v calls=%d", result, err, capability.selectCalls)
	}
}

func TestRunModelSelectorReplacesAgentAndCapabilitySelectors(t *testing.T) {
	capability := &selectionCapability{model: textModel("capability", "wrong", nil)}
	agent := ai.NewAgent[deps, string](textModel("default", "wrong", nil), ai.WithCapabilities(capability))
	agent.AddModelSelector(func(
		context.Context, ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		t.Fatal("agent selector was called")
		return ai.ModelSelection{}, nil
	})
	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelSelector(func(
		_ context.Context, selection ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		if selection.Step != 1 || selection.Model.Name() != "default" {
			t.Fatalf("unexpected run selection context: %+v", selection)
		}
		return ai.ModelSelection{Model: textModel("run", "run", nil)}, nil
	}))
	if err != nil || result.Output != "run" || capability.selectCalls != 0 {
		t.Fatalf("run selector did not replace lower layers: result=%+v err=%v", result, err)
	}
}

func TestModelIDSelectionIsResolvedOncePerRun(t *testing.T) {
	selected := toolCallingModel("resolved")
	resolutions := 0
	var selectedIDs []string
	agent := ai.NewAgent[deps, string](textModel("default", "wrong", nil))
	agent.AddModelSelector(func(
		_ context.Context, selection ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		selectedIDs = append(selectedIDs, selection.ModelID)
		return ai.ModelSelection{ID: "tenant-model"}, nil
	})
	agent.AddModelIDResolver(func(
		_ context.Context, resolution ai.ModelResolutionContext[deps], modelID string,
	) (ai.Model, error) {
		resolutions++
		if resolution.Deps.Location != "tenant" || modelID != "tenant-model" {
			t.Fatalf("unexpected resolution context: %+v id=%q", resolution, modelID)
		}
		return selected, nil
	})
	ai.AddSimpleTool(agent, "advance", func(context.Context, struct{}) (string, error) { return "advanced", nil })
	agent.AddModelSelector(func(
		_ context.Context, selection ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		if selection.ModelID != "tenant-model" || selection.Model.Name() != "resolved" {
			t.Fatalf("later selector did not see resolved model: %+v", selection)
		}
		if selection.Step == 2 {
			return ai.ModelSelection{Model: textModel("finish", "done", nil)}, nil
		}
		return ai.ModelSelection{}, nil
	})

	result, err := agent.Run(t.Context(), "go", deps{Location: "tenant"})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected ID-selected result=%+v err=%v", result, err)
	}
	if resolutions != 1 || !slices.Equal(selectedIDs, []string{"", "tenant-model"}) {
		t.Fatalf("unexpected ID selection state: resolutions=%d prior_ids=%v", resolutions, selectedIDs)
	}
}

func TestCapabilityModelIDResolutionAndStaticRunID(t *testing.T) {
	first := &selectionCapability{resolveID: "other", resolved: textModel("other", "wrong", nil)}
	second := &selectionCapability{resolveID: "alias", resolved: textModel("resolved", "done", nil)}
	agent := ai.NewAgent[deps, string](
		textModel("default", "wrong", nil), ai.WithCapabilities(dynamicInstructions{}, first, second),
	)
	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelID("alias"))
	if err != nil || result.Output != "done" || first.resolutions != 1 || second.resolutions != 1 {
		t.Fatalf("capability resolver chain failed: result=%+v err=%v calls=%d/%d", result, err, first.resolutions, second.resolutions)
	}
}

type invalidSelectionCapability struct{}

func (invalidSelectionCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (invalidSelectionCapability) SelectModel(
	context.Context, *ai.RunInfo, ai.ModelSelectionInfo,
) (ai.ModelSelection, error) {
	return ai.ModelSelection{Model: fakes.NewTestModel(), ID: "both"}, nil
}

func TestModelSelectionFailures(t *testing.T) {
	t.Run("selector", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		agent.AddModelSelector(func(context.Context, ai.ModelSelectionContext[deps]) (ai.ModelSelection, error) {
			return ai.ModelSelection{}, errors.New("selection failed")
		})
		if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || err.Error() != "ai: select model: selection failed" {
			t.Fatalf("unexpected selector error: %v", err)
		}
	})

	t.Run("capability selector", func(t *testing.T) {
		capability := &selectionCapability{selectErr: errors.New("capability selection failed")}
		agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
		if _, err := agent.Run(t.Context(), "go", deps{}); err == nil ||
			err.Error() != "ai: select model: capability selection failed" {
			t.Fatalf("unexpected capability selector error: %v", err)
		}
	})

	t.Run("invalid selection", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		agent.AddModelSelector(func(context.Context, ai.ModelSelectionContext[deps]) (ai.ModelSelection, error) {
			return ai.ModelSelection{Model: fakes.NewTestModel(), ID: "both"}, nil
		})
		if _, err := agent.Run(t.Context(), "go", deps{}); err == nil ||
			err.Error() != "ai: model selection must contain either Model or ID, not both" {
			t.Fatalf("unexpected invalid selection error: %v", err)
		}
	})

	t.Run("invalid run selection", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelSelector(func(
			context.Context, ai.ModelSelectionContext[deps],
		) (ai.ModelSelection, error) {
			return ai.ModelSelection{Model: fakes.NewTestModel(), ID: "both"}, nil
		}))
		if err == nil || err.Error() != "ai: model selection must contain either Model or ID, not both" {
			t.Fatalf("unexpected invalid run selection error: %v", err)
		}
	})

	t.Run("invalid capability selection", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](
			fakes.NewTestModel(), ai.WithCapabilities(invalidSelectionCapability{}),
		)
		if _, err := agent.Run(t.Context(), "go", deps{}); err == nil ||
			err.Error() != "ai: model selection must contain either Model or ID, not both" {
			t.Fatalf("unexpected invalid capability selection error: %v", err)
		}
	})

	t.Run("typed nil model", func(t *testing.T) {
		var selected *fakes.TestModel
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		agent.AddModelSelector(func(context.Context, ai.ModelSelectionContext[deps]) (ai.ModelSelection, error) {
			return ai.ModelSelection{Model: selected}, nil
		})
		if _, err := agent.Run(t.Context(), "go", deps{}); err == nil ||
			err.Error() != "ai: selected model must not be nil" {
			t.Fatalf("unexpected nil selection error: %v", err)
		}
	})

	t.Run("unknown ID", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelID("missing"))
		var unknown *ai.UnknownModelIDError
		if !errors.Is(err, ai.ErrUnknownModelID) || !errors.As(err, &unknown) || unknown.ID != "missing" {
			t.Fatalf("unexpected unknown model ID error: %v", err)
		}
	})

	t.Run("agent resolver", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		agent.AddModelIDResolver(func(
			context.Context, ai.ModelResolutionContext[deps], string,
		) (ai.Model, error) {
			return nil, errors.New("agent lookup failed")
		})
		if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelID("alias")); err == nil ||
			err.Error() != `ai: resolve model ID "alias": agent lookup failed` {
			t.Fatalf("unexpected agent resolver error: %v", err)
		}
	})

	t.Run("resolver", func(t *testing.T) {
		capability := &selectionCapability{resolveErr: errors.New("lookup failed")}
		agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
		if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelID("alias")); err == nil ||
			err.Error() != `ai: resolve model ID "alias": lookup failed` {
			t.Fatalf("unexpected resolver error: %v", err)
		}
	})

	t.Run("run dependency mismatch", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelSelector(func(
			context.Context, ai.ModelSelectionContext[string],
		) (ai.ModelSelection, error) {
			return ai.ModelSelection{}, nil
		}))
		if err == nil || err.Error() != "ai: select model: ai: run model selector dependencies do not match agent" {
			t.Fatalf("unexpected dependency mismatch: %v", err)
		}
	})
}

func TestRunModelSelectionOptionsAreExclusive(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	combinations := [][]ai.RunOption{
		{ai.WithRunModel(fakes.NewTestModel()), ai.WithRunModelID("id")},
		{ai.WithRunModel(fakes.NewTestModel()), ai.WithRunModelSelector(func(
			context.Context, ai.ModelSelectionContext[deps],
		) (ai.ModelSelection, error) {
			return ai.ModelSelection{}, nil
		})},
		{ai.WithRunModelID("id"), ai.WithRunModelSelector(func(
			context.Context, ai.ModelSelectionContext[deps],
		) (ai.ModelSelection, error) {
			return ai.ModelSelection{}, nil
		})},
	}
	for index, options := range combinations {
		if _, err := agent.Run(t.Context(), "go", deps{}, options...); err == nil ||
			err.Error() != "ai: run model, model ID, and model selector are mutually exclusive" {
			t.Fatalf("combination %d returned %v", index, err)
		}
	}
}

func TestRunModelOptionsRejectNilOrEmpty(t *testing.T) {
	for name, build := range map[string]func(){
		"empty ID":  func() { ai.WithRunModelID("") },
		"nil model": func() { ai.WithRunModel(nil) },
		"typed nil model": func() {
			var model *fakes.TestModel
			ai.WithRunModel(model)
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			build()
		})
	}
}

func TestModelSelectionMessagesAreDetached(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "old", ToolCallID: "old-call", Args: []byte(`{"value":1}`),
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{
				ToolName: "old", ToolCallID: "old-call", Content: "done", Metadata: map[string]any{"stable": true},
			},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.BinaryContent{Data: []byte("data"), MediaType: "text/plain"},
				ai.UploadedFile{FileID: "file-1", VendorMetadata: map[string]any{"stable": true}},
			}},
		}},
	}
	agent := ai.NewAgent[deps, string](textModel("selected", "done", nil))
	agent.AddModelSelector(func(
		_ context.Context, selection ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		response := selection.Messages[0].(ai.ModelResponse)
		call := response.Parts[0].(ai.ToolCallPart)
		call.Args[0] = 'X'
		request := selection.Messages[1].(ai.ModelRequest)
		result := request.Parts[0].(ai.ToolReturnPart)
		result.Metadata["stable"] = false
		prompt := request.Parts[1].(ai.UserPromptPart)
		prompt.Contents[0].(ai.BinaryContent).Data[0] = 'X'
		prompt.Contents[1].(ai.UploadedFile).VendorMetadata["stable"] = false
		selection.Messages = nil
		return ai.ModelSelection{}, nil
	})
	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithMessageHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	response := result.Messages()[0].(ai.ModelResponse)
	request := result.Messages()[1].(ai.ModelRequest)
	if response.Parts[0].(ai.ToolCallPart).Args[0] != '{' ||
		request.Parts[0].(ai.ToolReturnPart).Metadata["stable"] != true ||
		request.Parts[1].(ai.UserPromptPart).Contents[0].(ai.BinaryContent).Data[0] != 'd' ||
		request.Parts[1].(ai.UserPromptPart).Contents[1].(ai.UploadedFile).VendorMetadata["stable"] != true {
		t.Fatalf("selection context mutated history: %+v", result.Messages())
	}
}
