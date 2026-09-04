package ai_test

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type runReport struct {
	Answer string `json:"answer"`
}

type runCount struct {
	Count int `json:"count"`
}

type runOutputCapability struct{}

func (runOutputCapability) Setup(registry *ai.CapabilityRegistry) error {
	topP := 0.8
	registry.AddModelSettings(ai.ModelSettings{TopP: &topP})
	return nil
}

func TestRunAsSpecializesOneRunWithoutMutatingAgentOutput(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if params.Instructions != "agent\n\nrun" || params.Settings.MaxTokens != 42 ||
			params.Settings.TopP == nil || *params.Settings.TopP != 0.8 || len(params.Tools) != 1 {
			t.Fatalf("specialized run lost configuration: %+v", params)
		}
		if params.OutputTool == nil {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "plain"}}}, nil
		}
		if params.OutputTool.Name != "run_report" ||
			params.OutputTool.Schema["properties"].(map[string]any)["answer"].(map[string]any)["type"] != "string" {
			t.Fatalf("unexpected specialized output tool: %+v", params.OutputTool)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "run_report", ToolCallID: "report", Args: []byte(`{"answer":"typed"}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](
		model,
		ai.WithAgentName("specialized"),
		ai.WithAgentDescription("Specialized output agent"),
		ai.WithInstructions("agent"),
		ai.WithModelSettings(ai.ModelSettings{MaxTokens: 42}),
		ai.WithCapabilities(runOutputCapability{}),
	)
	agent.AddTool(ai.NewSimpleTool[deps]("unused", func(context.Context, struct{}) (string, error) {
		return "unused", nil
	}))
	agent.AddOutputToolPrepareFunc(func(
		_ context.Context, rc *ai.RunContext[deps], definition ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		if rc.AgentName != "specialized" || rc.AgentDescription != "Specialized output agent" {
			t.Fatalf("specialized run lost agent identity: %+v", rc)
		}
		definition.Name = "run_report"
		return &definition, nil
	})
	result, err := ai.RunAs[runReport](
		t.Context(), agent, "typed", deps{}, ai.WithRunInstructions("run"),
	)
	if err != nil || result.Output != (runReport{Answer: "typed"}) {
		t.Fatalf("unexpected typed result=%+v err=%v", result, err)
	}
	plain, err := agent.Run(t.Context(), "plain", deps{}, ai.WithRunInstructions("run"))
	if err != nil || plain.Output != "plain" || requests != 2 {
		t.Fatalf("agent output specialization leaked: result=%+v requests=%d err=%v", plain, requests, err)
	}
}

func TestRunPartsAsUsesMultimodalPrompt(t *testing.T) {
	contents := []ai.UserContent{
		ai.TextContent{Text: "inspect"},
		ai.ImageURL{URL: "https://example.com/image.png"},
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request := messages[0].(ai.ModelRequest)
		prompt := request.Parts[0].(ai.UserPromptPart)
		if !reflect.DeepEqual(prompt.Contents, contents) || params.OutputTool == nil {
			t.Fatalf("unexpected specialized multimodal request: messages=%+v params=%+v", messages, params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "report", Args: []byte(`{"answer":"image"}`),
		}}}, nil
	})
	result, err := ai.RunPartsAs[runReport](t.Context(), ai.NewAgent[deps, string](model), contents, deps{})
	if err != nil || result.Output.Answer != "image" {
		t.Fatalf("unexpected multimodal result=%+v err=%v", result, err)
	}
}

func TestRunStreamAsProducesTypedOutputs(t *testing.T) {
	model := &fakes.TestModel{CustomOutputArgs: []byte(`{"answer":"streamed"}`)}
	agent := ai.NewAgent[deps, string](model)
	stream := ai.RunStreamAs[runReport](t.Context(), agent, "go", deps{})
	var outputs []runReport
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	result := stream.Result()
	if result == nil || result.Output.Answer != "streamed" ||
		len(outputs) == 0 || outputs[len(outputs)-1].Answer != "streamed" {
		t.Fatalf("unexpected streamed specialized output: outputs=%+v result=%+v", outputs, result)
	}

	stopped := ai.RunStreamAs[runReport](t.Context(), agent, "stop", deps{})
	for range stopped.Events() {
		break
	}
	if stopped.Result() != nil {
		t.Fatal("stopped specialized stream produced a result")
	}
}

func TestRunStreamPartsAsProducesTypedResult(t *testing.T) {
	model := &fakes.TestModel{CustomOutputArgs: []byte(`{"answer":"parts"}`)}
	stream := ai.RunStreamPartsAs[runReport](
		t.Context(), ai.NewAgent[deps, string](model), []ai.UserContent{ai.TextContent{Text: "go"}}, deps{},
	)
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil || stream.Result().Output.Answer != "parts" {
		t.Fatalf("unexpected streamed parts result: %+v", stream.Result())
	}
}

func TestRunAsRejectsAgentOutputValidators(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], string) error { return nil })
	if _, err := ai.RunAs[runReport](t.Context(), agent, "go", deps{}); !errors.Is(err, ai.ErrOutputTypeOverrideWithValidators) {
		t.Fatalf("unexpected output validator error: %v", err)
	}
	if _, err := ai.RunPartsAs[runReport](
		t.Context(), agent, []ai.UserContent{ai.TextContent{Text: "go"}}, deps{},
	); !errors.Is(err, ai.ErrOutputTypeOverrideWithValidators) {
		t.Fatalf("unexpected multimodal output validator error: %v", err)
	}
	stream := ai.RunStreamAs[runReport](t.Context(), agent, "go", deps{})
	var streamErr error
	for _, err := range stream.Events() {
		streamErr = err
	}
	if !errors.Is(streamErr, ai.ErrOutputTypeOverrideWithValidators) || stream.Result() != nil {
		t.Fatalf("unexpected streamed output validator error: %v", streamErr)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("typed run did not freeze agent configuration")
		}
	}()
	agent.AddInstructionsFunc(func(context.Context, *ai.RunContext[deps]) (string, error) { return "late", nil })
}

func TestRunAsSupportsConcurrentOutputTypes(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		properties := params.OutputTool.Schema["properties"].(map[string]any)
		if _, ok := properties["answer"]; ok {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: params.OutputTool.Name, ToolCallID: "report", Args: []byte(`{"answer":"ok"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "count", Args: []byte(`{"count":7}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	var group sync.WaitGroup
	group.Add(2)
	var report *ai.RunResult[runReport]
	var count *ai.RunResult[runCount]
	var reportErr, countErr error
	go func() {
		defer group.Done()
		report, reportErr = ai.RunAs[runReport](t.Context(), agent, "report", deps{})
	}()
	go func() {
		defer group.Done()
		count, countErr = ai.RunAs[runCount](t.Context(), agent, "count", deps{})
	}()
	group.Wait()
	if reportErr != nil || countErr != nil || report.Output.Answer != "ok" || count.Output.Count != 7 {
		t.Fatalf("concurrent output types leaked: report=%+v count=%+v errors=%v/%v", report, count, reportErr, countErr)
	}
}
