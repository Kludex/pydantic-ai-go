package evals_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aievals "github.com/Kludex/pydantic-ai-go/ai/evals"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	pydanticevals "github.com/Kludex/pydantic-evals-go"
)

type attributedModel struct {
	mu       sync.Mutex
	messages []ai.ModelMessage
	params   ai.ModelRequestParams
	usage    ai.Usage
	err      error
}

func (*attributedModel) Name() string { return "configured-model" }

func (model *attributedModel) Request(
	_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.mu.Lock()
	defer model.mu.Unlock()
	model.messages = messages
	model.params = params
	if model.err != nil {
		return nil, model.err
	}
	return &ai.ModelResponse{
		Parts:              []ai.ResponsePart{ai.TextPart{Content: "answer"}},
		Usage:              model.usage,
		ModelName:          "response-model",
		ProviderName:       "provider",
		ProviderResponseID: "response-id",
	}, nil
}

func TestTextTaskEvaluatesAgentOutputAndRecordsRunData(t *testing.T) {
	cost := 0.25
	model := &attributedModel{usage: ai.Usage{
		Requests: 2, InputTokens: 4, OutputTokens: 5,
		CacheWriteTokens: 6, CacheReadTokens: 7, InputAudioTokens: 8,
		CacheAudioReadTokens: 9, OutputAudioTokens: 10, ReasoningTokens: 11,
		AcceptedPredictionTokens: 12, RejectedPredictionTokens: 13,
		Details: map[string]int{"custom": 14}, CostUSD: &cost,
	}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithMetadata(map[string]any{
		"suite": map[string]any{"name": "smoke"},
	}))
	task := aievals.NewTextTask(agent, struct{}{})
	dataset, err := pydanticevals.NewDataset[string, string, any]("agent", []pydanticevals.Case[string, string, any]{
		pydanticevals.NewCase[string, string, any]("question"),
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := dataset.Evaluate(t.Context(), task)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Cases) != 1 || report.Cases[0].Output != "answer" {
		t.Fatalf("unexpected evaluation report: %+v", report)
	}
	result := report.Cases[0]
	wantMetrics := map[string]float64{
		"pydantic_ai.requests":     2,
		"pydantic_ai.input_tokens": 4, "pydantic_ai.output_tokens": 5,
		"pydantic_ai.total_tokens": 9, "pydantic_ai.cache_write_tokens": 6,
		"pydantic_ai.cache_read_tokens": 7, "pydantic_ai.input_audio_tokens": 8,
		"pydantic_ai.cache_audio_read_tokens": 9, "pydantic_ai.output_audio_tokens": 10,
		"pydantic_ai.reasoning_tokens": 11, "pydantic_ai.accepted_prediction_tokens": 12,
		"pydantic_ai.rejected_prediction_tokens": 13,
		"pydantic_ai.usage.details.custom":       14, "pydantic_ai.cost_usd": 0.25,
	}
	for name, want := range wantMetrics {
		if result.Metrics[name] != want {
			t.Fatalf("metric %q=%v, want %v: %#v", name, result.Metrics[name], want, result.Metrics)
		}
	}
	for name, want := range map[string]string{
		"pydantic_ai.model_name": "response-model", "pydantic_ai.provider_name": "provider",
		"pydantic_ai.provider_response_id": "response-id",
	} {
		if result.Attributes[name] != want {
			t.Fatalf("attribute %q=%v, want %q", name, result.Attributes[name], want)
		}
	}
	if result.Attributes["pydantic_ai.run_id"] == "" || result.Attributes["pydantic_ai.conversation_id"] == "" {
		t.Fatalf("run identity was not recorded: %#v", result.Attributes)
	}
	metadata := result.Attributes["pydantic_ai.metadata"].(map[string]any)
	if metadata["suite"].(map[string]any)["name"] != "smoke" {
		t.Fatalf("application metadata was not recorded: %#v", metadata)
	}
	recordedUsage := result.Attributes["pydantic_ai.usage"].(ai.Usage)
	if recordedUsage.CostUSD == nil || *recordedUsage.CostUSD != 0.25 || recordedUsage.Details["custom"] != 14 {
		t.Fatalf("complete usage was not recorded: %+v", recordedUsage)
	}
}

func TestTaskMapsMultimodalInputsAndRunOptions(t *testing.T) {
	model := &attributedModel{}
	type deps struct{ Instruction string }
	agent := ai.NewAgent[deps, string](model)
	agent.AddInstructionsFunc(func(_ context.Context, runContext *ai.RunContext[deps]) (string, error) {
		return runContext.Deps.Instruction, nil
	})
	task := aievals.NewTask(agent, func(_ context.Context, input string) (aievals.AgentInput[deps], error) {
		return aievals.AgentInput[deps]{
			Prompt: ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: input}, ai.ImageURL{URL: "https://example.com/image.png"},
			}},
			Deps: deps{Instruction: "dependency instruction"},
			Options: []ai.RunOption{
				ai.WithRunInstructions("run instruction"),
			},
		}, nil
	})
	output, err := task(t.Context(), "describe")
	if err != nil || output != "answer" {
		t.Fatalf("unexpected mapped output=%q err=%v", output, err)
	}
	request := model.messages[0].(ai.ModelRequest)
	prompt := request.Parts[0].(ai.UserPromptPart)
	if len(prompt.Contents) != 2 || prompt.Contents[0].(ai.TextContent).Text != "describe" ||
		model.params.Instructions != "run instruction\n\ndependency instruction" {
		t.Fatalf("mapped input was not preserved: prompt=%+v params=%+v", prompt, model.params)
	}
}

func TestTextTaskRecordsSuccessfulToolCalls(t *testing.T) {
	var calls atomic.Int32
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if calls.Add(1) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"lookup", func(context.Context, struct{}) (string, error) { return "value", nil },
	))
	dataset, err := pydanticevals.NewDataset[string, string, any]("agent", []pydanticevals.Case[string, string, any]{
		pydanticevals.NewCase[string, string, any]("question"),
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := dataset.Evaluate(t.Context(), aievals.NewTextTask(agent, struct{}{}))
	if err != nil {
		t.Fatal(err)
	}
	if report.Cases[0].Metrics["pydantic_ai.tool_calls"] != 1 {
		t.Fatalf("tool usage was not recorded: %#v", report.Cases[0].Metrics)
	}
}

func newTestAgent() *ai.Agent[struct{}, string] {
	return ai.NewAgent[struct{}, string](fakes.NewTestModel())
}

func TestTaskErrors(t *testing.T) {
	assertPanic(t, "agent must not be nil", func() {
		var agent *ai.Agent[struct{}, string]
		aievals.NewTextTask(agent, struct{}{})
	})
	assertPanic(t, "input mapper must not be nil", func() {
		aievals.NewTask[string](newTestAgent(), nil)
	})

	mapperErr := errors.New("mapping failed")
	mappingTask := aievals.NewTask(newTestAgent(), func(
		context.Context, string,
	) (aievals.AgentInput[struct{}], error) {
		return aievals.AgentInput[struct{}]{}, mapperErr
	})
	if _, err := mappingTask(t.Context(), "input"); !errors.Is(err, mapperErr) ||
		!strings.Contains(err.Error(), "map task input") {
		t.Fatalf("unexpected mapper error: %v", err)
	}

	invalidPromptTask := aievals.NewTask(newTestAgent(), func(
		context.Context, string,
	) (aievals.AgentInput[struct{}], error) {
		return aievals.AgentInput[struct{}]{Prompt: ai.UserPromptPart{
			Content: "text", Contents: []ai.UserContent{ai.TextContent{Text: "other"}},
		}}, nil
	})
	if _, err := invalidPromptTask(t.Context(), "input"); err == nil ||
		err.Error() != "ai/evals: prompt cannot contain both text Content and multimodal Contents" {
		t.Fatalf("unexpected invalid-prompt error: %v", err)
	}

	requestErr := errors.New("request failed")
	failingTask := aievals.NewTextTask(
		ai.NewAgent[struct{}, string](&attributedModel{err: requestErr}), struct{}{},
	)
	if _, err := failingTask(t.Context(), "input"); !errors.Is(err, requestErr) {
		t.Fatalf("unexpected agent error: %v", err)
	}
}

func TestTaskRejectsDeferredAgentRun(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "approve", ToolCallID: "call", Args: []byte(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"approve", func(context.Context, struct{}) (string, error) { return "approved", nil },
		ai.WithApprovalRequired(),
	))
	_, err := aievals.NewTextTask(agent, struct{}{})(t.Context(), "input")
	if !errors.Is(err, aievals.ErrDeferredRun) || !strings.Contains(err.Error(), "1 tool calls") {
		t.Fatalf("unexpected deferred-run error: %v", err)
	}
}

func TestTextTaskOmitsZeroAndEmptyRunData(t *testing.T) {
	task := aievals.NewTextTask(newTestAgent(), struct{}{})
	dataset, err := pydanticevals.NewDataset[string, string, any]("agent", []pydanticevals.Case[string, string, any]{
		pydanticevals.NewCase[string, string, any]("question"),
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := dataset.Evaluate(t.Context(), task)
	if err != nil {
		t.Fatal(err)
	}
	result := report.Cases[0]
	if _, exists := result.Metrics["pydantic_ai.cost_usd"]; exists {
		t.Fatalf("unknown cost was recorded: %#v", result.Metrics)
	}
	if _, exists := result.Attributes["pydantic_ai.metadata"]; exists {
		t.Fatalf("empty metadata was recorded: %#v", result.Attributes)
	}
	if _, exists := result.Attributes["pydantic_ai.provider_name"]; exists {
		t.Fatalf("empty provider identity was recorded: %#v", result.Attributes)
	}
}

func assertPanic(t *testing.T, contains string, call func()) {
	t.Helper()
	defer func() {
		value := recover()
		if value == nil || !strings.Contains(value.(string), contains) {
			t.Fatalf("panic=%v, want containing %q", value, contains)
		}
	}()
	call()
}
