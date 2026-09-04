package ai_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestRunPersistsInstructionsOnEverySentRequest(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		latest := messages[len(messages)-1].(ai.ModelRequest)
		want := fmt.Sprintf("Static.\n\nDynamic step %d.", request)
		if latest.Instructions != want || params.Instructions != want {
			t.Fatalf("request %d instructions were not persisted: message=%q params=%q", request,
				latest.Instructions, params.Instructions)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithInstructions("Static."))
	agent.AddInstructionsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (string, error) {
		return fmt.Sprintf("Dynamic step %d.", rc.RunStep), nil
	})
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "ok", nil })
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	messages := result.Messages()
	if len(messages) != 4 ||
		messages[0].(ai.ModelRequest).Instructions != "Static.\n\nDynamic step 1." ||
		messages[2].(ai.ModelRequest).Instructions != "Static.\n\nDynamic step 2." {
		t.Fatalf("result history lost request instructions: %+v", messages)
	}
}

type rewriteInstructionsCapability struct{}

func (rewriteInstructionsCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (rewriteInstructionsCapability) WrapModelRequest(
	ctx context.Context,
	_ *ai.RunInfo,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	params.Instructions = "Wrapped instructions."
	return next(ctx, messages, params)
}

type appendSyntheticResponseCapability struct{}

func (appendSyntheticResponseCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (appendSyntheticResponseCapability) WrapModelRequest(
	ctx context.Context,
	_ *ai.RunInfo,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	request := messages[len(messages)-1].(ai.ModelRequest)
	request.Timestamp = time.Time{}
	request.RunID = ""
	request.ConversationID = ""
	messages[len(messages)-1] = request
	messages = append(messages, ai.ModelResponse{})
	return next(ctx, messages, params)
}

func TestModelRequestMiddlewarePersistsEffectiveInstructions(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request := messages[len(messages)-1].(ai.ModelRequest)
		if request.Instructions != "Wrapped instructions." || params.Instructions != request.Instructions {
			t.Fatalf("middleware instructions were not persisted: request=%+v params=%+v", request, params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model,
		ai.WithInstructions("Original instructions."),
		ai.WithCapabilities(rewriteInstructionsCapability{}),
	)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Messages()[0].(ai.ModelRequest).Instructions != "Wrapped instructions." {
		t.Fatalf("effective instructions missing from result: %+v", result.Messages())
	}
}

func TestEffectiveInstructionsFindLatestRequestInMiddlewareMessages(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request := messages[len(messages)-2].(ai.ModelRequest)
		if request.Instructions != "Instructions." {
			t.Fatalf("instructions were not attached before a synthetic response: %+v", messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model,
		ai.WithInstructions("Instructions."),
		ai.WithCapabilities(appendSyntheticResponseCapability{}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryKeepsRequestsWithDifferentInstructionsSeparate(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Instructions: "First.", Parts: []ai.RequestPart{ai.UserPromptPart{Content: "one"}}},
		ai.ModelRequest{Instructions: "Second.", Parts: []ai.RequestPart{ai.UserPromptPart{Content: "two"}}},
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) != 3 || messages[0].(ai.ModelRequest).Instructions != "First." ||
			messages[1].(ai.ModelRequest).Instructions != "Second." {
			t.Fatalf("requests with different instructions were merged: %+v", messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "three", deps{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryMergesCompatibleRequestInstructions(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Instructions: "Shared.", Parts: []ai.RequestPart{ai.UserPromptPart{Content: "one"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "two"}}},
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) != 2 || messages[0].(ai.ModelRequest).Instructions != "Shared." ||
			len(messages[0].(ai.ModelRequest).Parts) != 2 {
			t.Fatalf("compatible request instructions were not preserved: %+v", messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "three", deps{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
}
