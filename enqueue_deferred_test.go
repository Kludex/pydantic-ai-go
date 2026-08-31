package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestQueuedMessagesSurviveRepeatedDeferredPauses(t *testing.T) {
	modelCalls := 0
	var asapID string
	var idleID string
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		modelCalls++
		switch modelCalls {
		case 1:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "remote", ToolCallID: "remote", Args: json.RawMessage(`{}`)},
				ai.ToolCallPart{ToolName: "local", ToolCallID: "local", Args: json.RawMessage(`{}`)},
			}}, nil
		case 2:
			callIndex := -1
			resultIndex := -1
			asapIndex := -1
			for index, message := range messages {
				switch message := message.(type) {
				case ai.ModelResponse:
					if len(message.ToolCalls()) > 0 {
						callIndex = index
					}
					if _, exists := message.Metadata[ai.PendingMessagesMetadataKey]; exists {
						t.Fatalf("pending-message metadata reached the model: %+v", message.Metadata)
					}
				case ai.ModelRequest:
					for _, part := range message.Parts {
						switch part := part.(type) {
						case ai.ToolReturnPart:
							if part.ToolCallID == "remote" && part.Content == "external result" {
								resultIndex = index
							}
						case ai.UserPromptPart:
							if part.Content == "queued ASAP" {
								asapIndex = index
							}
						}
					}
				}
			}
			if callIndex < 0 || resultIndex <= callIndex || asapIndex < resultIndex {
				t.Fatalf(
					"deferred result and ASAP message were ordered incorrectly: call=%d result=%d asap=%d messages=%+v",
					callIndex, resultIndex, asapIndex, messages,
				)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "candidate"}}}, nil
		case 3:
			latest := messages[len(messages)-1].(ai.ModelRequest)
			if latest.Parts[0].(ai.UserPromptPart).Content != "queued when idle" {
				t.Fatalf("idle message was not preserved: %+v", latest)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "final"}}}, nil
		default:
			return nil, errors.New("unexpected model request")
		}
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "remote", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (any, error) {
		if rc.ToolCallApproved {
			return ai.RequestExternalToolExecution(map[string]any{"stage": 2}), nil
		}
		var err error
		asapID, err = rc.Enqueue(ai.UserPromptPart{Content: "queued ASAP"})
		if err != nil {
			return nil, err
		}
		idleID, err = rc.EnqueueWhenIdle(ai.UserPromptPart{Content: "queued when idle"})
		if err != nil {
			return nil, err
		}
		return ai.RequestToolApproval(map[string]any{"stage": 1}), nil
	}, ai.WithDynamicApproval(), ai.WithDynamicExternalExecution())
	ai.AddSimpleTool(agent, "local", func(context.Context, struct{}) (string, error) {
		return "local result", nil
	})

	first, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || first.Deferred() == nil || asapID == "" || idleID == "" {
		t.Fatalf("unexpected first pause result=%+v asap=%q idle=%q err=%v", first, asapID, idleID, err)
	}
	firstHistory, err := ai.MarshalMessages(first.Messages())
	if err != nil {
		t.Fatal(err)
	}
	decodedFirst, err := ai.UnmarshalMessages(firstHistory)
	if err != nil {
		t.Fatal(err)
	}

	second, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(decodedFirst),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"remote": ai.ApproveTool(),
		}}),
	)
	if err != nil || second.Deferred() == nil || modelCalls != 1 {
		t.Fatalf("queued messages escaped during re-deferral: result=%+v calls=%d err=%v", second, modelCalls, err)
	}
	secondHistory, err := ai.MarshalMessages(second.Messages())
	if err != nil {
		t.Fatal(err)
	}
	decodedSecond, err := ai.UnmarshalMessages(secondHistory)
	if err != nil {
		t.Fatal(err)
	}

	stream := agent.RunStream(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(decodedSecond),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{
			"remote": "external result",
		}}),
	)
	var enqueueIDs []string
	for event, eventErr := range stream.Events() {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if event, ok := event.(ai.EnqueuedMessagesEvent); ok {
			enqueueIDs = append(enqueueIDs, event.EnqueueID)
		}
	}
	if stream.Result() == nil || stream.Result().Output != "final" || modelCalls != 3 ||
		!slices.Equal(enqueueIDs, []string{asapID, idleID}) {
		t.Fatalf(
			"unexpected resumed queue result=%+v calls=%d enqueue_ids=%v",
			stream.Result(), modelCalls, enqueueIDs,
		)
	}
	for _, message := range stream.Result().Messages() {
		if response, ok := message.(ai.ModelResponse); ok {
			if _, exists := response.Metadata[ai.PendingMessagesMetadataKey]; exists {
				t.Fatalf("restored metadata remained in completed history: %+v", response.Metadata)
			}
		}
	}
}

func TestPendingMessagePersistenceErrorsFailDeferredRun(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "remote", ToolCallID: "remote", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "remote", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (any, error) {
		_, err := rc.Enqueue(ai.ModelRequest{
			Parts:    []ai.RequestPart{ai.UserPromptPart{Content: "later"}},
			Metadata: map[string]any{"invalid": make(chan struct{})},
		})
		if err != nil {
			return nil, err
		}
		return ai.RequestExternalToolExecution(nil), nil
	}, ai.WithDynamicExternalExecution())
	result, err := agent.Run(t.Context(), "go", deps{})
	if result != nil || err == nil || !strings.Contains(err.Error(), "ai: persist pending messages:") {
		t.Fatalf("unexpected pending persistence result=%+v err=%v", result, err)
	}
}

func TestPendingMessageMetadataValidation(t *testing.T) {
	validRequest, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{
		Parts: []ai.RequestPart{ai.UserPromptPart{Content: "later"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	responseOnly, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelResponse{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "wrong"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for name, test := range map[string]struct {
		value any
		want  string
	}{
		"type": {value: 42, want: "metadata has type int, expected string"},
		"json": {value: "{", want: "unexpected end of JSON input"},
		"id": {
			value: `[ {"enqueue_id":"","priority":"asap","messages":` + strconv.Quote(string(validRequest)) + `} ]`,
			want:  "enqueue ID must not be empty",
		},
		"priority": {
			value: `[ {"enqueue_id":"id","priority":"later","messages":` + strconv.Quote(string(validRequest)) + `} ]`,
			want:  `invalid priority "later"`,
		},
		"messages": {
			value: `[ {"enqueue_id":"id","priority":"asap","messages":"{\"bad\":true}"} ]`,
			want:  "cannot unmarshal object",
		},
		"empty": {
			value: `[ {"enqueue_id":"id","priority":"asap","messages":"[]"} ]`,
			want:  "message group must not be empty",
		},
		"ending": {
			value: `[ {"enqueue_id":"id","priority":"asap","messages":` + strconv.Quote(string(responseOnly)) + `} ]`,
			want:  "message group must end with a ModelRequest",
		},
	} {
		t.Run(name, func(t *testing.T) {
			history := []ai.ModelMessage{ai.ModelResponse{Metadata: map[string]any{
				ai.PendingMessagesMetadataKey: test.value,
			}}}
			_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
				t.Context(), "go", deps{}, ai.WithMessageHistory(history),
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected %s metadata error: %v", name, err)
			}
		})
	}
}

func TestStoppingRestoredEnqueueEventCancelsBeforeModelRequest(t *testing.T) {
	modelCalls := 0
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		modelCalls++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "approve", ToolCallID: "approve", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "approve", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (any, error) {
		if rc.ToolCallApproved {
			return "approved", nil
		}
		if _, err := rc.Enqueue(ai.UserPromptPart{Content: "later"}); err != nil {
			return nil, err
		}
		return ai.RequestToolApproval(nil), nil
	}, ai.WithDynamicApproval())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil {
		t.Fatalf("unexpected approval pause result=%+v err=%v", paused, err)
	}
	stream := agent.RunStream(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"approve": ai.ApproveTool(),
		}}),
	)
	stopped := false
	for event, eventErr := range stream.Events() {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if _, ok := event.(ai.EnqueuedMessagesEvent); ok {
			stopped = true
			break
		}
	}
	if !stopped || stream.Result() != nil || modelCalls != 1 {
		t.Fatalf("restored enqueue did not stop before request: stopped=%v result=%+v calls=%d", stopped, stream.Result(), modelCalls)
	}
}

func TestQueuedMessagesRemainInCancellationHistory(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "cancel", ToolCallID: "cancel", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "cancel", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		if _, err := rc.Enqueue(ai.UserPromptPart{Content: "survive cancellation"}); err != nil {
			return "", err
		}
		rc.Cancel()
		return "canceled", nil
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	var cancelled *ai.RunCancelledError
	if !errors.As(err, &cancelled) {
		t.Fatalf("expected cancellation snapshot, got %v", err)
	}
	encoded, err := ai.MarshalMessages(cancelled.Messages())
	if err != nil {
		t.Fatal(err)
	}
	history, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	seen := false
	resumeModel := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		for _, message := range messages {
			request, ok := message.(ai.ModelRequest)
			if !ok {
				continue
			}
			for _, part := range request.Parts {
				if prompt, ok := part.(ai.UserPromptPart); ok && prompt.Content == "survive cancellation" {
					seen = true
				}
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	result, err := ai.NewAgent[deps, string](resumeModel).Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
	)
	if err != nil || result.Output != "done" || !seen {
		t.Fatalf("queued cancellation message was lost: result=%+v seen=%v err=%v", result, seen, err)
	}
}
