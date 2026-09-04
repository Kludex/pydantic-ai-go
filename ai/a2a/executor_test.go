package a2a_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"iter"
	"strings"
	"testing"

	protocol "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	a2aintegration "github.com/Kludex/pydantic-ai-go/ai/a2a"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type eventModel struct {
	events []ai.ModelStreamEvent
	err    error
}

func (*eventModel) Name() string { return "event-model" }
func (*eventModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return nil, errors.New("unexpected static request")
}
func (model *eventModel) StreamRequest(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		for _, event := range model.events {
			if !yield(event, nil) {
				return
			}
		}
		if model.err != nil {
			yield(nil, model.err)
		}
	}, nil
}

type recordingQueue struct {
	events []protocol.Event
	failAt int
	writes int
}

func (queue *recordingQueue) Write(_ context.Context, event protocol.Event) error {
	queue.writes++
	if queue.failAt > 0 && queue.writes == queue.failAt {
		return errors.New("queue failed")
	}
	queue.events = append(queue.events, event)
	return nil
}
func (queue *recordingQueue) WriteVersioned(context.Context, protocol.Event, protocol.TaskVersion) error {
	return nil
}
func (queue *recordingQueue) Read(context.Context) (protocol.Event, protocol.TaskVersion, error) {
	return nil, protocol.TaskVersionMissing, errors.New("not implemented")
}
func (*recordingQueue) Close() error { return nil }

func TestExecutorStreamsArtifacts(t *testing.T) {
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "hello"},
			ai.FilePart{Content: ai.BinaryContent{Data: []byte("file"), MediaType: "text/plain"}},
		}}, nil
	})
	agent := ai.NewAgent[string, string](model)
	config := a2aintegration.Config[string]{
		ResolveDeps:         func(context.Context, *a2asrv.RequestContext) (string, error) { return "deps", nil },
		ArtifactName:        "answer",
		ArtifactDescription: "Generated answer",
		ArtifactExtensions:  []string{"urn:example:answer"},
		ArtifactMetadata:    map[string]any{"nested": map[string]any{"stable": true}},
	}
	executor := a2aintegration.NewExecutor(agent, config)
	config.ArtifactExtensions[0] = "changed"
	config.ArtifactMetadata["nested"].(map[string]any)["stable"] = false
	requestMessage := protocol.NewMessage(protocol.MessageRoleUser,
		protocol.TextPart{Text: "say hello"},
		protocol.DataPart{Data: map[string]any{"language": "en"}},
		protocol.FilePart{File: protocol.FileBytes{
			FileMeta: protocol.FileMeta{MimeType: "text/plain", Name: "input.txt"},
			Bytes:    base64.StdEncoding.EncodeToString([]byte("input")),
		}},
	)
	stored := &protocol.Task{History: []*protocol.Message{
		protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "old"}),
		protocol.NewMessage(protocol.MessageRoleAgent,
			protocol.TextPart{Text: "answer"}, protocol.DataPart{Data: map[string]any{"ok": true}},
			protocol.FilePart{File: protocol.FileBytes{
				FileMeta: protocol.FileMeta{MimeType: "text/plain"},
				Bytes:    base64.StdEncoding.EncodeToString([]byte("history")),
			}},
		),
		requestMessage,
	}}
	related := &protocol.Task{
		ID: "related", ContextID: "related-context",
		History: []*protocol.Message{
			nil,
			protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "related question"}),
			protocol.NewMessage(protocol.MessageRoleAgent, protocol.TextPart{Text: "related answer"}),
		},
		Artifacts: []*protocol.Artifact{nil, {}, {
			ID: "related-artifact", Parts: protocol.ContentParts{
				protocol.TextPart{Text: "artifact text"}, protocol.DataPart{Data: map[string]any{"value": true}},
			},
		}},
	}
	request := &a2asrv.RequestContext{
		Message: requestMessage, StoredTask: stored, TaskID: "task", ContextID: "context",
		RelatedTasks: []*protocol.Task{nil, {ID: "task"}, related, related},
	}
	queue := &recordingQueue{}
	if err := executor.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	if len(captured) < 5 {
		t.Fatalf("related history, task history, and prompt were not passed to the agent: %#v", captured)
	}
	relatedRequest := captured[0].(ai.ModelRequest)
	marker := relatedRequest.Parts[0].(ai.UserPromptPart)
	relatedUser := relatedRequest.Parts[1].(ai.UserPromptPart)
	relatedResponse := captured[1].(ai.ModelResponse)
	if marker.Content != `Related A2A task "related":` || relatedUser.Contents[0].(ai.TextContent).Text != "related question" ||
		relatedResponse.Parts[0].(ai.TextPart).Content != "related answer" || len(relatedResponse.Parts) != 3 ||
		relatedResponse.Parts[1].(ai.TextPart).Content != "artifact text" ||
		relatedResponse.Parts[2].(ai.TextPart).Content != `{"value":true}` {
		t.Fatalf("unexpected related task context: %#v", captured[:2])
	}
	if len(queue.events) != 4 {
		t.Fatalf("unexpected events: %#v", queue.events)
	}
	if status := queue.events[0].(*protocol.TaskStatusUpdateEvent); status.Status.State != protocol.TaskStateWorking {
		t.Fatalf("unexpected initial status: %#v", status)
	}
	firstArtifact := queue.events[1].(*protocol.TaskArtifactUpdateEvent)
	secondArtifact := queue.events[2].(*protocol.TaskArtifactUpdateEvent)
	if secondArtifact.Artifact.ID != firstArtifact.Artifact.ID || !secondArtifact.Append {
		t.Fatalf("artifact updates did not share identity: %#v %#v", firstArtifact, secondArtifact)
	}
	if firstArtifact.Artifact.Name != "answer" || firstArtifact.Artifact.Description != "Generated answer" ||
		firstArtifact.Artifact.Extensions[0] != "urn:example:answer" ||
		!firstArtifact.Artifact.Metadata["nested"].(map[string]any)["stable"].(bool) ||
		secondArtifact.Artifact.Name != "" || secondArtifact.Artifact.Metadata != nil {
		t.Fatalf("unexpected artifact presentation: %#v %#v", firstArtifact, secondArtifact)
	}
	if status := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent); status.Status.State != protocol.TaskStateCompleted || !status.Final {
		t.Fatalf("unexpected completion: %#v", status)
	}
}

func TestExecutorStreamingDeltas(t *testing.T) {
	file := ai.FilePart{Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}}
	model := &eventModel{events: []ai.ModelStreamEvent{
		ai.TextDeltaEvent{PartID: "text", Delta: "one"},
		ai.TextDeltaEvent{PartID: "text", Delta: "two"},
		ai.ThinkingDeltaEvent{PartID: "thinking", Delta: "hidden"},
		ai.FileEvent{PartID: "file", Part: file},
		ai.FileEvent{PartID: "file", Part: file, Replace: true},
		ai.FinishEvent{FinishReason: ai.FinishReasonStop},
	}}
	executor := a2aintegration.NewExecutor(
		ai.NewAgent[struct{}, string](model), a2aintegration.Config[struct{}]{},
	)
	request := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
		TaskID:  "task", ContextID: "context", StoredTask: &protocol.Task{},
	}
	queue := &recordingQueue{}
	if err := executor.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	artifacts := 0
	for _, event := range queue.events {
		if _, ok := event.(*protocol.TaskArtifactUpdateEvent); ok {
			artifacts++
		}
	}
	if artifacts != 4 {
		t.Fatalf("unexpected artifact count: %d events=%#v", artifacts, queue.events)
	}
}

func TestExecutorArtifactAndOutputFailures(t *testing.T) {
	request := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
		TaskID:  "task", ContextID: "context", StoredTask: &protocol.Task{},
	}
	streaming := a2aintegration.NewExecutor(
		ai.NewAgent[struct{}, string](&eventModel{events: []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "one"},
			ai.TextDeltaEvent{PartID: "text", Delta: "two"},
			ai.FinishEvent{},
		}}), a2aintegration.Config[struct{}]{},
	)
	if err := streaming.Execute(context.Background(), request, &recordingQueue{failAt: 3}); err == nil {
		t.Fatal("expected delta artifact failure")
	}

	model := fakes.NewTestModel()
	outputFunction := ai.NewOutputFunction[struct{}, struct {
		Value string `json:"value"`
	}, chan int]("output", func(context.Context, *ai.RunContext[struct{}], struct {
		Value string `json:"value"`
	}) (chan int, error) {
		return make(chan int), nil
	})
	agent := ai.NewOutputFunctionAgent(model, outputFunction)
	executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{})
	queue := &recordingQueue{}
	if err := executor.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	if queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent).Status.State != protocol.TaskStateFailed {
		t.Fatalf("unexpected output encoding failure: %#v", queue.events)
	}

	model = fakes.NewTestModel()
	type output struct {
		Answer string `json:"answer"`
	}
	structured := a2aintegration.NewExecutor(
		ai.NewAgent[struct{}, output](model), a2aintegration.Config[struct{}]{},
	)
	newRequest := *request
	newRequest.StoredTask = nil
	if err := structured.Execute(context.Background(), &newRequest, &recordingQueue{failAt: 3}); err == nil {
		t.Fatal("expected structured artifact failure")
	}
}

func TestExecutorNewTaskAndStructuredOutput(t *testing.T) {
	type output struct {
		Answer string `json:"answer"`
	}
	model := fakes.NewTestModel()
	model.CustomOutputArgs = []byte(`{"answer":"yes"}`)
	agent := ai.NewAgent[struct{}, output](model)
	executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{})
	request := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "answer"}),
		TaskID:  "task", ContextID: "context",
	}
	queue := &recordingQueue{}
	if err := executor.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	if len(queue.events) != 4 || queue.events[0].(*protocol.TaskStatusUpdateEvent).Status.State != protocol.TaskStateSubmitted {
		t.Fatalf("unexpected new task events: %#v", queue.events)
	}
	artifact := queue.events[2].(*protocol.TaskArtifactUpdateEvent)
	if artifact.Artifact.Parts[0].(protocol.TextPart).Text != `{"answer":"yes"}` {
		t.Fatalf("unexpected structured artifact: %#v", artifact)
	}
}

func TestExecutorDeferredAndCancellation(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{Name: "approve", Schema: map[string]any{"type": "object"}},
		func(context.Context, json.RawMessage) (any, error) { return "done", nil }, ai.WithApprovalRequired())
	executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{})
	request := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
		TaskID:  "task", ContextID: "context",
	}
	queue := &recordingQueue{}
	if err := executor.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	status := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent)
	if status.Status.State != protocol.TaskStateInputRequired || !status.Final || status.Status.Message == nil {
		t.Fatalf("unexpected deferred status: %#v", status)
	}
	state := status.Status.Message.Parts[0].(protocol.DataPart).Data
	approvals := state["approvals"].([]any)
	approvalID := approvals[0].(map[string]any)["id"].(string)
	resultPart, err := a2aintegration.NewDeferredResultsPart(a2aintegration.DeferredResults{
		Approvals: map[string]a2aintegration.DeferredApproval{approvalID: {
			Approved: true, OverrideArgs: json.RawMessage(`{"approved":true}`),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	resume := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, resultPart), TaskID: "task", ContextID: "context",
		StoredTask: &protocol.Task{ID: "task", Status: status.Status, History: []*protocol.Message{request.Message}},
	}
	queue = &recordingQueue{}
	if err := executor.Execute(context.Background(), resume, queue); err != nil {
		t.Fatal(err)
	}
	if resumed := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent); resumed.Status.State != protocol.TaskStateCompleted {
		t.Fatalf("approval did not resume the task: %#v", queue.events)
	}

	queue = &recordingQueue{}
	if err := executor.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	status = queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent)
	state = status.Status.Message.Parts[0].(protocol.DataPart).Data
	approvalID = state["approvals"].([]any)[0].(map[string]any)["id"].(string)
	resultPart, err = a2aintegration.NewDeferredResultsPart(a2aintegration.DeferredResults{
		Approvals: map[string]a2aintegration.DeferredApproval{approvalID: {Message: "not allowed"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	resume.Message = protocol.NewMessage(protocol.MessageRoleUser, resultPart)
	resume.StoredTask = &protocol.Task{ID: "task", Status: status.Status}
	queue = &recordingQueue{}
	if err := executor.Execute(context.Background(), resume, queue); err != nil {
		t.Fatal(err)
	}
	if denied := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent); denied.Status.State != protocol.TaskStateCompleted {
		t.Fatalf("denial did not resume the task: %#v", queue.events)
	}

	if err := executor.Execute(context.Background(), request, &recordingQueue{failAt: 3}); err == nil {
		t.Fatal("expected input-required queue error")
	}

	queue = &recordingQueue{}
	if err := executor.Cancel(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	status = queue.events[0].(*protocol.TaskStatusUpdateEvent)
	if status.Status.State != protocol.TaskStateCanceled || !status.Final {
		t.Fatalf("unexpected canceled status: %#v", status)
	}
}

func TestExecutorExternalDeferredResults(t *testing.T) {
	tests := []struct {
		name     string
		result   a2aintegration.DeferredCallResult
		partKind string
	}{
		{name: "success", result: a2aintegration.DeferredCallResult{
			Outcome: a2aintegration.DeferredCallSucceeded, Value: map[string]any{"answer": 42},
		}, partKind: "return"},
		{name: "failed", result: a2aintegration.DeferredCallResult{
			Outcome: a2aintegration.DeferredCallFailed, Message: "remote failed",
		}, partKind: "return"},
		{name: "retry", result: a2aintegration.DeferredCallResult{
			Outcome: a2aintegration.DeferredCallRetry, Message: "try again",
		}, partKind: "retry"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requests := 0
			model := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				requests++
				if requests == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: "external", ToolCallID: "external-call", Args: json.RawMessage(`{"value":1}`),
					}}}, nil
				}
				latest := messages[len(messages)-1].(ai.ModelRequest).Parts[0]
				if test.partKind == "return" {
					if _, ok := latest.(ai.ToolReturnPart); !ok {
						t.Fatalf("unexpected deferred return: %#v", latest)
					}
				} else if _, ok := latest.(ai.RetryPromptPart); !ok {
					t.Fatalf("unexpected deferred retry: %#v", latest)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			agent := ai.NewAgent[struct{}, string](model)
			agent.AddRawTool(ai.ToolDefinition{Name: "external", Schema: map[string]any{"type": "object"}},
				func(context.Context, json.RawMessage) (any, error) {
					return ai.RequestExternalToolExecution(map[string]any{"queue": "remote"}), nil
				}, ai.WithDynamicExternalExecution())
			executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{})
			request := &a2asrv.RequestContext{
				Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
				TaskID:  "task", ContextID: "context",
			}
			queue := &recordingQueue{}
			if err := executor.Execute(t.Context(), request, queue); err != nil {
				t.Fatal(err)
			}
			status := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent)
			state := status.Status.Message.Parts[0].(protocol.DataPart).Data
			pending := state["calls"].([]any)[0].(map[string]any)
			if pending["id"] != "external-call" || pending["name"] != "external" ||
				pending["metadata"].(map[string]any)["queue"] != "remote" {
				t.Fatalf("unexpected pending external call: %#v", pending)
			}
			part, err := a2aintegration.NewDeferredResultsPart(a2aintegration.DeferredResults{
				Calls:    map[string]a2aintegration.DeferredCallResult{"external-call": test.result},
				Metadata: map[string]map[string]any{"external-call": {"worker": "one"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			resume := &a2asrv.RequestContext{
				Message: protocol.NewMessage(protocol.MessageRoleUser, part), TaskID: "task", ContextID: "context",
				StoredTask: &protocol.Task{ID: "task", Status: status.Status},
			}
			queue = &recordingQueue{}
			if err := executor.Execute(t.Context(), resume, queue); err != nil {
				t.Fatal(err)
			}
			if final := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent); final.Status.State != protocol.TaskStateCompleted {
				t.Fatalf("external result did not resume: %#v", queue.events)
			}
		})
	}
}

func TestExecutorDeferredFailures(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "external", ToolCallID: "external-call", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddRawTool(ai.ToolDefinition{Name: "external", Schema: map[string]any{"type": "object"}},
		func(context.Context, json.RawMessage) (any, error) {
			return ai.RequestExternalToolExecution(nil), nil
		}, ai.WithDynamicExternalExecution())
	executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{})
	initial := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
		TaskID:  "task", ContextID: "context",
	}
	queue := &recordingQueue{}
	if err := executor.Execute(t.Context(), initial, queue); err != nil {
		t.Fatal(err)
	}
	status := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent)
	baseState := status.Status.Message.Parts[0].(protocol.DataPart).Data

	cloneState := func() map[string]any {
		encoded, err := json.Marshal(baseState)
		if err != nil {
			t.Fatal(err)
		}
		var cloned map[string]any
		if err := json.Unmarshal(encoded, &cloned); err != nil {
			t.Fatal(err)
		}
		return cloned
	}
	stored := func(data map[string]any) *protocol.Task {
		return &protocol.Task{ID: "task", Status: protocol.TaskStatus{
			State:   protocol.TaskStateInputRequired,
			Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.DataPart{Data: data}),
		}}
	}
	valid, err := a2aintegration.NewDeferredResultsPart(a2aintegration.DeferredResults{
		Calls: map[string]a2aintegration.DeferredCallResult{"external-call": {
			Outcome: a2aintegration.DeferredCallSucceeded, Value: "done",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	makePart := func(results a2aintegration.DeferredResults) protocol.DataPart {
		part, err := a2aintegration.NewDeferredResultsPart(results)
		if err != nil {
			t.Fatal(err)
		}
		return part
	}

	badStoredEncoding := cloneState()
	badStoredEncoding["calls"] = make(chan int)
	badStoredShape := cloneState()
	badStoredShape["calls"] = "invalid"
	missingMessages := cloneState()
	missingMessages["messages"] = ""
	invalidMessages := cloneState()
	invalidMessages["messages"] = "invalid"
	approvalState := cloneState()
	approvalState["approvals"] = approvalState["calls"]
	delete(approvalState, "calls")

	tests := []struct {
		name   string
		parts  []protocol.Part
		stored *protocol.Task
	}{
		{name: "results without state", parts: []protocol.Part{valid}},
		{name: "missing results", parts: []protocol.Part{protocol.TextPart{Text: "continue"}}, stored: stored(cloneState())},
		{name: "duplicate results", parts: []protocol.Part{valid, valid}, stored: stored(cloneState())},
		{name: "stored encoding", parts: []protocol.Part{valid}, stored: stored(badStoredEncoding)},
		{name: "stored shape", parts: []protocol.Part{valid}, stored: stored(badStoredShape)},
		{name: "missing messages", parts: []protocol.Part{valid}, stored: stored(missingMessages)},
		{name: "invalid messages", parts: []protocol.Part{valid}, stored: stored(invalidMessages)},
		{name: "result encoding", parts: []protocol.Part{protocol.DataPart{Data: map[string]any{
			"type": "pydantic-ai-go/deferred-tool-results", "calls": make(chan int),
		}}}, stored: stored(cloneState())},
		{name: "result shape", parts: []protocol.Part{protocol.DataPart{Data: map[string]any{
			"type": "pydantic-ai-go/deferred-tool-results", "calls": "invalid",
		}}}, stored: stored(cloneState())},
		{name: "incomplete calls", parts: []protocol.Part{makePart(a2aintegration.DeferredResults{})}, stored: stored(cloneState())},
		{name: "missing call", parts: []protocol.Part{makePart(a2aintegration.DeferredResults{Calls: map[string]a2aintegration.DeferredCallResult{
			"other": {Outcome: a2aintegration.DeferredCallSucceeded},
		}})}, stored: stored(cloneState())},
		{name: "invalid outcome", parts: []protocol.Part{makePart(a2aintegration.DeferredResults{Calls: map[string]a2aintegration.DeferredCallResult{
			"external-call": {Outcome: "unknown"},
		}})}, stored: stored(cloneState())},
		{name: "incomplete approvals", parts: []protocol.Part{makePart(a2aintegration.DeferredResults{})}, stored: stored(approvalState)},
		{name: "missing approval", parts: []protocol.Part{makePart(a2aintegration.DeferredResults{Approvals: map[string]a2aintegration.DeferredApproval{
			"other": {Approved: true},
		}})}, stored: stored(approvalState)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := &a2asrv.RequestContext{
				Message: protocol.NewMessage(protocol.MessageRoleUser, test.parts...), StoredTask: test.stored,
				TaskID: "task", ContextID: "context",
			}
			queue := &recordingQueue{}
			if err := executor.Execute(t.Context(), request, queue); err != nil {
				t.Fatal(err)
			}
			final := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent)
			if final.Status.State != protocol.TaskStateFailed {
				t.Fatalf("invalid deferred input was accepted: %#v", queue.events)
			}
		})
	}
	if requests != 1 {
		t.Fatalf("invalid deferred inputs reached the model %d times", requests-1)
	}

	plain := a2aintegration.NewExecutor(
		ai.NewAgent[struct{}, string](fakes.NewTestModel()), a2aintegration.Config[struct{}]{},
	)
	unrelatedState := &protocol.Task{ID: "task", Status: protocol.TaskStatus{
		State:   protocol.TaskStateInputRequired,
		Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.TextPart{Text: "application input"}),
	}}
	if err := plain.Execute(t.Context(), &a2asrv.RequestContext{
		Message:    protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "continue"}),
		StoredTask: unrelatedState, TaskID: "task", ContextID: "context",
	}, &recordingQueue{}); err != nil {
		t.Fatal(err)
	}

	badAgent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	badAgent.AddRawTool(ai.ToolDefinition{Name: "external", Schema: map[string]any{"type": "object"}},
		func(context.Context, json.RawMessage) (any, error) {
			return ai.RequestExternalToolExecution(map[string]any{"invalid": make(chan int)}), nil
		}, ai.WithDynamicExternalExecution())
	badExecutor := a2aintegration.NewExecutor(badAgent, a2aintegration.Config[struct{}]{})
	if err := badExecutor.Execute(t.Context(), initial, &recordingQueue{}); err == nil {
		t.Fatal("non-JSON deferred request metadata was accepted")
	}

	if _, err := a2aintegration.NewDeferredResultsPart(a2aintegration.DeferredResults{
		Calls: map[string]a2aintegration.DeferredCallResult{"call": {
			Outcome: a2aintegration.DeferredCallSucceeded, Value: make(chan int),
		}},
	}); err == nil {
		t.Fatal("non-JSON deferred result was accepted")
	}
}

func TestExecutorInputFilesAndFailures(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	})
	executor := a2aintegration.NewExecutor(
		ai.NewAgent[struct{}, string](model), a2aintegration.Config[struct{}]{},
	)
	uriParts := []protocol.Part{
		protocol.FilePart{File: protocol.FileURI{FileMeta: protocol.FileMeta{MimeType: "image/png"}, URI: "https://example.com/image"}},
		protocol.FilePart{File: protocol.FileURI{FileMeta: protocol.FileMeta{MimeType: "audio/wav"}, URI: "https://example.com/audio"}},
		protocol.FilePart{File: protocol.FileURI{FileMeta: protocol.FileMeta{MimeType: "video/mp4"}, URI: "https://example.com/video"}},
		protocol.FilePart{File: protocol.FileURI{FileMeta: protocol.FileMeta{MimeType: "application/pdf"}, URI: "https://example.com/document"}},
	}
	request := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, uriParts...),
		TaskID:  "task", ContextID: "context", StoredTask: &protocol.Task{},
	}
	if err := executor.Execute(context.Background(), request, &recordingQueue{}); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		message *protocol.Message
		history []*protocol.Message
		related []*protocol.Task
		options ai.MessageSanitizationOptions
	}{
		{name: "empty", message: protocol.NewMessage(protocol.MessageRoleUser)},
		{name: "unsupported part", message: protocol.NewMessage(protocol.MessageRoleUser, nil)},
		{name: "bad bytes", message: protocol.NewMessage(protocol.MessageRoleUser,
			protocol.FilePart{File: protocol.FileBytes{Bytes: "%%%"}})},
		{name: "missing file", message: protocol.NewMessage(protocol.MessageRoleUser,
			protocol.FilePart{})},
		{name: "bad user history", message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
			history: []*protocol.Message{protocol.NewMessage(protocol.MessageRoleUser)}},
		{name: "bad history role", message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
			history: []*protocol.Message{protocol.NewMessage(protocol.MessageRoleUnspecified, protocol.TextPart{Text: "bad"})}},
		{name: "bad agent URI", message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
			history: []*protocol.Message{protocol.NewMessage(protocol.MessageRoleAgent,
				protocol.FilePart{File: protocol.FileURI{URI: "https://example.com/file"}})}},
		{name: "bad agent bytes", message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
			history: []*protocol.Message{protocol.NewMessage(protocol.MessageRoleAgent,
				protocol.FilePart{File: protocol.FileBytes{Bytes: "%%%"}})}},
		{name: "related history", message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
			related: []*protocol.Task{{ID: "related", History: []*protocol.Message{
				protocol.NewMessage(protocol.MessageRoleUnspecified, protocol.TextPart{Text: "bad"}),
			}}}},
		{name: "related artifact", message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
			related: []*protocol.Task{{ID: "related", Artifacts: []*protocol.Artifact{{
				ID: "artifact", Parts: protocol.ContentParts{protocol.FilePart{File: protocol.FileURI{
					URI: "https://example.com/file",
				}}},
			}}}}},
		{name: "sanitization", message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
			options: ai.MessageSanitizationOptions{AllowedFileURLSchemes: []string{"bad scheme"}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := &a2asrv.RequestContext{
				Message: test.message, StoredTask: &protocol.Task{History: test.history}, RelatedTasks: test.related,
				TaskID: "task", ContextID: "context",
			}
			executor := a2aintegration.NewExecutor(
				ai.NewAgent[struct{}, string](model), a2aintegration.Config[struct{}]{Sanitization: test.options},
			)
			queue := &recordingQueue{}
			if err := executor.Execute(context.Background(), request, queue); err != nil {
				t.Fatal(err)
			}
			if queue.events[0].(*protocol.TaskStatusUpdateEvent).Status.State != protocol.TaskStateFailed {
				t.Fatalf("unexpected input failure: %#v", queue.events)
			}
		})
	}
}

func TestExecutorCancellationAndQueueFailures(t *testing.T) {
	request := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
		TaskID:  "task", ContextID: "context", StoredTask: &protocol.Task{},
	}
	canceled := a2aintegration.NewExecutor(
		ai.NewAgent[struct{}, string](&eventModel{err: context.Canceled}), a2aintegration.Config[struct{}]{},
	)
	queue := &recordingQueue{}
	if err := canceled.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	if status := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent); status.Status.State != protocol.TaskStateCanceled {
		t.Fatalf("unexpected canceled status: %#v", status)
	}

	resolved := a2aintegration.NewExecutor(
		ai.NewAgent[struct{}, string](fakes.NewTestModel()), a2aintegration.Config[struct{}]{
			ResolveDeps: func(context.Context, *a2asrv.RequestContext) (struct{}, error) {
				return struct{}{}, errors.New("deps failed")
			},
		},
	)
	queue = &recordingQueue{}
	if err := resolved.Execute(context.Background(), request, queue); err != nil {
		t.Fatal(err)
	}
	if queue.events[0].(*protocol.TaskStatusUpdateEvent).Status.State != protocol.TaskStateFailed {
		t.Fatalf("unexpected dependency failure: %#v", queue.events)
	}

	newRequest := *request
	newRequest.StoredTask = nil
	for _, test := range []struct {
		name   string
		queue  *recordingQueue
		stored bool
	}{
		{name: "submitted", queue: &recordingQueue{failAt: 1}},
		{name: "artifact", queue: &recordingQueue{failAt: 2}, stored: true},
		{name: "artifact update", queue: &recordingQueue{failAt: 3}, stored: true},
		{name: "completion", queue: &recordingQueue{failAt: 4}, stored: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			target := &newRequest
			if test.stored {
				target = request
			}
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{
					ai.TextPart{Content: "one"}, ai.TextPart{Content: "two"},
				}}, nil
			})
			executor := a2aintegration.NewExecutor(
				ai.NewAgent[struct{}, string](model), a2aintegration.Config[struct{}]{},
			)
			if err := executor.Execute(context.Background(), target, test.queue); err == nil {
				t.Fatal("expected queue error")
			}
		})
	}

	queue = &recordingQueue{failAt: 1}
	if err := canceled.Cancel(context.Background(), request, queue); err == nil {
		t.Fatal("expected cancellation queue error")
	}
	if err := canceled.Cancel(context.Background(), nil, queue); err == nil {
		t.Fatal("expected nil cancellation request error")
	}
	if err := canceled.Cancel(context.Background(), request, nil); err == nil {
		t.Fatal("expected nil cancellation queue error")
	}
}

func TestExecutorFailures(t *testing.T) {
	assertPanic(t, func() { a2aintegration.NewExecutor[struct{}, string](nil, a2aintegration.Config[struct{}]{}) })
	agent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("model failed")
	}))
	assertPanic(t, func() {
		a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{
			ArtifactMetadata: map[string]any{"invalid": make(chan int)},
		})
	})
	executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{})
	valid := &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleUser, protocol.TextPart{Text: "run"}),
		TaskID:  "task", ContextID: "context", StoredTask: &protocol.Task{},
	}
	queue := &recordingQueue{}
	if err := executor.Execute(context.Background(), valid, queue); err != nil {
		t.Fatal(err)
	}
	status := queue.events[len(queue.events)-1].(*protocol.TaskStatusUpdateEvent)
	if status.Status.State != protocol.TaskStateFailed || status.Status.Message == nil {
		t.Fatalf("unexpected failure status: %#v", status)
	}

	tests := []struct {
		name    string
		request *a2asrv.RequestContext
		queue   *recordingQueue
		match   string
	}{
		{name: "nil request", queue: &recordingQueue{}, match: "request message"},
		{name: "nil message", request: &a2asrv.RequestContext{}, queue: &recordingQueue{}, match: "request message"},
		{name: "working write", request: valid, queue: &recordingQueue{failAt: 1}, match: "working state"},
	}
	if err := executor.Execute(context.Background(), valid, nil); err == nil || !strings.Contains(err.Error(), "event queue") {
		t.Fatalf("unexpected nil queue error: %v", err)
	}
	badRoleQueue := &recordingQueue{}
	if err := executor.Execute(context.Background(), &a2asrv.RequestContext{
		Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.TextPart{Text: "bad"}),
	}, badRoleQueue); err != nil || badRoleQueue.events[0].(*protocol.TaskStatusUpdateEvent).Status.State != protocol.TaskStateFailed {
		t.Fatalf("unexpected bad-role failure: events=%#v err=%v", badRoleQueue.events, err)
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := executor.Execute(context.Background(), test.request, test.queue)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func assertPanic(t *testing.T, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	function()
}
