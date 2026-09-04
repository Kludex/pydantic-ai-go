package a2a_test

import (
	"context"
	"encoding/base64"
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	protocol "github.com/a2aproject/a2a-go/a2a"

	ai "github.com/Kludex/pydantic-ai-go"
	a2aintegration "github.com/Kludex/pydantic-ai-go/a2a"
)

func TestA2AModelStream(t *testing.T) {
	timestamp := time.Date(2026, 9, 4, 12, 0, 0, 0, time.FixedZone("test", 3600))
	task := &protocol.Task{
		ID: "task", ContextID: "context", Status: protocol.TaskStatus{State: protocol.TaskStateSubmitted},
		Artifacts: []*protocol.Artifact{nil, {
			ID: "artifact", Parts: protocol.ContentParts{protocol.TextPart{Text: "old"}},
		}, {
			ID: "artifact", Name: "answer", Parts: protocol.ContentParts{protocol.TextPart{Text: "start"}},
		}},
	}
	appendEvent := &protocol.TaskArtifactUpdateEvent{
		TaskID: "task", ContextID: "context", Append: true,
		Artifact: &protocol.Artifact{
			ID: "artifact", Name: "updated", Description: "Updated output",
			Extensions: []string{"urn:updated"}, Metadata: map[string]any{"updated": true},
			Parts: protocol.ContentParts{
				protocol.TextPart{Text: " continued"},
				protocol.DataPart{Data: map[string]any{"phase": "append"}},
			},
		},
	}
	replaceEvent := &protocol.TaskArtifactUpdateEvent{
		TaskID: "task", ContextID: "context",
		Artifact: &protocol.Artifact{ID: "artifact", Parts: protocol.ContentParts{
			protocol.FilePart{File: protocol.FileBytes{
				FileMeta: protocol.FileMeta{MimeType: "image/png", Name: "final.png"},
				Bytes:    base64.StdEncoding.EncodeToString([]byte("final")),
			}},
		}},
	}
	finalEvent := &protocol.TaskStatusUpdateEvent{
		TaskID: "task", ContextID: "context", Final: true,
		Status: protocol.TaskStatus{State: protocol.TaskStateCompleted, Timestamp: &timestamp},
	}
	model := a2aintegration.NewModel("remote", streamingClient(task, appendEvent, replaceEvent, finalEvent),
		a2aintegration.ModelConfig{ProviderURL: "https://agent.example"})

	sequence, err := model.StreamRequest(context.Background(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var metadata int
	var texts []ai.TextDeltaEvent
	var files []ai.FileEvent
	var finish *ai.FinishEvent
	for event, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		switch value := event.(type) {
		case ai.ResponseMetadataEvent:
			metadata++
			if value.ProviderName != "a2a" || value.ProviderResponseID != "task" {
				t.Fatalf("unexpected metadata: %#v", value)
			}
		case ai.TextDeltaEvent:
			texts = append(texts, value)
		case ai.FileEvent:
			files = append(files, value)
		case ai.FinishEvent:
			copy := value
			finish = &copy
		}
	}
	if metadata != 4 || len(texts) != 3 || len(files) != 1 || !files[0].Replace || finish == nil {
		t.Fatalf("unexpected stream lifecycle: metadata=%d texts=%#v files=%#v finish=%#v", metadata, texts, files, finish)
	}
	if texts[0].Delta != "start" || texts[1].Delta != " continued" || texts[2].Delta != `{"phase":"append"}` ||
		files[0].PartID != "artifact" || string(files[0].Part.Content.Data) != "final" {
		t.Fatalf("unexpected streamed parts: texts=%#v files=%#v", texts, files)
	}
	if finish.Timestamp != timestamp.UTC() || finish.ProviderURL != "https://agent.example" ||
		finish.ProviderResponseID != "task" || len(finish.Parts) != 1 ||
		string(finish.Parts[0].(ai.FilePart).Content.Data) != "final" {
		t.Fatalf("unexpected finish: %#v", finish)
	}
}

func TestA2AModelStreamTerminalResults(t *testing.T) {
	tests := []struct {
		name   string
		events []protocol.Event
		text   string
	}{
		{name: "message", events: []protocol.Event{&protocol.Message{
			ID: "message", Role: protocol.MessageRoleAgent,
			Parts: protocol.ContentParts{protocol.TextPart{Text: "message"}},
		}}, text: "message"},
		{name: "completed task", events: []protocol.Event{&protocol.Task{
			ID: "task", ContextID: "context", Status: protocol.TaskStatus{State: protocol.TaskStateCompleted},
			Artifacts: []*protocol.Artifact{{
				ID: "artifact", Parts: protocol.ContentParts{protocol.TextPart{Text: "task"}},
			}},
		}}, text: "task"},
		{name: "status message", events: []protocol.Event{&protocol.TaskStatusUpdateEvent{
			TaskID: "task", ContextID: "context", Final: true,
			Status: protocol.TaskStatus{
				State:   protocol.TaskStateCompleted,
				Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.TextPart{Text: "status"}),
			},
		}}, text: "status"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := a2aintegration.NewModel("remote", streamingClient(test.events...), a2aintegration.ModelConfig{})
			sequence, err := model.StreamRequest(context.Background(), promptMessages(), ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			var text string
			var finished bool
			for event, eventErr := range sequence {
				if eventErr != nil {
					t.Fatal(eventErr)
				}
				switch value := event.(type) {
				case ai.TextDeltaEvent:
					text += value.Delta
				case ai.FinishEvent:
					finished = true
				}
			}
			if text != test.text || !finished {
				t.Fatalf("unexpected terminal stream: text=%q finished=%t", text, finished)
			}
		})
	}
}

func TestA2AModelStreamErrors(t *testing.T) {
	transportErr := errors.New("disconnected")
	var nilTask *protocol.Task
	tests := []struct {
		name      string
		events    []protocol.Event
		streamErr error
		contains  string
	}{
		{name: "transport", streamErr: transportErr, contains: "disconnected"},
		{name: "nil event", events: []protocol.Event{nil}, contains: "nil event"},
		{name: "typed nil event", events: []protocol.Event{nilTask}, contains: "nil event"},
		{name: "failed task", events: []protocol.Event{&protocol.Task{
			ID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateFailed},
		}}, contains: "failed"},
		{name: "input required status", events: []protocol.Event{&protocol.TaskStatusUpdateEvent{
			TaskID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateInputRequired},
		}}, contains: "input-required"},
		{name: "nil artifact", events: []protocol.Event{&protocol.TaskArtifactUpdateEvent{
			TaskID: "task",
		}}, contains: "contains no artifact"},
		{name: "message URI", events: []protocol.Event{&protocol.Message{
			ID: "message", Parts: protocol.ContentParts{protocol.FilePart{
				File: protocol.FileURI{URI: "https://example.com/file"},
			}},
		}}, contains: "is not inline"},
		{name: "artifact URI", events: []protocol.Event{artifactEvent(protocol.FilePart{
			File: protocol.FileURI{URI: "https://example.com/file"},
		})}, contains: "is not inline"},
		{name: "artifact data", events: []protocol.Event{artifactEvent(protocol.DataPart{
			Data: map[string]any{"bad": make(chan int)},
		})}, contains: "encode response data"},
		{name: "task artifact", events: []protocol.Event{&protocol.Task{
			ID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateWorking},
			Artifacts: []*protocol.Artifact{{ID: "artifact", Parts: protocol.ContentParts{
				protocol.FilePart{File: protocol.FileBytes{Bytes: "!"}},
			}}},
		}}, contains: "decode response file"},
		{name: "invalid task artifact", events: []protocol.Event{&protocol.Task{
			ID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateWorking},
			Artifacts: []*protocol.Artifact{{ID: "artifact", Parts: protocol.ContentParts{
				protocol.DataPart{Data: map[string]any{"bad": make(chan int)}},
			}}},
		}}, contains: "retain task artifacts"},
		{name: "invalid appended artifact", events: []protocol.Event{
			artifactEvent(protocol.TextPart{Text: "valid"}),
			&protocol.TaskArtifactUpdateEvent{
				TaskID: "task", Append: true,
				Artifact: &protocol.Artifact{ID: "artifact", Parts: protocol.ContentParts{
					protocol.DataPart{Data: map[string]any{"bad": make(chan int)}},
				}},
			},
		}, contains: "retain artifact update"},
		{name: "completed task empty", events: []protocol.Event{&protocol.Task{
			ID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateCompleted},
		}}, contains: "contains no output"},
		{name: "completed status empty", events: []protocol.Event{&protocol.TaskStatusUpdateEvent{
			TaskID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateCompleted},
		}}, contains: "contains no output"},
		{name: "completed status bad message", events: []protocol.Event{&protocol.TaskStatusUpdateEvent{
			TaskID: "task", Status: protocol.TaskStatus{
				State: protocol.TaskStateCompleted,
				Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.FilePart{
					File: protocol.FileURI{URI: "https://example.com/file"},
				}),
			},
		}}, contains: "is not inline"},
		{name: "unfinished", events: []protocol.Event{&protocol.TaskStatusUpdateEvent{
			Status: protocol.TaskStatus{State: protocol.TaskStateWorking},
		}}, contains: "without a final event"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := streamingClient(test.events...)
			if test.streamErr != nil {
				client.stream = func(context.Context, *protocol.MessageSendParams) iter.Seq2[protocol.Event, error] {
					return func(yield func(protocol.Event, error) bool) { yield(nil, test.streamErr) }
				}
			}
			model := a2aintegration.NewModel("remote", client, a2aintegration.ModelConfig{})
			sequence, err := model.StreamRequest(context.Background(), promptMessages(), ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			var got error
			for _, eventErr := range sequence {
				if eventErr != nil {
					got = eventErr
				}
			}
			if got == nil || !strings.Contains(got.Error(), test.contains) {
				t.Fatalf("expected %q, got %v", test.contains, got)
			}
			if test.name == "transport" {
				var transport *ai.ModelTransportError
				if !errors.As(got, &transport) {
					t.Fatalf("stream error was not classified: %T", got)
				}
			}
		})
	}
}

func TestA2AModelStreamConsumerStops(t *testing.T) {
	tests := [][]protocol.Event{
		{&protocol.Message{ID: "message", Parts: protocol.ContentParts{protocol.TextPart{Text: "text"}}}},
		{&protocol.Message{ID: "message", Parts: protocol.ContentParts{protocol.FilePart{
			File: protocol.FileBytes{Bytes: base64.StdEncoding.EncodeToString([]byte("file"))},
		}}}},
		{&protocol.Task{
			ID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateCompleted},
			Artifacts: []*protocol.Artifact{{
				ID: "artifact", Parts: protocol.ContentParts{protocol.TextPart{Text: "task"}},
			}},
		}},
		{artifactEvent(protocol.TextPart{Text: "artifact"})},
		{&protocol.TaskStatusUpdateEvent{
			TaskID: "task", Status: protocol.TaskStatus{
				State:   protocol.TaskStateCompleted,
				Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.TextPart{Text: "status"}),
			},
		}},
	}
	for _, events := range tests {
		model := a2aintegration.NewModel("remote", streamingClient(events...), a2aintegration.ModelConfig{})
		sequence, err := model.StreamRequest(context.Background(), promptMessages(), ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		seen := 0
		sequence(func(ai.ModelStreamEvent, error) bool {
			seen++
			return seen < 2
		})
		if seen != 2 {
			t.Fatalf("consumer stop was ignored: %d", seen)
		}
	}

	model := a2aintegration.NewModel("remote", streamingClient(&protocol.Message{
		ID: "message", Parts: protocol.ContentParts{protocol.TextPart{Text: "text"}},
	}), a2aintegration.ModelConfig{})
	sequence, err := model.StreamRequest(context.Background(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	sequence(func(ai.ModelStreamEvent, error) bool {
		seen++
		return false
	})
	if seen != 1 {
		t.Fatalf("metadata consumer stop was ignored: %d", seen)
	}
}

func TestA2AModelStreamDetachesRetainedArtifacts(t *testing.T) {
	data := map[string]any{"value": "valid"}
	client := &modelClient{stream: func(
		context.Context, *protocol.MessageSendParams,
	) iter.Seq2[protocol.Event, error] {
		return func(yield func(protocol.Event, error) bool) {
			yield(artifactEvent(protocol.DataPart{Data: data}), nil)
			data["value"] = make(chan int)
			yield(&protocol.TaskStatusUpdateEvent{
				TaskID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateCompleted},
			}, nil)
		}
	}}
	model := a2aintegration.NewModel("remote", client, a2aintegration.ModelConfig{})
	sequence, err := model.StreamRequest(context.Background(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var finish ai.FinishEvent
	for event, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if value, ok := event.(ai.FinishEvent); ok {
			finish = value
		}
	}
	if finish.Parts[0].(ai.TextPart).Content != `{"value":"valid"}` {
		t.Fatalf("retained artifact was not detached: %#v", finish)
	}
}

func streamingClient(events ...protocol.Event) *modelClient {
	return &modelClient{stream: func(
		context.Context, *protocol.MessageSendParams,
	) iter.Seq2[protocol.Event, error] {
		return func(yield func(protocol.Event, error) bool) {
			for _, event := range events {
				if !yield(event, nil) {
					return
				}
			}
		}
	}}
}

func artifactEvent(part protocol.Part) *protocol.TaskArtifactUpdateEvent {
	return &protocol.TaskArtifactUpdateEvent{
		TaskID: "task", Artifact: &protocol.Artifact{ID: "artifact", Parts: protocol.ContentParts{part}},
	}
}
