package a2a_test

import (
	"context"
	"encoding/base64"
	"errors"
	"iter"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	protocol "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2asrv"

	ai "github.com/Kludex/pydantic-ai-go"
	a2aintegration "github.com/Kludex/pydantic-ai-go/a2a"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type modelClient struct {
	send   func(context.Context, *protocol.MessageSendParams) (protocol.SendMessageResult, error)
	stream func(context.Context, *protocol.MessageSendParams) iter.Seq2[protocol.Event, error]
}

func (client *modelClient) SendMessage(
	ctx context.Context, params *protocol.MessageSendParams,
) (protocol.SendMessageResult, error) {
	return client.send(ctx, params)
}

func (client *modelClient) SendStreamingMessage(
	ctx context.Context, params *protocol.MessageSendParams,
) iter.Seq2[protocol.Event, error] {
	return client.stream(ctx, params)
}

func TestA2AModelThroughOfficialJSONRPCClient(t *testing.T) {
	serverAgent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		prompt := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
		text := prompt.Content
		if len(prompt.Contents) > 0 {
			text = prompt.Contents[0].(ai.TextContent).Text
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "remote: " + text}}}, nil
	}))
	executor := a2aintegration.NewExecutor(serverAgent, a2aintegration.Config[struct{}]{})
	server := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))
	defer server.Close()

	client, err := a2aclient.NewFromEndpoints(context.Background(), []protocol.AgentInterface{{
		URL: server.URL, Transport: protocol.TransportProtocolJSONRPC,
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := client.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	model := a2aintegration.NewModel("remote", client, a2aintegration.ModelConfig{ProviderURL: server.URL})
	response, err := ai.RequestModel(context.Background(), model, promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "remote: question" || response.ProviderName != "a2a" ||
		response.ProviderResponseID == "" || response.ProviderDetails["context_id"] == "" {
		t.Fatalf("unexpected JSON-RPC response: %#v", response)
	}
	continued, err := ai.RequestModel(context.Background(), model, []ai.ModelMessage{
		promptMessages()[0], *response,
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "follow-up"}}},
	}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if continued.Text() != "remote: follow-up" || continued.ProviderResponseID == response.ProviderResponseID ||
		continued.ProviderDetails["context_id"] != response.ProviderDetails["context_id"] {
		t.Fatalf("remote task continuity was lost: %#v", continued)
	}
}

func TestA2AModelRequest(t *testing.T) {
	timestamp := time.Date(2026, 9, 4, 12, 0, 0, 0, time.FixedZone("test", 3600))
	blocking, historyLength := false, 3
	config := a2aintegration.ModelConfig{
		ProviderURL: "https://agent.example",
		SendConfig: &protocol.MessageSendConfig{
			AcceptedOutputModes: []string{"text/plain"}, Blocking: &blocking, HistoryLength: &historyLength,
			PushConfig: &protocol.PushConfig{
				URL: "https://push.example", Auth: &protocol.PushAuthInfo{Schemes: []string{"Bearer"}},
			},
		},
		Extensions:      []string{"urn:example:extension"},
		ReferenceTasks:  []protocol.TaskID{"related"},
		MessageMetadata: map[string]any{"nested": map[string]any{"value": "message"}},
		RequestMetadata: map[string]any{"nested": map[string]any{"value": "request"}},
	}
	var received *protocol.MessageSendParams
	client := &modelClient{send: func(
		_ context.Context, params *protocol.MessageSendParams,
	) (protocol.SendMessageResult, error) {
		received = params
		return &protocol.Task{
			ID: "task-2", ContextID: "context-2", Status: protocol.TaskStatus{
				State: protocol.TaskStateCompleted, Timestamp: &timestamp,
			}, Metadata: map[string]any{"task": map[string]any{"value": true}},
			Artifacts: []*protocol.Artifact{nil, {
				ID: "artifact", Name: "answer", Description: "Remote output",
				Extensions: []string{"urn:artifact"}, Metadata: map[string]any{"safe": true},
				Parts: protocol.ContentParts{
					protocol.TextPart{Text: "answer", Metadata: map[string]any{"text": true}},
					protocol.DataPart{Data: map[string]any{"count": 2}},
					protocol.FilePart{File: protocol.FileBytes{
						FileMeta: protocol.FileMeta{MimeType: "image/png", Name: "plot.png"},
						Bytes:    base64.StdEncoding.EncodeToString([]byte("png")),
					}},
				},
			}},
		}, nil
	}}
	model := a2aintegration.NewModel("remote-agent", client, config)

	config.Extensions[0] = "changed"
	config.ReferenceTasks[0] = "changed"
	config.MessageMetadata["nested"].(map[string]any)["value"] = "changed"
	config.RequestMetadata["nested"].(map[string]any)["value"] = "changed"
	config.SendConfig.AcceptedOutputModes[0] = "changed"
	config.SendConfig.PushConfig.Auth.Schemes[0] = "changed"

	transcript := "spoken"
	response, err := model.Request(context.Background(), []ai.ModelMessage{
		ai.ModelResponse{ProviderName: "other", ProviderDetails: map[string]any{"task_id": "ignored"}},
		ai.ModelResponse{ProviderName: "a2a", ProviderDetails: map[string]any{
			"task_id": "task-1", "context_id": "context-1",
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "legacy"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "question"},
				ai.BinaryContent{Data: []byte("raw"), MediaType: "application/octet-stream", Identifier: "raw.bin"},
				ai.ImageURL{URL: "https://example.com/image.png", MediaType: "image/png", Identifier: "image"},
				ai.AudioURL{URL: "https://example.com/audio.mp3", MediaType: "audio/mpeg", Identifier: "audio"},
				ai.VideoURL{URL: "https://example.com/video.mp4", MediaType: "video/mp4", Identifier: "video"},
				ai.DocumentURL{URL: "https://example.com/doc.pdf", MediaType: "application/pdf", Identifier: "doc"},
				ai.CachePoint{},
			}},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: &transcript},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &ai.BinaryContent{
				Data: []byte("speech"), MediaType: "audio/wav", Identifier: "speech.wav",
			}},
			ai.RetryPromptPart{Content: "try again"},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"later"}},
		}},
	}, ai.ModelRequestParams{Instructions: "Follow policy."})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "remote-agent" || model.ProviderName() != "a2a" ||
		model.ProviderURL() != "https://agent.example" ||
		model.ModelProfile().DefaultOutputMode != ai.OutputModePrompted {
		t.Fatalf("unexpected model identity: %#v", model)
	}
	if received.Message.TaskID != "task-1" || received.Message.ContextID != "context-1" ||
		received.Message.Extensions[0] != "urn:example:extension" || received.Message.ReferenceTasks[0] != "related" ||
		received.Config.AcceptedOutputModes[0] != "text/plain" || received.Config.Blocking == nil ||
		!*received.Config.Blocking || received.Config.PushConfig.Auth.Schemes[0] != "Bearer" ||
		received.Message.Metadata["nested"].(map[string]any)["value"] != "message" ||
		received.Metadata["nested"].(map[string]any)["value"] != "request" {
		t.Fatalf("request configuration was not detached: %#v", received)
	}
	if len(received.Message.Parts) != 11 || received.Message.Parts[0].(protocol.TextPart).Text != "Follow policy." ||
		received.Message.Parts[1].(protocol.TextPart).Text != "legacy" ||
		received.Message.Parts[2].(protocol.TextPart).Text != "question" ||
		received.Message.Parts[3].(protocol.FilePart).File.(protocol.FileBytes).Bytes !=
			base64.StdEncoding.EncodeToString([]byte("raw")) ||
		received.Message.Parts[9].(protocol.FilePart).File.(protocol.FileBytes).Name != "speech.wav" ||
		received.Message.Parts[10].(protocol.TextPart).Text == "" {
		t.Fatalf("unexpected A2A message parts: %#v", received.Message.Parts)
	}
	if response.Timestamp != timestamp.UTC() || response.ProviderResponseID != "task-2" ||
		response.ProviderDetails["status"] != "completed" || len(response.Parts) != 3 {
		t.Fatalf("unexpected response: %#v", response)
	}
	text := response.Parts[0].(ai.TextPart)
	data := response.Parts[1].(ai.TextPart)
	file := response.Parts[2].(ai.FilePart)
	if text.Content != "answer" || text.ID != "artifact:0" || data.Content != `{"count":2}` ||
		string(file.Content.Data) != "png" || file.Content.Identifier != "plot.png" ||
		text.ProviderDetails["artifact_name"] != "answer" ||
		text.ProviderDetails["metadata"].(map[string]any)["text"] != true {
		t.Fatalf("unexpected response parts: %#v", response.Parts)
	}
}

func TestA2AModelMessageAndStatusFallback(t *testing.T) {
	t.Run("message", func(t *testing.T) {
		model := a2aintegration.NewModel("remote", staticClient(&protocol.Message{
			ID: "message", Role: protocol.MessageRoleAgent, ContextID: "context",
			Metadata: map[string]any{"message": true},
			Parts:    protocol.ContentParts{protocol.TextPart{Text: "hello", Metadata: map[string]any{"part": true}}},
		}), a2aintegration.ModelConfig{})
		response, err := model.Request(context.Background(), promptMessages(), ai.ModelRequestParams{})
		if err != nil || response.Text() != "hello" || response.ProviderResponseID != "message" ||
			response.Parts[0].(ai.TextPart).ProviderDetails["metadata"].(map[string]any)["part"] != true {
			t.Fatalf("unexpected message response: %#v, %v", response, err)
		}
	})
	t.Run("status message", func(t *testing.T) {
		model := a2aintegration.NewModel("remote", staticClient(&protocol.Task{
			ID: "task", ContextID: "context", Status: protocol.TaskStatus{
				State:   protocol.TaskStateCompleted,
				Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.TextPart{Text: "status output"}),
			},
		}), a2aintegration.ModelConfig{})
		response, err := model.Request(context.Background(), promptMessages(), ai.ModelRequestParams{})
		if err != nil || response.Text() != "status output" {
			t.Fatalf("unexpected status response: %#v, %v", response, err)
		}
	})
}

func TestA2AModelRequestErrors(t *testing.T) {
	transportErr := errors.New("offline")
	tests := []struct {
		name     string
		messages []ai.ModelMessage
		params   ai.ModelRequestParams
		result   protocol.SendMessageResult
		err      error
		contains string
	}{
		{name: "no request", messages: []ai.ModelMessage{ai.ModelRequest{
			Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}},
		}, ai.ModelResponse{}}, result: messageResult(protocol.TextPart{Text: "ok"})},
		{name: "missing request", contains: "contains no request"},
		{name: "empty request", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"tool"}},
		}}}, contains: "no sendable content"},
		{name: "tools", messages: promptMessages(), params: ai.ModelRequestParams{
			Tools: []ai.ToolDefinition{{Name: "tool"}},
		}, contains: "caller-provided tools"},
		{name: "deferred tools", messages: promptMessages(), params: ai.ModelRequestParams{
			DeferredTools: []ai.ToolDefinition{{Name: "tool"}},
		}, contains: "caller-provided tools"},
		{name: "native tools", messages: promptMessages(), params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
		}, contains: "caller-provided tools"},
		{name: "output tool", messages: promptMessages(), params: ai.ModelRequestParams{
			OutputTool: &ai.ToolDefinition{Name: "output"},
		}, contains: "caller-provided tools"},
		{name: "settings", messages: promptMessages(), params: ai.ModelRequestParams{
			Settings: ai.ModelSettings{MaxTokens: 1},
		}, contains: "generation settings"},
		{name: "tool return", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "tool", ToolCallID: "call", Content: "done"},
		}}}, contains: "local tool results"},
		{name: "uploaded file", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Contents: []ai.UserContent{ai.UploadedFile{FileID: "file"}}},
		}}}, contains: "uploaded provider files"},
		{name: "transport", messages: promptMessages(), err: transportErr, contains: "offline"},
		{name: "nil result", messages: promptMessages(), contains: "unsupported send result"},
		{name: "nil task", messages: promptMessages(), result: (*protocol.Task)(nil), contains: "nil task"},
		{name: "failed task", messages: promptMessages(), result: &protocol.Task{
			ID: "task", ContextID: "context", Status: protocol.TaskStatus{State: protocol.TaskStateFailed,
				Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.TextPart{Text: "failed"})},
		}, contains: "failed"},
		{name: "empty task", messages: promptMessages(), result: &protocol.Task{
			ID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateCompleted},
		}, contains: "contains no output"},
		{name: "task file URI", messages: promptMessages(), result: &protocol.Task{
			ID: "task", Status: protocol.TaskStatus{State: protocol.TaskStateCompleted},
			Artifacts: []*protocol.Artifact{{ID: "artifact", Parts: protocol.ContentParts{protocol.FilePart{
				File: protocol.FileURI{URI: "https://example.com/file"},
			}}}},
		}, contains: "is not inline"},
		{name: "status file URI", messages: promptMessages(), result: &protocol.Task{
			ID: "task", Status: protocol.TaskStatus{
				State: protocol.TaskStateCompleted,
				Message: protocol.NewMessage(protocol.MessageRoleAgent, protocol.FilePart{
					File: protocol.FileURI{URI: "https://example.com/file"},
				}),
			},
		}, contains: "is not inline"},
		{name: "file URI", messages: promptMessages(), result: messageResult(protocol.FilePart{
			File: protocol.FileURI{URI: "https://example.com/file"},
		}), contains: "is not inline"},
		{name: "bad base64", messages: promptMessages(), result: messageResult(protocol.FilePart{
			File: protocol.FileBytes{Bytes: "!"},
		}), contains: "decode response file"},
		{name: "bad data", messages: promptMessages(), result: messageResult(protocol.DataPart{
			Data: map[string]any{"bad": make(chan int)},
		}), contains: "encode response data"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &modelClient{send: func(
				context.Context, *protocol.MessageSendParams,
			) (protocol.SendMessageResult, error) {
				return test.result, test.err
			}}
			model := a2aintegration.NewModel("remote", client, a2aintegration.ModelConfig{})
			_, err := model.Request(context.Background(), test.messages, test.params)
			if test.contains == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("expected %q, got %v", test.contains, err)
			}
			if test.name == "transport" {
				var modelErr *ai.ModelTransportError
				if !errors.As(err, &modelErr) || modelErr.ProviderName != "a2a" {
					t.Fatalf("transport error was not classified: %T %v", err, err)
				}
			}
			if test.name == "failed task" {
				var taskErr *a2aintegration.TaskError
				if !errors.As(err, &taskErr) || taskErr.TaskID != "task" || taskErr.ContextID != "context" ||
					taskErr.State != protocol.TaskStateFailed {
					t.Fatalf("task error was not inspectable: %T %v", err, err)
				}
			}
		})
	}
}

func TestA2AModelCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	model := a2aintegration.NewModel("remote", &modelClient{send: func(
		context.Context, *protocol.MessageSendParams,
	) (protocol.SendMessageResult, error) {
		return nil, context.Canceled
	}}, a2aintegration.ModelConfig{})
	_, err := model.Request(ctx, promptMessages(), ai.ModelRequestParams{})
	var transportErr *ai.ModelTransportError
	if !errors.Is(err, context.Canceled) || errors.As(err, &transportErr) {
		t.Fatalf("cancellation was classified as provider failure: %T %v", err, err)
	}
}

func TestA2AModelStreamRequestValidation(t *testing.T) {
	model := a2aintegration.NewModel("remote", &modelClient{}, a2aintegration.ModelConfig{})
	sequence, err := model.StreamRequest(context.Background(), promptMessages(), ai.ModelRequestParams{
		Settings: ai.ModelSettings{MaxTokens: 1},
	})
	if err == nil || sequence != nil {
		t.Fatalf("expected streaming request validation failure, got %v", err)
	}
}

func TestA2AModelConstructorValidation(t *testing.T) {
	assertPanic(t, func() { a2aintegration.NewModel("", &modelClient{}, a2aintegration.ModelConfig{}) })
	assertPanic(t, func() { a2aintegration.NewModel("remote", nil, a2aintegration.ModelConfig{}) })
	var nilClient *modelClient
	assertPanic(t, func() { a2aintegration.NewModel("remote", nilClient, a2aintegration.ModelConfig{}) })
	_ = a2aintegration.NewModel("remote", modelClientValue{}, a2aintegration.ModelConfig{})
	assertPanic(t, func() {
		a2aintegration.NewModel("remote", &modelClient{}, a2aintegration.ModelConfig{
			MessageMetadata: map[string]any{"invalid": make(chan int)},
		})
	})
}

func staticClient(result protocol.SendMessageResult) *modelClient {
	return &modelClient{send: func(
		context.Context, *protocol.MessageSendParams,
	) (protocol.SendMessageResult, error) {
		return result, nil
	}}
}

func promptMessages() []ai.ModelMessage {
	return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "question"}}}}
}

func messageResult(part protocol.Part) *protocol.Message {
	return protocol.NewMessage(protocol.MessageRoleAgent, part)
}

type modelClientValue struct{}

func (modelClientValue) SendMessage(
	context.Context, *protocol.MessageSendParams,
) (protocol.SendMessageResult, error) {
	return nil, nil
}

func (modelClientValue) SendStreamingMessage(
	context.Context, *protocol.MessageSendParams,
) iter.Seq2[protocol.Event, error] {
	return nil
}

var _ a2aintegration.ModelClient = (*modelClient)(nil)
