package ai_test

import (
	"context"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestRunRecordsRunAndConversationIDs(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		latest := messages[len(messages)-1].(ai.ModelRequest)
		if latest.RunID != "run-1" || latest.ConversationID != "conversation-1" || latest.Timestamp.IsZero() {
			t.Fatalf("request metadata missing on step %d: %+v", request, latest)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "work", func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
		if rc.RunID != "run-1" || rc.ConversationID != "conversation-1" {
			t.Fatalf("run context metadata missing: %+v", rc)
		}
		return "ok", nil
	})
	result, err := agent.Run(
		t.Context(), "go", deps{}, ai.WithRunID("run-1"), ai.WithConversationID("conversation-1"),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, message := range result.NewMessages() {
		switch message := message.(type) {
		case ai.ModelRequest:
			if message.RunID != "run-1" || message.ConversationID != "conversation-1" || message.Timestamp.IsZero() {
				t.Fatalf("request metadata missing from result: %+v", message)
			}
		case ai.ModelResponse:
			if message.RunID != "run-1" || message.ConversationID != "conversation-1" || message.Timestamp.IsZero() {
				t.Fatalf("response metadata missing from result: %+v", message)
			}
		}
	}
}

func TestConversationIDInheritanceAndReset(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{RunID: "old-request", ConversationID: "conversation-old"},
		ai.ModelResponse{RunID: "old-response", ConversationID: "conversation-latest"},
	}
	for name, test := range map[string]struct {
		option ai.RunOption
		old    bool
	}{
		"inherit": {old: true},
		"new":     {option: ai.WithConversationID("new")},
	} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				conversationID := messages[len(messages)-1].(ai.ModelRequest).ConversationID
				if test.old && conversationID != "conversation-latest" {
					t.Fatalf("conversation was not inherited: %q", conversationID)
				}
				if !test.old && (conversationID == "" || conversationID == "new" || conversationID == "conversation-latest") {
					t.Fatalf("new conversation was not generated: %q", conversationID)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			options := []ai.RunOption{ai.WithMessageHistory(history)}
			if test.option != nil {
				options = append(options, test.option)
			}
			if _, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{}, options...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRunRejectsDuplicateExplicitRunID(t *testing.T) {
	for name, history := range map[string][]ai.ModelMessage{
		"request":  {ai.ModelRequest{RunID: "duplicate"}},
		"response": {ai.ModelResponse{RunID: "duplicate"}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
				t.Context(), "go", deps{}, ai.WithMessageHistory(history), ai.WithRunID("duplicate"),
			)
			if err == nil || err.Error() != `ai: run ID "duplicate" already appears in message history` {
				t.Fatalf("unexpected duplicate ID error: %v", err)
			}
		})
	}
}

func TestRunPreservesProducerMessageIDs(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Timestamp: time.Unix(1, 0).UTC(),
			RunID: "producer-run", ConversationID: "producer-conversation",
		}, nil
	})
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	response := result.Messages()[1].(ai.ModelResponse)
	if response.RunID != "producer-run" || response.ConversationID != "producer-conversation" ||
		!response.Timestamp.Equal(time.Unix(1, 0).UTC()) {
		t.Fatalf("producer metadata was replaced: %+v", response)
	}
}

func TestRunIDOptionsRejectEmptyValues(t *testing.T) {
	for name, option := range map[string]func(){
		"run":          func() { ai.WithRunID("") },
		"conversation": func() { ai.WithConversationID("") },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected empty ID panic")
				}
			}()
			option()
		})
	}
}

func TestMessageMetadataSerializationAndLegacyAliases(t *testing.T) {
	stamp := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	messages := []ai.ModelMessage{
		ai.ModelRequest{
			Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}, Timestamp: stamp,
			Instructions: "Be concise.", RunID: "run", ConversationID: "conversation",
			Metadata: map[string]any{"request": true}, State: ai.RequestStateComplete,
		},
		ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Timestamp: stamp,
			ProviderName: "provider", ProviderURL: "https://provider.example",
			ProviderDetails: map[string]any{"tier": "fast"}, ProviderResponseID: "response",
			FinishReason: ai.FinishReasonStop, RunID: "run", ConversationID: "conversation",
			Metadata: map[string]any{"response": true}, State: ai.ModelResponseStateComplete,
		},
	}
	data, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	request := decoded[0].(ai.ModelRequest)
	response := decoded[1].(ai.ModelResponse)
	if request.RunID != "run" || request.Metadata["request"] != true || !request.Timestamp.Equal(stamp) ||
		response.ProviderName != "provider" || response.ProviderURL != "https://provider.example" ||
		response.ProviderDetails["tier"] != "fast" || response.ProviderResponseID != "response" ||
		response.FinishReason != ai.FinishReasonStop || response.State != ai.ModelResponseStateComplete ||
		response.Metadata["response"] != true {
		t.Fatalf("message metadata did not round trip: %+v", decoded)
	}

	legacy := strings.ReplaceAll(string(data), "provider_details", "vendor_details")
	legacy = strings.ReplaceAll(legacy, "provider_response_id", "vendor_id")
	decoded, err = ai.UnmarshalMessages([]byte(legacy))
	if err != nil {
		t.Fatal(err)
	}
	response = decoded[1].(ai.ModelResponse)
	if response.ProviderDetails["tier"] != "fast" || response.ProviderResponseID != "response" {
		t.Fatalf("legacy aliases were not accepted: %+v", response)
	}

}
