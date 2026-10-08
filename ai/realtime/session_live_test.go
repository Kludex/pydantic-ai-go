package realtime_test

import (
	"context"
	"errors"
	"iter"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

type endingConnection struct {
	*fakeConnection
	fail bool
}

func (connection *endingConnection) EndSession(context.Context) iter.Seq2[realtime.SessionUsage, error] {
	return func(yield func(realtime.SessionUsage, error) bool) {
		if connection.fail {
			yield(realtime.SessionUsage{}, errors.New("end failed"))
			return
		}
		ratio := 0.7
		yield(realtime.SessionUsage{Usage: ai.Usage{AudioSeconds: 3}, ResponseScoped: true, ContextWindowUsed: &ratio}, nil)
	}
}

func TestSynthesizedSessionHistoryAndFinalUsage(t *testing.T) {
	for _, fail := range []bool{false, true} {
		connection := &endingConnection{fakeConnection: newFakeConnection(), fail: fail}
		profile := fullProfile()
		profile.SynthesizesTurnBoundary = true
		profile.ImageInputRequiresResponse = true
		profile.SupportsManualTurnControl = false
		session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(context.Context, ai.ToolCallPart) (any, error) { return "ok", nil })))
		if err != nil {
			t.Fatal(err)
		}
		if err := session.Send(t.Context(), ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}); err == nil {
			t.Fatal("passive image accepted")
		}
		if err := session.Send(t.Context(), ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}, realtime.WithResponse(true)); err != nil {
			t.Fatal(err)
		}
		<-connection.sent
		connection.events <- realtime.AudioDelta{Data: []byte{100, 0}, ResponseID: "r"}
		connection.events <- realtime.OutputTranscript{Text: "spoken", ResponseID: "r"}
		connection.events <- realtime.ToolCall{ToolCallID: "c", ToolName: "tool", Arguments: "{}", ResponseID: "r", ResponseUsageFollows: true}
		<-connection.sent
		ratio := 0.4
		connection.events <- realtime.SessionUsage{Usage: ai.Usage{Requests: 1, InputTokens: 10}, ResponseScoped: true, ProviderDetails: map[string]any{"delegated_model": "gpt-5", "delegated_response_id": "b"}, ContextWindowUsed: &ratio}
		connection.events <- realtime.OutputTranscript{Text: "answer", ResponseID: "r"}
		connection.events <- realtime.ResponseDone{ProviderDetails: map[string]any{"status": "completed"}}
		for event, err := range session.Events(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := event.(realtime.TurnCompleteEvent); ok {
				break
			}
		}
		messages := session.NewMessages()
		found := false
		for _, message := range messages {
			if response, ok := message.(ai.ModelResponse); ok && response.ProviderDetails["delegated_model"] == "gpt-5" {
				found = true
			}
		}
		if !found {
			t.Fatal("backend metadata absent")
		}
		connection.events <- realtime.SessionReconnected{StateRestored: true}
		for event, err := range session.Events(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := event.(realtime.SessionReconnectEvent); ok {
				break
			}
		}
		if err := session.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
		if !fail && session.Usage().AudioSeconds != 3 {
			t.Fatal(session.Usage())
		}
	}
}

func TestRealtimeContextWindowValidation(t *testing.T) {
	profile := fullProfile()
	profile.ContextWindow = -1
	if _, err := realtime.Open(t.Context(), &fakeModel{connection: newFakeConnection(), profile: profile}, realtime.ConnectParams{}); err == nil {
		t.Fatal("negative window accepted")
	}
}
