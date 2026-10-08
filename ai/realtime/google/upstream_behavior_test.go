package google_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	googlert "github.com/Kludex/pydantic-ai-go/ai/realtime/google"
	"google.golang.org/genai"
)

type delayedConnector struct {
	first   *fakeSession
	second  *fakeSession
	started chan struct{}
	release chan struct{}
	calls   atomic.Int32
}

func (connector *delayedConnector) Connect(ctx context.Context, _ string, _ *genai.LiveConnectConfig) (googlert.LiveSession, error) {
	if connector.calls.Add(1) == 1 {
		return connector.first, nil
	}
	close(connector.started)
	select {
	case <-connector.release:
		return connector.second, nil
	case <-ctx.Done():
		return nil, context.Cause(ctx)
	}
}

func TestGoogleReconnectDropsAudioButRejectsText(t *testing.T) {
	first, second := newFakeSession(), newFakeSession()
	connector := &delayedConnector{first: first, second: second, started: make(chan struct{}), release: make(chan struct{})}
	connection, err := googlert.NewModel("gemini-3.8-live", googlert.WithConnector(connector)).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		AsyncToolCalls: ptr(true), Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond},
	}})
	if err != nil {
		t.Fatal(err)
	}
	recovery := connection.(realtime.ReconnectingConnection)
	if !recovery.CanReconnect() {
		t.Fatal("reconnect not available")
	}
	first.receive <- &genai.LiveServerMessage{GoAway: &genai.LiveServerGoAway{}}
	done := make(chan struct{})
	go func() {
		for event := range connection.Events(t.Context()) {
			if _, ok := event.(realtime.SessionReconnected); ok {
				break
			}
		}
		close(done)
	}()
	<-connector.started
	if !recovery.IsReconnecting() {
		t.Fatal("missing reconnect state")
	}
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{0, 0}}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.TextInput{Text: "late"}); err == nil {
		t.Fatal("text was accepted during reconnect")
	}
	close(connector.release)
	<-done
	if recovery.CanReconnect() {
		t.Fatal("reconnect budget was not spent")
	}
	_ = connection.Close(t.Context())
}

func TestGoogleRetriesLostToolSettlementBeforeInput(t *testing.T) {
	first, second := newFakeSession(), newFakeSession()
	second.err = errors.New("link lost")
	connector := &sequenceConnector{results: []connectorResult{{session: first}, {session: second}}}
	connection, err := googlert.NewModel("gemini-3.8-live", googlert.WithConnector(connector)).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond},
	}})
	if err != nil {
		t.Fatal(err)
	}
	first.receive <- &genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{NewHandle: "h", Resumable: true}}
	first.receive <- &genai.LiveServerMessage{ToolCall: &genai.LiveServerToolCall{FunctionCalls: []*genai.FunctionCall{{ID: "c", Name: "lookup"}}}}
	first.receive <- &genai.LiveServerMessage{GoAway: &genai.LiveServerGoAway{}}
	for event := range connection.Events(t.Context()) {
		if _, ok := event.(realtime.SessionReconnected); ok {
			break
		}
	}
	if err := connection.Send(t.Context(), realtime.TextInput{Text: "next"}); err == nil {
		t.Fatal("input overtook failed settlement")
	}
	second.err = nil
	if err := connection.Send(t.Context(), realtime.TextInput{Text: "next"}); err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
}

func TestGoogleRecentImageAndRichToolResult(t *testing.T) {
	live := newFakeSession()
	connection, err := googlert.NewModel("gemini-3.8-live", googlert.WithConnector(&fakeConnector{session: live})).Connect(t.Context(), realtime.ConnectParams{
		Settings: realtime.Settings{Provider: map[string]any{"google_affective_dialog": false}},
	})
	if err != nil {
		t.Fatal(err)
	}
	image := ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}
	for _, input := range []realtime.Input{realtime.ImageInput{Content: image}, realtime.TextContext{Text: "context"}, realtime.TextInput{Text: "question"}, realtime.TextInput{Text: "next"}} {
		if err := connection.Send(t.Context(), input); err != nil {
			t.Fatal(err)
		}
	}
	if len(live.client[1].Turns[0].Parts) != 2 || live.client[1].Turns[0].Parts[0].InlineData == nil || len(live.client[2].Turns[0].Parts) != 1 {
		t.Fatal("recent image was not sent once with the typed turn")
	}
	live.receive <- &genai.LiveServerMessage{ToolCall: &genai.LiveServerToolCall{FunctionCalls: []*genai.FunctionCall{{ID: "call", Name: "lookup"}}}}
	for event, eventErr := range connection.Events(t.Context()) {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if _, ok := event.(realtime.ToolCall); ok {
			break
		}
	}
	if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "call", Output: "result", Content: []ai.UserContent{ai.TextContent{Text: "note"}, image, ai.CachePoint{}}}); err != nil {
		t.Fatal(err)
	}
	response := live.tools[0].FunctionResponses[0]
	if response.Name != "lookup" || len(response.Parts) != 1 || response.Response["output"] != "result\n\nnote" {
		t.Fatalf("tool result: %+v", response)
	}
	for _, content := range []ai.UserContent{ai.BinaryContent{MediaType: "application/pdf"}, ai.AudioURL{URL: "https://example.com/audio"}} {
		if err := connection.Send(t.Context(), realtime.ToolResult{Content: []ai.UserContent{content}}); err == nil {
			t.Fatal("accepted unsupported media")
		}
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := connection.Send(cancelled, realtime.TextInput{Text: "no"}); err == nil {
		t.Fatal("ignored cancellation")
	}
	_ = connection.Close(t.Context())
}

func TestGoogleSeedsAudioAndRejectsWrongRate(t *testing.T) {
	for _, pcm := range [][]byte{{1, 0}, {1}} {
		live := newFakeSession()
		connector := &fakeConnector{session: live}
		connection, err := googlert.NewModel("gemini-3.1-flash-live-preview", googlert.WithConnector(connector)).Connect(t.Context(), realtime.ConnectParams{
			Messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &ai.BinaryContent{Data: pcm, MediaType: "audio/pcm"}}}}},
		})
		if len(pcm)%2 != 0 {
			if err == nil {
				t.Fatal("accepted odd PCM")
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if live.client[0].Turns[0].Parts[0].InlineData.MIMEType != "audio/pcm;rate=16000" {
			t.Fatal("seed audio was not converted")
		}
		_ = connection.Close(t.Context())
	}
	for _, name := range []string{"gemini-2.5-flash-native-audio-latest", "gemini-3.8-live-extended-thinking"} {
		if googlert.NewModel(name).Profile().SupportsSeedingAudio {
			t.Fatalf("unverified audio support: %s", name)
		}
	}
}

func TestGoogleUsagePrecedesTurnAndMapsFinishReason(t *testing.T) {
	for _, test := range []struct {
		reason genai.TurnCompleteReason
		finish ai.FinishReason
	}{
		{"MALFORMED_FUNCTION_CALL", ai.FinishReasonError}, {"STOP", ai.FinishReasonStop}, {"MAX_TOKENS", ai.FinishReasonLength},
		{"GENERATED_AUDIO_SAFETY", ai.FinishReasonContentFilter}, {"NEED_MORE_INPUT", ""},
	} {
		live := newFakeSession()
		connection, err := googlert.NewModel("gemini-3.8-live", googlert.WithConnector(&fakeConnector{session: live})).Connect(t.Context(), realtime.ConnectParams{})
		if err != nil {
			t.Fatal(err)
		}
		live.receive <- &genai.LiveServerMessage{UsageMetadata: &genai.UsageMetadata{PromptTokenCount: 10}, ServerContent: &genai.LiveServerContent{TurnComplete: true, TurnCompleteReason: test.reason}}
		used := false
		for event, eventErr := range connection.Events(t.Context()) {
			if eventErr != nil {
				t.Fatal(eventErr)
			}
			if _, ok := event.(realtime.SessionUsage); ok {
				used = true
			}
			if done, ok := event.(realtime.ResponseDone); ok {
				if !used || done.FinishReason != test.finish {
					t.Fatalf("terminal: %+v", done)
				}
				break
			}
		}
		_ = connection.Close(t.Context())
	}
}

func TestGoogleVertexToolBoundaryDoesNotEndAnswer(t *testing.T) {
	live := newFakeSession()
	connection, err := googlert.NewModel("gemini-live-2.5-flash", googlert.WithVertex("project", "global"), googlert.WithConnector(&fakeConnector{session: live})).Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	live.receive <- &genai.LiveServerMessage{ToolCall: &genai.LiveServerToolCall{FunctionCalls: []*genai.FunctionCall{{ID: "c", Name: "lookup"}}}}
	live.receive <- &genai.LiveServerMessage{ServerContent: &genai.LiveServerContent{TurnComplete: true}}
	live.receive <- &genai.LiveServerMessage{ServerContent: &genai.LiveServerContent{ModelTurn: &genai.Content{Parts: []*genai.Part{genai.NewPartFromText("answer")}}, TurnComplete: true}}
	boundaries := 0
	for event, eventErr := range connection.Events(t.Context()) {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if done, ok := event.(realtime.ResponseDone); ok {
			boundaries++
			if !done.MoreExpected {
				break
			}
		}
	}
	if boundaries != 2 {
		t.Fatalf("premature tool boundary: %d", boundaries)
	}
	_ = connection.Close(t.Context())
}

func TestGoogleReconnectSettlesLostCallsAndTypedTurns(t *testing.T) {
	for _, calls := range []bool{false, true} {
		first, second := newFakeSession(), newFakeSession()
		connector := &sequenceConnector{results: []connectorResult{{session: first}, {session: second}}}
		connection, err := googlert.NewModel("gemini-3.8-live", googlert.WithConnector(connector)).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
			Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond},
		}})
		if err != nil {
			t.Fatal(err)
		}
		if err := connection.Send(t.Context(), realtime.TextInput{Text: "question"}); err != nil {
			t.Fatal(err)
		}
		first.receive <- &genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{NewHandle: "h", Resumable: true}}
		first.receive <- &genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{}}
		if calls {
			first.receive <- &genai.LiveServerMessage{ToolCall: &genai.LiveServerToolCall{FunctionCalls: []*genai.FunctionCall{{ID: "c", Name: "lookup"}}}}
		}
		first.receive <- &genai.LiveServerMessage{GoAway: &genai.LiveServerGoAway{}}
		seenRejected, seenCancelled, seenInterrupted := false, false, false
		for event, eventErr := range connection.Events(t.Context()) {
			if eventErr != nil {
				t.Fatal(eventErr)
			}
			switch event := event.(type) {
			case realtime.InputRejected:
				seenRejected = true
			case realtime.ToolCallCancelled:
				seenCancelled = true
			case realtime.ResponseDone:
				seenInterrupted = seenInterrupted || event.Interrupted
			case realtime.SessionReconnected:
				if event.StateRestored || calls && (!seenCancelled || !seenInterrupted) || !calls && !seenRejected {
					t.Fatalf("lost state was not settled: %+v", event)
				}
			}
			if _, ok := event.(realtime.SessionReconnected); ok {
				break
			}
		}
		if calls && len(second.tools) != 1 {
			t.Fatal("lost call was not answered on resumed session")
		}
		second.receive <- &genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{NewHandle: "h2", Resumable: true}, ServerContent: &genai.LiveServerContent{TurnComplete: true}}
		for event := range connection.Events(t.Context()) {
			if _, ok := event.(realtime.ResponseDone); ok {
				break
			}
		}
		_ = connection.Close(t.Context())
	}
}

func TestGoogleTypedTurnHandleCoversAnswerAndFailedSend(t *testing.T) {
	live := newFakeSession()
	connection, err := googlert.NewModel("gemini-3.8-live", googlert.WithConnector(&fakeConnector{session: live})).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.TextInput{Text: "first"}); err != nil {
		t.Fatal(err)
	}
	live.receive <- &genai.LiveServerMessage{ServerContent: &genai.LiveServerContent{TurnComplete: true}}
	for event := range connection.Events(t.Context()) {
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	live.receive <- &genai.LiveServerMessage{SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{NewHandle: "h", Resumable: true}, ServerContent: &genai.LiveServerContent{TurnComplete: true}}
	for event := range connection.Events(t.Context()) {
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	live.err = errors.New("send failed")
	if err := connection.Send(t.Context(), realtime.TextInput{Text: "failed"}); err == nil {
		t.Fatal("accepted failed typed turn")
	}
	live.err = nil
	_ = connection.Close(t.Context())
}
