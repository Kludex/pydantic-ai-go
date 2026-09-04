package realtime_test

import (
	"context"
	"errors"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

type fakeConnection struct {
	events chan realtime.CodecEvent
	sent   chan realtime.Input
	err    error
	model  string
	close  sync.Once
}

func newFakeConnection() *fakeConnection {
	return &fakeConnection{events: make(chan realtime.CodecEvent, 32), sent: make(chan realtime.Input, 32)}
}

func (connection *fakeConnection) Send(ctx context.Context, input realtime.Input) error {
	if connection.err != nil {
		return connection.err
	}
	select {
	case connection.sent <- input:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

func (connection *fakeConnection) Events(ctx context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return func(yield func(realtime.CodecEvent, error) bool) {
		for {
			select {
			case <-ctx.Done():
				return
			case event, ok := <-connection.events:
				if !ok || !yield(event, nil) {
					return
				}
			}
		}
	}
}

func (connection *fakeConnection) Close(context.Context) error {
	connection.end()
	return connection.err
}
func (connection *fakeConnection) end()                      { connection.close.Do(func() { close(connection.events) }) }
func (connection *fakeConnection) ModelName() string         { return connection.model }
func (*fakeConnection) InputTranscriptionEnabled() bool      { return true }
func (*fakeConnection) ReconnectRestoresInFlightState() bool { return true }

type fakeModel struct {
	connection realtime.Connection
	profile    realtime.Profile
	err        error
	params     realtime.ConnectParams
}

func (*fakeModel) Name() string                    { return "live-model" }
func (*fakeModel) ProviderName() string            { return "test" }
func (model *fakeModel) Profile() realtime.Profile { return model.profile }
func (model *fakeModel) Connect(_ context.Context, params realtime.ConnectParams) (realtime.Connection, error) {
	model.params = params
	return model.connection, model.err
}

func fullProfile() realtime.Profile {
	profile := realtime.DefaultProfile()
	profile.SupportsImageInput = true
	profile.SupportsManualTurnControl = true
	profile.SupportsInterruption = true
	profile.SupportsOutputTruncation = true
	profile.SupportsSessionSeeding = true
	profile.AudioInputSampleRate = 24000
	profile.AudioOutputSampleRate = 24000
	return profile
}

func TestSessionLifecycle(t *testing.T) {
	connection := newFakeConnection()
	connection.model = "served-model"
	model := &fakeModel{connection: connection, profile: fullProfile()}
	seed := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "seed"}}}}
	providerSettings := map[string]any{
		"map":     map[string]any{"value": "original"},
		"list":    []any{map[string]any{"value": "original"}},
		"strings": []string{"original"}, "bytes": []byte("original"),
	}
	session, err := realtime.Open(t.Context(), model, realtime.ConnectParams{
		Messages: seed, Settings: realtime.Settings{Provider: providerSettings},
	},
		realtime.WithAudioRetention(realtime.AudioRetentionAll),
		realtime.WithImageRetention(1, 1),
		realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
			_ context.Context, call ai.ToolCallPart,
		) (any, error) {
			if call.ToolName != "lookup" || string(call.Args) != `{}` {
				t.Fatalf("unexpected tool call: %+v", call)
			}
			return ai.ToolReturn{
				ReturnValue: map[string]any{"answer": 42},
				Content:     []ai.UserContent{ai.TextContent{Text: "detail"}},
				Metadata:    map[string]any{"trace": "value"},
			}, nil
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	providerSettings["map"].(map[string]any)["value"] = "mutated"
	providerSettings["list"].([]any)[0].(map[string]any)["value"] = "mutated"
	providerSettings["strings"].([]string)[0] = "mutated"
	providerSettings["bytes"].([]byte)[0] = 'x'
	captured := model.params.Settings.Provider
	if captured["map"].(map[string]any)["value"] != "original" ||
		captured["list"].([]any)[0].(map[string]any)["value"] != "original" ||
		captured["strings"].([]string)[0] != "original" || string(captured["bytes"].([]byte)) != "original" {
		t.Fatalf("provider settings were not detached: %+v", captured)
	}
	if session.Profile().AudioInputSampleRate != 24000 || session.AudioInputSampleRate() != 24000 ||
		session.AudioOutputSampleRate() != 24000 || !session.InputTranscriptionEnabled() ||
		!session.ReconnectRestoresInFlightState() || session.Closed() || session.Err() != nil {
		t.Fatalf("unexpected opened session state")
	}

	audioReady := make(chan struct{})
	audioDone := make(chan []byte, 1)
	audioCtx, cancelAudio := context.WithCancel(t.Context())
	defer cancelAudio()
	go func() {
		close(audioReady)
		for chunk, err := range session.StreamAudio(audioCtx) {
			if err == nil {
				audioDone <- chunk
				return
			}
		}
	}()
	transcriptReady := make(chan struct{})
	transcriptDone := make(chan realtime.TranscriptUpdate, 1)
	transcriptCtx, cancelTranscript := context.WithCancel(t.Context())
	defer cancelTranscript()
	go func() {
		close(transcriptReady)
		for update, err := range session.StreamTranscripts(transcriptCtx) {
			if err == nil {
				transcriptDone <- update
				return
			}
		}
	}()
	<-audioReady
	<-transcriptReady
	time.Sleep(time.Millisecond)

	if err := session.Send(t.Context(), "hello"); err != nil {
		t.Fatal(err)
	}
	pcm := []byte{1, 0, 2, 0}
	if err := session.SendAudio(t.Context(), pcm, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	wav := testWAV(pcm, 24000)
	if err := session.Send(t.Context(), ai.BinaryContent{Data: wav, MediaType: "audio/wav"}); err != nil {
		t.Fatal(err)
	}
	firstImage := ai.BinaryContent{Data: []byte("first"), MediaType: "image/png"}
	secondImage := ai.BinaryContent{Data: []byte("second"), MediaType: "image/png"}
	if err := session.Send(t.Context(), firstImage); err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), secondImage); err != nil {
		t.Fatal(err)
	}
	if err := session.CommitAudio(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := session.ClearAudio(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := session.CreateResponse(t.Context()); err != nil {
		t.Fatal(err)
	}
	played := 12
	if err := session.Interrupt(t.Context(), &played); err != nil {
		t.Fatal(err)
	}

	connection.events <- realtime.InputSpeechStarted{ItemID: "user-1"}
	connection.events <- realtime.InputTranscript{Text: "Hel", ItemID: "user-1"}
	connection.events <- realtime.InputTranscript{Text: "Hello", Final: true, Cumulative: true, ItemID: "user-1"}
	connection.events <- realtime.InputSpeechEnded{ItemID: "user-1"}
	connection.events <- realtime.OutputSpeechStarted{}
	connection.events <- realtime.AudioDelta{Data: pcm, ItemID: "assistant-1"}
	connection.events <- realtime.OutputTranscript{Text: "Hi", ItemID: "assistant-1"}
	connection.events <- realtime.OutputTranscript{Text: " there", Final: true, ItemID: "assistant-1"}
	connection.events <- realtime.OutputSpeechEnded{}
	connection.events <- realtime.ToolCall{ToolCallID: "call-1", ToolName: "lookup"}

	var toolResult realtime.ToolResult
	deadline := time.After(2 * time.Second)
	for toolResult.ToolCallID == "" {
		select {
		case input := <-connection.sent:
			if result, ok := input.(realtime.ToolResult); ok {
				toolResult = result
			}
		case <-deadline:
			t.Fatal("tool result was not sent")
		}
	}
	if !strings.Contains(toolResult.Output, "42") || len(toolResult.Content) != 1 {
		t.Fatalf("unexpected tool result: %+v", toolResult)
	}
	connection.events <- realtime.SessionUsage{
		Usage:              ai.Usage{Requests: 1, InputTokens: 3, OutputTokens: 2},
		ProviderResponseID: "response-1", FinishReason: ai.FinishReasonStop, ResponseScoped: true,
	}
	connection.events <- realtime.ResponseDone{ProviderDetails: map[string]any{"status": "ok"}}
	connection.events <- realtime.InputTranscriptionError{ItemID: "bad", Err: errors.New("transcription")}
	connection.events <- realtime.SessionReconnected{StateRestored: true}
	connection.events <- realtime.ConversationCreated{ConversationID: "conversation"}
	connection.events <- realtime.ConversationItemCreated{ItemID: "replayed", Replayed: true}
	connection.events <- realtime.SessionError{Err: errors.New("recoverable"), Recoverable: true}
	connection.end()

	var events []realtime.Event
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if len(events) < 15 {
		t.Fatalf("too few translated events: %d", len(events))
	}
	if got := <-audioDone; !slices.Equal(got, pcm) {
		t.Fatalf("unexpected audio tap: %v", got)
	}
	if got := <-transcriptDone; got.Transcript == "" {
		t.Fatalf("unexpected transcript tap: %+v", got)
	}
	if !session.Closed() || session.Usage().InputTokens != 3 {
		t.Fatalf("unexpected final state: closed=%v usage=%+v", session.Closed(), session.Usage())
	}
	messages := session.Messages()
	if len(messages) < 6 || messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "seed" {
		t.Fatalf("unexpected messages: %+v", messages)
	}
	foundServed := false
	foundSecondImage := false
	foundFirstImage := false
	for _, message := range session.NewMessages() {
		switch message := message.(type) {
		case ai.ModelResponse:
			foundServed = foundServed || message.ModelName == "served-model" && message.ProviderResponseID == "response-1"
		case ai.ModelRequest:
			for _, part := range message.Parts {
				prompt, ok := part.(ai.UserPromptPart)
				if !ok || len(prompt.Contents) != 1 {
					continue
				}
				binary, ok := prompt.Contents[0].(ai.BinaryContent)
				if ok {
					foundFirstImage = foundFirstImage || string(binary.Data) == "first"
					foundSecondImage = foundSecondImage || string(binary.Data) == "second"
				}
			}
		}
	}
	if !foundServed || foundFirstImage || !foundSecondImage {
		t.Fatalf("response identity or image retention mismatch: served=%v first=%v second=%v", foundServed, foundFirstImage, foundSecondImage)
	}
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestSessionToolFailuresAndCancellation(t *testing.T) {
	for name, execute := range map[string]realtime.ToolExecutorFunc{
		"retry":  func(context.Context, ai.ToolCallPart) (any, error) { return nil, ai.Retryf("again") },
		"failed": func(context.Context, ai.ToolCallPart) (any, error) { return nil, ai.ToolFailedf("gone") },
		"error":  func(context.Context, ai.ToolCallPart) (any, error) { return nil, errors.New("boom") },
	} {
		t.Run(name, func(t *testing.T) {
			connection := newFakeConnection()
			session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
				realtime.ConnectParams{}, realtime.WithToolExecutor(execute))
			if err != nil {
				t.Fatal(err)
			}
			connection.events <- realtime.ToolCall{ToolCallID: name, ToolName: name, Arguments: `{}`}
			select {
			case input := <-connection.sent:
				result := input.(realtime.ToolResult)
				if result.Output == "" {
					t.Fatalf("empty failure output: %+v", result)
				}
			case <-time.After(time.Second):
				t.Fatal("missing failure result")
			}
			connection.end()
			for range session.Events(t.Context()) {
			}
		})
	}

	connection := newFakeConnection()
	started := make(chan struct{})
	cancelled := make(chan struct{})
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
			ctx context.Context, _ ai.ToolCallPart,
		) (any, error) {
			close(started)
			<-ctx.Done()
			close(cancelled)
			return nil, ctx.Err()
		})))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ToolCall{ToolCallID: "cancel", ToolName: "slow", Arguments: `{}`}
	<-started
	connection.events <- realtime.ToolCallCancelled{ToolCallIDs: []string{"cancel"}}
	<-cancelled
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestSessionValidationAndErrors(t *testing.T) {
	profile := fullProfile()
	var typedNilConnection *fakeConnection
	tests := map[string]struct {
		model   *fakeModel
		params  realtime.ConnectParams
		options []realtime.SessionOption
	}{
		"nil model": {model: nil},
		"negative tokens": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, params: realtime.ConnectParams{
			Settings: realtime.Settings{MaxTokens: -1},
		}},
		"bad modality": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, params: realtime.ConnectParams{
			Settings: realtime.Settings{OutputModality: "video"},
		}},
		"bad timeout": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, params: realtime.ConnectParams{
			Settings: realtime.Settings{HandshakeTimeout: -time.Second},
		}},
		"bad sensitivity": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, params: realtime.ConnectParams{
			Settings: realtime.Settings{TurnDetection: &realtime.TurnDetection{Enabled: true, Sensitivity: "extreme"}},
		}},
		"bad turn duration": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, params: realtime.ConnectParams{
			Settings: realtime.Settings{TurnDetection: &realtime.TurnDetection{PrefixPadding: -time.Second}},
		}},
		"bad reconnect": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, params: realtime.ConnectParams{
			Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{MaxAttempts: -1}},
		}},
		"bad retention": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, options: []realtime.SessionOption{
			realtime.WithAudioRetention("future"),
		}},
		"bad image interval": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, options: []realtime.SessionOption{
			realtime.WithImageRetention(0, 1),
		}},
		"bad image maximum": {model: &fakeModel{connection: newFakeConnection(), profile: profile}, options: []realtime.SessionOption{
			realtime.WithImageRetention(1, -2),
		}},
		"bad sample rate":      {model: &fakeModel{connection: newFakeConnection(), profile: realtime.Profile{}}},
		"connect":              {model: &fakeModel{profile: profile, err: errors.New("dial")}},
		"nil connection":       {model: &fakeModel{profile: profile}},
		"typed nil connection": {model: &fakeModel{profile: profile, connection: typedNilConnection}},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := realtime.Open(t.Context(), test.model, test.params, test.options...); err == nil {
				t.Fatal("expected error")
			}
		})
	}

	textUnsupported := profile
	textUnsupported.SupportsTextOutput = false
	if _, err := realtime.Open(t.Context(), &fakeModel{connection: newFakeConnection(), profile: textUnsupported},
		realtime.ConnectParams{Settings: realtime.Settings{OutputModality: realtime.OutputModalityText}}); err == nil {
		t.Fatal("expected text output error")
	}
	seedUnsupported := profile
	seedUnsupported.SupportsSessionSeeding = false
	if _, err := realtime.Open(t.Context(), &fakeModel{connection: newFakeConnection(), profile: seedUnsupported},
		realtime.ConnectParams{Messages: []ai.ModelMessage{ai.ModelRequest{}}}); err == nil {
		t.Fatal("expected seeding error")
	}
	profile.SupportedNativeTools = map[string]bool{}
	if _, err := realtime.Open(t.Context(), &fakeModel{connection: newFakeConnection(), profile: profile},
		realtime.ConnectParams{Request: ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}}}); err == nil {
		t.Fatal("expected native tool error")
	}
}

func TestSessionInputAndControlErrors(t *testing.T) {
	profile := realtime.DefaultProfile()
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), ""); err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{
		42,
		ai.BinaryContent{MediaType: "image/png"},
		ai.BinaryContent{Data: []byte("x"), MediaType: "application/pdf"},
	} {
		if err := session.Send(t.Context(), value); err == nil {
			t.Fatalf("expected input error for %T", value)
		}
	}
	if err := session.Send(t.Context(), ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"}); err == nil {
		t.Fatal("expected image capability error")
	}
	for _, call := range []func() error{
		func() error { return session.CommitAudio(t.Context()) },
		func() error { return session.ClearAudio(t.Context()) },
		func() error { return session.CreateResponse(t.Context()) },
		func() error { return session.Interrupt(t.Context(), nil) },
	} {
		if err := call(); err == nil {
			t.Fatal("expected control capability error")
		}
	}
	if err := session.SendAudio(t.Context(), []byte{1}, "audio/pcm"); err == nil {
		t.Fatal("expected odd PCM error")
	}
	if err := session.SendAudio(t.Context(), []byte{1, 0}, "audio/mp3"); err == nil {
		t.Fatal("expected media error")
	}
	if err := session.SendAudio(t.Context(), []byte("not wave"), "audio/wav"); err == nil {
		t.Fatal("expected WAV error")
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
	if err := session.Send(t.Context(), "closed"); err == nil {
		t.Fatal("expected closed error")
	}
}

func TestProfileMergeAndError(t *testing.T) {
	base := realtime.DefaultProfile()
	base.SupportedNativeTools["search"] = true
	no := false
	yes := true
	merged := realtime.MergeProfile(base, realtime.ProfileOverride{
		SupportsTextOutput: &no, SupportsImageInput: &yes,
		SupportsManualTurnControl: &yes, SupportsInterruption: &yes,
		SupportsOutputTruncation: &yes, SupportsSessionSeeding: &yes,
		SupportsWebRTC: &yes, SupportsSeedingImages: &yes, SupportsSeedingAudio: &yes,
		SupportsThinking: &yes, SupportsAsyncToolCalls: &yes, SupportsToolReturnSchema: &yes,
		EmitsInputSpeechEvents: &yes, SupportedNativeTools: map[string]bool{"code": true},
		AudioInputSampleRate: 16000, AudioOutputSampleRate: 22050,
	})
	if merged.SupportsTextOutput || !merged.SupportsImageInput || !merged.SupportsManualTurnControl ||
		!merged.SupportsInterruption || !merged.SupportsOutputTruncation || !merged.SupportsSessionSeeding ||
		!merged.SupportsWebRTC || !merged.SupportsSeedingImages || !merged.SupportsSeedingAudio ||
		!merged.SupportsThinking || !merged.SupportsAsyncToolCalls || !merged.SupportsToolReturnSchema ||
		!merged.EmitsInputSpeechEvents || merged.AudioInputSampleRate != 16000 || merged.AudioOutputSampleRate != 22050 ||
		merged.SupportedNativeTools["search"] || !merged.SupportedNativeTools["code"] {
		t.Fatalf("unexpected merged profile: %+v", merged)
	}
	merged.SupportedNativeTools["mutated"] = true
	if base.SupportedNativeTools["mutated"] {
		t.Fatal("profile map was not detached")
	}
	root := errors.New("root")
	err := &realtime.Error{Provider: "test", Model: "model", Message: "failed", Err: root}
	if !errors.Is(err, root) || !strings.Contains(err.Error(), "test/model") {
		t.Fatalf("unexpected realtime error: %v", err)
	}
	if (&realtime.Error{Message: "failed"}).Error() != "realtime: failed" {
		t.Fatal("unexpected plain error")
	}
	session := realtime.WebRTCSession{Provider: "openai", ID: "call"}
	if session.ProviderName() != "openai" || session.SessionID() != "call" {
		t.Fatalf("unexpected WebRTC session: %+v", session)
	}
}

func testWAV(pcm []byte, rate int) []byte {
	data := make([]byte, 44+len(pcm))
	copy(data[:4], "RIFF")
	data[4] = byte(36 + len(pcm))
	copy(data[8:12], "WAVE")
	copy(data[12:16], "fmt ")
	data[16] = 16
	data[20] = 1
	data[22] = 1
	data[24] = byte(rate)
	data[25] = byte(rate >> 8)
	data[28] = byte(rate * 2)
	data[29] = byte(rate * 2 >> 8)
	data[32] = 2
	data[34] = 16
	copy(data[36:40], "data")
	data[40] = byte(len(pcm))
	copy(data[44:], pcm)
	return data
}
