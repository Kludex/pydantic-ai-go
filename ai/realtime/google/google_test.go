package google_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	googlert "github.com/Kludex/pydantic-ai-go/ai/realtime/google"
	"google.golang.org/genai"
)

type fakeConnector struct {
	session *fakeSession
	model   string
	config  *genai.LiveConnectConfig
	err     error
}

func (connector *fakeConnector) Connect(
	_ context.Context, model string, config *genai.LiveConnectConfig,
) (googlert.LiveSession, error) {
	connector.model = model
	connector.config = config
	if connector.err != nil {
		return nil, connector.err
	}
	return connector.session, nil
}

type fakeSession struct {
	mutex    sync.Mutex
	receive  chan *genai.LiveServerMessage
	closed   chan struct{}
	once     sync.Once
	client   []genai.LiveClientContentInput
	realtime []genai.LiveRealtimeInput
	tools    []genai.LiveToolResponseInput
	err      error
}

func newFakeSession() *fakeSession {
	return &fakeSession{receive: make(chan *genai.LiveServerMessage, 16), closed: make(chan struct{})}
}

func (session *fakeSession) SendClientContent(input genai.LiveClientContentInput) error {
	session.mutex.Lock()
	defer session.mutex.Unlock()
	session.client = append(session.client, input)
	return session.err
}
func (session *fakeSession) SendRealtimeInput(input genai.LiveRealtimeInput) error {
	session.mutex.Lock()
	defer session.mutex.Unlock()
	session.realtime = append(session.realtime, input)
	return session.err
}
func (session *fakeSession) SendToolResponse(input genai.LiveToolResponseInput) error {
	session.mutex.Lock()
	defer session.mutex.Unlock()
	session.tools = append(session.tools, input)
	return session.err
}
func (session *fakeSession) counts() (clients, tools int) {
	session.mutex.Lock()
	defer session.mutex.Unlock()
	return len(session.client), len(session.tools)
}
func (session *fakeSession) Receive() (*genai.LiveServerMessage, error) {
	message, ok := <-session.receive
	if !ok {
		return nil, errors.New("closed")
	}
	return message, nil
}
func (session *fakeSession) Close() error {
	session.once.Do(func() {
		close(session.closed)
		close(session.receive)
	})
	return session.err
}

func TestThinkingProfileExcludesHalfCascadeLiveModel(t *testing.T) {
	for _, test := range []struct {
		name string
		want bool
	}{
		{name: "gemini-2.5-flash-native-audio-latest", want: true},
		{name: "gemini-3.1-flash-live-preview", want: true},
		{name: "gemini-live-2.5-flash"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := googlert.NewModel(test.name).Profile().SupportsThinking; got != test.want {
				t.Fatalf("SupportsThinking = %v, want %v", got, test.want)
			}
		})
	}
}

func TestGoogleRealtimeSession(t *testing.T) {
	live := newFakeSession()
	connector := &fakeConnector{session: live}
	inputTranscription := true
	outputTranscription := true
	affective := true
	proactive := true
	resume := true
	temperature := float32(0.5)
	model := googlert.NewModel("gemini-2.5-flash-native-audio-latest",
		googlert.WithConnector(connector),
		googlert.WithSettings(googlert.Settings{
			Temperature: &temperature, Voice: "Puck", LanguageCode: "en-US",
			InputTranscription: &inputTranscription, OutputTranscription: &outputTranscription,
			AffectiveDialog: &affective, ProactiveAudio: &proactive,
			AsyncToolCalls: true, EnableSessionResumption: &resume,
		}),
	)
	transcript := "seeded"
	params := realtime.ConnectParams{
		Messages: []ai.ModelMessage{
			ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Content: "history"},
				ai.UserPromptPart{Contents: []ai.UserContent{
					ai.TextContent{Text: "rich"}, ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
				}},
				ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: &transcript},
				ai.ToolReturnPart{ToolName: "tool", ToolCallID: "return", Content: map[string]any{"ok": true}},
				ai.RetryPromptPart{ToolName: "tool", ToolCallID: "retry", Content: "retry"},
			}},
			ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.TextPart{Content: "answer"},
				ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant, Transcript: &transcript},
				ai.ThinkingPart{Content: "thought"},
				ai.ToolCallPart{ToolName: "tool", ToolCallID: "call"},
			}},
		},
		Settings: realtime.Settings{
			MaxTokens: 100, Thinking: ai.ThinkingLevelHigh,
			TurnDetection: &realtime.TurnDetection{
				Enabled: true, Sensitivity: "high", PrefixPadding: 10 * time.Millisecond,
			},
		},
		Request: ai.ModelRequestParams{
			Instructions: "Be concise.",
			Tools: []ai.ToolDefinition{{
				Name: "lookup", Schema: map[string]any{"type": "object"},
				ReturnSchema: map[string]any{"type": "object"},
			}},
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
		},
	}
	session, err := realtime.Open(t.Context(), model, params, realtime.WithToolExecutor(
		realtime.ToolExecutorFunc(func(context.Context, ai.ToolCallPart) (any, error) { return "result", nil }),
	))
	if err != nil {
		t.Fatal(err)
	}
	clientCount, _ := live.counts()
	if connector.model == "" || connector.config.MaxOutputTokens != 100 || connector.config.SpeechConfig == nil ||
		len(connector.config.Tools) != 2 || connector.config.ThinkingConfig == nil || clientCount != 1 {
		t.Fatalf("unexpected Gemini config: %+v client_count=%d", connector.config, clientCount)
	}
	if session.AudioInputSampleRate() != 16000 || session.Profile().SupportsTextOutput ||
		!session.Profile().SupportsToolReturnSchema || !session.InputTranscriptionEnabled() ||
		!session.ReconnectRestoresInFlightState() {
		t.Fatalf("unexpected Gemini profile: %+v", session.Profile())
	}
	if err := session.SendAudio(t.Context(), []byte{1, 0}, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}); err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), "hello"); err != nil {
		t.Fatal(err)
	}
	live.receive <- &genai.LiveServerMessage{ServerContent: &genai.LiveServerContent{
		InterimInputTranscription: &genai.Transcription{Text: "hel"},
		InputTranscription:        &genai.Transcription{Text: "hello", Finished: true},
		OutputTranscription:       &genai.Transcription{Text: "hi", Finished: true},
		ModelTurn: &genai.Content{Parts: []*genai.Part{
			{InlineData: &genai.Blob{Data: []byte{1, 0}, MIMEType: "audio/pcm"}},
			{Text: "plain"},
		}},
		GroundingMetadata: &genai.GroundingMetadata{}, TurnComplete: true,
	}}
	live.receive <- &genai.LiveServerMessage{ToolCall: &genai.LiveServerToolCall{FunctionCalls: []*genai.FunctionCall{{
		ID: "call", Name: "lookup", Args: map[string]any{},
	}}}}
	live.receive <- &genai.LiveServerMessage{ToolCallCancellation: &genai.LiveServerToolCallCancellation{IDs: []string{"missing"}}}
	live.receive <- &genai.LiveServerMessage{UsageMetadata: &genai.UsageMetadata{
		PromptTokenCount: 3, ResponseTokenCount: 2, CachedContentTokenCount: 1, ThoughtsTokenCount: 1,
		PromptTokensDetails:   []*genai.ModalityTokenCount{{Modality: "TEXT", TokenCount: 3}},
		ResponseTokensDetails: []*genai.ModalityTokenCount{{Modality: "AUDIO", TokenCount: 2}},
	}}

	turns := 0
	calls := 0
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		switch event.(type) {
		case realtime.TurnCompleteEvent:
			turns++
		case ai.FunctionToolCallEvent:
			calls++
		}
		if turns == 1 && calls == 1 {
			break
		}
	}
	deadline := time.After(time.Second)
	for {
		_, toolCount := live.counts()
		if toolCount > 0 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("tool result was not sent")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	if session.Usage().InputTokens != 3 || session.Usage().Details["input_text_tokens"] != 3 {
		t.Fatalf("unexpected Gemini usage: %+v", session.Usage())
	}
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestGoogleConnectionAndValidation(t *testing.T) {
	live := newFakeSession()
	connection := &googlert.Connection{}
	_ = connection

	connector := &fakeConnector{session: live}
	model := googlert.NewModel("model", googlert.WithConnector(connector))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		TurnDetection: &realtime.TurnDetection{Enabled: false},
	}}); err == nil {
		t.Fatal("expected manual turn error")
	}
	no := false
	if _, err := googlert.NewModel("model", googlert.WithConnector(connector), googlert.WithSettings(
		googlert.Settings{EnableSessionResumption: &no},
	)).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Second},
	}}); err == nil {
		t.Fatal("expected resumption error")
	}
	if _, err := googlert.NewModel("", googlert.WithConnector(connector)).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected model name error")
	}
	connector.err = errors.New("dial")
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected connector error")
	}
	connector.err = nil
	connector.session = nil
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected nil session error")
	}
}
