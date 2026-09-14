package realtime

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"sync"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Event is one session event. Values are ai.PartStartEvent, ai.PartDeltaEvent,
// ai.PartEndEvent, ai.FunctionToolCallEvent, ai.FunctionToolResultEvent, or one
// of the realtime lifecycle events in this package.
type Event = any

// TurnCompleteEvent contains the finalized provider response.
type TurnCompleteEvent struct {
	// Response is the detached finalized provider response.
	Response ai.ModelResponse
}

// InputSpeechStartEvent reports server-detected user speech.
type InputSpeechStartEvent struct {
	// ItemID identifies the detected input segment when available.
	ItemID string
}

// InputSpeechEndEvent reports the end of server-detected user speech.
type InputSpeechEndEvent struct {
	// ItemID identifies the detected input segment when available.
	ItemID string
}

// OutputSpeechStartEvent reports model audio playback starting.
type OutputSpeechStartEvent struct{}

// OutputSpeechEndEvent reports model audio playback ending.
type OutputSpeechEndEvent struct{}

// ResponseInterruptedEvent reports a cancelled response.
type ResponseInterruptedEvent struct {
	// PlayedMilliseconds is the amount of audio retained in provider history.
	PlayedMilliseconds *int
}

// InputTranscriptionErrorEvent reports a recoverable transcription failure.
type InputTranscriptionErrorEvent struct {
	// ItemID identifies the input segment that failed.
	ItemID string
	// Err describes the recoverable transcription failure.
	Err error
}

// SessionReconnectEvent reports a successful transport reconnect.
type SessionReconnectEvent struct {
	// StateRestored reports that the provider resumed in-flight state.
	StateRestored bool
}

// SessionErrorEvent reports a recoverable provider error.
type SessionErrorEvent struct {
	// Err describes a recoverable provider or protocol error.
	Err error
}

// TranscriptUpdate is an incremental, render-ready transcript update.
type TranscriptUpdate struct {
	// Index identifies one speech part for the session lifetime.
	Index int
	// Speaker identifies the user or assistant turn.
	Speaker ai.SpeechSpeaker
	// Delta is newly appended text and is empty for a revision.
	Delta string
	// Transcript is the complete render-ready text so far.
	Transcript string
}

// ToolExecutor executes provider-requested local tools.
type ToolExecutor interface {
	// ExecuteTool runs one provider-requested local function.
	ExecuteTool(ctx context.Context, call ai.ToolCallPart) (any, error)
}

// ToolExecutorFunc adapts a function into ToolExecutor.
type ToolExecutorFunc func(ctx context.Context, call ai.ToolCallPart) (any, error)

// ExecuteTool calls the adapted function.
func (function ToolExecutorFunc) ExecuteTool(ctx context.Context, call ai.ToolCallPart) (any, error) {
	return function(ctx, call)
}

// SessionOption configures one session.
type SessionOption func(*sessionConfig)

type sessionConfig struct {
	audioRetention    AudioRetention
	toolExecutor      ToolExecutor
	retainImagesEvery int
	retainImagesMax   int
	handleBargeIn     bool
}

// WithAudioRetention controls which raw audio streams are retained in history.
func WithAudioRetention(retention AudioRetention) SessionOption {
	return func(config *sessionConfig) { config.audioRetention = retention }
}

// WithToolExecutor enables concurrent local function-tool execution.
func WithToolExecutor(executor ToolExecutor) SessionOption {
	return func(config *sessionConfig) { config.toolExecutor = executor }
}

// WithImageRetention samples image frames and bounds retained image history.
// every must be at least one. A maximum of -1 keeps every sampled frame.
func WithImageRetention(every, maximum int) SessionOption {
	return func(config *sessionConfig) {
		config.retainImagesEvery = every
		config.retainImagesMax = maximum
	}
}

// WithBargeIn makes supported models flush unheard audio and interrupt output when user speech starts.
func WithBargeIn(enabled bool) SessionOption {
	return func(config *sessionConfig) { config.handleBargeIn = enabled }
}

// SendOption configures one Session.Send call.
type SendOption func(*sendConfig)

type sendConfig struct{ respond *bool }

// WithResponse controls whether text or image input asks the model to respond.
func WithResponse(respond bool) SendOption {
	return func(config *sendConfig) { config.respond = &respond }
}

// Session owns one live connection, event pump, tool tasks, and portable history.
type Session struct {
	model      Model
	connection Connection
	profile    Profile
	config     sessionConfig

	ctx      context.Context
	cancel   context.CancelCauseFunc
	done     chan struct{}
	events   chan eventResult
	eventsMu sync.RWMutex

	sendMu sync.Mutex
	mu     sync.RWMutex
	closed bool
	err    error
	usage  ai.Usage

	seeded              []ai.ModelMessage
	history             []ai.ModelMessage
	responseParts       []ai.ResponsePart
	activeAssistant     *activeSpeech
	userTurns           map[string]*activeSpeech
	anonymousUser       *activeSpeech
	nextPartIndex       int
	inputAudio          []byte
	outputAudio         []byte
	imageCount          int
	retainedImages      []ai.ModelRequest
	pendingUsage        ai.Usage
	pendingResponseID   string
	pendingFinish       ai.FinishReason
	pendingToolResults  map[string]ai.ModelRequest
	providerPartIndexes map[string]int
	responseActive      bool
	serverCancelling    bool

	toolMu          sync.Mutex
	toolCancels     map[string]context.CancelFunc
	toolWG          sync.WaitGroup
	closingFromTool bool

	tapMu                     sync.Mutex
	audioTaps                 map[*audioTap]struct{}
	transcriptTaps            map[chan TranscriptUpdate]struct{}
	emittedAudioBytes         int
	turnAudioStartBytes       int
	audioPartIndex            int
	hasAudioPart              bool
	interruptedAudioPartIndex int
	hasInterruptedAudioPart   bool

	enqueueMu  sync.Mutex
	deliveryMu sync.Mutex
	enqueued   []queuedPrompt
}

type audioTap struct {
	queue               chan []byte
	subscribedAtBytes   int
	droppedBytes        int
	pendingDroppedBytes int
	playedBytes         int
}

type queuedPrompt struct {
	id       string
	priority ai.PendingMessagePriority
	text     string
}

type activeSpeech struct {
	index      int
	partID     string
	itemID     string
	speaker    ai.SpeechSpeaker
	transcript string
	audio      []byte
	text       bool
}

type eventResult struct {
	event Event
	err   error
}

type toolContextKey struct{}

type toolContextValue struct {
	session *Session
	callID  string
}

// Open validates settings, opens a provider connection, and starts its receive pump.
func Open(ctx context.Context, model Model, params ConnectParams, options ...SessionOption) (*Session, error) {
	params, profile, err := validateConnect(model, params)
	if err != nil {
		return nil, err
	}
	config := sessionConfig{
		audioRetention:    AudioRetentionTranscriptOnly,
		retainImagesEvery: 1,
		retainImagesMax:   100,
	}
	for _, option := range options {
		option(&config)
	}
	if err := validateSessionConfig(config); err != nil {
		return nil, err
	}
	connection, err := model.Connect(ctx, params)
	if err != nil {
		return nil, &Error{Provider: model.ProviderName(), Model: model.Name(), Message: "connect", Err: err}
	}
	if connectionIsNil(connection) {
		return nil, fmt.Errorf("realtime: model %q returned a nil connection", model.Name())
	}
	runCtx, cancel := context.WithCancelCause(ctx)
	session := &Session{
		model: model, connection: connection, profile: profile, config: config,
		ctx: runCtx, cancel: cancel, done: make(chan struct{}), events: make(chan eventResult, 128),
		seeded: params.Messages, userTurns: map[string]*activeSpeech{},
		pendingToolResults: map[string]ai.ModelRequest{}, providerPartIndexes: map[string]int{},
		toolCancels: map[string]context.CancelFunc{}, audioTaps: map[*audioTap]struct{}{},
		transcriptTaps: map[chan TranscriptUpdate]struct{}{},
	}
	if historyAware, ok := connection.(HistoryAwareConnection); ok {
		historyAware.SetMessageHistory(session.Messages)
	}
	go session.pump()
	return session, nil
}

func connectionIsNil(connection Connection) bool {
	if connection == nil {
		return true
	}
	value := reflect.ValueOf(connection)
	return value.Kind() == reflect.Pointer && value.IsNil()
}

func validateSessionConfig(config sessionConfig) error {
	switch config.audioRetention {
	case AudioRetentionTranscriptOnly, AudioRetentionInput, AudioRetentionOutput, AudioRetentionAll:
	default:
		return fmt.Errorf("realtime: unsupported audio retention %q", config.audioRetention)
	}
	if config.retainImagesEvery < 1 {
		return fmt.Errorf("realtime: image retention interval must be at least one")
	}
	if config.retainImagesMax < -1 {
		return fmt.Errorf("realtime: image retention maximum must be at least zero, or -1 for no limit")
	}
	return nil
}

// Profile returns a detached capability profile.
func (session *Session) Profile() Profile { return cloneProfile(session.profile) }

// AudioInputSampleRate returns the required raw PCM input rate.
func (session *Session) AudioInputSampleRate() int { return session.profile.AudioInputSampleRate }

// AudioOutputSampleRate returns the raw PCM output rate.
func (session *Session) AudioOutputSampleRate() int { return session.profile.AudioOutputSampleRate }

// InputTranscriptionEnabled reports whether user transcript events are expected.
func (session *Session) InputTranscriptionEnabled() bool {
	info, ok := session.connection.(ConnectionInfo)
	return !ok || info.InputTranscriptionEnabled()
}

// ReconnectRestoresInFlightState reports whether reconnect preserves unfinished work.
func (session *Session) ReconnectRestoresInFlightState() bool {
	info, ok := session.connection.(ConnectionInfo)
	return !ok || info.ReconnectRestoresInFlightState()
}

// Closed reports whether the pump and tool tasks have stopped.
func (session *Session) Closed() bool {
	session.mu.RLock()
	defer session.mu.RUnlock()
	return session.closed
}

// Err returns the terminal pump or transport error.
func (session *Session) Err() error {
	session.mu.RLock()
	defer session.mu.RUnlock()
	return session.err
}

// Usage returns detached cumulative session usage.
func (session *Session) Usage() ai.Usage {
	session.mu.RLock()
	defer session.mu.RUnlock()
	return session.usage.Clone()
}

// Messages returns seeded and newly accumulated portable history.
func (session *Session) Messages() []ai.ModelMessage {
	session.mu.RLock()
	defer session.mu.RUnlock()
	return (ai.ModelRequestContext{Messages: append(slices.Clone(session.seeded), session.history...)}).Clone().Messages
}

// NewMessages returns history produced by this session only.
func (session *Session) NewMessages() []ai.ModelMessage {
	session.mu.RLock()
	defer session.mu.RUnlock()
	return (ai.ModelRequestContext{Messages: session.history}).Clone().Messages
}

// Events yields session events. Only one consumer may iterate it.
func (session *Session) Events(ctx context.Context) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		for {
			select {
			case <-ctx.Done():
				yield(nil, context.Cause(ctx))
				return
			case result, ok := <-session.events:
				if !ok {
					if err := session.Err(); err != nil {
						yield(nil, err)
					}
					return
				}
				if !yield(result.event, result.err) || result.err != nil {
					return
				}
			}
		}
	}
}

// Send writes one text turn, image frame, or audio value.
// Text asks for a response by default. Images and audio do not.
func (session *Session) Send(ctx context.Context, content any, options ...SendOption) error {
	config := sendConfig{}
	for _, option := range options {
		option(&config)
	}
	switch value := any(content).(type) {
	case string:
		if value == "" {
			return nil
		}
		respond := config.respond == nil || *config.respond
		var input Input = TextInput{Text: value}
		if !respond {
			input = TextContext{Text: value}
		}
		if respond {
			session.setResponseActive(true)
		}
		if err := session.send(ctx, input); err != nil {
			if respond {
				session.setResponseActive(false)
			}
			return err
		}
		session.mu.Lock()
		session.history = append(session.history, ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: value}}})
		session.mu.Unlock()
		return nil
	case ai.BinaryContent:
		if len(value.Data) == 0 {
			return fmt.Errorf("realtime: binary input must not be empty")
		}
		if len(value.MediaType) >= 6 && value.MediaType[:6] == "image/" {
			return session.sendImage(ctx, value, config.respond != nil && *config.respond)
		}
		if len(value.MediaType) >= 6 && value.MediaType[:6] == "audio/" {
			if config.respond != nil && *config.respond {
				return fmt.Errorf("realtime: WithResponse(true) cannot be used with audio")
			}
			return session.SendAudio(ctx, value.Data, value.MediaType)
		}
		return fmt.Errorf("realtime: unsupported binary input media type %q", value.MediaType)
	default:
		return fmt.Errorf("realtime: unsupported input %T", value)
	}
}

// SendAudio sends one raw PCM chunk or one PCM WAV payload.
func (session *Session) SendAudio(ctx context.Context, data []byte, mediaType string) error {
	if len(data) == 0 {
		return nil
	}
	pcm := slices.Clone(data)
	if mediaType == "audio/wav" || mediaType == "audio/x-wav" {
		decoded, rate, err := decodePCMWAV(pcm)
		if err != nil {
			return err
		}
		if rate != session.profile.AudioInputSampleRate {
			return fmt.Errorf("realtime: WAV sample rate %d does not match required %d", rate, session.profile.AudioInputSampleRate)
		}
		pcm = decoded
	} else if mediaType != "" && mediaType != "audio/pcm" && mediaType != "audio/L16" {
		return fmt.Errorf("realtime: audio must be PCM16 or PCM WAV, got %q", mediaType)
	}
	if len(pcm)%2 != 0 {
		return fmt.Errorf("realtime: PCM16 audio length must be even")
	}
	if err := session.send(ctx, AudioInput{Data: pcm}); err != nil {
		return err
	}
	if session.config.audioRetention == AudioRetentionInput || session.config.audioRetention == AudioRetentionAll {
		session.mu.Lock()
		session.inputAudio = append(session.inputAudio, pcm...)
		session.mu.Unlock()
	}
	return nil
}

// SendStream sends text, images, or audio values from a caller-owned iterator.
func (session *Session) SendStream(ctx context.Context, inputs iter.Seq2[any, error]) error {
	for input, err := range inputs {
		if err != nil {
			return err
		}
		if err := session.Send(ctx, input); err != nil {
			return err
		}
	}
	return nil
}

// SendAudioStream sends every chunk from a caller-owned iterator.
func (session *Session) SendAudioStream(ctx context.Context, chunks iter.Seq2[[]byte, error]) error {
	for chunk, err := range chunks {
		if err != nil {
			return err
		}
		if err := session.SendAudio(ctx, chunk, "audio/pcm"); err != nil {
			return err
		}
	}
	return nil
}

func (session *Session) sendImage(ctx context.Context, content ai.BinaryContent, respond bool) error {
	if !session.profile.SupportsImageInput {
		return fmt.Errorf("realtime: model %q does not support image input", session.model.Name())
	}
	if respond && !session.profile.SupportsManualTurnControl {
		return fmt.Errorf(
			"realtime: cannot ask for an image response: model %q does not support manual turn control",
			session.model.Name(),
		)
	}
	if respond {
		session.setResponseActive(true)
	}
	if err := session.send(ctx, ImageInput{Content: content, Respond: respond}); err != nil {
		if respond {
			session.setResponseActive(false)
		}
		return err
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	session.imageCount++
	if session.imageCount%session.config.retainImagesEvery != 0 {
		return nil
	}
	request := ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{content}}}}
	session.history = append(session.history, request)
	session.retainedImages = append(session.retainedImages, request)
	if session.config.retainImagesMax >= 0 && len(session.retainedImages) > session.config.retainImagesMax {
		oldest := session.retainedImages[0]
		session.retainedImages = session.retainedImages[1:]
		for index, message := range session.history {
			if sameImageRequest(message, oldest) {
				session.history = append(session.history[:index], session.history[index+1:]...)
				break
			}
		}
	}
	return nil
}

// CommitAudio commits buffered input for manual turn taking.
func (session *Session) CommitAudio(ctx context.Context) error {
	if err := session.require(session.profile.SupportsManualTurnControl, "commit audio", "manual turn control"); err != nil {
		return err
	}
	return session.send(ctx, CommitAudio{})
}

// ClearAudio discards provider and locally retained uncommitted input.
func (session *Session) ClearAudio(ctx context.Context) error {
	if err := session.require(session.profile.SupportsManualTurnControl, "clear audio", "manual turn control"); err != nil {
		return err
	}
	if err := session.send(ctx, ClearAudio{}); err != nil {
		return err
	}
	session.mu.Lock()
	session.inputAudio = nil
	session.mu.Unlock()
	return nil
}

// CreateResponse asks the model to answer now.
func (session *Session) CreateResponse(ctx context.Context) error {
	if err := session.require(session.profile.SupportsManualTurnControl, "create response", "manual turn control"); err != nil {
		return err
	}
	session.setResponseActive(true)
	if err := session.send(ctx, CreateResponse{}); err != nil {
		session.setResponseActive(false)
		return err
	}
	return nil
}

// Interrupt cancels output and optionally truncates provider history to heard audio.
func (session *Session) Interrupt(ctx context.Context, playedMilliseconds *int) error {
	if err := session.require(session.profile.SupportsInterruption, "interrupt", "interruption"); err != nil {
		return err
	}
	if playedMilliseconds != nil {
		if *playedMilliseconds < 0 {
			return fmt.Errorf("realtime: played milliseconds must not be negative")
		}
		if !session.profile.SupportsOutputTruncation {
			return fmt.Errorf("realtime: model %q does not support output truncation", session.model.Name())
		}
	}
	inputs := make([]Input, 0, 2)
	if playedMilliseconds != nil {
		inputs = append(inputs, TruncateOutput{AudioEndMilliseconds: *playedMilliseconds})
	}
	session.mu.RLock()
	serverCancelling := session.serverCancelling
	session.mu.RUnlock()
	if !serverCancelling {
		inputs = append(inputs, CancelResponse{})
	}
	if len(inputs) > 0 {
		if err := session.sendInputs(ctx, inputs...); err != nil {
			return err
		}
	}
	session.publish(ResponseInterruptedEvent{PlayedMilliseconds: cloneInt(playedMilliseconds)})
	return nil
}

func sameImageRequest(message ai.ModelMessage, expected ai.ModelRequest) bool {
	return reflect.DeepEqual(message, expected)
}

func (session *Session) require(supported bool, method, feature string) error {
	if !supported {
		return fmt.Errorf("realtime: cannot %s: model %q does not support %s", method, session.model.Name(), feature)
	}
	return nil
}

func (session *Session) setResponseActive(active bool) {
	session.mu.Lock()
	session.responseActive = active
	session.mu.Unlock()
}

func (session *Session) send(ctx context.Context, input Input) error {
	return session.sendInputs(ctx, input)
}

func (session *Session) sendInputs(ctx context.Context, inputs ...Input) error {
	session.mu.RLock()
	closed := session.closed
	session.mu.RUnlock()
	if closed {
		return fmt.Errorf("realtime: session is closed")
	}
	session.sendMu.Lock()
	defer session.sendMu.Unlock()
	for _, input := range inputs {
		_ = input.RealtimeInputKind()
		if err := session.connection.Send(ctx, input); err != nil {
			return &Error{Provider: session.model.ProviderName(), Model: session.model.Name(), Message: "send", Err: err}
		}
	}
	return nil
}

// StreamAudio yields model PCM chunks. The bounded subscription starts when this method is called.
func (session *Session) StreamAudio(ctx context.Context) iter.Seq2[[]byte, error] {
	tap := &audioTap{queue: make(chan []byte, 32)}
	session.tapMu.Lock()
	tap.subscribedAtBytes = session.emittedAudioBytes
	session.mu.RLock()
	closed := session.closed
	session.mu.RUnlock()
	if closed {
		close(tap.queue)
	} else {
		session.audioTaps[tap] = struct{}{}
	}
	session.tapMu.Unlock()
	return func(yield func([]byte, error) bool) {
		defer session.removeAudioTap(tap)
		for {
			select {
			case <-ctx.Done():
				yield(nil, context.Cause(ctx))
				return
			case chunk, ok := <-tap.queue:
				if !ok {
					return
				}
				session.tapMu.Lock()
				tap.droppedBytes += tap.pendingDroppedBytes
				tap.pendingDroppedBytes = 0
				session.tapMu.Unlock()
				if !yield(slices.Clone(chunk), nil) {
					return
				}
				session.tapMu.Lock()
				tap.playedBytes += len(chunk)
				session.tapMu.Unlock()
			}
		}
	}
}

// StreamTranscripts yields render-ready transcript updates.
// The bounded subscription starts when this method is called.
func (session *Session) StreamTranscripts(ctx context.Context) iter.Seq2[TranscriptUpdate, error] {
	queue := make(chan TranscriptUpdate, 512)
	session.tapMu.Lock()
	session.mu.RLock()
	closed := session.closed
	session.mu.RUnlock()
	if closed {
		close(queue)
	} else {
		session.transcriptTaps[queue] = struct{}{}
	}
	session.tapMu.Unlock()
	return func(yield func(TranscriptUpdate, error) bool) {
		defer func() {
			session.tapMu.Lock()
			delete(session.transcriptTaps, queue)
			session.tapMu.Unlock()
		}()
		for {
			select {
			case <-ctx.Done():
				var zero TranscriptUpdate
				yield(zero, context.Cause(ctx))
				return
			case update, ok := <-queue:
				if !ok || !yield(update, nil) {
					return
				}
			}
		}
	}
}

func (session *Session) removeAudioTap(tap *audioTap) {
	session.tapMu.Lock()
	delete(session.audioTaps, tap)
	session.tapMu.Unlock()
}

// Close stops the pump, cancels tools, closes the transport, and waits for cleanup.
func (session *Session) Close(ctx context.Context) error {
	current, fromTool := ctx.Value(toolContextKey{}).(toolContextValue)
	fromOwnTool := fromTool && current.session == session
	waitCtx := ctx
	if fromOwnTool {
		waitCtx = context.WithoutCancel(ctx)
	}
	// Publish tool-owned cleanup before cancellation wakes the pump.
	session.toolMu.Lock()
	if fromOwnTool {
		session.closingFromTool = true
	}
	for callID, cancel := range session.toolCancels {
		if !fromOwnTool || callID != current.callID {
			cancel()
		}
	}
	session.toolMu.Unlock()
	session.cancel(context.Canceled)
	closeErr := session.connection.Close(waitCtx)
	select {
	case <-session.done:
	case <-waitCtx.Done():
		return errors.Join(closeErr, context.Cause(waitCtx))
	}
	return errors.Join(closeErr, session.Err())
}

func (session *Session) pump() {
	defer func() {
		session.cancel(nil)
		_ = session.connection.Close(context.Background())
		session.toolMu.Lock()
		closingFromTool := session.closingFromTool
		session.toolMu.Unlock()
		if !closingFromTool {
			session.toolWG.Wait()
		}
		session.mu.Lock()
		session.closed = true
		session.mu.Unlock()
		session.tapMu.Lock()
		for tap := range session.audioTaps {
			for len(tap.queue) > 0 {
				<-tap.queue
			}
			close(tap.queue)
		}
		for tap := range session.transcriptTaps {
			for len(tap) > 0 {
				<-tap
			}
			close(tap)
		}
		session.audioTaps = map[*audioTap]struct{}{}
		session.transcriptTaps = map[chan TranscriptUpdate]struct{}{}
		session.tapMu.Unlock()
		session.eventsMu.Lock()
		close(session.events)
		session.eventsMu.Unlock()
		close(session.done)
	}()
	for event, err := range session.connection.Events(session.ctx) {
		if err != nil {
			session.fail(&Error{Provider: session.model.ProviderName(), Model: session.model.Name(), Message: "receive", Err: err})
			return
		}
		if !session.handle(event) {
			return
		}
	}
}

func (session *Session) handle(event CodecEvent) bool {
	_ = event.RealtimeCodecEventKind()
	switch event := event.(type) {
	case AudioDelta:
		session.handleAudio(event)
	case OutputTranscript:
		session.handleOutputTranscript(event)
	case InputTranscript:
		session.handleInputTranscript(event)
	case ToolCall:
		session.handleToolCall(event)
	case ToolCallCancelled:
		session.cancelTools(event.ToolCallIDs)
	case SessionUsage:
		session.handleUsage(event)
	case ResponseDone:
		session.finishResponse(event)
	case InputSpeechStarted:
		session.mu.Lock()
		interrupts, ok := session.connection.(SpeechInterruptionConnection)
		session.serverCancelling = ok && interrupts.InterruptsResponseOnSpeech()
		session.mu.Unlock()
		if session.config.handleBargeIn && session.profile.SupportsInterruption {
			if played, err := session.PlayedAudioBytes(); err == nil {
				if _, err := session.InterruptAtAudio(session.ctx, played); err != nil {
					session.fail(err)
					return false
				}
			}
		}
		session.publish(InputSpeechStartEvent(event))
	case InputSpeechEnded:
		session.publish(InputSpeechEndEvent(event))
	case OutputSpeechStarted:
		session.publish(OutputSpeechStartEvent{})
	case OutputSpeechEnded:
		session.publish(OutputSpeechEndEvent{})
	case InputTranscriptionError:
		session.publish(InputTranscriptionErrorEvent(event))
	case SessionReconnected:
		session.mu.Lock()
		session.serverCancelling = false
		session.mu.Unlock()
		session.publish(SessionReconnectEvent(event))
	case PartStarted:
		session.mu.Lock()
		started := event.Event
		started.Index = session.nextPartIndex
		session.nextPartIndex++
		session.providerPartIndexes[started.PartID] = started.Index
		session.responseParts = append(session.responseParts, started.Part)
		session.mu.Unlock()
		session.publish(started)
	case PartEnded:
		session.mu.Lock()
		ended := event.Event
		ended.Index = session.providerPartIndexes[ended.PartID]
		delete(session.providerPartIndexes, ended.PartID)
		session.mu.Unlock()
		session.publish(ended)
	case ResponseStarted:
		session.mu.Lock()
		session.pendingResponseID = event.ResponseID
		session.responseActive = true
		session.serverCancelling = false
		session.mu.Unlock()
	case ConversationCreated, ConversationItemCreated:
	case SessionError:
		if event.Recoverable {
			session.publish(SessionErrorEvent{Err: event.Err})
		} else {
			session.fail(&Error{Provider: session.model.ProviderName(), Model: session.model.Name(), Message: "session", Err: event.Err})
			return false
		}
	default:
		session.fail(fmt.Errorf("realtime: unsupported codec event %T", event))
		return false
	}
	return true
}

func (session *Session) handleAudio(event AudioDelta) {
	session.mu.Lock()
	session.responseActive = true
	active := session.ensureAssistantLocked(false, event.ItemID)
	active.audio = append(active.audio, event.Data...)
	if session.config.audioRetention == AudioRetentionOutput || session.config.audioRetention == AudioRetentionAll {
		session.outputAudio = append(session.outputAudio, event.Data...)
	}
	index, partID := active.index, active.partID
	session.mu.Unlock()
	session.publish(ai.PartDeltaEvent{Index: index, PartID: partID, Delta: ai.SpeechPartDelta{
		Speaker: ai.SpeechSpeakerAssistant, AudioChunk: slices.Clone(event.Data),
	}})
	session.publishAudio(index, event.Data)
}

func (session *Session) handleOutputTranscript(event OutputTranscript) {
	session.mu.Lock()
	session.responseActive = true
	active := session.ensureAssistantLocked(event.OutputText, event.ItemID)
	previous := active.transcript
	transcript, delta := accumulateTranscript(previous, event.Text, event.Final)
	active.transcript = transcript
	index, partID, speaker := active.index, active.partID, active.speaker
	session.mu.Unlock()
	if delta != "" || transcript != previous {
		if event.OutputText {
			session.publish(ai.PartDeltaEvent{Index: index, PartID: partID, Delta: ai.TextPartDelta{ContentDelta: delta}})
		} else {
			full := transcript
			session.publish(ai.PartDeltaEvent{Index: index, PartID: partID, Delta: ai.SpeechPartDelta{
				Speaker: speaker, TranscriptDelta: delta, Transcript: &full,
			}})
		}
		session.publishTranscript(TranscriptUpdate{Index: index, Speaker: speaker, Delta: delta, Transcript: transcript})
	}
	if event.Final {
		session.finishAssistantPart()
	}
}

func (session *Session) handleInputTranscript(event InputTranscript) {
	session.mu.Lock()
	active := session.userTurnLocked(event.ItemID)
	previous := active.transcript
	transcript, delta := accumulateTranscript(previous, event.Text, event.Cumulative)
	active.transcript = transcript
	index, partID := active.index, active.partID
	session.mu.Unlock()
	if delta != "" || transcript != previous {
		full := transcript
		session.publish(ai.PartDeltaEvent{Index: index, PartID: partID, Delta: ai.SpeechPartDelta{
			Speaker: ai.SpeechSpeakerUser, TranscriptDelta: delta, Transcript: &full,
		}})
		session.publishTranscript(TranscriptUpdate{
			Index: index, Speaker: ai.SpeechSpeakerUser, Delta: delta, Transcript: transcript,
		})
	}
	if event.Final {
		session.finishUser(event.ItemID)
	}
}

func (session *Session) ensureAssistantLocked(text bool, itemID string) *activeSpeech {
	if session.activeAssistant != nil && session.activeAssistant.text == text &&
		(itemID == "" || session.activeAssistant.itemID == "" || session.activeAssistant.itemID == itemID) {
		if session.activeAssistant.itemID == "" {
			session.activeAssistant.itemID = itemID
		}
		return session.activeAssistant
	}
	if session.activeAssistant != nil {
		session.finishAssistantPartLocked()
	}
	index := session.nextPartIndex
	session.nextPartIndex++
	partID := fmt.Sprintf("assistant-%d", index)
	active := &activeSpeech{
		index: index, partID: partID, itemID: itemID, speaker: ai.SpeechSpeakerAssistant, text: text,
	}
	session.activeAssistant = active
	if text {
		session.publishLocked(ai.PartStartEvent{Index: index, PartID: partID, Part: ai.TextPart{}})
	} else {
		session.publishLocked(ai.PartStartEvent{Index: index, PartID: partID, Part: ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant,
		}})
	}
	return active
}

func (session *Session) userTurnLocked(itemID string) *activeSpeech {
	var active *activeSpeech
	if itemID == "" {
		active = session.anonymousUser
	} else {
		active = session.userTurns[itemID]
	}
	if active != nil {
		return active
	}
	index := session.nextPartIndex
	session.nextPartIndex++
	active = &activeSpeech{index: index, partID: fmt.Sprintf("user-%d", index), itemID: itemID, speaker: ai.SpeechSpeakerUser}
	if itemID == "" {
		session.anonymousUser = active
	} else {
		session.userTurns[itemID] = active
	}
	session.publishLocked(ai.PartStartEvent{Index: index, PartID: active.partID, Part: ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}})
	return active
}

func (session *Session) finishAssistantPart() {
	session.mu.Lock()
	session.finishAssistantPartLocked()
	session.mu.Unlock()
}

func (session *Session) finishAssistantPartLocked() {
	active := session.activeAssistant
	if active == nil {
		return
	}
	var part ai.ResponsePart
	if active.text {
		part = ai.TextPart{Content: active.transcript}
	} else {
		transcript := active.transcript
		speech := ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant}
		if transcript != "" {
			speech.Transcript = &transcript
		}
		if len(active.audio) > 0 && (session.config.audioRetention == AudioRetentionOutput || session.config.audioRetention == AudioRetentionAll) {
			speech.Audio = &ai.BinaryContent{Data: pcmToWAV(active.audio, session.profile.AudioOutputSampleRate), MediaType: "audio/wav"}
		}
		part = speech
	}
	session.responseParts = append(session.responseParts, part)
	session.publishLocked(ai.PartEndEvent{Index: active.index, PartID: active.partID, Part: part})
	session.activeAssistant = nil
	session.outputAudio = nil
}

func (session *Session) finishUser(itemID string) {
	session.mu.Lock()
	var active *activeSpeech
	if itemID == "" {
		active = session.anonymousUser
		session.anonymousUser = nil
	} else {
		active = session.userTurns[itemID]
		delete(session.userTurns, itemID)
	}
	transcript := active.transcript
	part := ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}
	if transcript != "" {
		part.Transcript = &transcript
	}
	if len(session.inputAudio) > 0 && (session.config.audioRetention == AudioRetentionInput || session.config.audioRetention == AudioRetentionAll) {
		part.Audio = &ai.BinaryContent{Data: pcmToWAV(session.inputAudio, session.profile.AudioInputSampleRate), MediaType: "audio/wav"}
		session.inputAudio = nil
	}
	session.history = append(session.history, ai.ModelRequest{Parts: []ai.RequestPart{part}})
	session.publishLocked(ai.PartEndEvent{Index: active.index, PartID: active.partID, Part: part})
	session.mu.Unlock()
}

func (session *Session) handleUsage(event SessionUsage) {
	session.mu.Lock()
	session.usage.Add(event.Usage)
	if event.ResponseScoped {
		session.pendingUsage.Add(event.Usage)
	}
	if event.ProviderResponseID != "" {
		session.pendingResponseID = event.ProviderResponseID
	}
	if event.FinishReason != "" {
		session.pendingFinish = event.FinishReason
	}
	session.mu.Unlock()
}

func (session *Session) finishResponse(event ResponseDone) {
	if event.Interrupted && session.config.handleBargeIn {
		session.flushAudioTaps()
	}
	session.mu.Lock()
	session.finishAssistantPartLocked()
	responseID := event.ProviderResponseID
	if responseID == "" {
		responseID = session.pendingResponseID
	}
	finish := event.FinishReason
	if finish == "" {
		finish = session.pendingFinish
	}
	response := ai.ModelResponse{
		Parts: slices.Clone(session.responseParts), Usage: session.pendingUsage.Clone(),
		ModelName: session.model.Name(), ProviderName: session.model.ProviderName(),
		ProviderResponseID: responseID, FinishReason: finish,
		ProviderDetails: cloneAnyMap(event.ProviderDetails), Timestamp: time.Now().UTC(),
	}
	if info, ok := session.connection.(ConnectionInfo); ok && info.ModelName() != "" {
		response.ModelName = info.ModelName()
	}
	if event.Interrupted {
		response.State = ai.ModelResponseStateIncomplete
	} else {
		response.State = ai.ModelResponseStateComplete
	}
	if len(response.Parts) > 0 {
		session.history = append(session.history, response)
		for _, part := range response.Parts {
			call, ok := part.(ai.ToolCallPart)
			if !ok {
				continue
			}
			if result, exists := session.pendingToolResults[call.ToolCallID]; exists {
				session.history = append(session.history, result)
				delete(session.pendingToolResults, call.ToolCallID)
			}
		}
	}
	session.responseParts = nil
	session.pendingUsage = ai.Usage{}
	session.pendingResponseID = ""
	session.pendingFinish = ""
	session.responseActive = false
	session.serverCancelling = false
	session.mu.Unlock()
	session.publish(TurnCompleteEvent{Response: *cloneResponse(&response)})
	go session.deliverEnqueued()
}

func (session *Session) handleToolCall(event ToolCall) {
	session.setResponseActive(true)
	call := ai.ToolCallPart{
		ToolName: event.ToolName, ToolCallID: event.ToolCallID, Args: json.RawMessage(event.Arguments),
	}
	if len(call.Args) == 0 {
		call.Args = json.RawMessage(`{}`)
	}
	session.mu.Lock()
	session.finishAssistantPartLocked()
	index := session.nextPartIndex
	session.nextPartIndex++
	partID := fmt.Sprintf("tool-%d", index)
	session.responseParts = append(session.responseParts, call)
	session.publishLocked(ai.PartStartEvent{Index: index, PartID: partID, Part: call})
	session.publishLocked(ai.PartEndEvent{Index: index, PartID: partID, Part: call})
	session.publishLocked(ai.FunctionToolCallEvent{Part: call})
	session.mu.Unlock()
	if session.config.toolExecutor == nil {
		result := ai.ToolReturnPart{
			ToolName: call.ToolName, ToolCallID: call.ToolCallID,
			Content: "No realtime tool executor is configured.", Outcome: ai.ToolReturnOutcomeFailed,
		}
		session.completeTool(call, result, nil)
		return
	}
	toolCtx, cancel := context.WithCancel(context.WithValue(
		session.ctx, toolContextKey{}, toolContextValue{session: session, callID: call.ToolCallID},
	))
	session.toolMu.Lock()
	session.toolCancels[call.ToolCallID] = cancel
	session.toolMu.Unlock()
	session.toolWG.Add(1)
	go func() {
		defer session.toolWG.Done()
		defer func() {
			session.toolMu.Lock()
			delete(session.toolCancels, call.ToolCallID)
			session.toolMu.Unlock()
			cancel()
		}()
		value, err := session.config.toolExecutor.ExecuteTool(toolCtx, call)
		part, extra := normalizeToolResult(call, value, err)
		session.completeTool(call, part, extra)
	}()
}

func normalizeToolResult(call ai.ToolCallPart, value any, err error) (ai.RequestPart, []ai.UserContent) {
	if err != nil {
		var retry *ai.RetryError
		if errors.As(err, &retry) {
			return ai.RetryPromptPart{ToolName: call.ToolName, ToolCallID: call.ToolCallID, Content: retry.Message}, nil
		}
		var failed *ai.ToolFailedError
		if errors.As(err, &failed) {
			return ai.ToolReturnPart{
				ToolName: call.ToolName, ToolCallID: call.ToolCallID,
				Content: failed.Message, Outcome: ai.ToolReturnOutcomeFailed,
			}, nil
		}
		return ai.ToolReturnPart{
			ToolName: call.ToolName, ToolCallID: call.ToolCallID,
			Content: err.Error(), Outcome: ai.ToolReturnOutcomeFailed,
		}, nil
	}
	if rich, ok := value.(ai.ToolReturn); ok {
		return ai.ToolReturnPart{
			ToolName: call.ToolName, ToolCallID: call.ToolCallID, Content: rich.ReturnValue,
			Metadata: cloneAnyMap(rich.Metadata), Outcome: ai.ToolReturnOutcomeSuccess, Timestamp: time.Now().UTC(),
		}, slices.Clone(rich.Content)
	}
	return ai.ToolReturnPart{
		ToolName: call.ToolName, ToolCallID: call.ToolCallID, Content: value,
		Outcome: ai.ToolReturnOutcomeSuccess, Timestamp: time.Now().UTC(),
	}, nil
}

func (session *Session) completeTool(call ai.ToolCallPart, part ai.RequestPart, extra []ai.UserContent) {
	output := renderToolPart(part)
	if err := session.send(session.ctx, ToolResult{
		ToolCallID: call.ToolCallID, Output: output, Content: slices.Clone(extra),
	}); err != nil && !errors.Is(err, context.Canceled) {
		session.mu.RLock()
		closed := session.closed
		session.mu.RUnlock()
		if !closed {
			session.fail(err)
		}
		return
	}
	session.mu.Lock()
	if session.closed {
		session.mu.Unlock()
		return
	}
	requestParts := []ai.RequestPart{part}
	if len(extra) > 0 {
		requestParts = append(requestParts, ai.UserPromptPart{Contents: slices.Clone(extra)})
	}
	request := ai.ModelRequest{Parts: requestParts}
	if responseIndex := responseWithToolCall(session.history, call.ToolCallID); responseIndex >= 0 {
		insertAt := responseIndex + 1
		for insertAt < len(session.history) && isToolResultRequest(session.history[insertAt]) {
			insertAt++
		}
		session.history = append(session.history, nil)
		copy(session.history[insertAt+1:], session.history[insertAt:])
		session.history[insertAt] = request
	} else {
		session.pendingToolResults[call.ToolCallID] = request
	}
	session.mu.Unlock()
	session.publish(ai.FunctionToolResultEvent{Part: part})
}

func responseWithToolCall(messages []ai.ModelMessage, callID string) int {
	for index, message := range messages {
		response, ok := message.(ai.ModelResponse)
		if !ok {
			continue
		}
		for _, part := range response.Parts {
			call, ok := part.(ai.ToolCallPart)
			if ok && call.ToolCallID == callID {
				return index
			}
		}
	}
	return -1
}

func isToolResultRequest(message ai.ModelMessage) bool {
	request, ok := message.(ai.ModelRequest)
	if !ok {
		return false
	}
	parts := append(slices.Clone(request.Parts), nil)
	_, returned := parts[0].(ai.ToolReturnPart)
	_, retried := parts[0].(ai.RetryPromptPart)
	return returned || retried
}

func renderToolPart(part ai.RequestPart) string {
	var value any
	if returned, ok := part.(ai.ToolReturnPart); ok {
		value = returned.Content
	} else {
		value = part.(ai.RetryPromptPart).Content
	}
	if text, ok := value.(string); ok {
		return text
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(encoded)
}

func (session *Session) cancelTools(ids []string) {
	session.toolMu.Lock()
	defer session.toolMu.Unlock()
	for _, id := range ids {
		if cancel := session.toolCancels[id]; cancel != nil {
			cancel()
		}
	}
}

func (session *Session) publish(event Event) {
	session.eventsMu.RLock()
	defer session.eventsMu.RUnlock()
	select {
	case session.events <- eventResult{event: event}:
	case <-session.ctx.Done():
	}
}

func (session *Session) publishLocked(event Event) {
	session.publish(event)
}

func (session *Session) fail(err error) {
	session.mu.Lock()
	if session.err == nil {
		session.err = err
	}
	session.mu.Unlock()
	session.eventsMu.RLock()
	select {
	case session.events <- eventResult{err: err}:
	default:
	}
	session.eventsMu.RUnlock()
	session.cancel(err)
}

func (session *Session) publishAudio(partIndex int, data []byte) {
	session.tapMu.Lock()
	defer session.tapMu.Unlock()
	if session.hasInterruptedAudioPart && partIndex == session.interruptedAudioPartIndex {
		return
	}
	if !session.hasAudioPart || partIndex != session.audioPartIndex {
		session.audioPartIndex = partIndex
		session.hasAudioPart = true
		session.turnAudioStartBytes = session.emittedAudioBytes
	}
	session.emittedAudioBytes += len(data)
	for tap := range session.audioTaps {
		chunk := slices.Clone(data)
		select {
		case tap.queue <- chunk:
		default:
			select {
			case dropped := <-tap.queue:
				tap.pendingDroppedBytes += len(dropped)
			default:
			}
			tap.queue <- chunk
		}
	}
}

func (session *Session) publishTranscript(update TranscriptUpdate) {
	session.tapMu.Lock()
	defer session.tapMu.Unlock()
	for tap := range session.transcriptTaps {
		select {
		case tap <- update:
		default:
			select {
			case <-tap:
			default:
			}
			tap <- update
		}
	}
}

func accumulateTranscript(previous, text string, cumulative bool) (string, string) {
	if cumulative {
		if text == previous {
			return previous, ""
		}
		if len(text) >= len(previous) && text[:len(previous)] == previous {
			return text, text[len(previous):]
		}
		return text, ""
	}
	return previous + text, text
}

func pcmToWAV(pcm []byte, sampleRate int) []byte {
	output := make([]byte, 44+len(pcm))
	copy(output[0:4], "RIFF")
	binary.LittleEndian.PutUint32(output[4:8], uint32(36+len(pcm)))
	copy(output[8:12], "WAVE")
	copy(output[12:16], "fmt ")
	binary.LittleEndian.PutUint32(output[16:20], 16)
	binary.LittleEndian.PutUint16(output[20:22], 1)
	binary.LittleEndian.PutUint16(output[22:24], 1)
	binary.LittleEndian.PutUint32(output[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(output[28:32], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(output[32:34], 2)
	binary.LittleEndian.PutUint16(output[34:36], 16)
	copy(output[36:40], "data")
	binary.LittleEndian.PutUint32(output[40:44], uint32(len(pcm)))
	copy(output[44:], pcm)
	return output
}

func decodePCMWAV(data []byte) ([]byte, int, error) {
	if len(data) < 44 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("realtime: invalid WAV audio")
	}
	var format, channels, bits uint16
	var rate uint32
	var pcm []byte
	for offset := 12; offset+8 <= len(data); {
		kind := string(data[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		start := offset + 8
		end := start + size
		if end > len(data) {
			return nil, 0, fmt.Errorf("realtime: truncated WAV chunk")
		}
		switch kind {
		case "fmt ":
			if size < 16 {
				return nil, 0, fmt.Errorf("realtime: invalid WAV format chunk")
			}
			format = binary.LittleEndian.Uint16(data[start : start+2])
			channels = binary.LittleEndian.Uint16(data[start+2 : start+4])
			rate = binary.LittleEndian.Uint32(data[start+4 : start+8])
			bits = binary.LittleEndian.Uint16(data[start+14 : start+16])
		case "data":
			pcm = slices.Clone(data[start:end])
		}
		offset = end + size%2
	}
	if format != 1 || channels != 1 || bits != 16 || rate == 0 || pcm == nil {
		return nil, 0, fmt.Errorf("realtime: WAV must contain mono PCM16 audio")
	}
	return pcm, int(rate), nil
}

func cloneResponse(response *ai.ModelResponse) *ai.ModelResponse {
	messages := (ai.ModelRequestContext{Messages: []ai.ModelMessage{*response}}).Clone().Messages
	cloned := messages[0].(ai.ModelResponse)
	return &cloned
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
