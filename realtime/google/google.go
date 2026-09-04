// Package google implements Gemini Live realtime sessions with the official Google Gen AI Go SDK.
package google

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"math/rand/v2"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/realtime"
	"google.golang.org/genai"
)

// Settings configures Gemini Live-specific generation, speech, and session behavior.
type Settings struct {
	// Temperature controls response randomness.
	Temperature *float32
	// TopP controls nucleus sampling.
	TopP *float32
	// TopK limits the candidate tokens considered at each step.
	TopK *float32
	// Seed requests best-effort deterministic sampling.
	Seed *int32
	// Voice selects a Gemini prebuilt voice.
	Voice string
	// LanguageCode is the BCP 47 speech output language.
	LanguageCode string
	// InputTranscription enables native user speech transcription.
	InputTranscription *bool
	// OutputTranscription enables model speech transcription.
	OutputTranscription *bool
	// AffectiveDialog enables emotion-aware delivery where supported.
	AffectiveDialog *bool
	// ProactiveAudio allows the model to decide whether input needs a response.
	ProactiveAudio *bool
	// AsyncToolCalls keeps supported native-audio generation active while tools run.
	AsyncToolCalls bool
	// EnableSessionResumption explicitly controls provider resume handles.
	EnableSessionResumption *bool
	// ConfigOverrides applies advanced official SDK configuration last.
	ConfigOverrides func(*genai.LiveConnectConfig)
}

// LiveSession is the official SDK session surface used by Connection.
type LiveSession interface {
	SendClientContent(input genai.LiveClientContentInput) error
	SendRealtimeInput(input genai.LiveRealtimeInput) error
	SendToolResponse(input genai.LiveToolResponseInput) error
	Receive() (*genai.LiveServerMessage, error)
	Close() error
}

// Connector opens official SDK live sessions.
type Connector interface {
	Connect(ctx context.Context, model string, config *genai.LiveConnectConfig) (LiveSession, error)
}

type sdkConnector struct{ live *genai.Live }

func (connector sdkConnector) Connect(
	ctx context.Context, model string, config *genai.LiveConnectConfig,
) (LiveSession, error) {
	return connector.live.Connect(ctx, model, config)
}

// Option configures a Model.
type Option func(*Model)

// WithClient uses an existing official Google Gen AI client.
func WithClient(client *genai.Client) Option {
	return func(model *Model) {
		model.client = client
		if client != nil {
			model.connector = sdkConnector{live: client.Live}
		}
	}
}

// WithConnector injects an official-SDK-compatible live connector.
func WithConnector(connector Connector) Option {
	return func(model *Model) { model.connector = connector }
}

// WithAPIKey sets a Gemini Developer API key instead of reading GOOGLE_API_KEY or GEMINI_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.apiKey = key } }

// WithVertex selects Vertex AI and its project and location.
func WithVertex(project, location string) Option {
	return func(model *Model) {
		model.vertex = true
		model.project = project
		model.location = location
	}
}

// WithHTTPClient selects the client used by the official SDK.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.httpClient = client }
}

// WithBaseURL sets a custom official SDK endpoint.
func WithBaseURL(baseURL string) Option { return func(model *Model) { model.baseURL = baseURL } }

// WithSettings adds model-level Gemini Live defaults.
func WithSettings(settings Settings) Option { return func(model *Model) { model.settings = settings } }

// WithProfile applies a partial profile override.
func WithProfile(override realtime.ProfileOverride) Option {
	return func(model *Model) { model.profile = realtime.MergeProfile(model.profile, override) }
}

// Model opens Gemini Live sessions.
type Model struct {
	name       string
	apiKey     string
	vertex     bool
	project    string
	location   string
	baseURL    string
	httpClient *http.Client
	client     *genai.Client
	connector  Connector
	settings   Settings
	profile    realtime.Profile
}

// NewModel creates a Gemini Live model.
func NewModel(name string, options ...Option) *Model {
	profile := realtime.DefaultProfile()
	profile.SupportsImageInput = true
	profile.SupportsTextOutput = false
	profile.SupportsSessionSeeding = true
	profile.SupportsSeedingImages = true
	profile.SupportsThinking = true
	profile.SupportsAsyncToolCalls = strings.Contains(name, "native-audio")
	profile.SupportsToolReturnSchema = true
	profile.SupportedNativeTools = map[string]bool{"web_search": true}
	profile.AudioInputSampleRate = 16000
	model := &Model{
		name: name, apiKey: firstNonEmpty(os.Getenv("GOOGLE_API_KEY"), os.Getenv("GEMINI_API_KEY")),
		httpClient: http.DefaultClient, profile: profile,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the requested Gemini Live model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (model *Model) ProviderName() string {
	if model.vertex {
		return "google-vertex"
	}
	return "google-gla"
}

// Profile returns detached realtime capabilities.
func (model *Model) Profile() realtime.Profile {
	return realtime.MergeProfile(model.profile, realtime.ProfileOverride{})
}

// Connect opens and configures a Gemini Live session.
func (model *Model) Connect(ctx context.Context, params realtime.ConnectParams) (realtime.Connection, error) {
	if model.name == "" {
		return nil, fmt.Errorf("google realtime: model name must not be empty")
	}
	connector, err := model.resolveConnector(ctx)
	if err != nil {
		return nil, err
	}
	settings := model.resolveSettings(params.Settings)
	config, err := model.liveConfig(params.Request, params.Settings, settings)
	if err != nil {
		return nil, err
	}
	session, err := connector.Connect(ctx, model.name, config)
	if err != nil {
		return nil, fmt.Errorf("google realtime: connect: %w", err)
	}
	if session == nil || (reflect.ValueOf(session).Kind() == reflect.Pointer && reflect.ValueOf(session).IsNil()) {
		return nil, fmt.Errorf("google realtime: connector returned a nil session")
	}
	connection := &Connection{
		session: session, connector: connector, config: config, reconnect: normalizedReconnect(params.Settings.Reconnect),
		provider: model.ProviderName(), model: model.name,
		inputTranscription: config.InputAudioTranscription != nil,
	}
	if len(params.Messages) > 0 {
		turns, err := seedTurns(params.Messages, model.profile)
		if err != nil {
			_ = session.Close()
			return nil, err
		}
		complete := false
		if err := session.SendClientContent(genai.LiveClientContentInput{Turns: turns, TurnComplete: &complete}); err != nil {
			_ = session.Close()
			return nil, fmt.Errorf("google realtime: seed history: %w", err)
		}
	}
	return connection, nil
}

func (model *Model) resolveConnector(ctx context.Context) (Connector, error) {
	if model.connector != nil {
		return model.connector, nil
	}
	backend := genai.BackendGeminiAPI
	config := &genai.ClientConfig{APIKey: model.apiKey, HTTPClient: model.httpClient}
	if model.vertex {
		backend = genai.BackendVertexAI
		config.Project = model.project
		config.Location = model.location
	}
	config.Backend = backend
	config.HTTPOptions.BaseURL = model.baseURL
	client, err := genai.NewClient(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("google realtime: create client: %w", err)
	}
	model.client = client
	model.connector = sdkConnector{live: client.Live}
	return model.connector, nil
}

func (model *Model) liveConfig(
	request ai.ModelRequestParams, common realtime.Settings, settings Settings,
) (*genai.LiveConnectConfig, error) {
	config := &genai.LiveConnectConfig{
		ResponseModalities: []genai.Modality{genai.ModalityAudio},
		SystemInstruction:  &genai.Content{Parts: []*genai.Part{genai.NewPartFromText(request.Instructions)}},
		Temperature:        settings.Temperature, TopP: settings.TopP, TopK: settings.TopK, Seed: settings.Seed,
	}
	if common.MaxTokens > 0 {
		config.MaxOutputTokens = int32(common.MaxTokens)
	}
	if settings.Voice != "" || settings.LanguageCode != "" {
		config.SpeechConfig = &genai.SpeechConfig{LanguageCode: settings.LanguageCode}
		if settings.Voice != "" {
			config.SpeechConfig.VoiceConfig = &genai.VoiceConfig{
				PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{VoiceName: settings.Voice},
			}
		}
	}
	inputTranscription := true
	if settings.InputTranscription != nil {
		inputTranscription = *settings.InputTranscription
	} else if common.InputTranscriptionModel != nil && *common.InputTranscriptionModel == "" {
		inputTranscription = false
	}
	if inputTranscription {
		config.InputAudioTranscription = &genai.AudioTranscriptionConfig{}
	}
	outputTranscription := true
	if settings.OutputTranscription != nil {
		outputTranscription = *settings.OutputTranscription
	}
	if outputTranscription {
		config.OutputAudioTranscription = &genai.AudioTranscriptionConfig{}
	}
	config.EnableAffectiveDialog = settings.AffectiveDialog
	if settings.ProactiveAudio != nil {
		config.Proactivity = &genai.ProactivityConfig{ProactiveAudio: settings.ProactiveAudio}
	}
	if common.Reconnect != nil {
		if settings.EnableSessionResumption != nil && !*settings.EnableSessionResumption {
			return nil, fmt.Errorf("google realtime: reconnect requires session resumption")
		}
		config.SessionResumption = &genai.SessionResumptionConfig{}
	} else if settings.EnableSessionResumption != nil && *settings.EnableSessionResumption {
		config.SessionResumption = &genai.SessionResumptionConfig{}
	}
	if common.TurnDetection != nil {
		if !common.TurnDetection.Enabled {
			return nil, fmt.Errorf("google realtime: manual turn control is unsupported")
		}
		config.RealtimeInputConfig = &genai.RealtimeInputConfig{
			AutomaticActivityDetection: googleVAD(*common.TurnDetection),
		}
	}
	if common.Thinking != "" {
		config.ThinkingConfig = thinkingConfig(common.Thinking)
	}
	if len(request.Tools) > 0 {
		declarations := make([]*genai.FunctionDeclaration, len(request.Tools))
		for index, tool := range request.Tools {
			declaration := &genai.FunctionDeclaration{
				Name: tool.Name, Description: tool.Description, ParametersJsonSchema: tool.Schema,
			}
			if tool.ReturnSchema != nil {
				declaration.ResponseJsonSchema = tool.ReturnSchema
			}
			if settings.AsyncToolCalls && model.profile.SupportsAsyncToolCalls {
				declaration.Behavior = genai.BehaviorNonBlocking
			}
			declarations[index] = declaration
		}
		config.Tools = append(config.Tools, &genai.Tool{FunctionDeclarations: declarations})
	}
	for _, tool := range request.NativeTools {
		if tool.Kind() == "web_search" {
			config.Tools = append(config.Tools, &genai.Tool{GoogleSearch: &genai.GoogleSearch{}})
		}
	}
	if settings.ConfigOverrides != nil {
		settings.ConfigOverrides(config)
	}
	return config, nil
}

func normalizedReconnect(policy *realtime.ReconnectPolicy) *realtime.ReconnectPolicy {
	if policy == nil {
		return nil
	}
	resolved := *policy
	if resolved.MaxAttempts == 0 {
		resolved.MaxAttempts = 3
	}
	if resolved.MaxReconnects == 0 {
		resolved.MaxReconnects = 50
	}
	if resolved.BaseDelay == 0 {
		resolved.BaseDelay = 500 * time.Millisecond
	}
	if resolved.MaxDelay == 0 {
		resolved.MaxDelay = 30 * time.Second
	}
	return &resolved
}

func (model *Model) resolveSettings(common realtime.Settings) Settings {
	settings := model.settings
	if value, ok := common.Provider["google_voice"].(string); ok {
		settings.Voice = value
	}
	if value, ok := common.Provider["google_language_code"].(string); ok {
		settings.LanguageCode = value
	}
	if value, ok := common.Provider["google_async_tool_calls"].(bool); ok {
		settings.AsyncToolCalls = value
	}
	return settings
}

// Connection adapts one official SDK live session.
type Connection struct {
	mutex              sync.RWMutex
	session            LiveSession
	connector          Connector
	config             *genai.LiveConnectConfig
	reconnect          *realtime.ReconnectPolicy
	provider           string
	model              string
	inputTranscription bool
	resumptionHandle   string
	reconnects         int
	closed             bool
	close              sync.Once
	closeErr           error
}

// ModelName returns the requested model because Gemini Live does not report a replacement.
func (connection *Connection) ModelName() string { return connection.model }

// InputTranscriptionEnabled reports configured native transcription.
func (connection *Connection) InputTranscriptionEnabled() bool { return connection.inputTranscription }

// ReconnectRestoresInFlightState reports Gemini's native session behavior.
func (*Connection) ReconnectRestoresInFlightState() bool { return true }

// Send writes one normalized input through the official SDK.
func (connection *Connection) Send(_ context.Context, input realtime.Input) error {
	connection.mutex.RLock()
	session := connection.session
	closed := connection.closed
	connection.mutex.RUnlock()
	if closed {
		return fmt.Errorf("google realtime: connection is closed")
	}
	switch input := input.(type) {
	case realtime.AudioInput:
		return session.SendRealtimeInput(genai.LiveRealtimeInput{
			Audio: &genai.Blob{Data: slices.Clone(input.Data), MIMEType: "audio/pcm;rate=16000"},
		})
	case realtime.ImageInput:
		return session.SendRealtimeInput(genai.LiveRealtimeInput{
			Video: &genai.Blob{Data: slices.Clone(input.Content.Data), MIMEType: input.Content.MediaType},
		})
	case realtime.TextInput:
		complete := true
		return session.SendClientContent(genai.LiveClientContentInput{
			Turns:        []*genai.Content{{Role: "user", Parts: []*genai.Part{genai.NewPartFromText(input.Text)}}},
			TurnComplete: &complete,
		})
	case realtime.ToolResult:
		response := map[string]any{"output": input.Output}
		return session.SendToolResponse(genai.LiveToolResponseInput{FunctionResponses: []*genai.FunctionResponse{{
			ID: input.ToolCallID, Response: response,
		}}})
	default:
		return fmt.Errorf("google realtime: unsupported %T input", input)
	}
}

// Events receives, maps, and reconnects official SDK sessions.
func (connection *Connection) Events(ctx context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return func(yield func(realtime.CodecEvent, error) bool) {
		closed := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				_ = connection.Close(context.Background())
			case <-closed:
			}
		}()
		defer close(closed)
		for {
			connection.mutex.RLock()
			session := connection.session
			connection.mutex.RUnlock()
			message, err := session.Receive()
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				restored, reconnectErr := connection.tryReconnect(ctx)
				if reconnectErr != nil {
					yield(nil, reconnectErr)
					return
				}
				if restored != nil {
					if !yield(realtime.SessionReconnected{StateRestored: *restored}, nil) {
						return
					}
					continue
				}
				yield(nil, fmt.Errorf("google realtime: receive: %w", err))
				return
			}
			if message == nil {
				continue
			}
			if update := message.SessionResumptionUpdate; update != nil && update.Resumable && update.NewHandle != "" {
				connection.mutex.Lock()
				connection.resumptionHandle = update.NewHandle
				connection.mutex.Unlock()
			}
			if message.GoAway != nil && connection.reconnect != nil {
				restored, reconnectErr := connection.tryReconnect(ctx)
				if reconnectErr != nil {
					yield(nil, reconnectErr)
					return
				}
				if restored != nil {
					if !yield(realtime.SessionReconnected{StateRestored: *restored}, nil) {
						return
					}
					continue
				}
			}
			for _, event := range mapServerMessage(message, connection.provider) {
				if !yield(event, nil) {
					return
				}
			}
		}
	}
}

func (connection *Connection) tryReconnect(ctx context.Context) (*bool, error) {
	policy := connection.reconnect
	if policy == nil {
		return nil, nil
	}
	connection.mutex.RLock()
	reconnects := connection.reconnects
	handle := connection.resumptionHandle
	connection.mutex.RUnlock()
	if reconnects >= policy.MaxReconnects {
		return nil, fmt.Errorf("google realtime: reconnect limit reached")
	}
	var last error
	for attempt := 0; attempt < policy.MaxAttempts; attempt++ {
		delay := policy.BaseDelay << attempt
		if delay > policy.MaxDelay {
			delay = policy.MaxDelay
		}
		if policy.Jitter && delay > 0 {
			delay = time.Duration(rand.Int64N(int64(delay) + 1))
		}
		if delay > 0 {
			timer := time.NewTimer(delay)
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				return nil, context.Cause(ctx)
			}
		}
		config := *connection.config
		config.SessionResumption = &genai.SessionResumptionConfig{Handle: handle}
		session, err := connection.connector.Connect(ctx, connection.model, &config)
		if err != nil {
			last = err
			continue
		}
		if session == nil || (reflect.ValueOf(session).Kind() == reflect.Pointer && reflect.ValueOf(session).IsNil()) {
			last = fmt.Errorf("google realtime: reconnect returned a nil session")
			continue
		}
		connection.mutex.Lock()
		old := connection.session
		connection.session = session
		connection.reconnects++
		connection.mutex.Unlock()
		_ = old.Close()
		restored := handle != ""
		return &restored, nil
	}
	return nil, fmt.Errorf("google realtime: reconnect failed: %w", last)
}

// Close closes the official SDK session once.
func (connection *Connection) Close(context.Context) error {
	connection.close.Do(func() {
		connection.mutex.Lock()
		connection.closed = true
		session := connection.session
		connection.mutex.Unlock()
		connection.closeErr = session.Close()
	})
	return connection.closeErr
}

func mapServerMessage(message *genai.LiveServerMessage, provider string) []realtime.CodecEvent {
	var events []realtime.CodecEvent
	if content := message.ServerContent; content != nil {
		if transcript := content.InterimInputTranscription; transcript != nil {
			events = append(events, realtime.InputTranscript{Text: transcript.Text, Cumulative: true})
		}
		if transcript := content.InputTranscription; transcript != nil {
			events = append(events, realtime.InputTranscript{
				Text: transcript.Text, Final: transcript.Finished, Cumulative: true,
			})
		}
		if transcript := content.OutputTranscription; transcript != nil {
			events = append(events, realtime.OutputTranscript{Text: transcript.Text, Final: transcript.Finished})
		}
		if turn := content.ModelTurn; turn != nil {
			for _, part := range turn.Parts {
				if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
					events = append(events, realtime.AudioDelta{Data: slices.Clone(part.InlineData.Data)})
				}
				if part.Text != "" && !part.Thought {
					events = append(events, realtime.OutputTranscript{Text: part.Text, OutputText: true})
				}
			}
		}
		if content.GroundingMetadata != nil {
			callID := "google-search"
			metadata := jsonCompatible(content.GroundingMetadata)
			call := ai.NativeToolCallPart{
				ToolName: "web_search", ToolCallID: callID, ToolKind: ai.ToolPartKindWebSearch,
				ProviderName: provider,
			}
			result := ai.NativeToolReturnPart{
				ToolName: "web_search", ToolCallID: callID, ToolKind: ai.ToolPartKindWebSearch,
				ProviderName: provider, Content: metadata,
			}
			events = append(events,
				ai.PartStartEvent{PartID: callID + "-call", Part: call},
				ai.PartEndEvent{PartID: callID + "-call", Part: call},
				ai.PartStartEvent{PartID: callID + "-result", Part: result},
				ai.PartEndEvent{PartID: callID + "-result", Part: result},
			)
		}
		if content.Interrupted || content.TurnComplete {
			events = append(events, realtime.ResponseDone{
				Interrupted:     content.Interrupted,
				ProviderDetails: map[string]any{"turn_complete_reason": content.TurnCompleteReason},
			})
		}
	}
	if calls := message.ToolCall; calls != nil {
		for _, call := range calls.FunctionCalls {
			arguments, _ := json.Marshal(call.Args)
			events = append(events, realtime.ToolCall{
				ToolCallID: call.ID, ToolName: call.Name, Arguments: string(arguments),
			})
		}
	}
	if cancelled := message.ToolCallCancellation; cancelled != nil {
		events = append(events, realtime.ToolCallCancelled{ToolCallIDs: slices.Clone(cancelled.IDs)})
	}
	if usage := message.UsageMetadata; usage != nil {
		events = append(events, realtime.SessionUsage{Usage: mapUsage(usage), ResponseScoped: true})
	}
	if message.GoAway != nil {
		events = append(events, realtime.SessionError{Err: fmt.Errorf("google realtime: server is going away")})
	}
	return events
}

func mapUsage(usage *genai.UsageMetadata) ai.Usage {
	result := ai.Usage{
		Requests: 1, InputTokens: int(usage.PromptTokenCount), OutputTokens: int(usage.ResponseTokenCount),
		CacheReadTokens: int(usage.CachedContentTokenCount), ReasoningTokens: int(usage.ThoughtsTokenCount),
		Details: map[string]int{},
	}
	for _, detail := range usage.PromptTokensDetails {
		result.Details["input_"+strings.ToLower(string(detail.Modality))+"_tokens"] += int(detail.TokenCount)
	}
	for _, detail := range usage.ResponseTokensDetails {
		result.Details["output_"+strings.ToLower(string(detail.Modality))+"_tokens"] += int(detail.TokenCount)
	}
	return result
}

func seedTurns(messages []ai.ModelMessage, profile realtime.Profile) ([]*genai.Content, error) {
	var turns []*genai.Content
	for _, message := range messages {
		content := &genai.Content{}
		switch message := message.(type) {
		case ai.ModelRequest:
			content.Role = "user"
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.UserPromptPart:
					if part.Content != "" {
						content.Parts = append(content.Parts, genai.NewPartFromText(part.Content))
					}
					for _, value := range part.Contents {
						switch value := value.(type) {
						case ai.TextContent:
							content.Parts = append(content.Parts, genai.NewPartFromText(value.Text))
						case ai.BinaryContent:
							if !profile.SupportsSeedingImages || !strings.HasPrefix(value.MediaType, "image/") {
								return nil, fmt.Errorf("google realtime: unsupported seeded content %q", value.MediaType)
							}
							content.Parts = append(content.Parts, genai.NewPartFromBytes(value.Data, value.MediaType))
						}
					}
				case ai.SpeechPart:
					if part.Transcript == nil {
						return nil, fmt.Errorf("google realtime: seeded speech requires a transcript")
					}
					content.Parts = append(content.Parts, genai.NewPartFromText(*part.Transcript))
				case ai.ToolReturnPart:
					content.Parts = append(content.Parts, genai.NewPartFromText("[Tool returned: "+render(part.Content)+"]"))
				case ai.RetryPromptPart:
					content.Parts = append(content.Parts, genai.NewPartFromText("[Tool error: "+render(part.Content)+"]"))
				}
			}
		case ai.ModelResponse:
			content.Role = "model"
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.TextPart:
					content.Parts = append(content.Parts, genai.NewPartFromText(part.Content))
				case ai.SpeechPart:
					if part.Transcript == nil {
						return nil, fmt.Errorf("google realtime: seeded speech requires a transcript")
					}
					content.Parts = append(content.Parts, genai.NewPartFromText(*part.Transcript))
				case ai.ThinkingPart:
					content.Parts = append(content.Parts, genai.NewPartFromText("<think>\n"+part.Content+"\n</think>"))
				case ai.ToolCallPart:
					content.Parts = append(content.Parts, genai.NewPartFromText("[Tool call: "+part.ToolName+"("+string(part.Args)+")]"))
				case ai.FilePart:
					return nil, fmt.Errorf("google realtime: generated files cannot be seeded")
				}
			}
		}
		if len(content.Parts) > 0 {
			turns = append(turns, content)
		}
	}
	return turns, nil
}

func googleVAD(settings realtime.TurnDetection) *genai.AutomaticActivityDetection {
	vad := &genai.AutomaticActivityDetection{}
	if settings.PrefixPadding > 0 {
		vad.PrefixPaddingMs = ptr(int32(settings.PrefixPadding.Milliseconds()))
	}
	if settings.SilenceDuration > 0 {
		vad.SilenceDurationMs = ptr(int32(settings.SilenceDuration.Milliseconds()))
	}
	switch settings.Sensitivity {
	case "high":
		vad.StartOfSpeechSensitivity = genai.StartSensitivityHigh
		vad.EndOfSpeechSensitivity = genai.EndSensitivityHigh
	case "low":
		vad.StartOfSpeechSensitivity = genai.StartSensitivityLow
		vad.EndOfSpeechSensitivity = genai.EndSensitivityLow
	}
	return vad
}

func thinkingConfig(level ai.ThinkingLevel) *genai.ThinkingConfig {
	if level == ai.ThinkingLevelDisabled {
		return &genai.ThinkingConfig{ThinkingBudget: ptr(int32(0))}
	}
	return &genai.ThinkingConfig{IncludeThoughts: true}
}

func jsonCompatible(value any) any {
	data, _ := json.Marshal(value)
	var result any
	_ = json.Unmarshal(data, &result)
	return result
}

func render(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(data)
}

func ptr[T any](value T) *T { return &value }

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

var _ realtime.Model = (*Model)(nil)
