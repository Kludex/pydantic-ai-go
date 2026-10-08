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

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/contextwindow"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
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

// WithAPIVersion selects the Google API version. Gemini Developer API
// proactive audio requires v1alpha. The default is v1beta.
func WithAPIVersion(version string) Option { return func(model *Model) { model.apiVersion = version } }

// WithSettings adds model-level Gemini Live defaults.
func WithSettings(settings Settings) Option { return func(model *Model) { model.settings = settings } }

// WithProfile applies a partial profile override.
func WithProfile(override realtime.ProfileOverride) Option {
	return func(model *Model) {
		model.profile = realtime.MergeProfile(model.profile, override)
		model.textOutputOverridden = override.SupportsTextOutput != nil
	}
}

// Model opens Gemini Live sessions.
type Model struct {
	name                 string
	apiKey               string
	vertex               bool
	project              string
	location             string
	baseURL              string
	apiVersion           string
	httpClient           *http.Client
	client               *genai.Client
	connector            Connector
	settings             Settings
	profile              realtime.Profile
	textOutputOverridden bool
}

// NewModel creates a Gemini Live model.
func NewModel(name string, options ...Option) *Model {
	profile := realtime.DefaultProfile()
	profile.SupportsImageInput = true
	profile.SupportsTextOutput = false
	profile.SupportsSessionSeeding = true
	profile.SupportsSeedingImages = true
	normalizedName := normalizedModelName(name)
	extendedThinking := strings.HasPrefix(normalizedName, "gemini-3.8-live-extended-thinking")
	gemini38 := strings.HasPrefix(normalizedName, "gemini-3.8-live")
	profile.SupportsSeedingAudio = (strings.HasPrefix(normalizedName, "gemini-3.1-flash-live") || gemini38) && !extendedThinking
	textOutputCapable := strings.HasPrefix(normalizedName, "gemini-live-2.5-flash")
	profile.SupportsThinking = extendedThinking || strings.Contains(normalizedName, "native-audio") ||
		!strings.HasPrefix(normalizedName, "gemini-live-2.5") && !gemini38
	profile.SupportsAsyncToolCalls = strings.Contains(normalizedName, "native-audio") || gemini38
	if extendedThinking {
		profile.AsyncToolCallMode = realtime.AsyncToolCallsAlways
	} else if profile.SupportsAsyncToolCalls {
		profile.AsyncToolCallMode = realtime.AsyncToolCallsOptional
	}
	profile.SupportsToolReturnSchema = true
	profile.SupportedNativeTools = map[string]bool{"web_search": true}
	profile.AudioInputSampleRate = 16000
	profile.ContextWindow = contextwindow.Lookup(name, "google", "")
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	model := &Model{
		name: name, apiKey: apiKey,
		httpClient: http.DefaultClient, profile: profile,
	}
	for _, option := range options {
		option(model)
	}
	if !model.textOutputOverridden {
		model.profile.SupportsTextOutput = model.vertex && textOutputCapable
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
	settings := model.resolveSettings(params.Settings)
	apiVersion := model.apiVersion
	if apiVersion == "" {
		apiVersion = "v1beta"
	}
	if settings.AffectiveDialog != nil && *settings.AffectiveDialog && model.rejectsAffectiveDialog() {
		return nil, fmt.Errorf("google realtime: affective dialog is not supported by model %q", model.name)
	}
	if settings.ProactiveAudio != nil && *settings.ProactiveAudio && !model.vertex && apiVersion != "v1alpha" {
		return nil, fmt.Errorf(
			"google realtime: proactive audio requires Gemini Developer API v1alpha; use WithAPIVersion(\"v1alpha\")",
		)
	}
	connector, err := model.resolveConnector(ctx)
	if err != nil {
		return nil, err
	}
	config, err := model.liveConfig(params.Request, params.Settings, settings)
	if err != nil {
		return nil, err
	}
	turns, err := seedTurns(params.Messages, model.profile)
	if err != nil {
		return nil, err
	}
	timeout := params.Settings.HandshakeTimeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	handshakeCtx, cancel := context.WithTimeout(ctx, timeout)
	session, err := connector.Connect(handshakeCtx, model.name, config)
	cancel()
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
		asyncToolCalls:     model.requiresAsyncToolCalls() || settings.AsyncToolCalls && model.profile.SupportsAsyncToolCalls,
		supportsScheduling: !model.requiresAsyncToolCalls(),
		handshakeTimeout:   timeout, calls: map[string]*genai.FunctionCall{},
		closesToolTurn: model.vertex && strings.HasPrefix(normalizedModelName(model.name), "gemini-live-2.5-flash"),
		missingVideo: strings.HasPrefix(normalizedModelName(model.name), "gemini-2.5-flash-native-audio") ||
			strings.HasPrefix(normalizedModelName(model.name), "gemini-3.") ||
			strings.HasPrefix(normalizedModelName(model.name), "gemini-live-2.5-flash"),
	}
	if len(turns) > 0 {
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
	if model.apiVersion != "" {
		config.HTTPOptions.APIVersion = model.apiVersion
	}
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
	modality := genai.ModalityAudio
	if common.OutputModality == realtime.OutputModalityText {
		modality = genai.ModalityText
	}
	config := &genai.LiveConnectConfig{
		ResponseModalities: []genai.Modality{modality},
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
	if model.requiresThinking() {
		config.ThinkingConfig = requiredThinkingConfig(common.Thinking)
	} else if common.Thinking != "" {
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
			if model.requiresAsyncToolCalls() || settings.AsyncToolCalls && model.profile.SupportsAsyncToolCalls {
				declaration.Behavior = genai.BehaviorNonBlocking
			} else if model.asyncToolCallsByDefault() {
				declaration.Behavior = genai.BehaviorBlocking
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

func normalizedModelName(name string) string {
	parts := strings.Split(strings.Trim(name, "/"), "/")
	return parts[len(parts)-1]
}

func (model *Model) requiresThinking() bool {
	return strings.HasPrefix(normalizedModelName(model.name), "gemini-3.8-live-extended-thinking")
}

func (model *Model) rejectsAffectiveDialog() bool {
	name := normalizedModelName(model.name)
	return strings.HasPrefix(name, "gemini-3.1-flash-live") || strings.HasPrefix(name, "gemini-3.8-live")
}

func (model *Model) requiresAsyncToolCalls() bool { return model.requiresThinking() }

func (model *Model) asyncToolCallsByDefault() bool {
	return strings.HasPrefix(normalizedModelName(model.name), "gemini-3.8-live")
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
	if common.AsyncToolCalls != nil {
		settings.AsyncToolCalls = *common.AsyncToolCalls
	}
	if value, ok := common.Provider["google_affective_dialog"].(bool); ok {
		settings.AffectiveDialog = &value
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
	asyncToolCalls     bool
	supportsScheduling bool
	resumptionHandle   string
	reconnects         int
	closed             bool
	close              sync.Once
	closeErr           error
	handshakeTimeout   time.Duration
	reconnecting       bool
	gaveUp             bool
	sendMu             sync.Mutex
	calls              map[string]*genai.FunctionCall
	lostCalls          []*genai.FunctionResponse
	lostAnswered       bool
	typedTurns         []typedTurn
	inputsReceived     int
	withholdsHandles   bool
	turnOpen           bool
	closesToolTurn     bool
	toolTurn           bool
	missingVideo       bool
	recentImage        *genai.Part
	recentImageAt      time.Time
}

type typedTurn struct {
	index    int
	answered bool
}

// ModelName returns the requested model because Gemini Live does not report a replacement.
func (connection *Connection) ModelName() string { return connection.model }

// InputTranscriptionEnabled reports configured native transcription.
func (connection *Connection) InputTranscriptionEnabled() bool { return connection.inputTranscription }

// IsReconnecting reports that the current transport is being replaced.
func (connection *Connection) IsReconnecting() bool {
	connection.mutex.RLock()
	defer connection.mutex.RUnlock()
	return connection.reconnecting
}

// CanReconnect reports whether another transport recovery is available.
func (connection *Connection) CanReconnect() bool {
	connection.mutex.RLock()
	defer connection.mutex.RUnlock()
	return !connection.gaveUp && connection.reconnect != nil && connection.reconnects < connection.reconnect.MaxReconnects
}

// ReconnectRestoresInFlightState reports Gemini's native session behavior.
func (*Connection) ReconnectRestoresInFlightState() bool { return true }

// Send writes one normalized input through the official SDK.
func (connection *Connection) Send(ctx context.Context, input realtime.Input) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	connection.sendMu.Lock()
	defer connection.sendMu.Unlock()
	connection.mutex.Lock()
	inputIndex := connection.inputsReceived
	connection.inputsReceived++
	connection.mutex.Unlock()
	if err := connection.answerLostCalls(); err != nil {
		return err
	}
	connection.mutex.RLock()
	session := connection.session
	closed := connection.closed
	reconnecting := connection.reconnecting
	connection.mutex.RUnlock()
	if reconnecting {
		if _, audio := input.(realtime.AudioInput); audio {
			return nil
		}
		return fmt.Errorf("google realtime: connection is reconnecting")
	}
	if closed {
		return fmt.Errorf("google realtime: connection is closed")
	}
	switch input := input.(type) {
	case realtime.AudioInput:
		return session.SendRealtimeInput(genai.LiveRealtimeInput{
			Audio: &genai.Blob{Data: slices.Clone(input.Data), MIMEType: "audio/pcm;rate=16000"},
		})
	case realtime.ImageInput:
		if err := session.SendRealtimeInput(genai.LiveRealtimeInput{
			Video: &genai.Blob{Data: slices.Clone(input.Content.Data), MIMEType: input.Content.MediaType},
		}); err != nil {
			return err
		}
		if connection.missingVideo {
			connection.recentImage = genai.NewPartFromBytes(slices.Clone(input.Content.Data), input.Content.MediaType)
			connection.recentImageAt = time.Now()
		}
		return nil
	case realtime.TextInput:
		complete := true
		parts := []*genai.Part{genai.NewPartFromText(input.Text)}
		if connection.recentImage != nil && time.Since(connection.recentImageAt) <= 10*time.Second {
			parts = append([]*genai.Part{connection.recentImage}, parts...)
		}
		connection.mutex.Lock()
		if connection.reconnect != nil {
			connection.typedTurns = append(connection.typedTurns, typedTurn{index: inputIndex})
		}
		connection.mutex.Unlock()
		err := session.SendClientContent(genai.LiveClientContentInput{
			Turns: []*genai.Content{{Role: "user", Parts: parts}}, TurnComplete: &complete,
		})
		if err == nil {
			connection.recentImage = nil
		} else {
			connection.mutex.Lock()
			connection.typedTurns = slices.DeleteFunc(connection.typedTurns, func(turn typedTurn) bool { return turn.index == inputIndex })
			connection.mutex.Unlock()
		}
		return err
	case realtime.TextContext:
		complete := false
		return session.SendClientContent(genai.LiveClientContentInput{
			Turns:        []*genai.Content{{Role: "user", Parts: []*genai.Part{genai.NewPartFromText(input.Text)}}},
			TurnComplete: &complete,
		})
	case realtime.ToolResult:
		output := input.Output
		var media []*genai.FunctionResponsePart
		for _, content := range input.Content {
			switch content := content.(type) {
			case ai.TextContent:
				output += "\n\n" + content.Text
			case ai.BinaryContent:
				modern := strings.HasPrefix(normalizedModelName(connection.model), "gemini-3.1-flash-live") ||
					strings.HasPrefix(normalizedModelName(connection.model), "gemini-3.8-live")
				if !modern || !slices.Contains([]string{"image/png", "image/jpeg", "image/webp", "text/plain"}, content.MediaType) {
					return fmt.Errorf("google realtime: unsupported tool result media %q", content.MediaType)
				}
				media = append(media, genai.NewFunctionResponsePartFromBytes(slices.Clone(content.Data), content.MediaType))
			case ai.CachePoint:
			default:
				return fmt.Errorf("google realtime: unsupported tool result content %T", content)
			}
		}
		connection.mutex.RLock()
		call := connection.calls[input.ToolCallID]
		connection.mutex.RUnlock()
		functionResponse := &genai.FunctionResponse{ID: input.ToolCallID, Response: map[string]any{"output": output}, Parts: media}
		if call != nil {
			functionResponse.Name = call.Name
			functionResponse.ID = call.ID
		}
		if connection.asyncToolCalls && connection.supportsScheduling {
			functionResponse.Scheduling = genai.FunctionResponseSchedulingInterrupt
		}
		if err := session.SendToolResponse(genai.LiveToolResponseInput{FunctionResponses: []*genai.FunctionResponse{
			functionResponse,
		}}); err != nil {
			return err
		}
		connection.mutex.Lock()
		delete(connection.calls, input.ToolCallID)
		connection.mutex.Unlock()
		return nil
	default:
		return fmt.Errorf("google realtime: unsupported %T input", input)
	}
}

// Events receives, maps, and reconnects official SDK sessions.
func (connection *Connection) answerLostCalls() error {
	connection.mutex.RLock()
	calls := slices.Clone(connection.lostCalls)
	answered := connection.lostAnswered
	session := connection.session
	connection.mutex.RUnlock()
	if len(calls) == 0 || answered {
		return nil
	}
	if err := session.SendToolResponse(genai.LiveToolResponseInput{FunctionResponses: calls}); err != nil {
		return err
	}
	connection.mutex.Lock()
	connection.lostAnswered = true
	connection.mutex.Unlock()
	return nil
}

// AnswersToolCallsPerResponse reports that blocking tool results share one reply.
func (connection *Connection) AnswersToolCallsPerResponse() bool { return !connection.asyncToolCalls }

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
					for _, event := range connection.reconnectEvents(*restored) {
						if !yield(event, nil) {
							return
						}
					}
					continue
				}
				yield(nil, fmt.Errorf("google realtime: receive: %w", err))
				return
			}
			if message == nil {
				continue
			}
			if update := message.SessionResumptionUpdate; update != nil {
				connection.mutex.Lock()
				if update.Resumable && update.NewHandle != "" {
					connection.resumptionHandle = update.NewHandle
					connection.typedTurns = slices.DeleteFunc(connection.typedTurns, func(turn typedTurn) bool { return turn.answered })
					if connection.lostAnswered {
						connection.lostCalls = nil
					}
				} else if slices.ContainsFunc(connection.typedTurns, func(turn typedTurn) bool { return !turn.answered }) {
					connection.withholdsHandles = true
				}
				connection.mutex.Unlock()
			}
			if message.GoAway != nil && connection.reconnect != nil {
				restored, reconnectErr := connection.tryReconnect(ctx)
				if reconnectErr != nil {
					yield(nil, reconnectErr)
					return
				}
				if restored != nil {
					for _, event := range connection.reconnectEvents(*restored) {
						if !yield(event, nil) {
							return
						}
					}
					continue
				}
			}
			for _, event := range mapServerMessage(message, connection.provider, connection.inputTranscription) {
				connection.mutex.Lock()
				skip := false
				switch value := event.(type) {
				case realtime.ToolCall:
					connection.calls[value.ToolCallID] = &genai.FunctionCall{ID: value.ToolCallID, Name: value.ToolName}
					connection.toolTurn = true
					connection.turnOpen = true
				case realtime.ToolCallCancelled:
					for _, id := range value.ToolCallIDs {
						delete(connection.calls, id)
					}
					connection.toolTurn = false
				case realtime.AudioDelta, realtime.OutputTranscript, realtime.PartStarted:
					connection.toolTurn = false
					connection.turnOpen = true
				case realtime.ResponseDone:
					if value.FinishReason != ai.FinishReasonToolCall {
						skip = connection.closesToolTurn && connection.toolTurn && !value.Interrupted && !value.MoreExpected
						connection.toolTurn = false
						if !skip {
							connection.turnOpen = value.MoreExpected
							if !value.MoreExpected {
								for index := range connection.typedTurns {
									if !connection.typedTurns[index].answered {
										connection.typedTurns[index].answered = true
										break
									}
								}
							}
						}
					}
				}
				connection.mutex.Unlock()
				if skip {
					continue
				}
				if !yield(event, nil) {
					return
				}
			}
		}
	}
}

func (connection *Connection) reconnectEvents(resumed bool) []realtime.CodecEvent {
	connection.sendMu.Lock()
	defer connection.sendMu.Unlock()
	connection.mutex.Lock()
	var events []realtime.CodecEvent
	var ids []string
	for id, call := range connection.calls {
		ids = append(ids, id)
		if resumed {
			connection.lostCalls = append(connection.lostCalls, &genai.FunctionResponse{
				ID: call.ID, Name: call.Name, Response: map[string]any{"error": "Tool call interrupted by reconnect."},
			})
		}
	}
	clear(connection.calls)
	if len(ids) > 0 {
		events = append(events, realtime.ToolCallCancelled{ToolCallIDs: ids})
	}
	if connection.turnOpen {
		events = append(events, realtime.ResponseDone{Interrupted: true})
	}
	lostTurn := false
	for _, turn := range connection.typedTurns {
		if !resumed || connection.withholdsHandles {
			lostTurn = true
			if !turn.answered && !connection.turnOpen {
				events = append(events, realtime.InputRejected{InputIndex: turn.index, Response: true})
			}
		}
	}
	connection.typedTurns = nil
	connection.turnOpen = false
	connection.toolTurn = false
	connection.lostAnswered = false
	if !resumed {
		connection.lostCalls = nil
	}
	connection.mutex.Unlock()
	_ = connection.answerLostCalls()
	return append(events, realtime.SessionReconnected{StateRestored: resumed && len(ids) == 0 && !lostTurn})
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
	connection.mutex.Lock()
	connection.reconnecting = true
	connection.mutex.Unlock()
	defer func() {
		connection.mutex.Lock()
		connection.reconnecting = false
		connection.mutex.Unlock()
	}()
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
		handshakeCtx, cancel := context.WithTimeout(ctx, connection.handshakeTimeout)
		session, err := connection.connector.Connect(handshakeCtx, connection.model, &config)
		cancel()
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
	connection.mutex.Lock()
	connection.gaveUp = true
	connection.mutex.Unlock()
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

func mapServerMessage(
	message *genai.LiveServerMessage, provider string, inputTranscriptionEnabled bool,
) []realtime.CodecEvent {
	var events []realtime.CodecEvent
	var terminal *realtime.ResponseDone
	if content := message.ServerContent; content != nil {
		if transcript := content.InterimInputTranscription; inputTranscriptionEnabled && transcript != nil {
			events = append(events, realtime.InputTranscript{Text: transcript.Text, Cumulative: true})
		}
		if transcript := content.InputTranscription; inputTranscriptionEnabled && transcript != nil {
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
				realtime.PartStarted{Event: ai.PartStartEvent{PartID: callID + "-call", Part: call}},
				realtime.PartEnded{Event: ai.PartEndEvent{PartID: callID + "-call", Part: call}},
				realtime.PartStarted{Event: ai.PartStartEvent{PartID: callID + "-result", Part: result}},
				realtime.PartEnded{Event: ai.PartEndEvent{PartID: callID + "-result", Part: result}},
			)
		}
		if content.Interrupted || content.TurnComplete {
			reason := string(content.TurnCompleteReason)
			finish := ai.FinishReason("")
			switch reason {
			case "MALFORMED_FUNCTION_CALL":
				finish = ai.FinishReasonError
			case "STOP":
				finish = ai.FinishReasonStop
			case "MAX_TOKENS":
				finish = ai.FinishReasonLength
			default:
				if strings.Contains(reason, "PROHIBITED") || strings.Contains(reason, "SAFETY") ||
					strings.Contains(reason, "BLOCKLIST") || strings.Contains(reason, "CELEBRITY") ||
					strings.Contains(reason, "MINORS") || strings.Contains(reason, "IDENTIFIABLE_PEOPLE") {
					finish = ai.FinishReasonContentFilter
				}
			}
			terminal = &realtime.ResponseDone{
				Interrupted:  content.Interrupted,
				MoreExpected: content.InteractionStatus == genai.InteractionStatusInProgress && !content.Interrupted,
				FinishReason: finish, ProviderDetails: map[string]any{"turn_complete_reason": content.TurnCompleteReason, "finish_reason": reason},
			}
		}
	}
	if calls := message.ToolCall; calls != nil {
		for _, call := range calls.FunctionCalls {
			arguments, _ := json.Marshal(call.Args)
			events = append(events, realtime.ToolCall{
				ToolCallID: call.ID, ToolName: call.Name, Arguments: string(arguments), ResponseUsageFollows: true,
			})
		}
	}
	if cancelled := message.ToolCallCancellation; cancelled != nil {
		events = append(events, realtime.ToolCallCancelled{ToolCallIDs: slices.Clone(cancelled.IDs)})
	}
	if usage := message.UsageMetadata; usage != nil {
		events = append(events, realtime.SessionUsage{Usage: mapUsage(usage), ResponseScoped: true})
	}
	if message.ToolCall != nil && len(message.ToolCall.FunctionCalls) > 0 {
		events = append(events, realtime.ResponseDone{MoreExpected: true, FinishReason: ai.FinishReasonToolCall})
	}
	if terminal != nil {
		events = append(events, *terminal)
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
					if part.Transcript != nil {
						content.Parts = append(content.Parts, genai.NewPartFromText(*part.Transcript))
					} else if part.Audio != nil {
						if !profile.SupportsSeedingAudio {
							return nil, fmt.Errorf("google realtime: seeded speech requires a transcript")
						}
						pcm, err := realtime.AudioPCM(*part.Audio, profile.AudioInputSampleRate)
						if err != nil {
							return nil, err
						}
						content.Parts = append(content.Parts, genai.NewPartFromBytes(pcm, "audio/pcm;rate=16000"))
					}
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

func requiredThinkingConfig(level ai.ThinkingLevel) *genai.ThinkingConfig {
	resolved := map[ai.ThinkingLevel]genai.ThinkingLevel{
		ai.ThinkingLevelEnabled: genai.ThinkingLevelMedium, ai.ThinkingLevelMedium: genai.ThinkingLevelMedium,
		ai.ThinkingLevelHigh: genai.ThinkingLevelHigh, ai.ThinkingLevelXHigh: genai.ThinkingLevelHigh,
	}[level]
	if resolved == "" {
		resolved = genai.ThinkingLevelLow
	}
	return &genai.ThinkingConfig{ThinkingLevel: resolved}
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

var _ realtime.Model = (*Model)(nil)
