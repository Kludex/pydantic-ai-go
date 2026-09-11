// Package xai implements xAI Grok Voice realtime sessions.
package xai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/internal/openaiprotocol"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

const defaultBaseURL = "https://api.x.ai/v1"

// Settings configures xAI-specific voice and VAD behavior.
type Settings struct {
	// Voice selects a built-in name or custom xAI voice ID.
	Voice string
	// TurnDetection replaces portable server VAD configuration.
	TurnDetection map[string]any
}

// Option configures a Model.
type Option func(*Model)

// WithAPIKey sets the bearer credential instead of reading XAI_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.apiKey = key } }

// WithBaseURL replaces the canonical xAI API base URL.
func WithBaseURL(baseURL string) Option { return func(model *Model) { model.baseURL = baseURL } }

// WithHTTPClient selects the websocket handshake client.
func WithHTTPClient(client *http.Client) Option { return func(model *Model) { model.client = client } }

// WithHeaders adds detached websocket handshake headers.
func WithHeaders(headers http.Header) Option {
	cloned := openaiprotocol.CloneHeader(headers)
	return func(model *Model) { model.headers = openaiprotocol.CloneHeader(cloned) }
}

// WithSettings adds model-level Grok Voice defaults.
func WithSettings(settings Settings) Option {
	settings.TurnDetection = cloneMap(settings.TurnDetection)
	return func(model *Model) { model.settings = settings }
}

// WithProfile applies a partial profile override.
func WithProfile(override realtime.ProfileOverride) Option {
	return func(model *Model) { model.profile = realtime.MergeProfile(model.profile, override) }
}

// Model opens xAI Grok Voice websocket sessions.
type Model struct {
	name     string
	apiKey   string
	baseURL  string
	client   *http.Client
	headers  http.Header
	settings Settings
	profile  realtime.Profile
}

// NewModel creates an xAI Grok Voice model.
func NewModel(name string, options ...Option) *Model {
	profile := realtime.DefaultProfile()
	profile.SupportsTextOutput = false
	profile.SupportsManualTurnControl = true
	profile.SupportsInterruption = true
	profile.SupportsSessionSeeding = true
	profile.SupportsThinking = name == "grok-voice-latest" || strings.HasPrefix(name, "grok-voice-think-")
	profile.EmitsInputSpeechEvents = true
	model := &Model{
		name: name, apiKey: strings.TrimSpace(os.Getenv("XAI_API_KEY")), baseURL: defaultBaseURL,
		client: http.DefaultClient, headers: http.Header{}, profile: profile,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the requested Grok Voice model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (*Model) ProviderName() string { return "xai" }

// Profile returns detached realtime capabilities.
func (model *Model) Profile() realtime.Profile {
	return realtime.MergeProfile(model.profile, realtime.ProfileOverride{})
}

// Connect opens and configures an xAI realtime websocket.
func (model *Model) Connect(ctx context.Context, params realtime.ConnectParams) (realtime.Connection, error) {
	if model.name == "" {
		return nil, fmt.Errorf("xai realtime: model name must not be empty")
	}
	if model.apiKey == "" {
		return nil, fmt.Errorf("xai realtime: XAI_API_KEY is not set")
	}
	if params.Settings.OutputModality == realtime.OutputModalityText {
		return nil, fmt.Errorf("xai realtime: Grok Voice does not support text output")
	}
	settings := model.resolveSettings(params.Settings)
	state := &dialState{}
	dial := func(ctx context.Context, history []ai.ModelMessage) (*websocket.Conn, string, error) {
		return model.dial(ctx, state, history, params.Request, params.Settings, settings)
	}
	socket, serverModel, err := dial(ctx, params.Messages)
	if err != nil {
		return nil, err
	}
	config := sessionConfig(params.Request, params.Settings, settings, model.profile)
	return openaiprotocol.New(openaiprotocol.Config{
		Provider: "xAI Grok Voice", Model: model.name, Socket: socket, ServerModel: serverModel,
		Dial: dial, Mapper: MapEvent, Reconnect: params.Settings.Reconnect,
		InputTranscriptionEnabled:  transcriptionEnabled(params.Settings),
		RestoresInFlightState:      true,
		InterruptsResponseOnSpeech: openaiprotocol.InterruptsResponseOnSpeech(config, true),
		SupportsImages:             false,
		OutputSampleRate:           model.profile.AudioOutputSampleRate,
	})
}

type dialState struct {
	mutex          sync.RWMutex
	conversationID string
}

func (state *dialState) get() string {
	state.mutex.RLock()
	defer state.mutex.RUnlock()
	return state.conversationID
}

func (state *dialState) set(value string) {
	state.mutex.Lock()
	defer state.mutex.Unlock()
	state.conversationID = value
}

func (model *Model) dial(
	ctx context.Context,
	state *dialState,
	history []ai.ModelMessage,
	request ai.ModelRequestParams,
	common realtime.Settings,
	settings Settings,
) (*websocket.Conn, string, error) {
	endpoint, err := websocketURL(model.baseURL, model.name, state.get())
	if err != nil {
		return nil, "", err
	}
	headers := openaiprotocol.CloneHeader(model.headers)
	headers.Set("Authorization", "Bearer "+model.apiKey)
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(headers))
	timeout := common.HandshakeTimeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	handshakeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	socket, response, err := openaiprotocol.DialSocket(handshakeCtx, endpoint, headers, model.client)
	if err != nil {
		if response != nil {
			return nil, "", fmt.Errorf("xai realtime: websocket handshake returned %s: %w", response.Status, err)
		}
		return nil, "", fmt.Errorf("xai realtime: websocket handshake: %w", err)
	}
	if state.get() != "" {
		history = nil
	}
	serverModel, conversationID, err := configureSocket(
		handshakeCtx, socket, history, request, common, settings, model.profile,
	)
	if err != nil {
		_ = socket.Close(websocket.StatusPolicyViolation, "handshake failed")
		return nil, "", err
	}
	if conversationID != "" {
		state.set(conversationID)
	}
	return socket, serverModel, nil
}

func configureSocket(
	ctx context.Context,
	socket *websocket.Conn,
	history []ai.ModelMessage,
	request ai.ModelRequestParams,
	common realtime.Settings,
	settings Settings,
	profile realtime.Profile,
) (string, string, error) {
	created, err := expect(ctx, socket, "session.created")
	if err != nil {
		return "", "", err
	}
	serverModel := stringValue(object(created["session"])["model"])
	if err := writeJSON(ctx, socket, map[string]any{
		"type": "session.update", "session": sessionConfig(request, common, settings, profile),
	}); err != nil {
		return "", "", err
	}
	conversationID := ""
	for {
		frame, err := readFrame(ctx, socket)
		if err != nil {
			return "", "", err
		}
		switch stringValue(frame["type"]) {
		case "conversation.created":
			conversationID = stringValue(object(frame["conversation"])["id"])
		case "session.updated":
			for _, item := range openairt.SeedItems(history, profile) {
				if err == nil {
					err = writeJSON(ctx, socket, map[string]any{"type": "conversation.item.create", "item": item})
				}
			}
			return serverModel, conversationID, err
		case "error":
			return "", "", fmt.Errorf("xai realtime: handshake error: %s", stringValue(object(frame["error"])["message"]))
		}
	}
}

func sessionConfig(
	request ai.ModelRequestParams, common realtime.Settings, settings Settings, profile realtime.Profile,
) map[string]any {
	input := map[string]any{"format": map[string]any{"type": "audio/pcm", "rate": profile.AudioInputSampleRate}}
	if common.InputTranscriptionModel == nil {
		input["transcription"] = map[string]any{"model": "grok-transcribe"}
	} else if *common.InputTranscriptionModel != "" {
		model := *common.InputTranscriptionModel
		if model == "auto" {
			model = "grok-transcribe"
		}
		input["transcription"] = map[string]any{"model": model}
	}
	config := map[string]any{
		"instructions": request.Instructions,
		"audio": map[string]any{
			"input": input,
			"output": map[string]any{"format": map[string]any{
				"type": "audio/pcm", "rate": profile.AudioOutputSampleRate,
			}},
		},
	}
	if settings.Voice != "" {
		config["voice"] = settings.Voice
	}
	switch {
	case settings.TurnDetection != nil:
		config["turn_detection"] = cloneMap(settings.TurnDetection)
	case common.TurnDetection != nil && common.TurnDetection.Enabled:
		config["turn_detection"] = portableTurnDetection(*common.TurnDetection)
	case common.TurnDetection != nil:
		config["turn_detection"] = nil
	default:
		config["turn_detection"] = map[string]any{"type": "server_vad"}
	}
	if common.MaxTokens > 0 {
		config["max_output_tokens"] = common.MaxTokens
	}
	if common.ParallelToolCalls != nil {
		config["parallel_tool_calls"] = *common.ParallelToolCalls
	}
	if common.ToolChoice != "" {
		config["tool_choice"] = common.ToolChoice
	}
	if len(request.Tools) > 0 {
		tools := make([]any, len(request.Tools))
		for index, tool := range request.Tools {
			tools[index] = map[string]any{
				"type": "function", "name": tool.Name, "description": tool.Description, "parameters": tool.Schema,
			}
		}
		config["tools"] = tools
	}
	if common.Thinking != "" && profile.SupportsThinking {
		effort := "high"
		if common.Thinking == ai.ThinkingLevelDisabled {
			effort = "none"
		}
		config["reasoning"] = map[string]any{"effort": effort}
	}
	if common.Reconnect != nil {
		config["resumption"] = map[string]any{"enabled": true}
	}
	return config
}

// MapEvent translates one xAI Grok Voice frame into provider-neutral codec events.
func MapEvent(data []byte) ([]realtime.CodecEvent, error) {
	var frame map[string]any
	if err := json.Unmarshal(data, &frame); err != nil {
		return nil, fmt.Errorf("xai realtime: decode event: %w", err)
	}
	kind := stringValue(frame["type"])
	if kind == "conversation.item.input_audio_transcription.updated" {
		return []realtime.CodecEvent{realtime.InputTranscript{
			Text: stringValue(frame["transcript"]), Cumulative: true, ItemID: stringValue(frame["item_id"]),
		}}, nil
	}
	events, err := openairt.MapEvent(data)
	if err != nil {
		return nil, err
	}
	if kind == "conversation.item.input_audio_transcription.completed" {
		for index, event := range events {
			if transcript, ok := event.(realtime.InputTranscript); ok {
				transcript.Cumulative = true
				events[index] = transcript
			}
		}
	}
	if kind == "response.done" {
		usageData := object(object(frame["response"])["usage"])
		input := object(usageData["input_token_details"])
		output := object(usageData["output_token_details"])
		for index, event := range events {
			usage, ok := event.(realtime.SessionUsage)
			if !ok {
				continue
			}
			for key, value := range map[string]int{
				"input_grok_tokens":      integer(input["grok_tokens"]),
				"output_grok_tokens":     integer(output["grok_tokens"]),
				"billable_audio_seconds": integer(usageData["billable_audio_seconds"]),
			} {
				if value != 0 {
					usage.Usage.Details[key] = value
				}
			}
			events[index] = usage
		}
	}
	return events, nil
}

func (model *Model) resolveSettings(common realtime.Settings) Settings {
	settings := model.settings
	settings.TurnDetection = cloneMap(settings.TurnDetection)
	if value, ok := common.Provider["xai_voice"].(string); ok {
		settings.Voice = value
	}
	if value, ok := common.Provider["xai_turn_detection"].(map[string]any); ok {
		settings.TurnDetection = cloneMap(value)
	}
	return settings
}

func websocketURL(baseURL, model, conversationID string) (string, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" {
		return "", fmt.Errorf("xai realtime: invalid base URL %q", baseURL)
	}
	switch parsed.Scheme {
	case "http":
		parsed.Scheme = "ws"
	case "https":
		parsed.Scheme = "wss"
	case "ws", "wss":
	default:
		return "", fmt.Errorf("xai realtime: unsupported base URL scheme %q", parsed.Scheme)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/realtime"
	query := parsed.Query()
	query.Set("model", model)
	if conversationID != "" {
		query.Set("conversation_id", conversationID)
	}
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func portableTurnDetection(settings realtime.TurnDetection) map[string]any {
	threshold := map[string]float64{"low": 0.7, "medium": 0.5, "high": 0.3}[settings.Sensitivity]
	config := map[string]any{"type": "server_vad"}
	if threshold != 0 {
		config["threshold"] = threshold
	}
	if settings.PrefixPadding > 0 {
		config["prefix_padding_ms"] = settings.PrefixPadding.Milliseconds()
	}
	if settings.SilenceDuration > 0 {
		config["silence_duration_ms"] = settings.SilenceDuration.Milliseconds()
	}
	return config
}

func expect(ctx context.Context, socket *websocket.Conn, expected string) (map[string]any, error) {
	for {
		frame, err := readFrame(ctx, socket)
		if err != nil {
			return nil, fmt.Errorf("xai realtime: wait for %s: %w", expected, err)
		}
		if stringValue(frame["type"]) == expected {
			return frame, nil
		}
	}
}

func readFrame(ctx context.Context, socket *websocket.Conn) (map[string]any, error) {
	kind, data, err := socket.Read(ctx)
	if err != nil {
		return nil, err
	}
	if kind != websocket.MessageText {
		return map[string]any{}, nil
	}
	var frame map[string]any
	if err := json.Unmarshal(data, &frame); err != nil {
		return nil, err
	}
	return frame, nil
}

func writeJSON(ctx context.Context, socket *websocket.Conn, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return socket.Write(ctx, websocket.MessageText, data)
}

func transcriptionEnabled(settings realtime.Settings) bool {
	return settings.InputTranscriptionModel == nil || *settings.InputTranscriptionModel != ""
}

func cloneMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	cloned := make(map[string]any, len(value))
	for key, item := range value {
		cloned[key] = item
	}
	return cloned
}

func object(value any) map[string]any {
	object, _ := value.(map[string]any)
	return object
}

func stringValue(value any) string {
	text, _ := value.(string)
	return text
}

func integer(value any) int {
	number, _ := value.(float64)
	return int(number)
}

var _ realtime.Model = (*Model)(nil)
