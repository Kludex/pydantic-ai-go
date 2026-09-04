// Package openai implements OpenAI Realtime websocket sessions.
package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/realtime"
	"github.com/Kludex/pydantic-ai-go/realtime/internal/openaiprotocol"
	"github.com/coder/websocket"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Settings configures OpenAI-specific realtime behavior.
type Settings struct {
	// Voice selects a built-in name or provider custom voice ID.
	Voice string
	// InputNoiseReduction selects near-field or far-field processing.
	InputNoiseReduction string
	// OutputSpeed is the playback speed multiplier from 0.25 through 1.5.
	OutputSpeed float64
	// TurnDetection replaces portable VAD configuration.
	TurnDetection map[string]any
	// Truncation configures provider conversation-window retention.
	Truncation any
}

// Option configures a Model.
type Option func(*Model)

// WithAPIKey sets the bearer credential instead of reading OPENAI_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.apiKey = key } }

// WithBaseURL replaces the default OpenAI API base URL.
func WithBaseURL(baseURL string) Option { return func(model *Model) { model.baseURL = baseURL } }

// WithHTTPClient selects the client used for websocket handshakes and WebRTC HTTP calls.
func WithHTTPClient(client *http.Client) Option { return func(model *Model) { model.client = client } }

// WithHeaders adds detached handshake headers.
func WithHeaders(headers http.Header) Option {
	cloned := openaiprotocol.CloneHeader(headers)
	return func(model *Model) { model.headers = openaiprotocol.CloneHeader(cloned) }
}

// WithSettings adds model-level OpenAI realtime defaults.
func WithSettings(settings Settings) Option {
	settings.TurnDetection = cloneMap(settings.TurnDetection)
	return func(model *Model) { model.settings = settings }
}

// WithProfile applies a partial profile override.
func WithProfile(override realtime.ProfileOverride) Option {
	return func(model *Model) { model.profile = realtime.MergeProfile(model.profile, override) }
}

// Model opens OpenAI Realtime sessions.
type Model struct {
	name     string
	apiKey   string
	baseURL  string
	client   *http.Client
	headers  http.Header
	settings Settings
	profile  realtime.Profile
}

// NewModel creates an OpenAI Realtime model.
func NewModel(name string, options ...Option) *Model {
	profile := realtime.DefaultProfile()
	profile.SupportsImageInput = true
	profile.SupportsManualTurnControl = true
	profile.SupportsInterruption = true
	profile.SupportsOutputTruncation = true
	profile.SupportsSessionSeeding = true
	profile.SupportsWebRTC = true
	profile.SupportsSeedingImages = true
	profile.SupportsSeedingAudio = true
	profile.SupportsAsyncToolCalls = true
	profile.EmitsInputSpeechEvents = true
	profile.SupportsThinking = strings.HasPrefix(name, "gpt-realtime-2")
	model := &Model{
		name: name, apiKey: getenv("OPENAI_API_KEY"), baseURL: defaultBaseURL,
		client: http.DefaultClient, headers: http.Header{}, profile: profile,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the requested OpenAI model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (*Model) ProviderName() string { return "openai" }

// Profile returns detached realtime capabilities.
func (model *Model) Profile() realtime.Profile {
	return realtime.MergeProfile(model.profile, realtime.ProfileOverride{})
}

// Connect opens and configures an OpenAI Realtime websocket.
func (model *Model) Connect(ctx context.Context, params realtime.ConnectParams) (realtime.Connection, error) {
	if model.name == "" {
		return nil, fmt.Errorf("openai realtime: model name must not be empty")
	}
	if model.apiKey == "" {
		return nil, fmt.Errorf("openai realtime: OPENAI_API_KEY is not set")
	}
	settings := model.resolveSettings(params.Settings)
	dial := func(ctx context.Context, history []ai.ModelMessage) (*websocket.Conn, string, error) {
		return model.dial(ctx, history, params.Request, params.Settings, settings)
	}
	socket, serverModel, err := dial(ctx, params.Messages)
	if err != nil {
		return nil, err
	}
	return openaiprotocol.New(openaiprotocol.Config{
		Provider: "OpenAI Realtime", Model: model.name, Socket: socket, ServerModel: serverModel,
		Dial: dial, Mapper: MapEvent, Reconnect: params.Settings.Reconnect,
		InputTranscriptionEnabled: transcriptionEnabled(params.Settings),
		RestoresInFlightState:     false, SupportsImages: true,
		OutputSampleRate: model.profile.AudioOutputSampleRate,
	})
}

// CreateClientSecret mints an ephemeral browser credential.
func (model *Model) CreateClientSecret(
	ctx context.Context,
	instructions string,
	tools []ai.ToolDefinition,
	common realtime.Settings,
	expiresAfter time.Duration,
) (realtime.ClientSecret, error) {
	if model.apiKey == "" {
		return realtime.ClientSecret{}, fmt.Errorf("openai realtime: OPENAI_API_KEY is not set")
	}
	if common.OutputModality == "" {
		common.OutputModality = realtime.OutputModalityAudio
	}
	config, err := sessionConfig(
		ai.ModelRequestParams{Instructions: instructions, Tools: tools}, common,
		model.resolveSettings(common), model.profile,
	)
	if err != nil {
		return realtime.ClientSecret{}, err
	}
	payload := map[string]any{"session": config}
	if expiresAfter > 0 {
		payload["expires_after"] = map[string]any{
			"anchor": "created_at", "seconds": int(expiresAfter.Seconds()),
		}
	}
	var response struct {
		Value     string         `json:"value"`
		ExpiresAt int64          `json:"expires_at"`
		Session   map[string]any `json:"session"`
	}
	if _, err := model.postJSON(ctx, "realtime/client_secrets", payload, &response); err != nil {
		return realtime.ClientSecret{}, err
	}
	if response.Value == "" || response.ExpiresAt == 0 {
		return realtime.ClientSecret{}, fmt.Errorf("openai realtime: client-secret response is incomplete")
	}
	return realtime.ClientSecret{
		Value: response.Value, ExpiresAt: time.Unix(response.ExpiresAt, 0).UTC(),
		ProviderDetails: response.Session,
	}, nil
}

// AnswerWebRTCOffer relays a browser SDP offer through the authenticated server.
func (model *Model) AnswerWebRTCOffer(
	ctx context.Context,
	offer string,
	instructions string,
	tools []ai.ToolDefinition,
	common realtime.Settings,
) (realtime.WebRTCAnswer, error) {
	if offer == "" {
		return realtime.WebRTCAnswer{}, fmt.Errorf("openai realtime: SDP offer must not be empty")
	}
	if common.OutputModality == "" {
		common.OutputModality = realtime.OutputModalityAudio
	}
	config, err := sessionConfig(
		ai.ModelRequestParams{Instructions: instructions, Tools: tools}, common,
		model.resolveSettings(common), model.profile,
	)
	if err != nil {
		return realtime.WebRTCAnswer{}, err
	}
	configData, err := json.Marshal(config)
	if err != nil {
		return realtime.WebRTCAnswer{}, err
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	sdp, _ := writer.CreateFormField("sdp")
	_, _ = io.WriteString(sdp, offer)
	session, _ := writer.CreateFormField("session")
	_, _ = session.Write(configData)
	_ = writer.Close()
	endpoint, err := httpEndpoint(model.baseURL, "realtime/calls")
	if err != nil {
		return realtime.WebRTCAnswer{}, err
	}
	request, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, &body)
	request.Header = openaiprotocol.CloneHeader(model.headers)
	request.Header.Set("Authorization", "Bearer "+model.apiKey)
	request.Header.Set("Accept", "application/sdp")
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response, err := model.client.Do(request)
	if err != nil {
		return realtime.WebRTCAnswer{}, fmt.Errorf("openai realtime: negotiate WebRTC: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	answer, err := io.ReadAll(response.Body)
	if err != nil {
		return realtime.WebRTCAnswer{}, fmt.Errorf("openai realtime: read WebRTC answer: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return realtime.WebRTCAnswer{}, fmt.Errorf(
			"openai realtime: WebRTC negotiation returned %s: %s", response.Status, strings.TrimSpace(string(answer)),
		)
	}
	location := response.Header.Get("Location")
	callID := parseCallID(location)
	if callID == "" {
		return realtime.WebRTCAnswer{}, fmt.Errorf("openai realtime: WebRTC response contained no call ID")
	}
	return realtime.WebRTCAnswer{
		SDP: string(answer), Session: realtime.WebRTCSession{
			Provider: "openai", ID: callID, Details: map[string]any{"location": location},
		},
	}, nil
}

// ConnectWebRTC attaches a server-side control channel to an existing call.
func (model *Model) ConnectWebRTC(
	ctx context.Context, session realtime.ProviderSession, params realtime.ConnectParams,
) (realtime.Connection, error) {
	if session == nil || session.ProviderName() != "openai" || session.SessionID() == "" {
		return nil, fmt.Errorf("openai realtime: WebRTC session must be an OpenAI call")
	}
	endpoint, err := websocketURLForCall(model.baseURL, session.SessionID())
	if err != nil {
		return nil, err
	}
	headers := openaiprotocol.CloneHeader(model.headers)
	headers.Set("Authorization", "Bearer "+model.apiKey)
	socket, response, err := openaiprotocol.DialSocket(ctx, endpoint, headers, model.client)
	if err != nil {
		if response != nil {
			return nil, fmt.Errorf("openai realtime: sideband handshake returned %s: %w", response.Status, err)
		}
		return nil, fmt.Errorf("openai realtime: sideband handshake: %w", err)
	}
	serverModel, err := configureSocket(
		ctx, socket, params.Messages, params.Request, params.Settings,
		model.resolveSettings(params.Settings), model.profile,
	)
	if err != nil {
		_ = socket.Close(websocket.StatusPolicyViolation, "handshake failed")
		return nil, err
	}
	return openaiprotocol.New(openaiprotocol.Config{
		Provider: "OpenAI Realtime", Model: model.name, Socket: socket, ServerModel: serverModel,
		Mapper: MapEvent, InputTranscriptionEnabled: transcriptionEnabled(params.Settings),
		SupportsImages: true, OutputSampleRate: model.profile.AudioOutputSampleRate,
	})
}

func (model *Model) postJSON(
	ctx context.Context, path string, payload any, output any,
) (*http.Response, error) {
	endpoint, err := httpEndpoint(model.baseURL, path)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	request, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	request.Header = openaiprotocol.CloneHeader(model.headers)
	request.Header.Set("Authorization", "Bearer "+model.apiKey)
	request.Header.Set("Content-Type", "application/json")
	response, err := model.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("openai realtime: request %s: %w", path, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		data, _ := io.ReadAll(response.Body)
		return response, fmt.Errorf(
			"openai realtime: request %s returned %s: %s", path, response.Status, strings.TrimSpace(string(data)),
		)
	}
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		return response, fmt.Errorf("openai realtime: decode %s response: %w", path, err)
	}
	return response, nil
}

func (model *Model) dial(
	ctx context.Context,
	history []ai.ModelMessage,
	request ai.ModelRequestParams,
	common realtime.Settings,
	settings Settings,
) (*websocket.Conn, string, error) {
	endpoint, err := websocketURL(model.baseURL, model.name)
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
			return nil, "", fmt.Errorf("openai realtime: websocket handshake returned %s: %w", response.Status, err)
		}
		return nil, "", fmt.Errorf("openai realtime: websocket handshake: %w", err)
	}
	serverModel, err := configureSocket(handshakeCtx, socket, history, request, common, settings, model.profile)
	if err != nil {
		_ = socket.Close(websocket.StatusPolicyViolation, "handshake failed")
		return nil, "", err
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
) (string, error) {
	created, err := expect(ctx, socket, "session.created")
	if err != nil {
		return "", err
	}
	sessionCreated := object(created["session"])
	serverModel := stringValue(sessionCreated["model"])
	config, err := sessionConfig(request, common, settings, profile)
	if err != nil {
		return "", err
	}
	if err := writeJSON(ctx, socket, map[string]any{"type": "session.update", "session": config}); err != nil {
		return "", err
	}
	if _, err := expect(ctx, socket, "session.updated"); err != nil {
		return "", err
	}
	for _, item := range SeedItems(history, profile) {
		if err == nil {
			err = writeJSON(ctx, socket, map[string]any{"type": "conversation.item.create", "item": item})
		}
	}
	return serverModel, err
}

func sessionConfig(
	request ai.ModelRequestParams, common realtime.Settings, settings Settings, profile realtime.Profile,
) (map[string]any, error) {
	config := map[string]any{
		"type": "realtime", "instructions": request.Instructions,
		"output_modalities": []string{string(common.OutputModality)},
		"audio": map[string]any{
			"input":  map[string]any{"format": map[string]any{"type": "audio/pcm", "rate": profile.AudioInputSampleRate}},
			"output": map[string]any{"format": map[string]any{"type": "audio/pcm", "rate": profile.AudioOutputSampleRate}},
		},
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
	audio := object(config["audio"])
	input := object(audio["input"])
	output := object(audio["output"])
	if common.InputTranscriptionModel == nil {
		input["transcription"] = map[string]any{"model": "gpt-realtime-whisper"}
	} else if *common.InputTranscriptionModel != "" {
		model := *common.InputTranscriptionModel
		if model == "auto" {
			model = "gpt-realtime-whisper"
		}
		input["transcription"] = map[string]any{"model": model}
	}
	if settings.InputNoiseReduction != "" {
		input["noise_reduction"] = map[string]any{"type": settings.InputNoiseReduction}
	}
	if settings.TurnDetection != nil {
		input["turn_detection"] = cloneMap(settings.TurnDetection)
	} else if common.TurnDetection != nil {
		if common.TurnDetection.Enabled {
			input["turn_detection"] = portableTurnDetection(*common.TurnDetection)
		} else {
			input["turn_detection"] = nil
		}
	}
	if settings.Voice != "" {
		output["voice"] = settings.Voice
	}
	if settings.OutputSpeed != 0 {
		if settings.OutputSpeed < 0.25 || settings.OutputSpeed > 1.5 {
			return nil, fmt.Errorf("openai realtime: output speed must be between 0.25 and 1.5")
		}
		output["speed"] = settings.OutputSpeed
	}
	if settings.Truncation != nil {
		config["truncation"] = settings.Truncation
	}
	if common.Thinking != "" && profile.SupportsThinking && common.Thinking != ai.ThinkingLevelDisabled {
		config["reasoning"] = map[string]any{"effort": common.Thinking}
	}
	return config, nil
}

func (model *Model) resolveSettings(common realtime.Settings) Settings {
	settings := model.settings
	settings.TurnDetection = cloneMap(settings.TurnDetection)
	if common.Provider == nil {
		return settings
	}
	if value, ok := common.Provider["openai_voice"].(string); ok {
		settings.Voice = value
	}
	if value, ok := common.Provider["openai_input_noise_reduction"].(string); ok {
		settings.InputNoiseReduction = value
	}
	if value, ok := common.Provider["openai_output_speed"].(float64); ok {
		settings.OutputSpeed = value
	}
	if value, ok := common.Provider["openai_turn_detection"].(map[string]any); ok {
		settings.TurnDetection = cloneMap(value)
	}
	if value, ok := common.Provider["openai_truncation"]; ok {
		settings.Truncation = value
	}
	return settings
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

// SeedItems converts portable history to OpenAI-protocol conversation items.
func SeedItems(messages []ai.ModelMessage, profile realtime.Profile) []map[string]any {
	var items []map[string]any
	for _, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.UserPromptPart:
					content := seedUserContent(part, profile)
					if len(content) > 0 {
						items = append(items, map[string]any{"type": "message", "role": "user", "content": content})
					}
				case ai.SpeechPart:
					if part.Transcript != nil {
						items = append(items, textItem("user", *part.Transcript, "input_text"))
					} else if profile.SupportsSeedingAudio && part.Audio != nil {
						items = append(items, map[string]any{
							"type": "message", "role": "user", "content": []any{map[string]any{
								"type": "input_audio", "audio": base64.StdEncoding.EncodeToString(part.Audio.Data),
							}},
						})
					}
				case ai.ToolReturnPart:
					items = append(items, map[string]any{
						"type": "function_call_output", "call_id": part.ToolCallID, "output": render(part.Content),
					})
				case ai.RetryPromptPart:
					items = append(items, map[string]any{
						"type": "function_call_output", "call_id": part.ToolCallID, "output": render(part.Content),
					})
				}
			}
		case ai.ModelResponse:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.TextPart:
					items = append(items, textItem("assistant", part.Content, "text"))
				case ai.SpeechPart:
					if part.Transcript != nil {
						items = append(items, textItem("assistant", *part.Transcript, "text"))
					}
				case ai.ToolCallPart:
					items = append(items, map[string]any{
						"type": "function_call", "call_id": part.ToolCallID,
						"name": part.ToolName, "arguments": string(part.Args),
					})
				}
			}
		}
	}
	return items
}

func seedUserContent(part ai.UserPromptPart, profile realtime.Profile) []any {
	if len(part.Contents) == 0 {
		if part.Content == "" {
			return nil
		}
		return []any{map[string]any{"type": "input_text", "text": part.Content}}
	}
	var content []any
	for _, item := range part.Contents {
		switch item := item.(type) {
		case ai.TextContent:
			content = append(content, map[string]any{"type": "input_text", "text": item.Text})
		case ai.BinaryContent:
			if profile.SupportsSeedingImages && strings.HasPrefix(item.MediaType, "image/") {
				content = append(content, map[string]any{
					"type": "input_image", "image_url": "data:" + item.MediaType + ";base64," +
						base64.StdEncoding.EncodeToString(item.Data),
				})
			}
		}
	}
	return content
}

func textItem(role, text, kind string) map[string]any {
	return map[string]any{
		"type": "message", "role": role, "content": []any{map[string]any{"type": kind, "text": text}},
	}
}

func httpEndpoint(baseURL, path string) (string, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" {
		return "", fmt.Errorf("openai realtime: invalid base URL %q", baseURL)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("openai realtime: unsupported HTTP base URL scheme %q", parsed.Scheme)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/" + strings.TrimLeft(path, "/")
	return parsed.String(), nil
}

func websocketURLForCall(baseURL, callID string) (string, error) {
	endpoint, err := websocketURL(baseURL, "")
	if err != nil {
		return "", err
	}
	parsed, _ := url.Parse(endpoint)
	query := parsed.Query()
	query.Del("model")
	query.Set("call_id", callID)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func parseCallID(location string) string {
	parsed, err := url.Parse(location)
	if err != nil {
		return ""
	}
	if value := parsed.Query().Get("call_id"); value != "" {
		return value
	}
	parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	if len(parts) >= 2 && parts[len(parts)-2] == "calls" {
		return parts[len(parts)-1]
	}
	return ""
}

func websocketURL(baseURL, model string) (string, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" {
		return "", fmt.Errorf("openai realtime: invalid base URL %q", baseURL)
	}
	switch parsed.Scheme {
	case "http":
		parsed.Scheme = "ws"
	case "https":
		parsed.Scheme = "wss"
	case "ws", "wss":
	default:
		return "", fmt.Errorf("openai realtime: unsupported base URL scheme %q", parsed.Scheme)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/realtime"
	query := parsed.Query()
	query.Set("model", model)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func expect(ctx context.Context, socket *websocket.Conn, expected string) (map[string]any, error) {
	for {
		kind, data, err := socket.Read(ctx)
		if err != nil {
			return nil, fmt.Errorf("openai realtime: wait for %s: %w", expected, err)
		}
		if kind != websocket.MessageText {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal(data, &frame); err != nil {
			return nil, fmt.Errorf("openai realtime: decode handshake event: %w", err)
		}
		if stringValue(frame["type"]) == "error" {
			return nil, fmt.Errorf("openai realtime: handshake error: %s", stringValue(object(frame["error"])["message"]))
		}
		if stringValue(frame["type"]) == expected {
			return frame, nil
		}
	}
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

var getenv = func(key string) string { return strings.TrimSpace(os.Getenv(key)) }

var _ realtime.Model = (*Model)(nil)
var _ realtime.WebRTCModel = (*Model)(nil)
