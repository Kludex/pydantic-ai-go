// Package azure implements Azure OpenAI GA and Azure AI Voice Live realtime sessions.
package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/internal/openaiprotocol"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

// TokenProvider returns a fresh Microsoft Entra token for one request.
type TokenProvider func(ctx context.Context) (string, error)

// Config configures Azure resources and authentication.
type Config struct {
	// Endpoint is the Azure OpenAI resource origin.
	Endpoint string
	// APIKey authenticates the Azure OpenAI resource.
	APIKey string
	// TokenProvider supplies Microsoft Entra tokens instead of APIKey.
	TokenProvider TokenProvider
	// VoiceLiveEndpoint is the separate Azure AI Voice Live resource origin.
	VoiceLiveEndpoint string
	// VoiceLiveAPIKey authenticates the Voice Live resource.
	VoiceLiveAPIKey string
	// VoiceLiveAPIVersion selects the Voice Live preview protocol.
	VoiceLiveAPIVersion string
	// HTTPClient performs websocket handshakes and WebRTC signaling.
	HTTPClient *http.Client
	// Headers adds detached request headers.
	Headers http.Header
}

// Settings configures Azure realtime behavior.
type Settings struct {
	// OpenAI configures fields shared with the OpenAI GA protocol.
	OpenAI openairt.Settings
	// VoiceLive explicitly selects Azure AI Voice Live.
	VoiceLive bool
	// VoiceLiveTurnDetection replaces portable VAD on Voice Live.
	VoiceLiveTurnDetection map[string]any
}

// Option configures a Model.
type Option func(*Model)

// WithSettings adds model-level Azure realtime defaults.
func WithSettings(settings Settings) Option {
	settings.OpenAI.TurnDetection = cloneMap(settings.OpenAI.TurnDetection)
	settings.VoiceLiveTurnDetection = cloneMap(settings.VoiceLiveTurnDetection)
	return func(model *Model) { model.settings = settings }
}

// WithProfile applies a partial realtime profile override.
func WithProfile(override realtime.ProfileOverride) Option {
	return func(model *Model) { model.profile = realtime.MergeProfile(model.profile, override) }
}

// Model opens Azure OpenAI or Voice Live realtime sessions.
type Model struct {
	name     string
	config   Config
	settings Settings
	profile  realtime.Profile
}

// NewModel creates an Azure realtime deployment model.
func NewModel(name string, config Config, options ...Option) (*Model, error) {
	if name == "" {
		return nil, fmt.Errorf("azure realtime: deployment must not be empty")
	}
	if config.Endpoint == "" {
		config.Endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	if config.APIKey == "" && config.TokenProvider == nil {
		config.APIKey = os.Getenv("AZURE_OPENAI_API_KEY")
	}
	if config.APIKey != "" && config.TokenProvider != nil {
		return nil, fmt.Errorf("azure realtime: API key and token provider cannot both be set")
	}
	if config.APIKey == "" && config.TokenProvider == nil {
		return nil, fmt.Errorf("azure realtime: API key or token provider is required")
	}
	if _, err := resourceOrigin(config.Endpoint); err != nil {
		return nil, err
	}
	if config.VoiceLiveEndpoint == "" {
		config.VoiceLiveEndpoint = os.Getenv("AZURE_VOICELIVE_ENDPOINT")
	}
	if config.VoiceLiveAPIKey == "" {
		config.VoiceLiveAPIKey = os.Getenv("AZURE_VOICELIVE_API_KEY")
	}
	if config.VoiceLiveAPIVersion == "" {
		config.VoiceLiveAPIVersion = os.Getenv("AZURE_VOICELIVE_API_VERSION")
		if config.VoiceLiveAPIVersion == "" {
			config.VoiceLiveAPIVersion = "2025-10-01"
		}
	}
	if config.HTTPClient == nil {
		config.HTTPClient = http.DefaultClient
	}
	config.Headers = openaiprotocol.CloneHeader(config.Headers)
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
	if modelAPIs(name) == voiceLiveOnly {
		profile.SupportsWebRTC = false
	}
	model := &Model{name: name, config: config, profile: profile}
	for _, option := range options {
		option(model)
	}
	if model.settings.VoiceLive {
		model.profile.SupportsWebRTC = false
	}
	return model, nil
}

// Name returns the Azure deployment name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable Azure provider identity.
func (*Model) ProviderName() string { return "azure" }

// Profile returns detached realtime capabilities.
func (model *Model) Profile() realtime.Profile {
	return realtime.MergeProfile(model.profile, realtime.ProfileOverride{})
}

// Connect opens and configures an Azure realtime websocket.
func (model *Model) Connect(ctx context.Context, params realtime.ConnectParams) (realtime.Connection, error) {
	settings := model.resolveSettings(params.Settings)
	voiceLive, err := model.useVoiceLive(settings)
	if err != nil {
		return nil, err
	}
	dial := func(ctx context.Context, history []ai.ModelMessage) (*websocket.Conn, string, error) {
		return model.dial(ctx, history, params.Request, params.Settings, settings, voiceLive, "")
	}
	socket, serverModel, err := dial(ctx, params.Messages)
	if err != nil {
		return nil, err
	}
	return openaiprotocol.New(openaiprotocol.Config{
		Provider: "Azure Realtime", Model: model.name, Socket: socket, ServerModel: serverModel,
		Dial: dial, Mapper: openairt.MapEvent, Reconnect: params.Settings.Reconnect,
		InputTranscriptionEnabled: transcriptionEnabled(params.Settings), RestoresInFlightState: false,
		SupportsImages: true, OutputSampleRate: model.profile.AudioOutputSampleRate,
	})
}

// CreateClientSecret mints an ephemeral Azure OpenAI browser credential.
func (model *Model) CreateClientSecret(
	ctx context.Context,
	instructions string,
	tools []ai.ToolDefinition,
	common realtime.Settings,
	expiresAfter time.Duration,
) (realtime.ClientSecret, error) {
	settings := model.resolveSettings(common)
	if voiceLive, err := model.useVoiceLive(settings); err != nil {
		return realtime.ClientSecret{}, err
	} else if voiceLive {
		return realtime.ClientSecret{}, fmt.Errorf("azure realtime: WebRTC is unavailable through Voice Live")
	}
	config, err := sessionConfig(
		ai.ModelRequestParams{Instructions: instructions, Tools: tools}, common, settings, false, model.profile, model.name,
	)
	if err != nil {
		return realtime.ClientSecret{}, err
	}
	payload := map[string]any{"session": config}
	if expiresAfter > 0 {
		payload["expires_after"] = map[string]any{"anchor": "created_at", "seconds": int(expiresAfter.Seconds())}
	}
	var response struct {
		Value     string         `json:"value"`
		ExpiresAt int64          `json:"expires_at"`
		Session   map[string]any `json:"session"`
	}
	if err := model.postJSON(ctx, "realtime/client_secrets", payload, &response); err != nil {
		return realtime.ClientSecret{}, err
	}
	if response.Value == "" || response.ExpiresAt == 0 {
		return realtime.ClientSecret{}, fmt.Errorf("azure realtime: client-secret response is incomplete")
	}
	return realtime.ClientSecret{
		Value: response.Value, ExpiresAt: time.Unix(response.ExpiresAt, 0).UTC(),
		ProviderDetails: response.Session,
	}, nil
}

// AnswerWebRTCOffer relays an SDP offer with a short-lived Azure credential.
func (model *Model) AnswerWebRTCOffer(
	ctx context.Context,
	offer string,
	instructions string,
	tools []ai.ToolDefinition,
	settings realtime.Settings,
) (realtime.WebRTCAnswer, error) {
	if offer == "" {
		return realtime.WebRTCAnswer{}, fmt.Errorf("azure realtime: SDP offer must not be empty")
	}
	secret, err := model.CreateClientSecret(ctx, instructions, tools, settings, 0)
	if err != nil {
		return realtime.WebRTCAnswer{}, err
	}
	endpoint := model.httpURL("realtime/calls")
	request, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(offer))
	request.Header.Set("Authorization", "Bearer "+secret.Value)
	request.Header.Set("Content-Type", "application/sdp")
	response, err := model.config.HTTPClient.Do(request)
	if err != nil {
		return realtime.WebRTCAnswer{}, fmt.Errorf("azure realtime: negotiate WebRTC: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return realtime.WebRTCAnswer{}, fmt.Errorf("azure realtime: read WebRTC answer: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return realtime.WebRTCAnswer{}, fmt.Errorf(
			"azure realtime: WebRTC negotiation returned %s: %s", response.Status, strings.TrimSpace(string(body)),
		)
	}
	location := response.Header.Get("Location")
	callID := parseCallID(location)
	if callID == "" {
		return realtime.WebRTCAnswer{}, fmt.Errorf("azure realtime: WebRTC response contained no call ID")
	}
	return realtime.WebRTCAnswer{
		SDP: string(body), Session: realtime.WebRTCSession{
			Provider: "azure", ID: callID, Details: map[string]any{"location": location},
		},
	}, nil
}

// ConnectWebRTC attaches a server-side control channel to an Azure OpenAI call.
func (model *Model) ConnectWebRTC(
	ctx context.Context, session realtime.ProviderSession, params realtime.ConnectParams,
) (realtime.Connection, error) {
	if session == nil || session.ProviderName() != "azure" || session.SessionID() == "" {
		return nil, fmt.Errorf("azure realtime: WebRTC session must be an Azure call")
	}
	settings := model.resolveSettings(params.Settings)
	if voiceLive, err := model.useVoiceLive(settings); err != nil {
		return nil, err
	} else if voiceLive {
		return nil, fmt.Errorf("azure realtime: WebRTC is unavailable through Voice Live")
	}
	socket, serverModel, err := model.dial(
		ctx, params.Messages, params.Request, params.Settings, settings, false, session.SessionID(),
	)
	if err != nil {
		return nil, err
	}
	return openaiprotocol.New(openaiprotocol.Config{
		Provider: "Azure Realtime", Model: model.name, Socket: socket, ServerModel: serverModel,
		Mapper: openairt.MapEvent, InputTranscriptionEnabled: transcriptionEnabled(params.Settings),
		SupportsImages: true, OutputSampleRate: model.profile.AudioOutputSampleRate,
	})
}

func (model *Model) dial(
	ctx context.Context,
	history []ai.ModelMessage,
	request ai.ModelRequestParams,
	common realtime.Settings,
	settings Settings,
	voiceLive bool,
	callID string,
) (*websocket.Conn, string, error) {
	endpoint, err := model.websocketURL(voiceLive, callID)
	if err != nil {
		return nil, "", err
	}
	headers, err := model.authHeaders(ctx, voiceLive)
	if err != nil {
		return nil, "", err
	}
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(headers))
	timeout := common.HandshakeTimeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	handshakeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	socket, response, err := openaiprotocol.DialSocket(handshakeCtx, endpoint, headers, model.config.HTTPClient)
	if err != nil {
		if response != nil {
			return nil, "", fmt.Errorf("azure realtime: websocket handshake returned %s: %w", response.Status, err)
		}
		return nil, "", fmt.Errorf("azure realtime: websocket handshake: %w", err)
	}
	serverModel, err := configureSocket(
		handshakeCtx, socket, history, request, common, settings, voiceLive, model.profile, model.name,
	)
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
	voiceLive bool,
	profile realtime.Profile,
	modelName string,
) (string, error) {
	created, err := expect(ctx, socket, "session.created")
	if err != nil {
		return "", err
	}
	serverModel := stringValue(object(created["session"])["model"])
	config, err := sessionConfig(request, common, settings, voiceLive, profile, modelName)
	if err != nil {
		return "", err
	}
	if err := writeJSON(ctx, socket, map[string]any{"type": "session.update", "session": config}); err != nil {
		return "", err
	}
	if _, err := expect(ctx, socket, "session.updated"); err != nil {
		return "", err
	}
	for _, item := range openairt.SeedItems(history, profile) {
		if err == nil {
			err = writeJSON(ctx, socket, map[string]any{"type": "conversation.item.create", "item": item})
		}
	}
	return serverModel, err
}

func sessionConfig(
	request ai.ModelRequestParams,
	common realtime.Settings,
	settings Settings,
	voiceLive bool,
	profile realtime.Profile,
	modelName string,
) (map[string]any, error) {
	if voiceLive {
		return voiceLiveSessionConfig(request, common, settings, profile, modelName), nil
	}
	config := map[string]any{
		"type": "realtime", "instructions": request.Instructions,
		"output_modalities": []string{string(defaultOutput(common.OutputModality))},
		"audio": map[string]any{
			"input":  map[string]any{"format": map[string]any{"type": "audio/pcm", "rate": profile.AudioInputSampleRate}},
			"output": map[string]any{"format": map[string]any{"type": "audio/pcm", "rate": profile.AudioOutputSampleRate}},
		},
	}
	applyCommonConfig(config, request, common)
	input := object(object(config["audio"])["input"])
	output := object(object(config["audio"])["output"])
	applyTranscription(input, "transcription", common, "gpt-realtime-whisper")
	if settings.OpenAI.InputNoiseReduction != "" {
		input["noise_reduction"] = map[string]any{"type": settings.OpenAI.InputNoiseReduction}
	}
	if settings.OpenAI.TurnDetection != nil {
		input["turn_detection"] = cloneMap(settings.OpenAI.TurnDetection)
	} else if common.TurnDetection != nil {
		input["turn_detection"] = portableTurnDetection(*common.TurnDetection)
	}
	if settings.OpenAI.Voice != "" {
		output["voice"] = settings.OpenAI.Voice
	}
	if settings.OpenAI.OutputSpeed != 0 {
		if settings.OpenAI.OutputSpeed < 0.25 || settings.OpenAI.OutputSpeed > 1.5 {
			return nil, fmt.Errorf("azure realtime: output speed must be between 0.25 and 1.5")
		}
		output["speed"] = settings.OpenAI.OutputSpeed
	}
	if settings.OpenAI.Truncation != nil {
		config["truncation"] = settings.OpenAI.Truncation
	}
	if common.Thinking != "" && profile.SupportsThinking && common.Thinking != ai.ThinkingLevelDisabled {
		config["reasoning"] = map[string]any{"effort": common.Thinking}
	}
	return config, nil
}

func voiceLiveSessionConfig(
	request ai.ModelRequestParams, common realtime.Settings, settings Settings, profile realtime.Profile, modelName string,
) map[string]any {
	modalities := []string{"text", "audio"}
	if defaultOutput(common.OutputModality) == realtime.OutputModalityText {
		modalities = []string{"text"}
	}
	turnDetection := settings.VoiceLiveTurnDetection
	if turnDetection == nil && common.TurnDetection != nil {
		turnDetection = portableTurnDetection(*common.TurnDetection)
	}
	if turnDetection == nil {
		turnDetection = map[string]any{"type": "server_vad"}
	}
	config := map[string]any{
		"instructions": request.Instructions, "modalities": modalities,
		"input_audio_format": "pcm16", "output_audio_format": "pcm16",
		"input_audio_sampling_rate": profile.AudioInputSampleRate,
		"turn_detection":            turnDetection,
	}
	if common.MaxTokens > 0 {
		config["max_response_output_tokens"] = common.MaxTokens
	}
	if common.ToolChoice != "" {
		config["tool_choice"] = common.ToolChoice
	}
	applyTools(config, request)
	automatic := "azure-speech"
	if strings.HasPrefix(modelName, "gpt-realtime") {
		automatic = "whisper-1"
	}
	applyTranscription(config, "input_audio_transcription", common, automatic)
	if settings.OpenAI.Voice != "" {
		config["voice"] = map[string]any{"type": "openai", "name": settings.OpenAI.Voice}
	}
	return config
}

func applyCommonConfig(config map[string]any, request ai.ModelRequestParams, common realtime.Settings) {
	if common.MaxTokens > 0 {
		config["max_output_tokens"] = common.MaxTokens
	}
	if common.ParallelToolCalls != nil {
		config["parallel_tool_calls"] = *common.ParallelToolCalls
	}
	if common.ToolChoice != "" {
		config["tool_choice"] = common.ToolChoice
	}
	applyTools(config, request)
}

func applyTools(config map[string]any, request ai.ModelRequestParams) {
	if len(request.Tools) == 0 {
		return
	}
	tools := make([]any, len(request.Tools))
	for index, tool := range request.Tools {
		tools[index] = map[string]any{
			"type": "function", "name": tool.Name, "description": tool.Description, "parameters": tool.Schema,
		}
	}
	config["tools"] = tools
}

func applyTranscription(target map[string]any, key string, common realtime.Settings, automatic string) {
	if common.InputTranscriptionModel == nil {
		target[key] = map[string]any{"model": automatic}
		return
	}
	if *common.InputTranscriptionModel == "" {
		return
	}
	name := *common.InputTranscriptionModel
	if name == "auto" {
		name = automatic
	}
	target[key] = map[string]any{"model": name}
}

func (model *Model) resolveSettings(common realtime.Settings) Settings {
	settings := model.settings
	settings.OpenAI.TurnDetection = cloneMap(settings.OpenAI.TurnDetection)
	settings.VoiceLiveTurnDetection = cloneMap(settings.VoiceLiveTurnDetection)
	if value, ok := common.Provider["azure_voice_live"].(bool); ok {
		settings.VoiceLive = value
	}
	if value, ok := common.Provider["azure_voice_live_turn_detection"].(map[string]any); ok {
		settings.VoiceLiveTurnDetection = cloneMap(value)
	}
	if value, ok := common.Provider["openai_voice"].(string); ok {
		settings.OpenAI.Voice = value
	}
	if value, ok := common.Provider["openai_input_noise_reduction"].(string); ok {
		settings.OpenAI.InputNoiseReduction = value
	}
	if value, ok := common.Provider["openai_output_speed"].(float64); ok {
		settings.OpenAI.OutputSpeed = value
	}
	if value, ok := common.Provider["openai_turn_detection"].(map[string]any); ok {
		settings.OpenAI.TurnDetection = cloneMap(value)
	}
	if value, ok := common.Provider["openai_truncation"]; ok {
		settings.OpenAI.Truncation = value
	}
	return settings
}

type servingAPIs int

const (
	bothAPIs servingAPIs = iota
	azureOpenAIOnly
	voiceLiveOnly
)

func modelAPIs(name string) servingAPIs {
	for _, base := range []string{
		"gpt-4o-realtime", "gpt-4o-mini-realtime", "gpt-realtime-2", "gpt-realtime-translate",
		"gpt-realtime-whisper", "gpt-live-transcribe",
	} {
		if nameMatches(name, base) {
			return azureOpenAIOnly
		}
	}
	for _, base := range []string{"phi4-mm-realtime", "azure-realtime", "gpt-4o", "gpt-4.1", "gpt-5", "phi4-mini"} {
		if nameMatches(name, base) {
			return voiceLiveOnly
		}
	}
	return bothAPIs
}

func nameMatches(name, base string) bool {
	return name == base || (strings.HasPrefix(name, base) && len(name) > len(base) &&
		(name[len(base)] == '-' || name[len(base)] == '.'))
}

func (model *Model) useVoiceLive(settings Settings) (bool, error) {
	apis := modelAPIs(model.name)
	if settings.VoiceLive && apis == azureOpenAIOnly {
		return false, fmt.Errorf("azure realtime: deployment %q is not served by Voice Live", model.name)
	}
	return settings.VoiceLive || apis == voiceLiveOnly, nil
}

func (model *Model) authHeaders(ctx context.Context, voiceLive bool) (http.Header, error) {
	headers := openaiprotocol.CloneHeader(model.config.Headers)
	if model.config.TokenProvider != nil {
		token, err := model.config.TokenProvider(ctx)
		if err != nil {
			return nil, fmt.Errorf("azure realtime: get Entra token: %w", err)
		}
		if token == "" {
			return nil, fmt.Errorf("azure realtime: get Entra token: empty token")
		}
		headers.Set("Authorization", "Bearer "+token)
		return headers, nil
	}
	key := model.config.APIKey
	if voiceLive {
		key = model.config.VoiceLiveAPIKey
		if key == "" {
			return nil, fmt.Errorf("azure realtime: Voice Live API key is required")
		}
	}
	headers.Set("api-key", key)
	return headers, nil
}

func (model *Model) websocketURL(voiceLive bool, callID string) (string, error) {
	endpoint := model.config.Endpoint
	path := "/openai/v1/realtime"
	if voiceLive {
		endpoint = model.config.VoiceLiveEndpoint
		path = "/voice-live/realtime"
		if endpoint == "" {
			return "", fmt.Errorf("azure realtime: Voice Live endpoint is required")
		}
	}
	origin, err := resourceOrigin(endpoint)
	if err != nil {
		return "", err
	}
	origin.Scheme = map[string]string{"http": "ws", "https": "wss"}[origin.Scheme]
	origin.Path = path
	query := origin.Query()
	if callID != "" {
		query.Set("call_id", callID)
	} else {
		query.Set("model", model.name)
	}
	if voiceLive {
		query.Set("api-version", model.config.VoiceLiveAPIVersion)
	}
	origin.RawQuery = query.Encode()
	return origin.String(), nil
}

func (model *Model) httpURL(path string) string {
	origin, _ := resourceOrigin(model.config.Endpoint)
	origin.Path = "/openai/v1/" + strings.TrimLeft(path, "/")
	if strings.TrimLeft(path, "/") == "realtime/calls" {
		query := origin.Query()
		query.Set("webrtcfilter", "on")
		origin.RawQuery = query.Encode()
	}
	return origin.String()
}

func (model *Model) postJSON(ctx context.Context, path string, payload, output any) error {
	endpoint := model.httpURL(path)
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	headers, err := model.authHeaders(ctx, false)
	if err != nil {
		return err
	}
	request.Header = headers
	request.Header.Set("Content-Type", "application/json")
	response, err := model.config.HTTPClient.Do(request)
	if err != nil {
		return fmt.Errorf("azure realtime: request %s: %w", path, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		data, _ := io.ReadAll(response.Body)
		return fmt.Errorf(
			"azure realtime: request %s returned %s: %s", path, response.Status, strings.TrimSpace(string(data)),
		)
	}
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		return fmt.Errorf("azure realtime: decode %s response: %w", path, err)
	}
	return nil
}

func resourceOrigin(raw string) (*url.URL, error) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, fmt.Errorf("azure realtime: endpoint must be an absolute HTTP URL")
	}
	parsed.Path = ""
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed, nil
}

func portableTurnDetection(settings realtime.TurnDetection) map[string]any {
	if !settings.Enabled {
		return nil
	}
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
		kind, data, err := socket.Read(ctx)
		if err != nil {
			return nil, fmt.Errorf("azure realtime: wait for %s: %w", expected, err)
		}
		if kind != websocket.MessageText {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal(data, &frame); err != nil {
			return nil, fmt.Errorf("azure realtime: decode handshake event: %w", err)
		}
		if stringValue(frame["type"]) == "error" {
			return nil, fmt.Errorf("azure realtime: handshake error: %s", stringValue(object(frame["error"])["message"]))
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

func defaultOutput(modality realtime.OutputModality) realtime.OutputModality {
	if modality == "" {
		return realtime.OutputModalityAudio
	}
	return modality
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

var _ realtime.Model = (*Model)(nil)
var _ realtime.WebRTCModel = (*Model)(nil)
