package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/internal/openaiprotocol"
	"github.com/coder/websocket"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

// LiveSettings configures GPT-Live's voice and delegated Responses backend.
type LiveSettings struct {
	Voice        string
	Instructions string
	Delegation   map[string]any
	TurnSilence  time.Duration
	Store        bool
	DataChannel  map[string]any
}

// WithLiveSettings sets GPT-Live defaults. Session provider settings take precedence.
func WithLiveSettings(settings LiveSettings) Option {
	settings.Delegation = cloneLiveMap(settings.Delegation)
	settings.DataChannel = cloneLiveMap(settings.DataChannel)
	return func(model *Model) { model.liveSettings = settings }
}

// LiveModel opens GPT-Live sessions, not OpenAI Realtime API sessions.
type LiveModel struct {
	config  *Model
	backend string
}

// NewLiveModel creates a voice model. A + suffix selects its Responses backend.
func NewLiveModel(name string, options ...Option) *LiveModel {
	name, backend, _ := strings.Cut(name, "+")
	profile := realtime.DefaultProfile()
	profile.SupportsTextOutput = false
	profile.SupportsImageInput = true
	profile.ImageInputRequiresResponse = true
	profile.SupportsSessionSeeding = true
	profile.SupportsWebRTC = true
	profile.SupportsThinking = true
	profile.SupportsAsyncToolCalls = true
	profile.SynthesizesTurnBoundary = true
	profile.UsageExcludesContextWindow = true
	profile.SupportedNativeTools["web_search"] = true
	config := &Model{name: name, apiKey: getenv("OPENAI_API_KEY"), baseURL: defaultBaseURL,
		client: http.DefaultClient, headers: http.Header{}, profile: profile}
	for _, option := range options {
		option(config)
	}
	return &LiveModel{config: config, backend: backend}
}

// Name returns the voice model name without its backend suffix.
func (model *LiveModel) Name() string { return model.config.name }

// ProviderName returns the durable provider identity.
func (*LiveModel) ProviderName() string { return "openai" }

// ProviderURL returns the configured API base URL.
func (model *LiveModel) ProviderURL() string { return model.config.baseURL }

// Profile returns detached GPT-Live capabilities.
func (model *LiveModel) Profile() realtime.Profile { return model.config.Profile() }

// Connect starts one audio-driven GPT-Live websocket session.
func (model *LiveModel) Connect(ctx context.Context, params realtime.ConnectParams) (realtime.Connection, error) {
	if params.Settings.Reconnect != nil {
		policy := *params.Settings.Reconnect
		if policy.MaxAttempts == 0 {
			policy.MaxAttempts = 3
		}
		if policy.MaxReconnects == 0 {
			policy.MaxReconnects = 50
		}
		if policy.BaseDelay == 0 {
			policy.BaseDelay = 500 * time.Millisecond
		}
		if policy.MaxDelay == 0 {
			policy.MaxDelay = 30 * time.Second
		}
		if policy.MaxAttempts < 1 || policy.MaxReconnects < 1 || policy.BaseDelay < 0 || policy.MaxDelay < policy.BaseDelay {
			return nil, fmt.Errorf("openai GPT-Live: invalid reconnect policy")
		}
		params.Settings.Reconnect = &policy
	}
	settings, err := model.resolveLiveSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	config, err := model.liveConfig(params, settings, false)
	if err != nil {
		return nil, err
	}
	socket, started, err := model.openLiveSocket(ctx, "live/sessions", config, params.Settings.HandshakeTimeout)
	if err != nil {
		return nil, err
	}
	dial := func(ctx context.Context, forkFrom string, seed []map[string]any) (*websocket.Conn, string, error) {
		path := "live/sessions"
		replacement := cloneLiveMap(config)
		delete(replacement, "input")
		if forkFrom != "" {
			path += "/" + url.PathEscape(forkFrom) + "/fork"
			replacement = map[string]any{}
		} else if len(seed) > 0 {
			replacement["input"] = seed
		}
		socket, started, err := model.openLiveSocket(ctx, path, replacement, params.Settings.HandshakeTimeout)
		return socket, stringValue(started["id"]), err
	}
	if started["delegation"] == nil {
		started["delegation"] = config["delegation"]
	}
	return newLiveConnection(socket, started, params.Settings, settings, model.config.profile.AudioOutputSampleRate,
		model.config.baseURL, false, dial), nil
}

func (model *LiveModel) openLiveSocket(ctx context.Context, path string, config map[string]any, timeout time.Duration) (*websocket.Conn, map[string]any, error) {
	if model.Name() == "" || model.config.apiKey == "" {
		return nil, nil, fmt.Errorf("openai GPT-Live: model name and OPENAI_API_KEY are required")
	}
	endpoint, err := websocketURL(model.config.baseURL, "")
	if err != nil {
		return nil, nil, err
	}
	parsed, _ := url.Parse(endpoint)
	parsed.Path = strings.TrimSuffix(parsed.Path, "/realtime")
	parsed.RawPath = ""
	parsed.RawQuery = ""
	endpoint = strings.TrimRight(parsed.String(), "/") + "/" + path
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	handshakeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	headers := openaiprotocol.CloneHeader(model.config.headers)
	headers.Set("Authorization", "Bearer "+model.config.apiKey)
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(headers))
	socket, _, err := openaiprotocol.DialSocket(handshakeCtx, endpoint, headers, model.config.client)
	if err != nil {
		return nil, nil, fmt.Errorf("openai GPT-Live: handshake: %w", err)
	}
	if config != nil {
		err = writeJSON(handshakeCtx, socket, map[string]any{"type": "session.start", "session": config})
	}
	var frame map[string]any
	if err == nil {
		frame, err = expect(handshakeCtx, socket, "session.started")
	}
	started := object(frame["session"])
	if err == nil && (stringValue(started["id"]) == "" || stringValue(started["model"]) == "") {
		err = fmt.Errorf("openai GPT-Live: malformed session.started")
	}
	if err != nil {
		_ = socket.CloseNow()
		if strings.Contains(err.Error(), "handshake error") || strings.Contains(err.Error(), "malformed session.started") {
			err = &realtimeHandshakeError{err: err}
		}
		return nil, nil, err
	}
	return socket, started, nil
}

func cloneLiveMap(value map[string]any) map[string]any {
	data, err := json.Marshal(value)
	if err != nil {
		return maps.Clone(value)
	}
	var cloned map[string]any
	_ = json.Unmarshal(data, &cloned)
	return cloned
}

var _ realtime.WebRTCModel = (*LiveModel)(nil)
