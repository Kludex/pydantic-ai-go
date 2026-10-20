// Package elevenlabs runs realtime sessions on hosted ElevenLabs Agents.
//
// An ElevenLabs agent runs a cascaded pipeline (speech recognition, an LLM,
// ElevenLabs text-to-speech, and a server-side turn model) behind an agent ID.
// The model wraps that hosted agent and takes over as much of each
// conversation as the agent's override toggles allow, failing before the
// socket is dialed where they do not.
package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/coder/websocket"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

const defaultBaseURL = "https://api.elevenlabs.io"

// ToolSync says how the session's tools are reconciled with the agent's client tools.
type ToolSync string

const (
	// ToolSyncError fails the connect, describing every difference. It is the default.
	ToolSyncError ToolSync = "error"
	// ToolSyncSync makes the workspace match the session over REST before dialing.
	// It changes workspace state every conversation with the agent shares.
	ToolSyncSync ToolSync = "sync"
	// ToolSyncOff trusts the agent's own tool definitions.
	ToolSyncOff ToolSync = "off"
)

// TTSSettings are per-conversation text-to-speech overrides.
type TTSSettings struct {
	// ModelID selects the TTS model, such as eleven_v3_conversational.
	ModelID string
	// Stability ranges from zero (expressive) to one (consistent).
	Stability *float64
	// Speed multiplies playback speed.
	Speed *float64
	// SimilarityBoost ranges from zero to one.
	SimilarityBoost *float64
}

// Settings configures ElevenLabs Agents sessions. Each override setting needs
// its toggle under Security > Overrides in the agent's settings.
type Settings struct {
	// VoiceID selects the output voice (the tts.voice_id override).
	VoiceID string
	// Language sets the conversation language code (the agent.language override).
	Language string
	// FirstMessage sets the greeting the agent opens with (the agent.first_message override).
	FirstMessage string
	// LLM selects the pipeline LLM by ElevenLabs model ID (the agent.prompt.llm override).
	LLM string
	// TTS sets the TTS model and delivery (the tts.* overrides).
	TTS TTSSettings
	// DynamicVariables fill {{placeholders}} in the agent's prompts and messages. No toggle needed.
	DynamicVariables map[string]any
	// CustomLLMExtraBody is merged into a custom-LLM request. It needs the agent's
	// custom_llm_extra_body toggle.
	CustomLLMExtraBody map[string]any
	// UserID identifies the end user for ElevenLabs analytics. No toggle needed.
	UserID string
	// ToolSync reconciles the session's tools with the agent's. Empty means ToolSyncError.
	ToolSync ToolSync
	// KeepAudioTags keeps expressive tags such as [laughs] in transcripts. By
	// default they are removed and kept in ProviderDetails["audio_tags"].
	KeepAudioTags bool
	// ConfigOverride is deep-merged last into conversation_config_override. It is
	// not checked against the toggles; the server closes the socket for a
	// field the agent does not permit.
	ConfigOverride map[string]any
	// WarningHandler receives provider warnings. Nil uses slog.Warn.
	WarningHandler func(string)
}

// Option configures a Model.
type Option func(*Model)

// WithAPIKey sets the API key instead of reading ELEVENLABS_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.apiKey = key } }

// WithBaseURL selects a data-residency host such as
// https://api.eu.residency.elevenlabs.io, which pins both the REST calls and
// the conversation socket to the region.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.baseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient selects the caller-owned client for REST calls and the socket handshake.
func WithHTTPClient(client *http.Client) Option { return func(model *Model) { model.client = client } }

// WithSettings sets model-level defaults.
func WithSettings(settings Settings) Option {
	settings = settings.clone()
	return func(model *Model) { model.settings = settings }
}

// WithProfile applies a partial profile override. Use it to declare the audio
// sample rates of an agent configured away from 16 kHz PCM.
func WithProfile(override realtime.ProfileOverride) Option {
	return func(model *Model) { model.profile = realtime.MergeProfile(model.profile, override) }
}

// Model opens conversations with one hosted ElevenLabs agent.
type Model struct {
	agentID  string
	apiKey   string
	baseURL  string
	client   *http.Client
	settings Settings
	profile  realtime.Profile

	warnedContextUsage sync.Once
}

// NewModel creates a model for the agent with agentID, such as agent_0101k2...
func NewModel(agentID string, options ...Option) *Model {
	profile := realtime.DefaultProfile()
	// Barge-in is server-side, with no client verb to cancel or truncate a response.
	profile.SupportsInterruption = false
	profile.AudioInputSampleRate = 16000
	profile.AudioOutputSampleRate = 16000
	// Usage arrives as context_usage reports, never attached to a response.
	profile.UsageExcludesContextWindow = true
	model := &Model{
		agentID: agentID, apiKey: strings.TrimSpace(os.Getenv("ELEVENLABS_API_KEY")), baseURL: defaultBaseURL,
		client: http.DefaultClient, profile: profile,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the agent ID; ElevenLabs has no realtime model ID apart from the agent.
func (model *Model) Name() string { return model.agentID }

// ProviderName returns the durable provider identity.
func (*Model) ProviderName() string { return "elevenlabs" }

// ProviderURL returns the configured API base URL.
func (model *Model) ProviderURL() string { return model.baseURL }

// Profile returns detached realtime capabilities.
func (model *Model) Profile() realtime.Profile {
	return realtime.MergeProfile(model.profile, realtime.ProfileOverride{})
}

// Connect checks the agent over REST, reconciles its tools, and opens one conversation.
func (model *Model) Connect(ctx context.Context, params realtime.ConnectParams) (realtime.Connection, error) {
	settings, err := model.resolveSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	if model.agentID == "" {
		return nil, errors.New("elevenlabs: agent ID must not be empty")
	}
	if model.apiKey == "" {
		return nil, errors.New("elevenlabs: ELEVENLABS_API_KEY is not set; even a public agent's preflight needs it")
	}
	if len(params.Messages) > 0 {
		return nil, errors.New("elevenlabs: the conversation socket cannot seed history")
	}
	tools := params.Request.Tools
	if params.Settings.ToolChoice == realtime.ToolChoiceRequired {
		return nil, errors.New(
			"elevenlabs: a realtime session can't force a tool call, because ElevenLabs agents have no per-response " +
				"tool choice",
		)
	}
	// ToolChoiceNone narrows the agent's attached client tools to none for this conversation.
	restricts := params.Settings.ToolChoice == realtime.ToolChoiceNone

	agent, err := model.fetchAgent(ctx)
	if err != nil {
		return nil, err
	}
	if !agent.sendsContextUsage() {
		model.warnedContextUsage.Do(func() {
			settings.warn(fmt.Sprintf(
				"elevenlabs: agent %q doesn't send context_usage reports, so the session has no context usage, and a "+
					"spoken reply is taken to end after a second without audio rather than with its last audio. Add "+
					"context_usage to the agent's client events (Advanced > Client events in the agent's settings).",
				model.agentID,
			))
		})
	}
	instructions := params.Request.Instructions
	attached := agent.toolIDs()
	switch settings.ToolSync {
	case ToolSyncError, "":
		if mismatches := toolMismatches(tools, agent.clientTools()); len(mismatches) > 0 {
			return nil, fmt.Errorf(
				"elevenlabs: the tools of agent %q do not match this agent run:\n- %s\nUpdate the agent's client tools "+
					"to match, set ToolSync to %q to sync them from Pydantic AI, or set it to %q to trust the agent",
				model.agentID, strings.Join(mismatches, "\n- "), ToolSyncSync, ToolSyncOff,
			)
		}
	case ToolSyncSync:
		// The sync changes workspace state, so everything else that can refuse the conversation is checked first.
		var probe []string
		if restricts && len(tools) > 0 {
			probe = []string{}
		}
		if err := model.checkPermissions(agent, settings, requestedOverrides(instructions, settings, params.Settings, probe)); err != nil {
			return nil, err
		}
		if attached, err = model.syncTools(ctx, agent, tools); err != nil {
			return nil, err
		}
	case ToolSyncOff:
	default:
		return nil, fmt.Errorf("elevenlabs: unknown tool sync mode %q", settings.ToolSync)
	}

	var restricted []string
	if restricts {
		// Judged against the attached tools rather than the run's: with ToolSyncOff the agent can have client
		// tools the run doesn't define, which the choice leaves out all the same.
		allowed, err := model.restrictedToolIDs(ctx, attached, map[string]bool{})
		if err != nil {
			return nil, err
		}
		if !slices.Equal(allowed, attached) {
			restricted = allowed
		}
	}
	requested := requestedOverrides(instructions, settings, params.Settings, restricted)
	if err := model.checkPermissions(agent, settings, requested); err != nil {
		return nil, err
	}
	payload := initiationPayload(requested, settings)

	signedURL, err := model.signedURL(ctx)
	if err != nil {
		return nil, err
	}
	timeout := params.Settings.HandshakeTimeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	handshakeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	headers := http.Header{}
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(headers))
	socket, response, err := websocket.Dial(handshakeCtx, signedURL, &websocket.DialOptions{HTTPHeader: headers, HTTPClient: model.client})
	if err != nil {
		if response != nil {
			return nil, &APIError{StatusCode: response.StatusCode, Body: readBody(response), Model: model.agentID}
		}
		return nil, &realtime.Error{Provider: "elevenlabs", Model: model.agentID, Message: "dial conversation", Err: err}
	}
	socket.SetReadLimit(16 << 20)
	metadata, err := handshake(handshakeCtx, socket, payload)
	if err == nil {
		err = model.checkAudioFormats(metadata)
	}
	if err != nil {
		_ = socket.CloseNow()
		var realtimeErr *realtime.Error
		if errors.As(err, &realtimeErr) {
			return nil, err
		}
		return nil, &realtime.Error{Provider: "elevenlabs", Model: model.agentID, Message: "handshake", Err: err}
	}
	return newConnection(socket, metadata.ConversationID, params.Settings.OutputModality == realtime.OutputModalityText,
		!settings.KeepAudioTags, settings.warn), nil
}

func (model *Model) resolveSettings(common realtime.Settings) (Settings, error) {
	settings := model.settings.clone()
	if common.Reconnect != nil {
		return settings, errors.New(
			"elevenlabs: conversations are not resumed after a drop, so a reconnect policy would silently start a " +
				"conversation that remembers nothing; remove it and open a new session instead",
		)
	}
	if common.TurnDetection != nil && !common.TurnDetection.Enabled {
		return settings, errors.New(
			"elevenlabs: the agent's server-side turn model is always on, so turn detection cannot be disabled " +
				"per conversation; configure turn-taking on the agent instead",
		)
	}
	if common.TurnDetection != nil && *common.TurnDetection != (realtime.TurnDetection{Enabled: true}) {
		return settings, errors.New(
			"elevenlabs: turn detection cannot be configured per conversation; configure turn-taking on the agent",
		)
	}
	if common.InputTranscriptionModel != nil && *common.InputTranscriptionModel == "" {
		return settings, errors.New(
			"elevenlabs: speech recognition drives the agent pipeline and cannot be disabled per conversation",
		)
	}
	if value, ok := common.Provider["elevenlabs_agents_tool_sync"].(string); ok {
		settings.ToolSync = ToolSync(value)
	}
	return settings, nil
}

type override struct {
	path   []string
	value  any
	source string
}

// requestedOverrides lists the conversation_config_override entries the session asks for.
func requestedOverrides(instructions string, settings Settings, common realtime.Settings, toolIDs []string) []override {
	var requested []override
	if toolIDs != nil {
		requested = append(requested, override{[]string{"agent", "prompt", "tool_ids"}, toolIDs, "the tool choice"})
	}
	if instructions != "" {
		requested = append(requested, override{[]string{"agent", "prompt", "prompt"}, instructions, "the agent instructions"})
	}
	if settings.LLM != "" {
		requested = append(requested, override{[]string{"agent", "prompt", "llm"}, settings.LLM, "the LLM setting"})
	}
	if settings.FirstMessage != "" {
		requested = append(requested, override{[]string{"agent", "first_message"}, settings.FirstMessage, "the first message setting"})
	}
	if settings.Language != "" {
		requested = append(requested, override{[]string{"agent", "language"}, settings.Language, "the language setting"})
	}
	if settings.VoiceID != "" {
		requested = append(requested, override{[]string{"tts", "voice_id"}, settings.VoiceID, "the voice setting"})
	}
	if settings.TTS.ModelID != "" {
		requested = append(requested, override{[]string{"tts", "model_id"}, settings.TTS.ModelID, "the TTS model_id setting"})
	}
	for _, knob := range []struct {
		name  string
		value *float64
	}{{"stability", settings.TTS.Stability}, {"speed", settings.TTS.Speed}, {"similarity_boost", settings.TTS.SimilarityBoost}} {
		if knob.value != nil {
			requested = append(requested, override{[]string{"tts", knob.name}, *knob.value, "the TTS " + knob.name + " setting"})
		}
	}
	if common.OutputModality == realtime.OutputModalityText {
		requested = append(requested, override{[]string{"conversation", "text_only"}, true, "text output"})
	}
	return requested
}

// checkPermissions names the toggle of a requested override the agent does not permit.
func (model *Model) checkPermissions(agent agentPreflight, settings Settings, requested []override) error {
	permissions := agent.overridePermissions()
	for _, entry := range requested {
		var node any = permissions
		for _, key := range entry.path {
			object, _ := node.(map[string]any)
			node = object[key]
		}
		if node != true {
			dotted := strings.Join(entry.path, ".")
			return fmt.Errorf(
				"elevenlabs: agent %q does not permit the %s conversation override required by %s. Enable the "+
					"toggle at platform_settings.overrides.conversation_config_override.%s (Security > Overrides in "+
					"the agent's settings), or remove the conflicting input",
				model.agentID, dotted, entry.source, dotted,
			)
		}
	}
	if settings.CustomLLMExtraBody != nil && agent.PlatformSettings.Overrides["custom_llm_extra_body"] != true {
		return fmt.Errorf(
			"elevenlabs: agent %q does not permit custom_llm_extra_body. Enable the toggle at "+
				"platform_settings.overrides.custom_llm_extra_body, or remove the setting",
			model.agentID,
		)
	}
	return nil
}

func initiationPayload(requested []override, settings Settings) map[string]any {
	overrides := map[string]any{}
	for _, entry := range requested {
		node := overrides
		for _, key := range entry.path[:len(entry.path)-1] {
			child, ok := node[key].(map[string]any)
			if !ok {
				child = map[string]any{}
				node[key] = child
			}
			node = child
		}
		node[entry.path[len(entry.path)-1]] = entry.value
	}
	deepMerge(overrides, settings.ConfigOverride)
	payload := map[string]any{
		"type": "conversation_initiation_client_data",
		// ElevenLabs' source enum names its own SDKs and integrations only.
		"source_info": map[string]any{"source": "unknown", "version": "pydantic-ai-go"},
	}
	if len(overrides) > 0 {
		payload["conversation_config_override"] = overrides
	}
	if settings.CustomLLMExtraBody != nil {
		payload["custom_llm_extra_body"] = settings.CustomLLMExtraBody
	}
	if settings.DynamicVariables != nil {
		payload["dynamic_variables"] = settings.DynamicVariables
	}
	if settings.UserID != "" {
		payload["user_id"] = settings.UserID
	}
	return payload
}

func deepMerge(base, override map[string]any) {
	for key, value := range override {
		child, isObject := value.(map[string]any)
		existing, hasObject := base[key].(map[string]any)
		if isObject && hasObject {
			deepMerge(existing, child)
		} else {
			base[key] = value
		}
	}
}

func (model *Model) checkAudioFormats(metadata initiationMetadata) error {
	profile := model.Profile()
	for _, format := range []struct {
		wire, direction string
		rate            int
	}{
		{metadata.UserInputAudioFormat, "input", profile.AudioInputSampleRate},
		{metadata.AgentOutputAudioFormat, "output", profile.AudioOutputSampleRate},
	} {
		if format.wire != "" && format.wire != fmt.Sprintf("pcm_%d", format.rate) {
			return &realtime.Error{Provider: "elevenlabs", Model: model.agentID, Message: fmt.Sprintf(
				"the agent is configured for %s audio format %q, but the model profile expects mono PCM16 at %d Hz. "+
					"Declare the agent's actual sample rates with WithProfile (PCM formats only; ulaw_8000 telephony "+
					"agents are not supported)", format.direction, format.wire, format.rate,
			)}
		}
	}
	return nil
}

func (settings Settings) clone() Settings {
	settings.DynamicVariables = maps.Clone(settings.DynamicVariables)
	settings.CustomLLMExtraBody = cloneMap(settings.CustomLLMExtraBody)
	settings.ConfigOverride = cloneMap(settings.ConfigOverride)
	return settings
}

func (settings Settings) warn(message string) {
	if settings.WarningHandler != nil {
		settings.WarningHandler(message)
		return
	}
	slog.Warn(message)
}

func cloneMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	data, _ := json.Marshal(value)
	var cloned map[string]any
	_ = json.Unmarshal(data, &cloned)
	return cloned
}

// APIError reports a non-2xx response from the ElevenLabs API, including a rejected socket upgrade.
type APIError struct {
	// StatusCode is the HTTP status.
	StatusCode int
	// Body is the response body.
	Body string
	// Model is the agent ID.
	Model string
}

// Error describes the API failure.
func (err *APIError) Error() string {
	return fmt.Sprintf("elevenlabs: agent %q: API returned status %d: %s", err.Model, err.StatusCode, err.Body)
}

// IsModelAPIError marks API failures as provider failures.
func (*APIError) IsModelAPIError() bool { return true }

func readBody(response *http.Response) string {
	data, _ := io.ReadAll(io.LimitReader(response.Body, 1<<16))
	_ = response.Body.Close()
	return strings.TrimSpace(string(data))
}

// rest performs one REST call. A 404 is returned to callers that tolerate it.
func (model *Model) rest(ctx context.Context, method, path string, query url.Values, body any, output any) (int, error) {
	var reader io.Reader
	if body != nil {
		data, _ := json.Marshal(body)
		reader = bytes.NewReader(data)
	}
	target := model.baseURL + path
	if len(query) > 0 {
		target += "?" + query.Encode()
	}
	request, err := http.NewRequestWithContext(ctx, method, target, reader)
	if err != nil {
		return 0, err
	}
	request.Header.Set("xi-api-key", model.apiKey)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := model.client.Do(request)
	if err != nil {
		return 0, &realtime.Error{Provider: "elevenlabs", Model: model.agentID, Message: "could not reach the ElevenLabs API", Err: err}
	}
	if response.StatusCode < 200 || response.StatusCode > 299 {
		return response.StatusCode, &APIError{StatusCode: response.StatusCode, Body: readBody(response), Model: model.agentID}
	}
	defer func() { _ = response.Body.Close() }()
	if output == nil {
		return response.StatusCode, nil
	}
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		return response.StatusCode, &realtime.Error{Provider: "elevenlabs", Model: model.agentID, Message: "decode " + path, Err: err}
	}
	return response.StatusCode, nil
}
