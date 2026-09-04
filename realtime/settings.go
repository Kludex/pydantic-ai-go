package realtime

import (
	"fmt"
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// ToolChoice controls whether and which function tools the model may call.
type ToolChoice string

const (
	// ToolChoiceAuto lets the provider choose whether to call a tool.
	ToolChoiceAuto ToolChoice = "auto"
	// ToolChoiceNone prevents function calls.
	ToolChoiceNone ToolChoice = "none"
	// ToolChoiceRequired requires at least one function call where supported.
	ToolChoiceRequired ToolChoice = "required"
)

// AudioRetention controls which raw audio streams are retained in portable history.
type AudioRetention string

const (
	// AudioRetentionTranscriptOnly drops raw audio after transcription.
	AudioRetentionTranscriptOnly AudioRetention = "transcript_only"
	// AudioRetentionInput additionally keeps user audio.
	AudioRetentionInput AudioRetention = "input_audio"
	// AudioRetentionOutput additionally keeps assistant audio.
	AudioRetentionOutput AudioRetention = "output_audio"
	// AudioRetentionAll keeps user and assistant audio.
	AudioRetentionAll AudioRetention = "all"
)

// OutputModality selects speech or plain text generation.
type OutputModality string

const (
	// OutputModalityAudio requests spoken output.
	OutputModalityAudio OutputModality = "audio"
	// OutputModalityText requests plain text output.
	OutputModalityText OutputModality = "text"
)

// TurnDetection configures cross-provider voice activity detection.
type TurnDetection struct {
	// Enabled selects automatic VAD instead of push-to-talk.
	Enabled bool
	// Sensitivity accepts low, medium, or high.
	Sensitivity string
	// PrefixPadding retains audio immediately before detected speech.
	PrefixPadding time.Duration
	// SilenceDuration controls how long silence must last to end a turn.
	SilenceDuration time.Duration
}

// ReconnectPolicy configures bounded transport reconnection.
type ReconnectPolicy struct {
	// MaxAttempts bounds dial attempts for one dropped connection.
	MaxAttempts int
	// MaxReconnects bounds successful reconnects for the session lifetime.
	MaxReconnects int
	// BaseDelay is the initial exponential backoff.
	BaseDelay time.Duration
	// MaxDelay caps exponential backoff.
	MaxDelay time.Duration
	// Jitter randomizes each backoff delay.
	Jitter bool
}

// Settings configures one realtime provider session.
type Settings struct {
	// MaxTokens bounds each model response. Zero uses the provider default.
	MaxTokens int
	// ParallelToolCalls controls concurrent provider function calls.
	ParallelToolCalls *bool
	// ToolChoice controls whether and which functions may be called.
	ToolChoice ToolChoice
	// InputTranscriptionModel selects a provider transcription model. An empty value disables it.
	InputTranscriptionModel *string
	// OutputModality selects speech or plain text.
	OutputModality OutputModality
	// Thinking configures provider reasoning where supported.
	Thinking ai.ThinkingLevel
	// TurnDetection configures automatic VAD or manual turns.
	TurnDetection *TurnDetection
	// HandshakeTimeout bounds initial protocol negotiation.
	HandshakeTimeout time.Duration
	// Reconnect enables bounded recovery from transport drops.
	Reconnect *ReconnectPolicy
	// Provider contains detached provider-prefixed settings.
	Provider map[string]any
}

func (settings Settings) normalized() (Settings, error) {
	if settings.MaxTokens < 0 {
		return Settings{}, fmt.Errorf("realtime: max tokens must not be negative")
	}
	if settings.OutputModality == "" {
		settings.OutputModality = OutputModalityAudio
	}
	if settings.OutputModality != OutputModalityAudio && settings.OutputModality != OutputModalityText {
		return Settings{}, fmt.Errorf("realtime: unsupported output modality %q", settings.OutputModality)
	}
	if settings.HandshakeTimeout < 0 {
		return Settings{}, fmt.Errorf("realtime: handshake timeout must not be negative")
	}
	if settings.HandshakeTimeout == 0 {
		settings.HandshakeTimeout = 30 * time.Second
	}
	if settings.TurnDetection != nil {
		turn := *settings.TurnDetection
		if turn.PrefixPadding < 0 || turn.SilenceDuration < 0 {
			return Settings{}, fmt.Errorf("realtime: turn detection durations must not be negative")
		}
		if turn.Sensitivity != "" && turn.Sensitivity != "low" && turn.Sensitivity != "medium" && turn.Sensitivity != "high" {
			return Settings{}, fmt.Errorf("realtime: unsupported turn detection sensitivity %q", turn.Sensitivity)
		}
		settings.TurnDetection = &turn
	}
	if settings.Reconnect != nil {
		policy := *settings.Reconnect
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
			return Settings{}, fmt.Errorf("realtime: invalid reconnect policy")
		}
		settings.Reconnect = &policy
	}
	settings.Provider = cloneAnyMap(settings.Provider)
	return settings, nil
}

func cloneAnyMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	cloned := make(map[string]any, len(values))
	for key, value := range values {
		cloned[key] = cloneAnyValue(value)
	}
	return cloned
}

func cloneAnyValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneAnyMap(value)
	case []any:
		cloned := make([]any, len(value))
		for index, item := range value {
			cloned[index] = cloneAnyValue(item)
		}
		return cloned
	case []string:
		return slices.Clone(value)
	case []byte:
		return slices.Clone(value)
	default:
		return value
	}
}
