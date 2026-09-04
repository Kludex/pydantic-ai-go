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
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// AudioRetention controls which raw audio streams are retained in portable history.
type AudioRetention string

const (
	AudioRetentionTranscriptOnly AudioRetention = "transcript_only"
	AudioRetentionInput          AudioRetention = "input_audio"
	AudioRetentionOutput         AudioRetention = "output_audio"
	AudioRetentionAll            AudioRetention = "all"
)

// OutputModality selects speech or plain text generation.
type OutputModality string

const (
	OutputModalityAudio OutputModality = "audio"
	OutputModalityText  OutputModality = "text"
)

// TurnDetection configures cross-provider voice activity detection.
type TurnDetection struct {
	Enabled         bool
	Sensitivity     string
	PrefixPadding   time.Duration
	SilenceDuration time.Duration
}

// ReconnectPolicy configures bounded transport reconnection.
type ReconnectPolicy struct {
	MaxAttempts   int
	MaxReconnects int
	BaseDelay     time.Duration
	MaxDelay      time.Duration
	Jitter        bool
}

// Settings configures one realtime provider session.
type Settings struct {
	MaxTokens               int
	ParallelToolCalls       *bool
	ToolChoice              ToolChoice
	InputTranscriptionModel *string
	OutputModality          OutputModality
	Thinking                ai.ThinkingLevel
	TurnDetection           *TurnDetection
	HandshakeTimeout        time.Duration
	Reconnect               *ReconnectPolicy
	Provider                map[string]any
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
