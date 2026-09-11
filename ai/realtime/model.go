package realtime

import (
	"context"
	"fmt"
	"reflect"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Error reports a realtime transport or protocol failure.
type Error struct {
	// Provider identifies the failed realtime service.
	Provider string
	// Model identifies the requested realtime model.
	Model string
	// Message describes the failed operation.
	Message string
	// Err is the underlying transport or protocol error.
	Err error
}

// Error describes the realtime failure.
func (err *Error) Error() string {
	prefix := "realtime"
	if err.Provider != "" {
		prefix += " " + err.Provider
	}
	if err.Model != "" {
		prefix += "/" + err.Model
	}
	if err.Err != nil {
		return prefix + ": " + err.Message + ": " + err.Err.Error()
	}
	return prefix + ": " + err.Message
}

// Unwrap returns the underlying transport error.
func (err *Error) Unwrap() error { return err.Err }

// ConnectParams contains detached session initialization state.
type ConnectParams struct {
	// Messages seeds portable conversation history.
	Messages []ai.ModelMessage
	// Settings configures the provider session.
	Settings Settings
	// Request carries instructions and available tools.
	Request ai.ModelRequestParams
}

// Model opens persistent bidirectional connections.
type Model interface {
	// Name returns the requested model identifier.
	Name() string
	// ProviderName returns the provider identity used in response history.
	ProviderName() string
	// Profile returns detached model capabilities.
	Profile() Profile
	// Connect opens and configures one provider session.
	Connect(ctx context.Context, params ConnectParams) (Connection, error)
}

// ClientSecret is a short-lived browser credential.
type ClientSecret struct {
	// Value is the short-lived browser credential.
	Value string
	// ExpiresAt is the credential expiration time.
	ExpiresAt time.Time
	// ProviderDetails contains detached non-secret response fields.
	ProviderDetails map[string]any
}

// ProviderSession identifies an existing provider-side media session.
type ProviderSession interface {
	// ProviderName returns the service that owns the media session.
	ProviderName() string
	// SessionID returns the opaque provider session identifier.
	SessionID() string
}

// WebRTCSession identifies a provider-side WebRTC call.
type WebRTCSession struct {
	// Provider identifies the service that owns the call.
	Provider string
	// ID is the provider-assigned call identifier.
	ID string
	// Details contains detached signaling response metadata.
	Details map[string]any
}

// ProviderName returns the provider that owns the call.
func (session WebRTCSession) ProviderName() string { return session.Provider }

// SessionID returns the provider-assigned call identifier.
func (session WebRTCSession) SessionID() string { return session.ID }

// WebRTCAnswer contains the provider SDP answer and sideband session handle.
type WebRTCAnswer struct {
	// SDP is the provider answer returned to the browser.
	SDP string
	// Session identifies the call for server-side control.
	Session WebRTCSession
}

// WebRTCModel adds browser signaling and server-side sideband control.
type WebRTCModel interface {
	Model
	CreateClientSecret(
		ctx context.Context, instructions string, tools []ai.ToolDefinition, settings Settings, expiresAfter time.Duration,
	) (ClientSecret, error)
	AnswerWebRTCOffer(
		ctx context.Context, offer string, instructions string, tools []ai.ToolDefinition, settings Settings,
	) (WebRTCAnswer, error)
	ConnectWebRTC(ctx context.Context, session ProviderSession, params ConnectParams) (Connection, error)
}

func validateConnect(model Model, params ConnectParams) (ConnectParams, Profile, error) {
	if model == nil || (reflect.ValueOf(model).Kind() == reflect.Pointer && reflect.ValueOf(model).IsNil()) {
		return ConnectParams{}, Profile{}, fmt.Errorf("realtime: model must not be nil")
	}
	settings, err := params.Settings.normalized()
	if err != nil {
		return ConnectParams{}, Profile{}, err
	}
	profile := cloneProfile(model.Profile())
	if profile.AudioInputSampleRate <= 0 || profile.AudioOutputSampleRate <= 0 {
		return ConnectParams{}, Profile{}, fmt.Errorf("realtime: model profile sample rates must be positive")
	}
	if profile.ContextWindow < 0 {
		return ConnectParams{}, Profile{}, fmt.Errorf("realtime: model profile context window must not be negative")
	}
	if settings.OutputModality == OutputModalityText && !profile.SupportsTextOutput {
		return ConnectParams{}, Profile{}, fmt.Errorf("realtime: model %q does not support text output", model.Name())
	}
	if len(params.Messages) > 0 && !profile.SupportsSessionSeeding {
		return ConnectParams{}, Profile{}, fmt.Errorf("realtime: model %q does not support session seeding", model.Name())
	}
	for _, tool := range params.Request.NativeTools {
		if !profile.SupportedNativeTools[tool.Kind()] && !tool.IsOptional() {
			return ConnectParams{}, Profile{}, fmt.Errorf(
				"realtime: model %q does not support native tool %q", model.Name(), tool.Kind(),
			)
		}
	}
	detached := (ai.ModelRequestContext{Messages: params.Messages, Params: params.Request}).Clone()
	params.Messages = detached.Messages
	params.Settings = settings
	params.Request = detached.Params
	return params, profile, nil
}
