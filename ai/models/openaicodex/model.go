package openaicodex

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

// Model uses the OpenAI Responses wire format with ChatGPT/Codex subscription authentication.
type Model struct {
	*ai.ModelWrapper
	model   *openai.ResponsesModel
	manager *credentialManager
}

type config struct {
	credentials      *Credentials
	credentialSet    bool
	credentialSource CredentialSource
	httpClient       *http.Client
	defaultSettings  ai.ModelSettings
}

// Option configures a Codex model.
type Option func(*config)

// WithCredentials keeps one credential set in memory. Rotated credentials do
// not survive the process. Use WithCredentialSource when they must be durable.
func WithCredentials(credentials Credentials) Option {
	return func(config *config) {
		value := credentials
		config.credentials = &value
		config.credentialSet = true
	}
}

// WithCredentialSource uses caller-owned storage for initial and rotated credentials.
func WithCredentialSource(source CredentialSource) Option {
	return func(config *config) { config.credentialSource = source }
}

// WithHTTPClient sets the caller-owned HTTP client. The model copies the client
// and wraps its transport without changing or closing the original.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = settings }
}

// NewModel creates a subscription-authenticated OpenAI Codex model. When no
// credential option is supplied, it reads the Codex CLI auth file without changing it.
func NewModel(name string, options ...Option) (*Model, error) {
	configuration := config{httpClient: http.DefaultClient}
	for _, option := range options {
		option(&configuration)
	}
	if name == "" {
		return nil, errors.New("openai-codex: model name must not be empty")
	}
	if configuration.credentialSet && configuration.credentialSource != nil {
		return nil, errors.New("openai-codex: credentials and credential source are mutually exclusive")
	}
	if configuration.httpClient == nil {
		return nil, errors.New("openai-codex: HTTP client must not be nil")
	}
	if configuration.credentialSet {
		if err := configuration.credentials.validate(); err != nil {
			return nil, errors.New("openai-codex: invalid credentials: " + err.Error())
		}
	} else if configuration.credentialSource == nil {
		credentials, err := LoadCodexCLICredentials()
		if err != nil {
			return nil, err
		}
		configuration.credentials = &credentials
	}

	baseTransport := configuration.httpClient.Transport
	if baseTransport == nil {
		baseTransport = http.DefaultTransport
	}
	tokenClient := *configuration.httpClient
	tokenClient.Transport = baseTransport
	manager := newCredentialManager(configuration.credentials, configuration.credentialSource, &tokenClient)
	requestClient := *configuration.httpClient
	requestClient.Transport = &codexTransport{base: baseTransport, manager: manager}
	delegate := openai.NewResponsesModel(name,
		openai.WithProvider(openai.ProviderConfig{
			Name: "openai-codex", BaseURL: defaultBaseURL, HTTPClient: &requestClient,
		}),
		openai.WithDefaultSettings(configuration.defaultSettings),
	)
	return &Model{ModelWrapper: ai.WrapModel(delegate), model: delegate, manager: manager}, nil
}

// Credentials returns the credential set currently held in memory. A model
// with a credential source has no current set until its first request loads one.
func (model *Model) Credentials() (Credentials, error) {
	credentials, ok := model.manager.current()
	if !ok {
		return Credentials{}, errors.New("openai-codex: credentials are unavailable before the credential source is loaded")
	}
	return credentials, nil
}

// ModelProfile reports the Responses output capabilities inherited from OpenAI.
func (model *Model) ModelProfile() ai.ModelProfile { return model.model.ModelProfile() }

// SupportsNativeTool reports native Responses tools accepted by Codex.
func (model *Model) SupportsNativeTool(tool ai.NativeTool) bool {
	return model.model.SupportsNativeTool(tool)
}

// Request drains the streaming-only Codex endpoint into a complete response.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	if historyEndsWithSuspendedCodexResponse(messages) {
		return nil, errors.New(
			"openai-codex: resuming a suspended run is not supported because Codex requires store=false",
		)
	}
	messages, params = prepareRequest(messages, params)
	events, err := model.model.StreamRequest(ctx, messages, params)
	if err != nil {
		return nil, err
	}
	return ai.CollectModelStream(events, params)
}

// StreamRequest streams one Codex Responses request.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	if historyEndsWithSuspendedCodexResponse(messages) {
		return nil, errors.New(
			"openai-codex: resuming a suspended run is not supported because Codex requires store=false",
		)
	}
	messages, params = prepareRequest(messages, params)
	return model.model.StreamRequest(ctx, messages, params)
}

// CountTokens reports that the Codex subscription backend does not expose the Responses token-count endpoint.
func (model *Model) CountTokens(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (ai.Usage, error) {
	return ai.Usage{}, fmt.Errorf(
		"openai-codex: server-side token counting is not available: %w", ai.ErrTokenCountingUnsupported,
	)
}

var (
	_ ai.Model                  = (*Model)(nil)
	_ ai.StreamingModel         = (*Model)(nil)
	_ ai.TokenCountingModel     = (*Model)(nil)
	_ ai.ModelDefaultSettings   = (*Model)(nil)
	_ ai.ModelProviderIdentity  = (*Model)(nil)
	_ ai.ModelProfiler          = (*Model)(nil)
	_ ai.NativeToolSupportModel = (*Model)(nil)
)
