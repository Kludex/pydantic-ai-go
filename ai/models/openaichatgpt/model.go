package openaichatgpt

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/oauthcreds"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

// Model calls the OpenAI Responses API with a signed-in user's ChatGPT plan.
// Plan usage serves a narrower dialect than the public API: requests stream
// with store=false, system prompts use the developer role, sampling, output
// limit, and service tier settings are dropped, token counting is unavailable,
// and web search is the only hosted tool. Deferred tools use local search.
type Model struct {
	*ai.ModelWrapper
	model   *openai.ResponsesModel
	manager *oauthcreds.Manager[Credentials]
}

type config struct {
	credentials      *Credentials
	credentialSource CredentialSource
	httpClient       *http.Client
	defaultSettings  ai.ModelSettings
}

// Option configures a Model.
type Option func(*config)

// WithCredentials keeps one credential set in memory. Rotated credentials do
// not survive the process. Use WithCredentialSource when they must be durable.
func WithCredentials(credentials Credentials) Option {
	return func(config *config) { config.credentials = &credentials }
}

// WithCredentialSource uses caller-owned storage for initial and rotated credentials.
func WithCredentialSource(source CredentialSource) Option {
	return func(config *config) { config.credentialSource = source }
}

// WithHTTPClient sets the caller-owned HTTP client. The model copies it and
// wraps its transport, injecting credentials only on HTTPS requests to
// api.openai.com, so the client can safely serve other destinations too.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = settings }
}

// NewModel creates a ChatGPT plan model such as gpt-6.1-sol. It never falls
// back to OPENAI_API_KEY: sign in with NewOAuthFlow first.
func NewModel(name string, options ...Option) (*Model, error) {
	configuration := config{httpClient: http.DefaultClient}
	for _, option := range options {
		option(&configuration)
	}
	switch {
	case name == "":
		return nil, errors.New("openai-chatgpt: model name must not be empty")
	case configuration.credentials != nil && configuration.credentialSource != nil:
		return nil, errors.New("openai-chatgpt: credentials and credential source are mutually exclusive")
	case configuration.credentials == nil && configuration.credentialSource == nil:
		return nil, errors.New("openai-chatgpt: credentials or a credential source are required: sign in with NewOAuthFlow first")
	case configuration.httpClient == nil:
		return nil, errors.New("openai-chatgpt: HTTP client must not be nil")
	}
	if configuration.credentials != nil {
		if err := configuration.credentials.validate(); err != nil {
			return nil, err
		}
	}

	base := configuration.httpClient.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	tokenClient := *configuration.httpClient
	tokenClient.Transport = base
	var source oauthcreds.Source[Credentials]
	if configuration.credentialSource != nil {
		source = configuration.credentialSource
	}
	manager := oauthcreds.New(configuration.credentials, source, &tokenClient, oauthcreds.Hooks[Credentials]{
		Refresh:     RefreshCredentials,
		Validate:    Credentials.validate,
		AccessToken: func(credentials Credentials) string { return credentials.AccessToken },
		Persistence: func(err error) error { return &CredentialsPersistenceError{err: err} },
	})
	requestClient := *configuration.httpClient
	requestClient.Transport = &oauthcreds.Transport[Credentials]{
		Base: base, Host: "api.openai.com", Manager: manager,
		Headers: func(credentials Credentials) map[string]string {
			return map[string]string{"Authorization": "Bearer " + credentials.AccessToken}
		},
	}
	delegate := openai.NewResponsesModel(name,
		openai.WithProvider(openai.ProviderConfig{
			Name: "openai-chatgpt", BaseURL: defaultBaseURL, HTTPClient: &requestClient,
		}),
		openai.WithChatCompatibility(openai.ChatCompatibility{DeveloperSystemPrompts: true}),
		openai.WithDeferredToolSupport(false),
		openai.WithDefaultSettings(configuration.defaultSettings),
	)
	return &Model{ModelWrapper: ai.WrapModel(delegate), model: delegate, manager: manager}, nil
}

// Credentials returns the credential set currently held in memory. A model
// with a credential source has none until its first request loads one.
func (model *Model) Credentials() (Credentials, error) {
	credentials, ok := model.manager.Current()
	if !ok {
		return Credentials{}, errors.New(
			"openai-chatgpt: credentials are unavailable before the credential source is loaded",
		)
	}
	return credentials, nil
}

// ModelProfile reports the inherited Responses output behavior. Plan usage
// offers no prompt caching to configure.
func (model *Model) ModelProfile() ai.ModelProfile { return model.model.ModelProfile() }

// SupportsNativeTool reports web search, the only hosted tool plan usage accepts.
func (model *Model) SupportsNativeTool(tool ai.NativeTool) bool {
	_, search := tool.(ai.WebSearchTool)
	return search && model.model.SupportsNativeTool(tool)
}

// Request drains the streaming-only endpoint into a complete response.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	events, err := model.StreamRequest(ctx, messages, params)
	if err != nil {
		return nil, err
	}
	return ai.CollectModelStream(events, params)
}

// StreamRequest streams one Responses request in the plan-usage dialect.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	if last := len(messages) - 1; last >= 0 {
		if response, ok := messages[last].(ai.ModelResponse); ok && response.State == ai.ModelResponseStateSuspended {
			return nil, errors.New(
				"openai-chatgpt: resuming a suspended run is not supported because plan usage requires store=false",
			)
		}
	}
	// Generic settings the route rejects are dropped for portability; explicit
	// provider settings in ExtraBody still reach it, so it reports them.
	params.Settings = params.Settings.Clone()
	params.Settings.MaxTokens = 0
	params.Settings.Temperature = nil
	params.Settings.TopP = nil
	params.Settings.ServiceTier = ""
	if params.Settings.ExtraBody == nil {
		params.Settings.ExtraBody = map[string]any{}
	}
	params.Settings.ExtraBody["store"] = false
	return model.model.StreamRequest(ctx, messages, params)
}

// CountTokens reports that plan usage serves only POST /v1/responses.
func (*Model) CountTokens(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (ai.Usage, error) {
	return ai.Usage{}, fmt.Errorf(
		"openai-chatgpt: server-side token counting is not available: %w", ai.ErrTokenCountingUnsupported,
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
