// Package snowflake implements ai.Model against Snowflake Cortex Chat Completions.
package snowflake

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"regexp"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const invalidBaseURL = "https://invalid.invalid/api/v2/cortex/v1"

// Model calls Snowflake Cortex through its OpenAI-compatible API.
type Model struct {
	*ai.ModelWrapper
	model     *openai.Model
	family    modelFamily
	configErr error
}

type modelFamily string

const (
	familyClaude modelFamily = "claude"
	familyOpenAI modelFamily = "openai"
	familyOther  modelFamily = "other"
)

type config struct {
	account         string
	token           string
	baseURL         string
	provider        *openai.ProviderConfig
	httpClient      *http.Client
	defaultSettings *ai.ModelSettings
}

// Option configures a Snowflake model.
type Option func(*config)

// WithAccount sets the Snowflake account identifier used to construct the Cortex endpoint.
func WithAccount(account string) Option {
	if err := validateAccount(account); err != nil {
		panic(err)
	}
	return func(config *config) {
		config.account = normalizeAccount(account)
		config.baseURL = ""
	}
}

// WithToken sets the Snowflake bearer token. The default is SNOWFLAKE_TOKEN.
func WithToken(token string) Option { return func(config *config) { config.token = token } }

// WithBaseURL sets a custom Cortex API root.
func WithBaseURL(baseURL string) Option {
	if baseURL == "" {
		panic("snowflake: base URL must not be empty")
	}
	return func(config *config) { config.baseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithProvider configures a gateway while retaining Snowflake model semantics.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	return func(config *config) { config.provider = &provider }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = &settings }
}

// NewProviderConfig resolves Snowflake account and token environment configuration.
func NewProviderConfig() (openai.ProviderConfig, error) {
	configuration := config{
		account: os.Getenv("SNOWFLAKE_ACCOUNT"), token: os.Getenv("SNOWFLAKE_TOKEN"),
		baseURL: os.Getenv("SNOWFLAKE_BASE_URL"),
	}
	return configuration.providerConfig()
}

// NewModel creates a Snowflake Cortex model.
func NewModel(name string, options ...Option) *Model {
	configuration := config{
		account: os.Getenv("SNOWFLAKE_ACCOUNT"), token: os.Getenv("SNOWFLAKE_TOKEN"),
		baseURL: os.Getenv("SNOWFLAKE_BASE_URL"),
	}
	for _, option := range options {
		option(&configuration)
	}
	provider, err := configuration.providerConfig()
	if err != nil {
		provider = openai.ProviderConfig{Name: "snowflake", BaseURL: invalidBaseURL}
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(provider), openai.WithStrictToolSupport(false),
		openai.WithDeferredToolSupport(false),
		openai.WithChatCompatibility(openai.ChatCompatibility{ReasoningDetails: true}),
	}
	if configuration.defaultSettings != nil {
		openAIOptions = append(openAIOptions, openai.WithDefaultSettings(*configuration.defaultSettings))
	}
	underlying := openai.NewModel(name, openAIOptions...)
	return &Model{
		ModelWrapper: ai.WrapModel(underlying), model: underlying,
		family: snowflakeModelFamily(name), configErr: err,
	}
}

// ModelProfile selects prompted output for Cortex families without function tools.
func (model *Model) ModelProfile() ai.ModelProfile {
	mode := ai.OutputModeTool
	if model.family == familyOther {
		mode = ai.OutputModePrompted
	}
	return ai.ModelProfile{DefaultOutputMode: mode}
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	params, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	response, err := model.model.Request(ctx, messages, params)
	if response != nil {
		model.normalizeResponse(response)
	}
	return response, err
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	params, err := model.prepareParams(params)
	if err != nil {
		return nil, err
	}
	stream, err := model.model.StreamRequest(ctx, messages, params)
	if err != nil {
		return nil, err
	}
	return model.normalizeStream(stream), nil
}

func (config config) providerConfig() (openai.ProviderConfig, error) {
	if config.provider != nil {
		provider := *config.provider
		provider.Headers = provider.Headers.Clone()
		return provider, nil
	}
	baseURL := strings.TrimRight(config.baseURL, "/")
	if baseURL == "" {
		if err := validateAccount(config.account); err != nil {
			return openai.ProviderConfig{}, err
		}
		baseURL = "https://" + normalizeAccount(config.account) + ".snowflakecomputing.com/api/v2/cortex/v1"
	}
	if config.token == "" {
		return openai.ProviderConfig{}, fmt.Errorf("snowflake: set SNOWFLAKE_TOKEN or use WithToken")
	}
	return openai.ProviderConfig{
		Name: "snowflake", BaseURL: baseURL, APIKey: config.token, HTTPClient: config.httpClient,
	}, nil
}

func validateAccount(account string) error {
	account = normalizeAccount(account)
	if account == "" {
		return fmt.Errorf("snowflake: set SNOWFLAKE_ACCOUNT or use WithAccount or WithBaseURL")
	}
	if !regexp.MustCompile(`^[A-Za-z0-9._-]+$`).MatchString(account) {
		return fmt.Errorf("snowflake: invalid account identifier %q", account)
	}
	return nil
}

func normalizeAccount(account string) string {
	account = strings.TrimSuffix(strings.TrimRight(account, "/"), ".snowflakecomputing.com")
	account = strings.TrimPrefix(strings.TrimPrefix(account, "https://"), "http://")
	return account
}

func snowflakeModelFamily(name string) modelFamily {
	name = strings.ToLower(name)
	switch {
	case strings.HasPrefix(name, "claude"):
		return familyClaude
	case strings.HasPrefix(name, "openai-"):
		return familyOpenAI
	default:
		return familyOther
	}
}
