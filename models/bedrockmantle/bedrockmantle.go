// Package bedrockmantle implements ai.Model against Amazon Bedrock Mantle's OpenAI-compatible APIs.
package bedrockmantle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

const invalidBaseURL = "https://invalid.invalid/v1"

type modelInterface string

const (
	interfaceChat            modelInterface = "chat"
	interfaceResponses       modelInterface = "responses"
	interfaceOpenAIResponses modelInterface = "openai-responses"
)

// Model routes an OpenAI model to its Bedrock Mantle endpoint family.
type Model struct {
	name            string
	providerURL     string
	delegate        ai.Model
	streaming       ai.StreamingModel
	interfaceName   modelInterface
	defaultSettings ai.ModelSettings
	configErr       error
	bearerToken     string
	region          string
	awsConfig       *aws.Config
	loadOptions     []func(*awsconfig.LoadOptions) error
	loadOnce        sync.Once
	loadErr         error
}

type config struct {
	region          string
	baseURL         string
	bearerToken     string
	httpClient      *http.Client
	awsConfig       *aws.Config
	loadOptions     []func(*awsconfig.LoadOptions) error
	provider        *openai.ProviderConfig
	defaultSettings ai.ModelSettings
}

// Option configures a Bedrock Mantle model.
type Option func(*config)

// WithRegion sets the AWS region used for endpoint construction and SigV4 authentication.
func WithRegion(region string) Option { return func(config *config) { config.region = region } }

// WithBaseURL sets a custom Mantle origin or either Mantle API root.
func WithBaseURL(baseURL string) Option { return func(config *config) { config.baseURL = baseURL } }

// WithAPIKey sets the Bedrock bearer token. The default is AWS_BEARER_TOKEN_BEDROCK.
func WithAPIKey(token string) Option {
	return func(config *config) {
		config.bearerToken = token
		if token != "" {
			config.awsConfig = nil
			config.loadOptions = nil
		}
	}
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithAWSConfig uses an already loaded AWS configuration for SigV4 authentication.
func WithAWSConfig(loaded aws.Config) Option {
	loaded = loaded.Copy()
	return func(config *config) {
		config.awsConfig = &loaded
		config.bearerToken = ""
	}
}

// WithAWSLoadOptions configures loading default AWS credentials for SigV4 authentication.
func WithAWSLoadOptions(options ...func(*awsconfig.LoadOptions) error) Option {
	cloned := append([]func(*awsconfig.LoadOptions) error(nil), options...)
	return func(config *config) {
		config.loadOptions = cloned
		config.bearerToken = ""
		config.awsConfig = nil
	}
}

// WithProvider configures a gateway while retaining Mantle routing and response semantics.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	return func(config *config) { config.provider = &provider }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = settings }
}

// NewModel creates a Bedrock Mantle model and selects Chat Completions or Responses from its model ID.
func NewModel(name string, options ...Option) *Model {
	configuration := config{
		region: os.Getenv("AWS_DEFAULT_REGION"), bearerToken: os.Getenv("AWS_BEARER_TOKEN_BEDROCK"),
	}
	if configuration.region == "" {
		configuration.region = os.Getenv("AWS_REGION")
	}
	for _, option := range options {
		option(&configuration)
	}
	if configuration.region == "" && configuration.awsConfig != nil {
		configuration.region = configuration.awsConfig.Region
	}
	model := &Model{name: name, defaultSettings: configuration.defaultSettings.Clone()}
	providerName, bareName := splitModelID(name)
	if providerName != "openai" {
		model.configErr = fmt.Errorf("bedrock-mantle: model %q is not an OpenAI Bedrock model ID", name)
	}
	model.interfaceName = interfaceForModel(bareName)
	origin := mantleOrigin(configuration.baseURL)
	if configuration.provider != nil {
		origin = mantleOrigin(configuration.provider.BaseURL)
	} else if origin == "" && configuration.region != "" {
		origin = "https://bedrock-mantle." + configuration.region + ".api.aws"
	}
	if origin == "" && model.configErr == nil {
		model.configErr = fmt.Errorf("bedrock-mantle: set AWS_DEFAULT_REGION, AWS_REGION, WithRegion, or WithBaseURL")
	}
	baseURL := origin + "/v1"
	if model.interfaceName == interfaceOpenAIResponses {
		baseURL = origin + "/openai/v1"
	}
	if model.configErr != nil {
		baseURL = invalidBaseURL
	}
	provider := openai.ProviderConfig{
		Name: "bedrock-mantle", BaseURL: baseURL, APIKey: configuration.bearerToken,
		HTTPClient: configuration.httpClient,
	}
	if configuration.provider != nil {
		provider = *configuration.provider
		provider.Name = "bedrock-mantle"
		provider.BaseURL = baseURL
		provider.Headers = provider.Headers.Clone()
	}
	model.providerURL = baseURL
	model.bearerToken = provider.APIKey
	model.region = configuration.region
	model.awsConfig = configuration.awsConfig
	model.loadOptions = append([]func(*awsconfig.LoadOptions) error(nil), configuration.loadOptions...)
	if model.bearerToken == "" && configuration.provider == nil {
		provider.PrepareRequest = model.signRequest
	}
	common := []openai.Option{
		openai.WithProvider(provider), openai.WithDefaultSettings(model.defaultSettings),
		openai.WithStrictToolSupport(true), openai.WithDeferredToolSupport(false),
	}
	if model.interfaceName == interfaceChat {
		delegate := openai.NewModel(name, common...)
		model.delegate, model.streaming = delegate, delegate
	} else {
		delegate := openai.NewResponsesModel(name, common...)
		model.delegate, model.streaming = delegate, delegate
	}
	return model
}

// Name returns the Bedrock Mantle model ID.
func (model *Model) Name() string { return model.name }

// ProviderName returns Bedrock Mantle's durable provider identity.
func (*Model) ProviderName() string { return "bedrock-mantle" }

// ProviderURL returns the selected Mantle endpoint family.
func (model *Model) ProviderURL() string { return model.providerURL }

// DefaultModelSettings returns detached request defaults.
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }

// ModelProfile reports native structured output without image output.
func (*Model) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool}
}

// SupportsNativeTool reports that portable OpenAI-native tools are not supported by Mantle.
func (*Model) SupportsNativeTool(ai.NativeTool) bool { return false }

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	if err := model.validateRequest(params); err != nil {
		return nil, err
	}
	response, err := model.delegate.Request(ctx, messages, params)
	if response != nil && model.interfaceName == interfaceOpenAIResponses {
		qualifyResponseToolCallIDs(response)
	}
	return response, err
}

// StreamRequest implements ai.StreamingModel.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	if err := model.validateRequest(params); err != nil {
		return nil, err
	}
	stream, err := model.streaming.StreamRequest(ctx, messages, params)
	if err != nil || model.interfaceName != interfaceOpenAIResponses {
		return stream, err
	}
	return qualifyStreamToolCallIDs(stream), nil
}

func (model *Model) validateRequest(params ai.ModelRequestParams) error {
	if model.configErr != nil {
		return model.configErr
	}
	if params.AllowImageOutput {
		return fmt.Errorf("bedrock-mantle: image output is not supported")
	}
	if len(params.NativeTools) > 0 {
		return fmt.Errorf("bedrock-mantle: portable native tools are not supported")
	}
	return nil
}

func (model *Model) signRequest(request *http.Request) error {
	model.loadOnce.Do(func() {
		if model.awsConfig != nil {
			return
		}
		loaded, err := awsconfig.LoadDefaultConfig(request.Context(), model.loadOptions...)
		if err != nil {
			model.loadErr = fmt.Errorf("bedrock-mantle: load AWS configuration: %w", err)
			return
		}
		model.awsConfig = &loaded
	})
	if model.loadErr != nil {
		return model.loadErr
	}
	region := model.region
	if region == "" {
		region = model.awsConfig.Region
	}
	credentials, err := model.awsConfig.Credentials.Retrieve(request.Context())
	if err != nil {
		return fmt.Errorf("bedrock-mantle: retrieve AWS credentials: %w", err)
	}
	body, _ := io.ReadAll(request.Body)
	request.Body = io.NopCloser(strings.NewReader(string(body)))
	hash := sha256.Sum256(body)
	return v4.NewSigner().SignHTTP(
		request.Context(), credentials, request, hex.EncodeToString(hash[:]), "bedrock", region, time.Now(),
	)
}

func mantleOrigin(baseURL string) string {
	origin := strings.TrimRight(baseURL, "/")
	for _, suffix := range []string{"/openai/v1", "/v1"} {
		origin = strings.TrimSuffix(origin, suffix)
	}
	return origin
}

func splitModelID(name string) (string, string) {
	for _, prefix := range []string{"us", "eu", "apac", "jp", "au", "ca", "global", "us-gov"} {
		name = strings.TrimPrefix(name, prefix+".")
	}
	provider, bare, found := strings.Cut(name, ".")
	if !found {
		return "", name
	}
	return provider, bare
}

func interfaceForModel(name string) modelInterface {
	switch {
	case strings.HasPrefix(name, "gpt-oss-safeguard"):
		return interfaceChat
	case strings.HasPrefix(name, "gpt-oss"):
		return interfaceResponses
	default:
		return interfaceOpenAIResponses
	}
}
