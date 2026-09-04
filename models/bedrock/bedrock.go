// Package bedrock implements ai.Model against Amazon Bedrock's Converse API.
package bedrock

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/middleware"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Client is the required Bedrock Runtime surface used by Model.
type Client interface {
	// Converse generates one complete response.
	Converse(
		ctx context.Context, input *bedrockruntime.ConverseInput, options ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error)
}

// EventStream is one Bedrock Converse response stream.
type EventStream interface {
	// Events returns provider events until the channel closes.
	Events() <-chan types.ConverseStreamOutput
	// Close releases the stream and must permit repeated calls.
	Close() error
	// Err returns the terminal stream-reader error.
	Err() error
}

// ResultMetadataEventStream exposes AWS operation metadata for a stream.
// Custom clients may implement it to preserve the provider request ID.
type ResultMetadataEventStream interface {
	EventStream
	// ResultMetadata returns detached AWS operation metadata.
	ResultMetadata() middleware.Metadata
}

// StreamingClient is the optional Bedrock Runtime streaming surface.
type StreamingClient interface {
	// ConverseStream starts one streaming response.
	ConverseStream(
		ctx context.Context, input *bedrockruntime.ConverseStreamInput, options ...func(*bedrockruntime.Options),
	) (EventStream, error)
}

// TokenCountingClient is the optional Bedrock Runtime token-counting surface.
type TokenCountingClient interface {
	// CountTokens returns the prospective input-token count.
	CountTokens(
		ctx context.Context, input *bedrockruntime.CountTokensInput, options ...func(*bedrockruntime.Options),
	) (*bedrockruntime.CountTokensOutput, error)
}

// Option configures a Model.
type Option func(*Model)

// WithClient uses a caller-owned concurrent-safe Bedrock Runtime client.
func WithClient(client Client) Option {
	if clientIsNil(client) {
		panic("bedrock: client must not be nil")
	}
	return func(model *Model) {
		if sdkClient, ok := client.(*bedrockruntime.Client); ok {
			model.client = &awsClient{client: sdkClient}
			return
		}
		model.client = client
	}
}

// WithProviderURL records the endpoint identity used by a custom client.
func WithProviderURL(providerURL string) Option {
	return func(model *Model) { model.setProviderURL(strings.TrimRight(providerURL, "/")) }
}

// WithAWSConfig uses an already loaded AWS SDK configuration.
func WithAWSConfig(config aws.Config) Option {
	config = config.Copy()
	if config.BaseEndpoint != nil {
		baseEndpoint := *config.BaseEndpoint
		config.BaseEndpoint = &baseEndpoint
	}
	return func(model *Model) {
		model.client = &awsClient{client: bedrockruntime.NewFromConfig(config)}
		model.setProviderURL(awsProviderURL(config))
	}
}

// WithAWSLoadOptions configures loading the default AWS SDK configuration.
// These options are ignored when a client or loaded configuration is supplied.
func WithAWSLoadOptions(options ...func(*awsconfig.LoadOptions) error) Option {
	cloned := append([]func(*awsconfig.LoadOptions) error(nil), options...)
	return func(model *Model) { model.loadOptions = cloned }
}

// WithDefaultSettings sets detached request defaults.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	return func(model *Model) { model.defaultSettings = settings.Clone() }
}

// Model calls Amazon Bedrock through the Converse API.
type Model struct {
	name            string
	client          Client
	providerURL     string
	providerURLMu   sync.RWMutex
	defaultSettings ai.ModelSettings
	loadOptions     []func(*awsconfig.LoadOptions) error
	loadOnce        sync.Once
	loadErr         error
}

// NewModel creates a Bedrock Converse model. If no client is supplied, the
// first operation loads the default AWS configuration.
func NewModel(name string, options ...Option) *Model {
	model := &Model{name: name}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the Bedrock model ID or inference-profile ARN.
func (model *Model) Name() string { return model.name }

// ProviderName returns Bedrock's durable provider identity.
func (*Model) ProviderName() string { return "bedrock" }

// SupportsNativeTool reports support for Nova code interpreter.
func (*Model) SupportsNativeTool(tool ai.NativeTool) bool {
	if err := ai.ValidateNativeTools([]ai.NativeTool{tool}); err != nil {
		return false
	}
	_, ok := tool.CloneNativeTool().(ai.CodeExecutionTool)
	return ok
}

// ProviderURL returns the configured Bedrock Runtime endpoint when known.
func (model *Model) ProviderURL() string {
	model.providerURLMu.RLock()
	defer model.providerURLMu.RUnlock()
	return model.providerURL
}

// DefaultModelSettings returns detached request defaults.
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }

// PromptCacheRetention reports the longest requested Bedrock cache lifetime.
func (*Model) PromptCacheRetention(settings ai.ModelSettings) (time.Duration, bool) {
	_, cache, _, err := extractSettings(settings)
	if err != nil {
		return 0, false
	}
	return cache.retention()
}

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	input, err := buildConverseInput(ctx, model.name, messages, params)
	if err != nil {
		return nil, err
	}
	client, err := model.resolveClient(ctx)
	if err != nil {
		return nil, err
	}
	output, err := client.Converse(ctx, input, requestOptions(params.Settings.ExtraHeaders))
	if err != nil {
		return nil, modelError(ctx, model, "request", err)
	}
	response, err := convertResponse(model, output)
	if err != nil {
		return nil, err
	}
	return response, nil
}

// CountTokens counts prospective input tokens through Bedrock Runtime.
func (model *Model) CountTokens(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (ai.Usage, error) {
	input, err := buildConverseInput(ctx, model.name, messages, params)
	if err != nil {
		return ai.Usage{}, err
	}
	client, err := model.resolveClient(ctx)
	if err != nil {
		return ai.Usage{}, err
	}
	counter, ok := client.(TokenCountingClient)
	if !ok {
		return ai.Usage{}, fmt.Errorf("%w by Bedrock client", ai.ErrTokenCountingUnsupported)
	}
	output, err := counter.CountTokens(ctx, countTokensInput(input), requestOptions(params.Settings.ExtraHeaders))
	if err != nil {
		return ai.Usage{}, modelError(ctx, model, "token count request", err)
	}
	if output == nil || output.InputTokens == nil {
		return ai.Usage{}, fmt.Errorf("bedrock: token count response omitted inputTokens")
	}
	return ai.Usage{InputTokens: int(*output.InputTokens)}, nil
}

func (model *Model) resolveClient(ctx context.Context) (Client, error) {
	model.loadOnce.Do(func() {
		if model.client != nil {
			return
		}
		config, err := awsconfig.LoadDefaultConfig(ctx, model.loadOptions...)
		if err != nil {
			model.loadErr = fmt.Errorf("bedrock: load AWS configuration: %w", err)
			return
		}
		model.client = &awsClient{client: bedrockruntime.NewFromConfig(config)}
		model.setProviderURL(awsProviderURL(config))
	})
	return model.client, model.loadErr
}

func (model *Model) setProviderURL(providerURL string) {
	model.providerURLMu.Lock()
	defer model.providerURLMu.Unlock()
	model.providerURL = providerURL
}

func clientIsNil(client Client) bool {
	if client == nil {
		return true
	}
	value := reflect.ValueOf(client)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
