// Package bedrock provides text embeddings through Amazon Bedrock Runtime.
package bedrock

import (
	"context"
	"reflect"
	"strings"
	"sync"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

// Request describes one Bedrock InvokeModel operation.
type Request struct {
	// ModelID is the Bedrock model or inference-profile identifier.
	ModelID string
	// Body is the detached InvokeModel JSON payload.
	Body []byte
	// Headers contains detached provider-specific request headers.
	Headers map[string]string
}

// Response contains one Bedrock InvokeModel response.
type Response struct {
	// Body is the detached InvokeModel response payload.
	Body []byte
	// InputTokens is the provider-reported input usage.
	InputTokens int
}

// Client invokes Bedrock models. Implementations must be safe for concurrent calls.
type Client interface {
	InvokeModel(ctx context.Context, request Request) (Response, error)
}

// Option configures a Bedrock embedding model.
type Option func(*Model)

// WithClient uses client instead of the default AWS credential chain.
func WithClient(client Client) Option {
	if clientIsNil(client) {
		panic("bedrock embeddings: client must not be nil")
	}
	return func(model *Model) { model.client = client }
}

// WithProviderURL records the endpoint identity used by a custom client.
func WithProviderURL(providerURL string) Option {
	return func(model *Model) { model.setProviderURL(strings.TrimRight(providerURL, "/")) }
}

// WithSettings sets detached model defaults.
func WithSettings(settings embeddings.Settings) Option {
	return func(model *Model) { model.settings = settings.Clone() }
}

// Model generates embeddings with Titan, Cohere, and Nova models on Bedrock.
type Model struct {
	modelName        string
	family           family
	settings         embeddings.Settings
	client           Client
	providerURL      string
	providerURLMutex sync.RWMutex
	loadOptions      []func(*awsconfig.LoadOptions) error
	loadOnce         sync.Once
	loadErr          error
}

// NewModel creates a Bedrock embedding model. If no client is supplied, the first request loads the default AWS configuration.
func NewModel(modelName string, options ...Option) (*Model, error) {
	family, err := modelFamily(modelName)
	if err != nil {
		return nil, err
	}
	model := &Model{modelName: modelName, family: family}
	for _, option := range options {
		option(model)
	}
	return model, nil
}

// Name returns the configured Bedrock model ID.
func (model *Model) Name() string { return model.modelName }

// ProviderName returns the durable provider identity.
func (*Model) ProviderName() string { return "bedrock" }

// ProviderURL returns the configured Bedrock Runtime endpoint when known.
func (model *Model) ProviderURL() string {
	model.providerURLMutex.RLock()
	defer model.providerURLMutex.RUnlock()
	return model.providerURL
}

// MaxInputTokens reports a known model input limit.
func (model *Model) MaxInputTokens(context.Context) (int, bool, error) {
	limit, exists := maxInputTokens[normalizedModelName(model.modelName)]
	return limit, exists, nil
}

func (model *Model) setProviderURL(providerURL string) {
	model.providerURLMutex.Lock()
	defer model.providerURLMutex.Unlock()
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

var maxInputTokens = map[string]int{
	"amazon.titan-embed-text-v1": 8192, "amazon.titan-embed-text-v2:0": 8192,
	"cohere.embed-english-v3": 512, "cohere.embed-multilingual-v3": 512,
	"cohere.embed-v4:0": 128000, "amazon.nova-2-multimodal-embeddings-v1:0": 8192,
}
