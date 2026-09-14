// Package xai implements images.Model against xAI's image generation gRPC API.
package xai

import (
	"crypto/tls"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/xai/internal/xaiapi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

const (
	defaultTarget      = "api.x.ai:443"
	defaultProviderURL = "https://api.x.ai/v1"
	maximumMessageSize = 20 << 20
)

// Model calls xAI's direct image generation gRPC API.
type Model struct {
	name        string
	providerURL string
	target      string
	apiKey      string
	connection  grpc.ClientConnInterface
	settings    images.Settings
	clientOnce  sync.Once
	client      xaiapi.ImageClient
	ownedConn   *grpc.ClientConn
	clientErr   error
}

// Option configures a Model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is XAI_API_KEY.
func WithAPIKey(apiKey string) Option { return func(model *Model) { model.apiKey = apiKey } }

// WithTarget sets the xAI-compatible gRPC target used by the model-owned connection.
func WithTarget(target string) Option {
	if strings.TrimSpace(target) == "" {
		panic("xai images: gRPC target must not be empty")
	}
	return func(model *Model) { model.target = target }
}

// WithProviderURL sets the endpoint identity used for pricing and telemetry.
func WithProviderURL(providerURL string) Option {
	return func(model *Model) { model.providerURL = strings.TrimRight(providerURL, "/") }
}

// WithClient sets a caller-owned gRPC connection. The model never closes it.
func WithClient(connection grpc.ClientConnInterface) Option {
	if connection == nil {
		panic("xai images: gRPC client connection must not be nil")
	}
	return func(model *Model) { model.connection = connection }
}

// WithDefaultSettings sets defaults overridden by each image request.
func WithDefaultSettings(settings images.Settings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.settings = settings.Clone() }
}

// NewModel creates an xAI Grok Imagine model.
func NewModel(name string, options ...Option) *Model {
	model := &Model{
		name: name, providerURL: defaultProviderURL, target: defaultTarget,
		apiKey: os.Getenv("XAI_API_KEY"),
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the image model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (*Model) ProviderName() string { return "xai" }

// ProviderURL returns the configured provider endpoint identity.
func (model *Model) ProviderURL() string { return model.providerURL }

// DefaultSettings returns detached model defaults.
func (model *Model) DefaultSettings() images.Settings { return model.settings.Clone() }

// Close releases only the default connection created by the model.
// A connection supplied through WithClient remains owned by the caller.
func (model *Model) Close() error {
	if model.ownedConn == nil {
		return nil
	}
	return model.ownedConn.Close()
}

func (model *Model) imageClient() (xaiapi.ImageClient, error) {
	model.clientOnce.Do(func() {
		connection := model.connection
		if connection == nil {
			model.ownedConn, model.clientErr = grpc.NewClient(
				model.target,
				grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})),
				grpc.WithDefaultCallOptions(
					grpc.MaxCallSendMsgSize(maximumMessageSize),
					grpc.MaxCallRecvMsgSize(maximumMessageSize),
				),
			)
			connection = model.ownedConn
		}
		if model.clientErr == nil {
			model.client = xaiapi.NewImageClient(connection)
		}
	})
	if model.clientErr != nil {
		return nil, fmt.Errorf("xai images: create gRPC client: %w", model.clientErr)
	}
	return model.client, nil
}
