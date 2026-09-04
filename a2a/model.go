package a2a

import (
	"context"
	"iter"
	"slices"
	"strings"

	protocol "github.com/a2aproject/a2a-go/a2a"

	ai "github.com/Kludex/pydantic-ai-go"
)

const modelProviderName = "a2a"

// ModelClient is the transport-agnostic subset of the official A2A client used by Model.
type ModelClient interface {
	SendMessage(ctx context.Context, params *protocol.MessageSendParams) (protocol.SendMessageResult, error)
	SendStreamingMessage(ctx context.Context, params *protocol.MessageSendParams) iter.Seq2[protocol.Event, error]
}

// ModelConfig controls requests to a remote A2A agent.
type ModelConfig struct {
	// ProviderURL identifies the remote agent endpoint in telemetry and persisted responses.
	ProviderURL string
	// SendConfig configures accepted output modes, returned history, and push delivery.
	// Model always replaces Blocking with true because ai.Model requires a terminal response.
	SendConfig *protocol.MessageSendConfig
	// Extensions lists negotiated extension URIs relevant to each message.
	Extensions []string
	// ReferenceTasks lists related remote task IDs relevant to each message.
	ReferenceTasks []protocol.TaskID
	// MessageMetadata contains detached extension metadata attached to each message.
	MessageMetadata map[string]any
	// RequestMetadata contains detached extension metadata attached to each send request.
	RequestMetadata map[string]any
}

// Model adapts a remote A2A agent to ai.Model and ai.StreamingModel.
type Model struct {
	name   string
	client ModelClient
	config ModelConfig
}

// NewModel creates a model backed by an official A2A client.
func NewModel(name string, client ModelClient, config ModelConfig) *Model {
	if strings.TrimSpace(name) == "" {
		panic("ai/a2a: model name must not be empty")
	}
	if nilModelClient(client) {
		panic("ai/a2a: model client must not be nil")
	}
	config.Extensions = slices.Clone(config.Extensions)
	config.ReferenceTasks = slices.Clone(config.ReferenceTasks)
	config.MessageMetadata = cloneModelMetadata(config.MessageMetadata)
	config.RequestMetadata = cloneModelMetadata(config.RequestMetadata)
	config.SendConfig = cloneSendConfig(config.SendConfig)
	return &Model{name: name, client: client, config: config}
}

// Name returns the remote agent identity used as the model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the stable A2A provider identity.
func (*Model) ProviderName() string { return modelProviderName }

// ProviderURL returns the configured remote agent endpoint.
func (model *Model) ProviderURL() string { return model.config.ProviderURL }

// ModelProfile uses prompted JSON because A2A does not define a structured-output request field.
func (*Model) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModePrompted}
}

// Request sends one blocking message to the remote agent.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	request, err := model.request(messages, params)
	if err != nil {
		return nil, err
	}
	result, err := model.client.SendMessage(ctx, request)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "send message", err)
	}
	return modelResponse(result, model.name, model.config.ProviderURL)
}

// StreamRequest streams message and artifact updates from the remote agent.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	request, err := model.request(messages, params)
	if err != nil {
		return nil, err
	}
	return model.stream(ctx, request), nil
}

var _ ai.StreamingModel = (*Model)(nil)
var _ ai.ModelProviderIdentity = (*Model)(nil)
var _ ai.ModelProfiler = (*Model)(nil)
