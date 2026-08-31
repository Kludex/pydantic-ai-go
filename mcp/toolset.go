// Package mcp connects Model Context Protocol servers as agent toolsets or caller-managed sessions.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	// ToolMetadataKey contains the MCP tool's detached _meta object.
	ToolMetadataKey = "mcp_meta"
	// ToolAnnotationsMetadataKey contains the MCP tool annotation hints.
	ToolAnnotationsMetadataKey = "mcp_annotations"
)

// ToolErrorBehavior controls how MCP protocol and tool errors affect the run.
type ToolErrorBehavior string

const (
	// ToolErrorRetry asks the model to call the tool again with corrected input.
	ToolErrorRetry ToolErrorBehavior = "retry"
	// ToolErrorFailed records a terminal failed tool result for the model.
	ToolErrorFailed ToolErrorBehavior = "failed"
	// ToolErrorAbort ends the agent run with the MCP error.
	ToolErrorAbort ToolErrorBehavior = "error"
)

// TransportFactory creates one MCP transport for each agent run.
type TransportFactory[Deps any] func(
	ctx context.Context, rc *ai.RunContext[Deps],
) (mcpsdk.Transport, error)

// Option configures MCP sessions and toolsets.
type Option func(*config)

type config struct {
	id               string
	implementation   *mcpsdk.Implementation
	clientOptions    *mcpsdk.ClientOptions
	samplingModel    ai.Model
	samplingHandler  SamplingHandler
	elicitation      ElicitationHandler
	oauthHandler     OAuthHandler
	sessionOptions   *mcpsdk.ClientSessionOptions
	initTimeout      time.Duration
	readTimeout      time.Duration
	errorBehavior    ToolErrorBehavior
	preferStructured bool
}

// WithID gives the toolset a stable identity for tracing and durable execution.
func WithID(id string) Option {
	return func(config *config) { config.id = id }
}

// WithImplementation sets the MCP client identity sent during initialization.
func WithImplementation(implementation *mcpsdk.Implementation) Option {
	return func(config *config) { config.implementation = implementation }
}

// WithClientOptions configures server-initiated MCP features and notifications.
func WithClientOptions(options *mcpsdk.ClientOptions) Option {
	return func(config *config) {
		if options == nil {
			config.clientOptions = nil
			return
		}
		cloned := *options
		cloned.Capabilities = cloneProtocol(options.Capabilities)
		if options.MultiRoundTrip != nil {
			multiRoundTrip := *options.MultiRoundTrip
			cloned.MultiRoundTrip = &multiRoundTrip
		}
		config.clientOptions = &cloned
	}
}

// WithSessionOptions configures each MCP client session.
func WithSessionOptions(options *mcpsdk.ClientSessionOptions) Option {
	return func(config *config) { config.sessionOptions = options }
}

// WithInitTimeout bounds transport connection and MCP initialization.
func WithInitTimeout(timeout time.Duration) Option {
	if timeout <= 0 {
		panic(fmt.Sprintf("ai/mcp: init timeout must be positive, got %s", timeout))
	}
	return func(config *config) { config.initTimeout = timeout }
}

// WithReadTimeout bounds tool discovery and each tool call.
func WithReadTimeout(timeout time.Duration) Option {
	if timeout <= 0 {
		panic(fmt.Sprintf("ai/mcp: read timeout must be positive, got %s", timeout))
	}
	return func(config *config) { config.readTimeout = timeout }
}

// WithToolErrorBehavior selects retry, failed-result, or abort behavior.
func WithToolErrorBehavior(behavior ToolErrorBehavior) Option {
	if behavior != ToolErrorRetry && behavior != ToolErrorFailed && behavior != ToolErrorAbort {
		panic(fmt.Sprintf("ai/mcp: invalid tool error behavior %q", behavior))
	}
	return func(config *config) { config.errorBehavior = behavior }
}

// WithPreferStructuredContent uses structuredContent even when an MCP result
// also contains non-text content.
func WithPreferStructuredContent(prefer bool) Option {
	return func(config *config) { config.preferStructured = prefer }
}

// Toolset exposes one MCP server through ai.Toolset. It creates an isolated
// MCP session for each agent run and is safe to reuse across agents.
type Toolset[Deps any] struct {
	factory TransportFactory[Deps]
	config  config
	client  *mcpsdk.Client
}

// NewToolset creates a reusable MCP toolset from a per-run transport factory.
func NewToolset[Deps any](factory TransportFactory[Deps], opts ...Option) *Toolset[Deps] {
	if factory == nil {
		panic("ai/mcp: transport factory must not be nil")
	}
	cfg := newConfig(opts)
	return &Toolset[Deps]{
		factory: factory, config: cfg,
		client: mcpsdk.NewClient(cfg.implementation, cfg.clientOptions),
	}
}

func newConfig(options []Option) config {
	cfg := config{
		implementation: &mcpsdk.Implementation{Name: "pydantic-ai-go", Version: "dev"},
		initTimeout:    5 * time.Second, readTimeout: 5 * time.Minute, errorBehavior: ToolErrorRetry,
	}
	for _, option := range options {
		option(&cfg)
	}
	if cfg.implementation == nil {
		cfg.implementation = &mcpsdk.Implementation{Name: "pydantic-ai-go", Version: "dev"}
	} else {
		cfg.implementation = cloneProtocol(cfg.implementation)
	}
	if cfg.sessionOptions != nil {
		sessionOptions := *cfg.sessionOptions
		cfg.sessionOptions = &sessionOptions
	}
	cfg.clientOptions = prepareClientOptions(cfg)
	return cfg
}

// NewStreamableHTTPToolset creates an MCP toolset for the recommended HTTP transport.
func NewStreamableHTTPToolset[Deps any](endpoint string, opts ...Option) *Toolset[Deps] {
	return NewToolset(func(context.Context, *ai.RunContext[Deps]) (mcpsdk.Transport, error) {
		return &mcpsdk.StreamableClientTransport{Endpoint: endpoint}, nil
	}, opts...)
}

// NewSSEToolset creates an MCP toolset for the deprecated HTTP+SSE transport.
func NewSSEToolset[Deps any](endpoint string, opts ...Option) *Toolset[Deps] {
	return NewToolset(func(context.Context, *ai.RunContext[Deps]) (mcpsdk.Transport, error) {
		return &mcpsdk.SSEClientTransport{Endpoint: endpoint}, nil
	}, opts...)
}

// NewCommandToolset creates an MCP toolset that starts a subprocess for each run.
func NewCommandToolset[Deps any](command string, args []string, opts ...Option) *Toolset[Deps] {
	return NewToolset(func(context.Context, *ai.RunContext[Deps]) (mcpsdk.Transport, error) {
		return &mcpsdk.CommandTransport{Command: exec.Command(command, args...)}, nil
	}, opts...)
}

// ToolsetID returns the configured stable identity.
func (t *Toolset[Deps]) ToolsetID() string { return t.config.id }

// ForRun creates an isolated transport before the run opens its resources.
func (t *Toolset[Deps]) ForRun(
	ctx context.Context, rc *ai.RunContext[Deps],
) (ai.Toolset[Deps], error) {
	transport, err := t.factory(ctx, rc)
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: create transport: %w", err)
	}
	if transportIsNil(transport) {
		return nil, errors.New("ai/mcp: transport factory returned nil")
	}
	transport, err = applyOAuthHandler(transport, t.config.oauthHandler)
	if err != nil {
		return nil, err
	}
	return &runToolset[Deps]{owner: t, transport: transport}, nil
}

// Tools returns an error because MCP tools require an open run-scoped session.
func (*Toolset[Deps]) Tools(context.Context, *ai.RunContext[Deps]) ([]ai.Tool[Deps], error) {
	return nil, errors.New("ai/mcp: toolset is not open")
}

type runToolset[Deps any] struct {
	owner     *Toolset[Deps]
	transport mcpsdk.Transport
	session   *mcpsdk.ClientSession
}

func (t *runToolset[Deps]) ToolsetID() string { return t.owner.config.id }

func (t *runToolset[Deps]) OpenToolset(
	ctx context.Context, _ *ai.RunContext[Deps],
) (ai.Toolset[Deps], ai.ToolsetCloseFunc, error) {
	connectCtx, cancel := context.WithTimeout(ctx, t.owner.config.initTimeout)
	defer cancel()
	session, err := t.owner.client.Connect(connectCtx, t.transport, t.owner.config.sessionOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("ai/mcp: connect: %w", err)
	}
	t.session = session
	return t, func(context.Context) error { return session.Close() }, nil
}

func (t *runToolset[Deps]) Tools(ctx context.Context, _ *ai.RunContext[Deps]) ([]ai.Tool[Deps], error) {
	if t.session == nil {
		return nil, errors.New("ai/mcp: toolset is not open")
	}
	readCtx, cancel := context.WithTimeout(ctx, t.owner.config.readTimeout)
	defer cancel()
	var tools []ai.Tool[Deps]
	for tool, err := range t.session.Tools(readCtx, nil) {
		if err != nil {
			return nil, fmt.Errorf("ai/mcp: list tools: %w", err)
		}
		tools, err = t.appendTool(tools, tool)
		if err != nil {
			return nil, err
		}
	}
	return tools, nil
}

func (t *runToolset[Deps]) ToolsetInstructions(
	context.Context, *ai.RunContext[Deps],
) ([]ai.InstructionPart, error) {
	if t.session == nil {
		return nil, errors.New("ai/mcp: toolset is not open")
	}
	result := t.session.InitializeResult()
	if result == nil || result.Instructions == "" {
		return nil, nil
	}
	return []ai.InstructionPart{{Content: result.Instructions, Dynamic: true}}, nil
}

func (t *runToolset[Deps]) appendTool(tools []ai.Tool[Deps], tool *mcpsdk.Tool) ([]ai.Tool[Deps], error) {
	converted, err := t.convertTool(tool)
	if err != nil {
		return nil, err
	}
	return append(tools, converted), nil
}

func (t *runToolset[Deps]) convertTool(tool *mcpsdk.Tool) (ai.Tool[Deps], error) {
	if tool == nil || tool.Name == "" {
		return ai.Tool[Deps]{}, errors.New("ai/mcp: server returned a tool without a name")
	}
	schema, err := schemaMap(tool.InputSchema)
	if err != nil {
		return ai.Tool[Deps]{}, fmt.Errorf("ai/mcp: tool %q input schema: %w", tool.Name, err)
	}
	returnSchema, err := schemaMap(tool.OutputSchema)
	if err != nil {
		return ai.Tool[Deps]{}, fmt.Errorf("ai/mcp: tool %q output schema: %w", tool.Name, err)
	}
	metadata := map[string]any{}
	if len(tool.Meta) > 0 {
		metadata[ToolMetadataKey] = cloneJSON(tool.Meta)
	}
	if tool.Annotations != nil {
		metadata[ToolAnnotationsMetadataKey] = cloneJSON(tool.Annotations)
	}
	definition := ai.ToolDefinition{
		Name: tool.Name, Description: tool.Description, Schema: schema, ReturnSchema: returnSchema,
		Metadata: metadata,
	}
	name := tool.Name
	return ai.NewRawTool[Deps](definition, func(ctx context.Context, rawArgs json.RawMessage) (any, error) {
		return t.callTool(ctx, name, rawArgs)
	}), nil
}

func (t *runToolset[Deps]) callTool(ctx context.Context, name string, rawArgs json.RawMessage) (any, error) {
	var arguments any
	if len(rawArgs) > 0 {
		if err := json.Unmarshal(rawArgs, &arguments); err != nil {
			return nil, fmt.Errorf("ai/mcp: decode tool %q arguments: %w", name, err)
		}
	}
	readCtx, cancel := context.WithTimeout(ctx, t.owner.config.readTimeout)
	defer cancel()
	result, err := t.session.CallTool(readCtx, &mcpsdk.CallToolParams{Name: name, Arguments: arguments})
	if err != nil {
		return nil, t.toolError(name, err)
	}
	return t.mapToolCallResult(name, result)
}

func (t *runToolset[Deps]) mapToolCallResult(name string, result *mcpsdk.CallToolResult) (any, error) {
	value, err := mapCallToolResult(result, t.owner.config.preferStructured)
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: tool %q result: %w", name, err)
	}
	if result.IsError {
		return nil, t.toolError(name, errors.New(formatToolResult(value)))
	}
	return ai.ToolReturn{ReturnValue: value, Metadata: map[string]any{ToolMetadataKey: cloneJSON(result.Meta)}}, nil
}

func (t *runToolset[Deps]) toolError(name string, err error) error {
	message := fmt.Sprintf("MCP tool %q failed: %v", name, err)
	switch t.owner.config.errorBehavior {
	case ToolErrorFailed:
		return ai.ToolFailedf("%s", message)
	case ToolErrorAbort:
		return errors.New(message)
	default:
		return ai.Retryf("%s", message)
	}
}

func schemaMap(value any) (map[string]any, error) {
	if value == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var schema map[string]any
	err = json.Unmarshal(encoded, &schema)
	return schema, err
}

func mapCallToolResult(result *mcpsdk.CallToolResult, preferStructured bool) (any, error) {
	if result == nil {
		return nil, errors.New("server returned no result")
	}
	allText := true
	for _, content := range result.Content {
		if _, ok := content.(*mcpsdk.TextContent); !ok {
			allText = false
			break
		}
	}
	if result.StructuredContent != nil && (preferStructured || allText) {
		if object, ok := result.StructuredContent.(map[string]any); ok && len(object) == 1 {
			if value, exists := object["result"]; exists {
				return cloneJSON(value), nil
			}
		}
		return cloneJSON(result.StructuredContent), nil
	}
	values := make([]any, len(result.Content))
	for index, content := range result.Content {
		value, err := mapContent(content)
		if err != nil {
			return nil, err
		}
		values[index] = value
	}
	if len(values) == 1 {
		return values[0], nil
	}
	return values, nil
}

func mapContent(content mcpsdk.Content) (any, error) {
	switch content := content.(type) {
	case *mcpsdk.TextContent:
		trimmed := strings.TrimSpace(content.Text)
		if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
			var value any
			if json.Unmarshal([]byte(trimmed), &value) == nil {
				return value, nil
			}
		}
		return content.Text, nil
	case *mcpsdk.ImageContent:
		return ai.BinaryContent{Data: append([]byte(nil), content.Data...), MediaType: content.MIMEType}, nil
	case *mcpsdk.AudioContent:
		return ai.BinaryContent{Data: append([]byte(nil), content.Data...), MediaType: content.MIMEType}, nil
	default:
		return cloneJSONValue(content)
	}
}

func cloneJSON(value any) any {
	cloned, _ := cloneJSONValue(value)
	return cloned
}

func cloneJSONValue(value any) (any, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var cloned any
	err = json.Unmarshal(encoded, &cloned)
	return cloned, err
}

func formatToolResult(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(encoded)
}
