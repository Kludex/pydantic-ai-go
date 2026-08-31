package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// InitializeResult is the server state negotiated when a Session connects.
type InitializeResult = mcpsdk.InitializeResult

// Prompt is a prompt template advertised by an MCP server.
type Prompt = mcpsdk.Prompt

// PromptResult is a rendered MCP prompt.
type PromptResult = mcpsdk.GetPromptResult

// Resource is a concrete resource advertised by an MCP server.
type Resource = mcpsdk.Resource

// ResourceTemplate is an RFC 6570 resource template advertised by an MCP server.
type ResourceTemplate = mcpsdk.ResourceTemplate

// ReadResourceResult contains text or binary content read from an MCP resource.
type ReadResourceResult = mcpsdk.ReadResourceResult

// Tool is a tool definition advertised by an MCP server.
type Tool = mcpsdk.Tool

// CallToolResult is the protocol-level result of a direct tool call.
type CallToolResult = mcpsdk.CallToolResult

// Session is a caller-managed connection to one MCP server. It can serve
// direct protocol operations and tools for several agent runs concurrently.
type Session struct {
	session *mcpsdk.ClientSession
	config  config
}

// Connect opens a reusable MCP session. Close the session after all direct
// operations and agent runs that use it have stopped.
func Connect(ctx context.Context, transport mcpsdk.Transport, options ...Option) (*Session, error) {
	if transportIsNil(transport) {
		return nil, errors.New("ai/mcp: transport must not be nil")
	}
	cfg := newConfig(options)
	client := mcpsdk.NewClient(cfg.implementation, cfg.clientOptions)
	connectCtx, cancel := context.WithTimeout(ctx, cfg.initTimeout)
	defer cancel()
	connected, err := client.Connect(connectCtx, transport, cfg.sessionOptions)
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: connect: %w", err)
	}
	return &Session{session: connected, config: cfg}, nil
}

// Close gracefully closes the session. It is safe to call more than once.
func (session *Session) Close() error {
	if session == nil || session.session == nil {
		return nil
	}
	return session.session.Close()
}

// SessionID returns the server-assigned transport session ID when available.
func (session *Session) SessionID() string {
	if session == nil || session.session == nil {
		return ""
	}
	return session.session.ID()
}

// InitializeResult returns a detached snapshot of negotiated server state.
func (session *Session) InitializeResult() *InitializeResult {
	if session == nil || session.session == nil {
		return nil
	}
	return cloneProtocol(session.session.InitializeResult())
}

// Ping checks that the server can respond before the read timeout.
func (session *Session) Ping(ctx context.Context) error {
	if _, err := session.initialized(); err != nil {
		return err
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	if err := session.session.Ping(readCtx, nil); err != nil {
		return fmt.Errorf("ai/mcp: ping: %w", err)
	}
	return nil
}

// ListPrompts returns all prompt templates and follows pagination.
// It returns an empty slice when the server does not advertise prompts.
func (session *Session) ListPrompts(ctx context.Context) ([]*Prompt, error) {
	initialized, err := session.initialized()
	if err != nil {
		return nil, err
	}
	if initialized.Capabilities == nil || initialized.Capabilities.Prompts == nil {
		return []*Prompt{}, nil
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	prompts := []*Prompt{}
	for prompt, listErr := range session.session.Prompts(readCtx, nil) {
		if listErr != nil {
			return nil, fmt.Errorf("ai/mcp: list prompts: %w", listErr)
		}
		prompts = append(prompts, cloneProtocol(prompt))
	}
	return prompts, nil
}

// GetPrompt renders one prompt template with string arguments.
func (session *Session) GetPrompt(
	ctx context.Context, name string, arguments map[string]string,
) (*PromptResult, error) {
	initialized, err := session.initialized()
	if err != nil {
		return nil, err
	}
	if initialized.Capabilities == nil || initialized.Capabilities.Prompts == nil {
		return nil, fmt.Errorf("ai/mcp: server does not advertise prompts; cannot get prompt %q", name)
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	result, err := session.session.GetPrompt(readCtx, &mcpsdk.GetPromptParams{
		Name: name, Arguments: cloneStringMap(arguments),
	})
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: get prompt %q: %w", name, err)
	}
	return cloneProtocol(result), nil
}

// ListResources returns all concrete resources and follows pagination.
// It returns an empty slice when the server does not advertise resources.
func (session *Session) ListResources(ctx context.Context) ([]*Resource, error) {
	initialized, err := session.initialized()
	if err != nil {
		return nil, err
	}
	if initialized.Capabilities == nil || initialized.Capabilities.Resources == nil {
		return []*Resource{}, nil
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	resources := []*Resource{}
	for resource, listErr := range session.session.Resources(readCtx, nil) {
		if listErr != nil {
			return nil, fmt.Errorf("ai/mcp: list resources: %w", listErr)
		}
		resources = append(resources, cloneProtocol(resource))
	}
	return resources, nil
}

// ListResourceTemplates returns all resource URI templates and follows pagination.
func (session *Session) ListResourceTemplates(ctx context.Context) ([]*ResourceTemplate, error) {
	initialized, err := session.initialized()
	if err != nil {
		return nil, err
	}
	if initialized.Capabilities == nil || initialized.Capabilities.Resources == nil {
		return []*ResourceTemplate{}, nil
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	templates := []*ResourceTemplate{}
	for template, listErr := range session.session.ResourceTemplates(readCtx, nil) {
		if listErr != nil {
			return nil, fmt.Errorf("ai/mcp: list resource templates: %w", listErr)
		}
		templates = append(templates, cloneProtocol(template))
	}
	return templates, nil
}

// ReadResource reads text or binary content for one URI.
func (session *Session) ReadResource(ctx context.Context, uri string) (*ReadResourceResult, error) {
	initialized, err := session.initialized()
	if err != nil {
		return nil, err
	}
	if initialized.Capabilities == nil || initialized.Capabilities.Resources == nil {
		return nil, fmt.Errorf("ai/mcp: server does not advertise resources; cannot read %q", uri)
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	result, err := session.session.ReadResource(readCtx, &mcpsdk.ReadResourceParams{URI: uri})
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: read resource %q: %w", uri, err)
	}
	return cloneProtocol(result), nil
}

// ListTools returns all tool definitions and follows pagination.
// It returns an empty slice when the server does not advertise tools.
func (session *Session) ListTools(ctx context.Context) ([]*Tool, error) {
	initialized, err := session.initialized()
	if err != nil {
		return nil, err
	}
	if initialized.Capabilities == nil || initialized.Capabilities.Tools == nil {
		return []*Tool{}, nil
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	tools := []*Tool{}
	for tool, listErr := range session.session.Tools(readCtx, nil) {
		if listErr != nil {
			return nil, fmt.Errorf("ai/mcp: list tools: %w", listErr)
		}
		tools = append(tools, cloneProtocol(tool))
	}
	return tools, nil
}

// CallTool invokes one server tool directly. Protocol-level tool failures are
// returned in CallToolResult.IsError rather than converted to agent retries.
func (session *Session) CallTool(ctx context.Context, name string, arguments any) (*CallToolResult, error) {
	if _, err := session.initialized(); err != nil {
		return nil, err
	}
	readCtx, cancel := session.readContext(ctx)
	defer cancel()
	result, err := session.session.CallTool(readCtx, &mcpsdk.CallToolParams{Name: name, Arguments: arguments})
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: call tool %q: %w", name, err)
	}
	return cloneProtocol(result), nil
}

func (session *Session) initialized() (*mcpsdk.InitializeResult, error) {
	if session == nil || session.session == nil || session.session.InitializeResult() == nil {
		return nil, errors.New("ai/mcp: session is not connected")
	}
	return session.session.InitializeResult(), nil
}

func (session *Session) readContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, session.config.readTimeout)
}

func transportIsNil(transport mcpsdk.Transport) bool {
	if transport == nil {
		return true
	}
	value := reflect.ValueOf(transport)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func cloneProtocol[T any](value *T) *T {
	encoded, _ := json.Marshal(value)
	var cloned T
	_ = json.Unmarshal(encoded, &cloned)
	return &cloned
}

func cloneStringMap(values map[string]string) map[string]string {
	if values == nil {
		return nil
	}
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

// SessionToolset exposes tools from a caller-managed Session. Agent runs never
// close the session.
type SessionToolset[Deps any] struct {
	toolset *runToolset[Deps]
}

// NewSessionToolset adapts a shared session to ai.Toolset.
func NewSessionToolset[Deps any](session *Session) *SessionToolset[Deps] {
	if session == nil || session.session == nil {
		panic("ai/mcp: session must not be nil")
	}
	owner := &Toolset[Deps]{config: session.config}
	return &SessionToolset[Deps]{toolset: &runToolset[Deps]{owner: owner, session: session.session}}
}

// ToolsetID returns the ID passed to Connect with WithID.
func (toolset *SessionToolset[Deps]) ToolsetID() string { return toolset.toolset.ToolsetID() }

// Tools lists tools from the shared session.
func (toolset *SessionToolset[Deps]) Tools(
	ctx context.Context, runContext *ai.RunContext[Deps],
) ([]ai.Tool[Deps], error) {
	return toolset.toolset.Tools(ctx, runContext)
}

// ToolsetInstructions returns the server instructions negotiated by the session.
func (toolset *SessionToolset[Deps]) ToolsetInstructions(
	ctx context.Context, runContext *ai.RunContext[Deps],
) ([]ai.InstructionPart, error) {
	return toolset.toolset.ToolsetInstructions(ctx, runContext)
}

var _ ai.Toolset[struct{}] = (*SessionToolset[struct{}])(nil)
