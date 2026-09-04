package mcp_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aimcp "github.com/Kludex/pydantic-ai-go/ai/mcp"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func connectSession(
	t *testing.T, server *mcpsdk.Server, options ...aimcp.Option,
) *aimcp.Session {
	t.Helper()
	clientTransport, serverTransport := mcpsdk.NewInMemoryTransports()
	serverSession, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = serverSession.Close() })
	session, err := aimcp.Connect(t.Context(), clientTransport, options...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close() })
	return session
}

func TestSessionDirectOperationsAndSharedToolset(t *testing.T) {
	server := mcpsdk.NewServer(
		&mcpsdk.Implementation{
			Name: "catalog", Version: "1",
			Icons: []mcpsdk.Icon{{Source: "https://example.com/icon.png", Sizes: []string{"32x32"}}},
		},
		&mcpsdk.ServerOptions{Instructions: "Use the catalog tools.", PageSize: 1},
	)
	server.AddPrompt(&mcpsdk.Prompt{
		Name: "review", Description: "Review code", Meta: mcpsdk.Meta{"source": "server"},
		Arguments: []*mcpsdk.PromptArgument{{Name: "language", Required: true}},
	}, func(_ context.Context, request *mcpsdk.GetPromptRequest) (*mcpsdk.GetPromptResult, error) {
		return &mcpsdk.GetPromptResult{
			Description: "Review " + request.Params.Arguments["language"],
			Meta:        mcpsdk.Meta{"rendered": true},
			Messages: []*mcpsdk.PromptMessage{{
				Role: mcpsdk.Role("user"), Content: &mcpsdk.TextContent{Text: "Check this code."},
			}},
		}, nil
	})
	server.AddPrompt(&mcpsdk.Prompt{Name: "second"}, func(
		context.Context, *mcpsdk.GetPromptRequest,
	) (*mcpsdk.GetPromptResult, error) {
		return &mcpsdk.GetPromptResult{Messages: []*mcpsdk.PromptMessage{}}, nil
	})
	server.AddPrompt(&mcpsdk.Prompt{Name: "error"}, func(
		context.Context, *mcpsdk.GetPromptRequest,
	) (*mcpsdk.GetPromptResult, error) {
		return nil, errors.New("prompt failed")
	})
	server.AddResource(&mcpsdk.Resource{
		URI: "file:///guide", Name: "guide", MIMEType: "text/plain", Size: 5,
		Meta: mcpsdk.Meta{"source": "server"},
	}, func(context.Context, *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
		return &mcpsdk.ReadResourceResult{Contents: []*mcpsdk.ResourceContents{
			{URI: "file:///guide", MIMEType: "text/plain", Text: "hello"},
			{URI: "file:///guide.bin", MIMEType: "application/octet-stream", Blob: []byte("blob")},
		}}, nil
	})
	server.AddResourceTemplate(&mcpsdk.ResourceTemplate{
		URITemplate: "file:///{name}", Name: "files", Meta: mcpsdk.Meta{"template": true},
	}, func(_ context.Context, request *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
		if request.Params.URI == "file:///error" {
			return nil, errors.New("resource failed")
		}
		return &mcpsdk.ReadResourceResult{Contents: []*mcpsdk.ResourceContents{{
			URI: request.Params.URI, Text: request.Params.URI,
		}}}, nil
	})
	server.AddTool(&mcpsdk.Tool{
		Name: "echo", Description: "Echo text", Meta: mcpsdk.Meta{"source": "server"},
		InputSchema: map[string]any{
			"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}},
			"required": []string{"text"},
		},
		OutputSchema: map[string]any{"type": "string"},
	}, func(_ context.Context, request *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
		var arguments map[string]any
		if err := json.Unmarshal(request.Params.Arguments, &arguments); err != nil {
			return nil, err
		}
		return &mcpsdk.CallToolResult{
			StructuredContent: map[string]any{"result": arguments["text"]},
			Meta:              mcpsdk.Meta{"called": true},
		}, nil
	})
	server.AddTool(&mcpsdk.Tool{Name: "fail", InputSchema: map[string]any{"type": "object"}}, func(
		context.Context, *mcpsdk.CallToolRequest,
	) (*mcpsdk.CallToolResult, error) {
		return nil, errors.New("tool failed")
	})

	session := connectSession(t, server, aimcp.WithID("catalog"), aimcp.WithPreferStructuredContent(true))
	if err := session.Ping(t.Context()); err != nil {
		t.Fatal(err)
	}
	initialized := session.InitializeResult()
	if initialized == nil || initialized.ServerInfo.Name != "catalog" ||
		initialized.Instructions != "Use the catalog tools." || initialized.Capabilities.Prompts == nil ||
		initialized.Capabilities.Resources == nil || initialized.Capabilities.Tools == nil {
		t.Fatalf("unexpected initialization: %+v", initialized)
	}
	initialized.ServerInfo.Icons[0].Sizes[0] = "changed"
	if session.InitializeResult().ServerInfo.Icons[0].Sizes[0] != "32x32" {
		t.Fatal("initialization result was not detached")
	}
	if session.SessionID() != "" {
		t.Fatalf("unexpected in-memory session ID %q", session.SessionID())
	}

	prompts, err := session.ListPrompts(t.Context())
	if err != nil || len(prompts) != 3 || prompts[0].Name != "error" ||
		prompts[1].Name != "review" || prompts[2].Name != "second" {
		t.Fatalf("unexpected prompts: %+v err=%v", prompts, err)
	}
	prompts[1].Meta["source"] = "changed"
	prompts, err = session.ListPrompts(t.Context())
	if err != nil || prompts[1].Meta["source"] != "server" {
		t.Fatal("listed prompts were not detached")
	}
	if empty, err := session.GetPrompt(t.Context(), "second", nil); err != nil || len(empty.Messages) != 0 {
		t.Fatalf("unexpected empty prompt: %+v err=%v", empty, err)
	}
	if _, err := session.GetPrompt(t.Context(), "error", nil); err == nil || !strings.Contains(err.Error(), "prompt failed") {
		t.Fatalf("unexpected prompt protocol error: %v", err)
	}
	arguments := map[string]string{"language": "Go"}
	prompt, err := session.GetPrompt(t.Context(), "review", arguments)
	arguments["language"] = "Rust"
	if err != nil || prompt.Description != "Review Go" || prompt.Meta["rendered"] != true ||
		prompt.Messages[0].Content.(*mcpsdk.TextContent).Text != "Check this code." {
		t.Fatalf("unexpected rendered prompt: %+v err=%v", prompt, err)
	}

	resources, err := session.ListResources(t.Context())
	if err != nil || len(resources) != 1 || resources[0].URI != "file:///guide" || resources[0].Size != 5 {
		t.Fatalf("unexpected resources: %+v err=%v", resources, err)
	}
	resources[0].Meta["source"] = "changed"
	resources, err = session.ListResources(t.Context())
	if err != nil || resources[0].Meta["source"] != "server" {
		t.Fatal("listed resources were not detached")
	}
	templates, err := session.ListResourceTemplates(t.Context())
	if err != nil || len(templates) != 1 || templates[0].URITemplate != "file:///{name}" ||
		templates[0].Meta["template"] != true {
		t.Fatalf("unexpected resource templates: %+v err=%v", templates, err)
	}
	resource, err := session.ReadResource(t.Context(), "file:///guide")
	if err != nil || len(resource.Contents) != 2 || resource.Contents[0].Text != "hello" ||
		string(resource.Contents[1].Blob) != "blob" {
		t.Fatalf("unexpected resource: %+v err=%v", resource, err)
	}
	if _, err := session.ReadResource(t.Context(), "file:///error"); err == nil ||
		!strings.Contains(err.Error(), "resource failed") {
		t.Fatalf("unexpected resource protocol error: %v", err)
	}
	resource.Contents[1].Blob[0] = 'X'
	resource, err = session.ReadResource(t.Context(), "file:///guide")
	if err != nil || string(resource.Contents[1].Blob) != "blob" {
		t.Fatal("resource result was not detached")
	}

	tools, err := session.ListTools(t.Context())
	if err != nil || len(tools) != 2 || tools[0].Name != "echo" || tools[1].Name != "fail" ||
		tools[0].Meta["source"] != "server" {
		t.Fatalf("unexpected tools: %+v err=%v", tools, err)
	}
	tools[0].Meta["source"] = "changed"
	tools, err = session.ListTools(t.Context())
	if err != nil || tools[0].Meta["source"] != "server" {
		t.Fatal("listed tools were not detached")
	}

	direct, err := session.CallTool(t.Context(), "echo", map[string]any{"text": "direct"})
	if err != nil || direct.IsError || direct.Meta["called"] != true ||
		direct.StructuredContent.(map[string]any)["result"] != "direct" {
		t.Fatalf("unexpected direct result: %+v err=%v", direct, err)
	}
	if _, err := session.CallTool(t.Context(), "fail", map[string]any{}); err == nil ||
		!strings.Contains(err.Error(), "tool failed") {
		t.Fatalf("unexpected tool protocol error: %v", err)
	}
	direct.Meta["called"] = false
	direct, err = session.CallTool(t.Context(), "echo", map[string]any{"text": "again"})
	if err != nil || direct.Meta["called"] != true {
		t.Fatal("direct tool result was not detached")
	}

	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !strings.Contains(params.Instructions, "catalog tools") || len(params.Tools) != 2 ||
			params.Tools[0].ToolsetID != "catalog" || params.Tools[1].ToolsetID != "catalog" {
			t.Fatalf("shared toolset missing: instructions=%q tools=%+v", params.Instructions, params.Tools)
		}
		for _, part := range messages[len(messages)-1].(ai.ModelRequest).Parts {
			if result, ok := part.(ai.ToolReturnPart); ok {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: result.Content.(string)}}}, nil
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "echo", ToolCallID: "echo", Args: json.RawMessage(`{"text":"shared"}`),
		}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddToolset(aimcp.NewSessionToolset[struct{}](session))
	for range 2 {
		result, err := agent.Run(t.Context(), "echo", struct{}{})
		if err != nil || result.Output != "shared" {
			t.Fatalf("shared session run failed: result=%+v err=%v", result, err)
		}
	}

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := session.ListPrompts(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected canceled prompt list error: %v", err)
	}
	if _, err := session.ListResources(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected canceled resource list error: %v", err)
	}
	if _, err := session.ListResourceTemplates(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected canceled resource template list error: %v", err)
	}
	if _, err := session.ListTools(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected canceled tool list error: %v", err)
	}
}

func TestSessionUnsupportedCapabilitiesAndErrors(t *testing.T) {
	server := mcpsdk.NewServer(
		&mcpsdk.Implementation{Name: "empty", Version: "1"},
		&mcpsdk.ServerOptions{Capabilities: &mcpsdk.ServerCapabilities{}},
	)
	session := connectSession(t, server, aimcp.WithImplementation(nil))
	if prompts, err := session.ListPrompts(t.Context()); err != nil || len(prompts) != 0 {
		t.Fatalf("unexpected prompts: %+v err=%v", prompts, err)
	}
	if resources, err := session.ListResources(t.Context()); err != nil || len(resources) != 0 {
		t.Fatalf("unexpected resources: %+v err=%v", resources, err)
	}
	if templates, err := session.ListResourceTemplates(t.Context()); err != nil || len(templates) != 0 {
		t.Fatalf("unexpected templates: %+v err=%v", templates, err)
	}
	if tools, err := session.ListTools(t.Context()); err != nil || len(tools) != 0 {
		t.Fatalf("unexpected tools: %+v err=%v", tools, err)
	}
	if _, err := session.GetPrompt(t.Context(), "missing", nil); err == nil ||
		!strings.Contains(err.Error(), "does not advertise prompts") {
		t.Fatalf("unexpected prompt error: %v", err)
	}
	if _, err := session.ReadResource(t.Context(), "missing:"); err == nil ||
		!strings.Contains(err.Error(), "does not advertise resources") {
		t.Fatalf("unexpected resource error: %v", err)
	}
	if _, err := session.CallTool(t.Context(), "missing", nil); err == nil ||
		!strings.Contains(err.Error(), `call tool "missing"`) {
		t.Fatalf("unexpected tool error: %v", err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.Ping(t.Context()); err == nil || !strings.Contains(err.Error(), "ping") {
		t.Fatalf("unexpected closed session error: %v", err)
	}
}

func TestSessionValidation(t *testing.T) {
	if _, err := aimcp.Connect(t.Context(), nil); err == nil || err.Error() != "ai/mcp: transport must not be nil" {
		t.Fatalf("unexpected nil transport error: %v", err)
	}
	var typedNil *mcpsdk.StreamableClientTransport
	if _, err := aimcp.Connect(t.Context(), typedNil); err == nil || err.Error() != "ai/mcp: transport must not be nil" {
		t.Fatalf("unexpected typed nil transport error: %v", err)
	}
	var session *aimcp.Session
	if session.InitializeResult() != nil || session.SessionID() != "" || session.Close() != nil {
		t.Fatal("nil session accessors changed shape")
	}
	if err := session.Ping(t.Context()); err == nil || err.Error() != "ai/mcp: session is not connected" {
		t.Fatalf("unexpected nil session error: %v", err)
	}
	if _, err := session.ListPrompts(t.Context()); err == nil {
		t.Fatal("nil session listed prompts")
	}
	if _, err := session.GetPrompt(t.Context(), "prompt", nil); err == nil {
		t.Fatal("nil session rendered a prompt")
	}
	if _, err := session.ListResources(t.Context()); err == nil {
		t.Fatal("nil session listed resources")
	}
	if _, err := session.ListResourceTemplates(t.Context()); err == nil {
		t.Fatal("nil session listed resource templates")
	}
	if _, err := session.ReadResource(t.Context(), "resource:"); err == nil {
		t.Fatal("nil session read a resource")
	}
	if _, err := session.ListTools(t.Context()); err == nil {
		t.Fatal("nil session listed tools")
	}
	if _, err := session.CallTool(t.Context(), "tool", nil); err == nil {
		t.Fatal("nil session called a tool")
	}
	func() {
		defer func() {
			if recovered := recover(); recovered != "ai/mcp: session must not be nil" {
				t.Fatalf("unexpected panic: %v", recovered)
			}
		}()
		aimcp.NewSessionToolset[struct{}](nil)
	}()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := aimcp.Connect(ctx, blockingTransport{}, aimcp.WithInitTimeout(time.Second)); err == nil ||
		!strings.Contains(err.Error(), "ai/mcp: connect") || !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected canceled connect error: %v", err)
	}
}

type blockingTransport struct{}

func (blockingTransport) Connect(ctx context.Context) (mcpsdk.Connection, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}
