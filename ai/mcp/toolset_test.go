package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

type testDeps struct{}

func serverFactory[Deps any](server *mcpsdk.Server) TransportFactory[Deps] {
	return func(ctx context.Context, _ *ai.RunContext[Deps]) (mcpsdk.Transport, error) {
		clientTransport, serverTransport := mcpsdk.NewInMemoryTransports()
		if _, err := server.Connect(ctx, serverTransport, nil); err != nil {
			return nil, err
		}
		return clientTransport, nil
	}
}

func TestMCPToolsetRunsToolsAndInstructions(t *testing.T) {
	server := mcpsdk.NewServer(
		&mcpsdk.Implementation{Name: "test-server", Version: "1"},
		&mcpsdk.ServerOptions{Instructions: "Use the remote calculator."},
	)
	readOnly := false
	server.AddTool(&mcpsdk.Tool{
		Name: "add", Description: "Add two integers", Meta: mcpsdk.Meta{"source": "test"},
		Annotations: &mcpsdk.ToolAnnotations{ReadOnlyHint: true, DestructiveHint: &readOnly},
		InputSchema: map[string]any{
			"type": "object", "properties": map[string]any{
				"a": map[string]any{"type": "integer"}, "b": map[string]any{"type": "integer"},
			}, "required": []string{"a", "b"}, "additionalProperties": false,
		},
		OutputSchema: map[string]any{"type": "integer"},
	}, func(_ context.Context, request *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
		var args struct {
			A int `json:"a"`
			B int `json:"b"`
		}
		if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
			return nil, err
		}
		return &mcpsdk.CallToolResult{
			Content:           []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"result":5}`}},
			StructuredContent: map[string]any{"result": args.A + args.B},
			Meta:              mcpsdk.Meta{"request": "metadata"},
		}, nil
	})

	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			if !strings.Contains(params.Instructions, "remote calculator") || len(params.Tools) != 1 {
				t.Fatalf("MCP setup missing: instructions=%q tools=%+v", params.Instructions, params.Tools)
			}
			definition := params.Tools[0]
			if definition.Name != "add" || definition.Description != "Add two integers" ||
				definition.ToolsetID != "calculator" || definition.ReturnSchema["type"] != "integer" ||
				definition.Metadata[ToolMetadataKey].(map[string]any)["source"] != "test" ||
				definition.Metadata[ToolAnnotationsMetadataKey] == nil {
				t.Fatalf("unexpected MCP definition: %+v", definition)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "add", ToolCallID: "call", Args: json.RawMessage(`{"a":2,"b":3}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		result := request.Parts[0].(ai.ToolReturnPart)
		if result.Content != float64(5) || result.Metadata[ToolMetadataKey].(map[string]any)["request"] != "metadata" {
			t.Fatalf("unexpected MCP result: %+v", result)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "five"}}}, nil
	})
	toolset := NewToolset(serverFactory[testDeps](server), WithID("calculator"))
	agent := ai.NewAgent[testDeps, string](model)
	agent.AddToolset(toolset)
	result, err := agent.Run(t.Context(), "add", testDeps{})
	if err != nil || result.Output != "five" || result.Usage().ToolCalls != 1 {
		t.Fatalf("unexpected MCP run result=%+v err=%v", result, err)
	}
}

func TestMCPToolErrors(t *testing.T) {
	for _, test := range []struct {
		name      string
		behavior  ToolErrorBehavior
		protocol  bool
		wantError bool
		outcome   ai.ToolReturnOutcome
	}{
		{name: "retry", behavior: ToolErrorRetry},
		{name: "failed", behavior: ToolErrorFailed, outcome: ai.ToolReturnOutcomeFailed},
		{name: "abort", behavior: ToolErrorAbort, wantError: true},
		{name: "protocol retry", behavior: ToolErrorRetry, protocol: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "server", Version: "1"}, nil)
			server.AddTool(&mcpsdk.Tool{Name: "fail", InputSchema: map[string]any{"type": "object"}},
				func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
					if test.protocol {
						return nil, errors.New("protocol failure")
					}
					return &mcpsdk.CallToolResult{IsError: true, Content: []mcpsdk.Content{
						&mcpsdk.TextContent{Text: "tool failure"},
					}}, nil
				})
			requests := 0
			model := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				requests++
				if requests == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: "fail", ToolCallID: "call", Args: json.RawMessage(`{}`),
					}}}, nil
				}
				request := messages[len(messages)-1].(ai.ModelRequest)
				if test.outcome != "" && request.Parts[0].(ai.ToolReturnPart).Outcome != test.outcome {
					t.Fatalf("unexpected failed outcome: %+v", request)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "recovered"}}}, nil
			})
			agent := ai.NewAgent[testDeps, string](model, ai.WithRetryLimits(ai.RetryLimits{Tools: 2, Output: 1}))
			agent.AddToolset(NewToolset(serverFactory[testDeps](server), WithToolErrorBehavior(test.behavior)))
			result, err := agent.Run(t.Context(), "go", testDeps{})
			if test.wantError {
				if err == nil || !strings.Contains(err.Error(), "MCP tool") {
					t.Fatalf("expected abort error, got result=%+v err=%v", result, err)
				}
				return
			}
			if err != nil || result.Output != "recovered" {
				t.Fatalf("unexpected recovery: result=%+v err=%v", result, err)
			}
		})
	}
}

func TestMCPResultMapping(t *testing.T) {
	image := &mcpsdk.ImageContent{Data: []byte("image"), MIMEType: "image/png"}
	audio := &mcpsdk.AudioContent{Data: []byte("audio"), MIMEType: "audio/mpeg"}
	link := &mcpsdk.ResourceLink{URI: "file:///a", Name: "a"}
	for _, test := range []struct {
		name   string
		result *mcpsdk.CallToolResult
		prefer bool
		want   any
	}{
		{name: "empty", result: &mcpsdk.CallToolResult{}, want: []any{}},
		{name: "text", result: &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "hello"}}}, want: "hello"},
		{name: "JSON text", result: &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: ` {"a":1} `}}}, want: map[string]any{"a": float64(1)}},
		{name: "invalid JSON text", result: &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "{bad"}}}, want: "{bad"},
		{name: "binary", result: &mcpsdk.CallToolResult{Content: []mcpsdk.Content{image}}, want: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}},
		{name: "multiple", result: &mcpsdk.CallToolResult{Content: []mcpsdk.Content{audio, link}}},
		{name: "structured unwrapped", result: &mcpsdk.CallToolResult{StructuredContent: map[string]any{"result": "yes"}}, want: "yes"},
		{name: "structured object", result: &mcpsdk.CallToolResult{StructuredContent: map[string]any{"a": 1, "b": 2}}, want: map[string]any{"a": float64(1), "b": float64(2)}},
		{name: "prefer structured", result: &mcpsdk.CallToolResult{StructuredContent: "yes", Content: []mcpsdk.Content{image}}, prefer: true, want: "yes"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := mapCallToolResult(test.result, test.prefer)
			if err != nil {
				t.Fatal(err)
			}
			if test.want != nil && !reflect.DeepEqual(got, test.want) {
				t.Fatalf("unexpected mapping: got=%#v want=%#v", got, test.want)
			}
		})
	}
	if _, err := mapCallToolResult(nil, false); err == nil {
		t.Fatal("expected nil result error")
	}
	owner := NewStreamableHTTPToolset[testDeps]("http://localhost/mcp")
	if _, err := (&runToolset[testDeps]{owner: owner}).mapToolCallResult("bad", nil); err == nil {
		t.Fatal("expected wrapped nil result error")
	}
	if formatToolResult("text") != "text" || formatToolResult(map[string]any{"a": 1}) != `{"a":1}` ||
		formatToolResult(make(chan int)) == "" {
		t.Fatal("unexpected formatted tool result")
	}
}

func TestMCPConfigurationAndFailures(t *testing.T) {
	assertPanic(t, "nil factory", func() { NewToolset[testDeps](nil) })
	assertPanic(t, "init timeout", func() { WithInitTimeout(0) })
	assertPanic(t, "read timeout", func() { WithReadTimeout(0) })
	assertPanic(t, "error behavior", func() { WithToolErrorBehavior("bad") })

	implementation := &mcpsdk.Implementation{Name: "custom", Version: "1"}
	clientOptions := &mcpsdk.ClientOptions{}
	sessionOptions := &mcpsdk.ClientSessionOptions{}
	toolset := NewToolset[testDeps](func(context.Context, *ai.RunContext[testDeps]) (mcpsdk.Transport, error) {
		return nil, errors.New("factory failure")
	}, WithID("id"), WithImplementation(implementation), WithClientOptions(clientOptions),
		WithSessionOptions(sessionOptions), WithInitTimeout(time.Second), WithReadTimeout(time.Second),
		WithPreferStructuredContent(true))
	if toolset.ToolsetID() != "id" {
		t.Fatalf("unexpected toolset ID %q", toolset.ToolsetID())
	}
	if _, err := toolset.Tools(t.Context(), nil); err == nil {
		t.Fatal("expected unopened toolset error")
	}
	if _, err := toolset.ForRun(t.Context(), nil); err == nil {
		t.Fatal("expected factory error")
	}

	nilTransport := NewToolset[testDeps](func(context.Context, *ai.RunContext[testDeps]) (mcpsdk.Transport, error) {
		return nil, nil
	})
	if _, err := nilTransport.ForRun(t.Context(), nil); err == nil {
		t.Fatal("expected nil transport error")
	}

	for _, created := range []*Toolset[testDeps]{
		NewStreamableHTTPToolset[testDeps]("http://localhost/mcp"),
		NewSSEToolset[testDeps]("http://localhost/sse"),
		NewCommandToolset[testDeps]("echo", []string{"ok"}),
	} {
		resolved, err := created.ForRun(t.Context(), nil)
		if err != nil || resolved == nil {
			t.Fatalf("transport constructor failed: resolved=%T err=%v", resolved, err)
		}
	}

	badConnect := NewToolset[testDeps](func(context.Context, *ai.RunContext[testDeps]) (mcpsdk.Transport, error) {
		return failingTransport{}, nil
	})
	resolved, err := badConnect.ForRun(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	opener := resolved.(ai.ToolsetOpener[testDeps])
	if _, _, err := opener.OpenToolset(t.Context(), nil); err == nil {
		t.Fatal("expected connect error")
	}
}

type failingTransport struct{}

func (failingTransport) Connect(context.Context) (mcpsdk.Connection, error) {
	return nil, errors.New("connect failure")
}

func assertPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s did not panic", name)
		}
	}()
	fn()
}

func TestMCPInternalSchemaAndClosedSessionErrors(t *testing.T) {
	if schema, err := schemaMap(nil); err != nil || schema != nil {
		t.Fatalf("unexpected nil schema: %#v %v", schema, err)
	}
	if _, err := schemaMap(make(chan int)); err == nil {
		t.Fatal("expected schema marshal error")
	}
	if _, err := schemaMap("not an object"); err == nil {
		t.Fatal("expected schema shape error")
	}
	if _, err := cloneJSONValue(make(chan int)); err == nil {
		t.Fatal("expected clone marshal error")
	}

	owner := NewStreamableHTTPToolset[testDeps]("http://localhost/mcp")
	run := &runToolset[testDeps]{owner: owner}
	if run.ToolsetID() != "" {
		t.Fatalf("unexpected run toolset ID %q", run.ToolsetID())
	}
	if _, err := run.Tools(t.Context(), nil); err == nil {
		t.Fatal("expected unopened list error")
	}
	if _, err := run.ToolsetInstructions(t.Context(), nil); err == nil {
		t.Fatal("expected unopened instructions error")
	}

}

type invalidJSONValue struct{}

func (invalidJSONValue) MarshalJSON() ([]byte, error) { return []byte("{"), nil }

func openMCPRunToolset(t *testing.T, server *mcpsdk.Server) (*runToolset[testDeps], ai.ToolsetCloseFunc) {
	t.Helper()
	owner := NewToolset(serverFactory[testDeps](server), WithReadTimeout(time.Second))
	resolved, err := owner.ForRun(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	opened, closeFunc, err := resolved.(ai.ToolsetOpener[testDeps]).OpenToolset(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	return opened.(*runToolset[testDeps]), closeFunc
}

func TestMCPInternalErrorBranches(t *testing.T) {
	owner := NewStreamableHTTPToolset[testDeps]("http://localhost/mcp")
	run := &runToolset[testDeps]{owner: owner}
	for _, tool := range []*mcpsdk.Tool{
		nil,
		{Name: ""},
		{Name: "bad-input", InputSchema: make(chan int)},
		{Name: "bad-output", InputSchema: map[string]any{"type": "object"}, OutputSchema: make(chan int)},
	} {
		if _, err := run.appendTool(nil, tool); err == nil {
			t.Fatalf("expected invalid tool error for %+v", tool)
		}
	}
	if _, err := cloneJSONValue(invalidJSONValue{}); err == nil {
		t.Fatal("expected invalid marshaled JSON error")
	}
	if _, err := mapCallToolResult(&mcpsdk.CallToolResult{Content: []mcpsdk.Content{
		&mcpsdk.ResourceLink{URI: "file:///bad", Name: "bad", Meta: mcpsdk.Meta{"bad": make(chan int)}},
	}}, false); err == nil {
		t.Fatal("expected malformed content error")
	}

	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "server", Version: "1"}, nil)
	server.AddTool(&mcpsdk.Tool{Name: "plain", InputSchema: map[string]any{"type": "object"}},
		func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "ok"}}}, nil
		})
	opened, closeFunc := openMCPRunToolset(t, server)
	defer func() {
		if err := closeFunc(t.Context()); err != nil {
			t.Fatal(err)
		}
	}()
	value, err := opened.callTool(t.Context(), "plain", nil)
	if err != nil || value.(ai.ToolReturn).ReturnValue != "ok" {
		t.Fatalf("empty arguments call failed: value=%#v err=%v", value, err)
	}
	if _, err := opened.callTool(t.Context(), "plain", json.RawMessage(`{`)); err == nil {
		t.Fatal("expected invalid argument error")
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := opened.Tools(canceled, nil); err == nil {
		t.Fatal("expected canceled list-tools error")
	}

	malformedServer := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "malformed", Version: "1"}, nil)
	malformedServer.AddTool(&mcpsdk.Tool{Name: "", InputSchema: map[string]any{"type": "object"}},
		func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return &mcpsdk.CallToolResult{}, nil
		})
	malformed, malformedClose := openMCPRunToolset(t, malformedServer)
	defer func() { _ = malformedClose(t.Context()) }()
	if _, err := malformed.Tools(t.Context(), nil); err == nil {
		t.Fatal("expected malformed listed tool error")
	}
}
