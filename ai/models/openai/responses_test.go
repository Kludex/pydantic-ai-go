package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func newResponsesServer(t *testing.T, handler http.HandlerFunc) *openai.ResponsesModel {
	t.Helper()
	return newResponsesServerWithOptions(t, handler)
}

func rawAnnotationSettings(t *testing.T) ai.ModelSettings {
	t.Helper()
	enabled := true
	settings, err := (openai.Settings{IncludeRawAnnotations: &enabled}).Build()
	if err != nil {
		t.Fatal(err)
	}
	return settings
}

func newResponsesServerWithOptions(
	t *testing.T, handler http.HandlerFunc, opts ...openai.Option,
) *openai.ResponsesModel {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	options := []openai.Option{
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
		openai.WithHTTPClient(server.Client()),
	}
	return openai.NewResponsesModel("gpt-5", append(options, opts...)...)
}

func TestResponsesCountTokens(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses/input_tokens" || request.Header.Get("X-Custom") != "value" {
			t.Errorf("unexpected token count request: %s headers=%v", request.URL.Path, request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"input_tokens":17}`))
	})
	if model.ProviderName() != "openai" || model.ProviderURL() == "" {
		t.Fatalf("unexpected provider identity: %q %q", model.ProviderName(), model.ProviderURL())
	}
	parallel := true
	usage, err := model.CountTokens(t.Context(), []ai.ModelMessage{ai.ModelRequest{
		Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "hello"}, ai.ImageURL{URL: "https://example.com/image.png"},
		}}},
	}}, ai.ModelRequestParams{
		Instructions: "Be brief.",
		Tools:        []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		NativeTools:  []ai.NativeTool{ai.WebSearchTool{SearchContextSize: ai.WebSearchContextLow}},
		Settings: ai.ModelSettings{
			MaxTokens: 100, ParallelToolCalls: &parallel,
			ExtraHeaders: map[string]string{"X-Custom": "value"}, ExtraBody: map[string]any{"store": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if usage.InputTokens != 17 || usage.Requests != 0 {
		t.Fatalf("unexpected token usage: %+v", usage)
	}
	if body["model"] != "gpt-5" || body["instructions"] != "Be brief." || body["store"] != true ||
		body["max_output_tokens"] != nil || body["background"] != nil || body["include"] != nil {
		t.Fatalf("unexpected token count body: %v", body)
	}
	content := body["input"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 2 || content[1].(map[string]any)["type"] != "input_image" {
		t.Fatalf("token count omitted multimodal content: %#v", content)
	}
	tools := body["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["type"] != "web_search" ||
		tools[1].(map[string]any)["type"] != "function" {
		t.Fatalf("token count omitted native or function tools: %#v", tools)
	}
	if _, err := model.CountTokens(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "cannot count tokens without messages") {
		t.Fatalf("unexpected empty token count error: %v", err)
	}
}

func TestResponsesCountTokensErrors(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}
	t.Run("payload", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5")
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: "extreme"},
		}})
		if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
			t.Fatalf("unexpected payload error: %v", err)
		}
	})
	t.Run("marshal", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5")
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"bad": make(chan struct{})},
		}})
		if err == nil || !strings.Contains(err.Error(), "marshal token count request") {
			t.Fatalf("unexpected marshal error: %v", err)
		}
	})
	t.Run("request", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(":"))
		if _, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected request construction error")
		}
	})
	t.Run("prepare", func(t *testing.T) {
		prepareErr := errors.New("prepare failed")
		model := openai.NewResponsesModel("gpt-5", openai.WithProvider(openai.ProviderConfig{
			Name: "compatible", BaseURL: "http://example.test",
			PrepareRequest: func(*http.Request) error { return prepareErr },
		}))
		if _, err := model.CountTokens(
			t.Context(), messages, ai.ModelRequestParams{},
		); !errors.Is(err, prepareErr) {
			t.Fatalf("unexpected prepare error: %v", err)
		}
	})
	for _, test := range []struct {
		name       string
		response   *http.Response
		requestErr error
		contains   string
	}{
		{name: "request failure", requestErr: errors.New("request failed"), contains: "token count request"},
		{name: "read failure", response: &http.Response{
			StatusCode: http.StatusOK, Body: compactionErrorBody{}, Header: make(http.Header),
		}, contains: "read token count response"},
		{name: "status", response: &http.Response{
			StatusCode: http.StatusBadRequest, Body: io.NopCloser(strings.NewReader("bad")), Header: make(http.Header),
		}, contains: "status 400"},
		{name: "decode", response: &http.Response{
			StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{")), Header: make(http.Header),
		}, contains: "decode token count response"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := &http.Client{Transport: compactionRoundTripFunc(func(*http.Request) (*http.Response, error) {
				return test.response, test.requestErr
			})}
			model := openai.NewResponsesModel(
				"gpt-5", openai.WithBaseURL("http://example.test"), openai.WithHTTPClient(client),
			)
			if _, err := model.CountTokens(
				t.Context(), messages, ai.ModelRequestParams{},
			); err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected token count error: %v", err)
			}
		})
	}
}

func TestResponsesRefusal(t *testing.T) {
	model := newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5",
			"output":[{"id":"message","type":"message","content":[{
				"type":"refusal","refusal":"I cannot help with that request."
			}]}]
		}`))
	})
	_, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "blocked", struct{}{})
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) || filtered.Response().FinishReason != ai.FinishReasonContentFilter ||
		filtered.Response().ProviderDetails["refusal"] != "I cannot help with that request." ||
		filtered.Response().ProviderDetails["finish_reason"] != nil {
		t.Fatalf("unexpected Responses refusal: %v response=%+v", err, filtered)
	}
}

type unsupportedNativeTool struct{ optional bool }

func (tool unsupportedNativeTool) Kind() string                   { return "unsupported" }
func (tool unsupportedNativeTool) UniqueID() string               { return "unsupported" }
func (tool unsupportedNativeTool) IsOptional() bool               { return tool.optional }
func (tool unsupportedNativeTool) CloneNativeTool() ai.NativeTool { return tool }

func TestResponsesWebSearchNativeTool(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[
				{"type":"web_search_call","id":"web-1","status":"completed","action":{"type":"search","query":"Go news"}},
				{"type":"message","id":"message","content":[{"type":"output_text","text":"done"}]}
			]
		}`))
	})
	external := false
	response, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "search"},
	}}}, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{
		SearchContextSize: ai.WebSearchContextHigh,
		UserLocation: &ai.WebSearchUserLocation{
			City: "Paris", Country: "FR", Region: "IDF", Timezone: "Europe/Paris",
		},
		AllowedDomains: []string{"go.dev"}, ExternalWebAccess: &external,
	}}})
	if err != nil {
		t.Fatal(err)
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	location := tool["user_location"].(map[string]any)
	filters := tool["filters"].(map[string]any)
	if tool["type"] != "web_search" || tool["search_context_size"] != "high" ||
		tool["external_web_access"] != false || location["type"] != "approximate" || location["city"] != "Paris" ||
		filters["allowed_domains"].([]any)[0] != "go.dev" {
		t.Fatalf("unexpected web search request: %#v", tool)
	}
	if len(response.Parts) != 3 {
		t.Fatalf("unexpected web search response: %#v", response.Parts)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if call.ToolKind != ai.ToolPartKindWebSearch || call.ToolCallID != "web-1" ||
		string(call.Args) != `{"type":"search","query":"Go news"}` || returned.ToolCallID != "web-1" ||
		returned.ToolKind != ai.ToolPartKindWebSearch || returned.Content.(map[string]any)["status"] != "completed" {
		t.Fatalf("unexpected normalized web search parts: call=%+v return=%+v", call, returned)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&ai.WebSearchTool{}},
	}); err != nil {
		t.Fatal(err)
	}
	tool = body["tools"].([]any)[0].(map[string]any)
	if tool["search_context_size"] != "medium" || tool["user_location"] != nil || tool["filters"] != nil ||
		tool["external_web_access"] != nil {
		t.Fatalf("unexpected default web search request: %#v", tool)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	replayed := body["input"].([]any)[0].(map[string]any)
	if replayed["type"] != "web_search_call" || replayed["id"] != "web-1" ||
		replayed["action"].(map[string]any)["query"] != "Go news" || replayed["status"] != "completed" {
		t.Fatalf("unexpected web search replay: %#v", replayed)
	}
}

func TestResponsesCodeExecutionNativeTool(t *testing.T) {
	var body map[string]any
	model := newResponsesServerWithOptions(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[
				{"type":"code_interpreter_call","id":"code-1","container_id":"container-1","code":"print(1)","status":"completed","outputs":[
					{"type":"logs","logs":"1\n"},{"type":"image","url":"data:image/png;base64,aW1hZ2U="}
				]},
				{"type":"message","id":"message","content":[{"type":"output_text","text":"done"}]}
			]
		}`))
	}, openai.WithResponsesCodeExecutionOutputs(true))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.CodeExecutionTool{Files: []ai.UploadedFile{
			{FileID: "file-openai", ProviderName: "openai"},
			{FileID: "file-anthropic", ProviderName: "anthropic"},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	container := tool["container"].(map[string]any)
	if tool["type"] != "code_interpreter" || container["type"] != "auto" ||
		container["file_ids"].([]any)[0] != "file-openai" ||
		body["include"].([]any)[0] != "code_interpreter_call.outputs" {
		t.Fatalf("unexpected code execution request: %#v", body)
	}
	if len(response.Parts) != 4 {
		t.Fatalf("unexpected code execution response parts: %#v", response.Parts)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	file := response.Parts[1].(ai.FilePart)
	returned := response.Parts[2].(ai.NativeToolReturnPart)
	if call.ToolKind != ai.ToolPartKindCodeExecution || call.ToolCallID != "code-1" ||
		string(call.Args) != `{"container_id":"container-1","code":"print(1)"}` ||
		file.Content.MediaType != "image/png" || string(file.Content.Data) != "image" || file.ID != "code-1" ||
		returned.ToolKind != ai.ToolPartKindCodeExecution || returned.Content.(map[string]any)["status"] != "completed" ||
		returned.Content.(map[string]any)["logs"].([]string)[0] != "1\n" {
		t.Fatalf("unexpected normalized code execution: call=%+v file=%+v return=%+v", call, file, returned)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&ai.CodeExecutionTool{}},
	}); err != nil {
		t.Fatal(err)
	}
	if container := body["tools"].([]any)[0].(map[string]any)["container"].(map[string]any); container["file_ids"] != nil {
		t.Fatalf("empty code container included files: %#v", container)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	replayed := body["input"].([]any)[0].(map[string]any)
	if replayed["type"] != "code_interpreter_call" || replayed["id"] != "code-1" ||
		replayed["container_id"] != "container-1" || replayed["code"] != "print(1)" ||
		replayed["status"] != "completed" || replayed["outputs"] != nil {
		t.Fatalf("unexpected code execution replay: %#v", replayed)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: "code_execution", ToolKind: ai.ToolPartKindCodeExecution,
			ProviderName: "openai", Args: json.RawMessage(`{bad`), ToolCallID: "bad",
		},
	}}}, ai.ModelRequestParams{}); err == nil || !strings.Contains(err.Error(), "parse code execution arguments") {
		t.Fatalf("unexpected malformed code replay error: %v", err)
	}
}

func TestResponsesCodeExecutionOutputErrors(t *testing.T) {
	for _, test := range []struct {
		name   string
		output string
		want   string
	}{
		{name: "unknown", output: `{"type":"audio"}`, want: "unknown code interpreter output type"},
		{name: "invalid URI", output: `{"type":"image","url":"https://example.com/image.png"}`, want: "invalid code interpreter image data URI"},
		{name: "invalid base64", output: `{"type":"image","url":"data:image/png;base64,!"}`, want: "decode code interpreter image"},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
				_, _ = response.Write([]byte(`{"output":[{"type":"code_interpreter_call","id":"code","outputs":[` +
					test.output + `]}]}`))
			})
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected code output error: %v", err)
			}
		})
	}
}

func TestResponsesMCPServerTool(t *testing.T) {
	var bodies []map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[
				{"type":"mcp_list_tools","id":"list-1","server_label":"docs","tools":[{
					"name":"search","description":"Search docs","input_schema":{"type":"object"},
					"annotations":{"read_only":true}}],"error":null},
				{"type":"mcp_call","id":"call-1","server_label":"docs","name":"search",
					"arguments":"{\"query\":\"Go\"}","output":{"text":"result"},"error":null}
			],"usage":{}
		}`))
	})
	emptyAllowed := []string{}
	params := ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.MCPServerTool{
			ID: "docs", URL: "https://example.com/mcp", Description: "Documentation",
			AllowedTools: emptyAllowed, Headers: map[string]string{"X-Tenant": "acme"},
		},
		&ai.MCPServerTool{
			ID: "calendar", URL: "x-openai-connector:connector_googlecalendar",
			AuthorizationToken: "token", AllowedTools: []string{"events"},
		},
	}}
	response, err := model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	tools := bodies[0]["tools"].([]any)
	first := tools[0].(map[string]any)
	second := tools[1].(map[string]any)
	if first["type"] != "mcp" || first["server_label"] != "docs" ||
		first["server_url"] != "https://example.com/mcp" || first["require_approval"] != "never" ||
		first["server_description"] != "Documentation" || len(first["allowed_tools"].([]any)) != 0 ||
		first["headers"].(map[string]any)["X-Tenant"] != "acme" ||
		second["connector_id"] != "connector_googlecalendar" || second["server_url"] != nil ||
		second["authorization"] != "token" || second["allowed_tools"].([]any)[0] != "events" {
		t.Fatalf("unexpected MCP tools: %#v", tools)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	listed := response.Parts[1].(ai.NativeToolReturnPart)
	called := response.Parts[2].(ai.NativeToolCallPart)
	returned := response.Parts[3].(ai.NativeToolReturnPart)
	listedContent := listed.Content.(map[string]any)
	listedTools := listedContent["tools"].([]map[string]any)
	if call.ToolName != "mcp_server:docs" || call.ToolKind != ai.ToolPartKindMCPServer ||
		string(call.Args) != `{"action":"list_tools"}` || listed.ToolCallID != "list-1" ||
		listedTools[0]["input_schema"].(map[string]any)["type"] != "object" || listedContent["error"] != nil ||
		called.ToolName != "mcp_server:docs" || called.ToolCallID != "call-1" ||
		string(called.Args) != `{"action":"call_tool","tool_args":{"query":"Go"},"tool_name":"search"}` ||
		returned.Content.(map[string]any)["output"].(map[string]any)["text"] != "result" {
		t.Fatalf("unexpected MCP parts: %#v", response.Parts)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	input := bodies[1]["input"].([]any)
	listReplay := input[0].(map[string]any)
	callReplay := input[1].(map[string]any)
	if listReplay["type"] != "mcp_list_tools" || listReplay["server_label"] != "docs" ||
		len(listReplay["tools"].([]any)) != 0 || callReplay["type"] != "mcp_call" ||
		callReplay["server_label"] != "docs" || callReplay["name"] != "search" ||
		callReplay["arguments"] != `{"query":"Go"}` {
		t.Fatalf("unexpected MCP replay: %#v", input)
	}
}

func TestResponsesMCPServerErrors(t *testing.T) {
	tests := []struct {
		name string
		part ai.NativeToolCallPart
		want string
	}{
		{name: "server name", part: ai.NativeToolCallPart{
			ToolName: "mcp_server", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{"action":"list_tools"}`), ProviderName: "openai",
		}, want: "invalid MCP server tool name"},
		{name: "arguments", part: ai.NativeToolCallPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{`), ProviderName: "openai",
		}, want: "parse MCP server arguments"},
		{name: "tool name", part: ai.NativeToolCallPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{"action":"call_tool"}`), ProviderName: "openai",
		}, want: "call tool name must not be empty"},
		{name: "action", part: ai.NativeToolCallPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{"action":"approve"}`), ProviderName: "openai",
		}, want: "invalid MCP server action"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
				_, _ = response.Write([]byte(`{"model":"gpt-5","output":[]}`))
			})
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{
				ProviderName: "openai", Parts: []ai.ResponsePart{test.part},
			}}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected MCP history error: %v", err)
			}
		})
	}

	model := newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"model":"gpt-5","output":[{
			"type":"mcp_call","id":"call","server_label":"docs","name":"search","arguments":"{"
		}]}`))
	})
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "parse MCP tool arguments") {
		t.Fatalf("unexpected malformed MCP response error: %v", err)
	}
	model = newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"model":"gpt-5","output":[{"type":"mcp_approval_request"}]}`))
	})
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "MCP approval requests are not supported") {
		t.Fatalf("unexpected MCP approval error: %v", err)
	}

	var body map[string]any
	model = newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"model":"gpt-5","output":[]}`))
	})
	_, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", Parts: []ai.ResponsePart{ai.NativeToolCallPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{"action":"call_tool","tool_name":"search"}`), ProviderName: "openai",
		}},
	}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if body["input"].([]any)[0].(map[string]any)["arguments"] != `{}` {
		t.Fatalf("empty MCP arguments were not replayed: %#v", body)
	}
}

func TestResponsesFileSearchNativeTool(t *testing.T) {
	var body map[string]any
	model := newResponsesServerWithOptions(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[
				{"type":"file_search_call","id":"search-1","status":"completed","queries":["quarterly revenue"],
				 "results":[{"file_id":"file-1","filename":"report.pdf","score":0.9,"text":"Revenue grew."}]},
				{"type":"message","id":"message","content":[{"type":"output_text","text":"done"}]}
			]
		}`))
	}, openai.WithResponsesFileSearchResults(true))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.FileSearchTool{FileStoreIDs: []string{"vs-1", "vs-2"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	stores := tool["vector_store_ids"].([]any)
	if tool["type"] != "file_search" || len(stores) != 2 || stores[0] != "vs-1" ||
		body["include"].([]any)[0] != "file_search_call.results" {
		t.Fatalf("unexpected file search request: %#v", body)
	}
	if len(response.Parts) != 3 {
		t.Fatalf("unexpected file search response: %#v", response.Parts)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	content := returned.Content.(map[string]any)
	results := content["results"].([]map[string]any)
	if call.ToolKind != ai.ToolPartKindFileSearch || call.ToolCallID != "search-1" ||
		string(call.Args) != `{"queries":["quarterly revenue"]}` || returned.ToolCallID != "search-1" ||
		returned.ToolKind != ai.ToolPartKindFileSearch || content["status"] != "completed" ||
		results[0]["filename"] != "report.pdf" || returned.Timestamp.IsZero() {
		t.Fatalf("unexpected normalized file search: call=%+v return=%+v", call, returned)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	replayed := body["input"].([]any)[0].(map[string]any)
	if replayed["type"] != "file_search_call" || replayed["id"] != "search-1" ||
		replayed["status"] != "completed" || replayed["queries"].([]any)[0] != "quarterly revenue" {
		t.Fatalf("unexpected file search replay: %#v", replayed)
	}
}

func TestResponsesFileSearchDefaultsAndErrors(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"status":"completed","output":[
			{"type":"file_search_call","id":"search","status":"in_progress","queries":[]}]}`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		&ai.FileSearchTool{FileStoreIDs: []string{"vs"}},
	}}); err != nil {
		t.Fatal(err)
	}
	if body["include"] != nil {
		t.Fatalf("file search results were requested by default: %#v", body)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: "file_search", ToolCallID: "bad", ToolKind: ai.ToolPartKindFileSearch,
			ProviderName: "openai", Args: json.RawMessage(`{`),
		},
	}}}, ai.ModelRequestParams{}); err == nil || !strings.Contains(err.Error(), "parse file search arguments") {
		t.Fatalf("unexpected malformed file search replay error: %v", err)
	}
}

func TestResponsesImageGenerationNativeTool(t *testing.T) {
	compression := 75
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[
				{"type":"image_generation_call","id":"image-1","status":"generating","background":"transparent",
				 "quality":"high","size":"1024x1536","revised_prompt":"a better prompt","output_format":"webp",
				 "result":"aW1hZ2U="},
				{"type":"message","id":"message","content":[{"type":"output_text","text":"done"}]}
			]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.ImageGenerationTool{
			Action: ai.ImageGenerationActionEdit, Background: ai.ImageGenerationBackgroundTransparent,
			InputFidelity: ai.ImageGenerationInputFidelityHigh, Moderation: ai.ImageGenerationModerationLow,
			Model: "gpt-image-2", OutputCompression: &compression, OutputFormat: ai.ImageGenerationOutputWebP,
			PartialImages: 2, Quality: ai.ImageGenerationQualityHigh, AspectRatio: ai.ImageAspectRatio2x3,
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "image_generation" || tool["action"] != "edit" || tool["background"] != "transparent" ||
		tool["input_fidelity"] != "high" || tool["moderation"] != "low" || tool["model"] != "gpt-image-2" ||
		tool["output_compression"] != float64(75) || tool["output_format"] != "webp" ||
		tool["partial_images"] != float64(2) || tool["quality"] != "high" || tool["size"] != "1024x1536" {
		t.Fatalf("unexpected image generation request: %#v", tool)
	}
	if len(response.Parts) != 4 {
		t.Fatalf("unexpected image generation response parts: %#v", response.Parts)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	file := response.Parts[1].(ai.FilePart)
	returned := response.Parts[2].(ai.NativeToolReturnPart)
	content := returned.Content.(map[string]any)
	if call.ToolKind != ai.ToolPartKindImageGeneration || call.ToolCallID != "image-1" || len(call.Args) != 0 ||
		file.ID != "image-1" || file.Content.MediaType != "image/webp" || string(file.Content.Data) != "image" ||
		returned.ToolKind != ai.ToolPartKindImageGeneration || content["status"] != "completed" ||
		content["background"] != "transparent" || content["quality"] != "high" ||
		content["size"] != "1024x1536" || content["revised_prompt"] != "a better prompt" {
		t.Fatalf("unexpected normalized image generation: call=%+v file=%+v return=%+v", call, file, returned)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	replayed := body["input"].([]any)[0].(map[string]any)
	if replayed["type"] != "image_generation_call" || replayed["id"] != "image-1" {
		t.Fatalf("unexpected image generation replay: %#v", replayed)
	}
}

func TestResponsesImageGenerationDefaultsAndErrors(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"status":"completed","output":[
			{"type":"image_generation_call","id":"image","status":"in_progress"}]}`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&ai.ImageGenerationTool{}},
	}); err != nil {
		t.Fatal(err)
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	if tool["action"] != "auto" || tool["background"] != "auto" || tool["moderation"] != "auto" ||
		tool["output_compression"] != float64(100) || tool["output_format"] != "png" ||
		tool["partial_images"] != float64(0) || tool["quality"] != "auto" || tool["size"] != "auto" ||
		tool["input_fidelity"] != nil || tool["model"] != nil {
		t.Fatalf("unexpected image generation defaults: %#v", tool)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.ImageGenerationTool{Size: ai.ImageGenerationSize1024x1024},
	}}); err != nil {
		t.Fatal(err)
	}
	if size := body["tools"].([]any)[0].(map[string]any)["size"]; size != "1024x1024" {
		t.Fatalf("unexpected explicit image size: %v", size)
	}
	for _, test := range []struct {
		name string
		tool ai.ImageGenerationTool
		want string
	}{
		{name: "Google size", tool: ai.ImageGenerationTool{Size: ai.ImageGenerationSize4K}, want: "unsupported image generation size"},
		{name: "Google aspect", tool: ai.ImageGenerationTool{AspectRatio: ai.ImageAspectRatio16x9}, want: "unsupported image generation aspect ratio"},
		{name: "conflict", tool: ai.ImageGenerationTool{
			AspectRatio: ai.ImageAspectRatio1x1, Size: ai.ImageGenerationSize1024x1536,
		}, want: "conflicts with size"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{test.tool}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected image tool error: %v", err)
			}
		})
	}

	model = newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"status":"completed","output":[
			{"type":"image_generation_call","id":"image","result":"!"}]}`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "decode generated image") {
		t.Fatalf("unexpected generated image error: %v", err)
	}
}

func TestResponsesNativeToolCompatibility(t *testing.T) {
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["tools"] != nil {
			t.Errorf("optional unsupported tool was sent: %#v", body)
		}
		_, _ = response.Write([]byte(`{"model":"gpt-5","output":[],"usage":{}}`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{unsupportedNativeTool{optional: true}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{unsupportedNativeTool{}},
	}); err == nil || !strings.Contains(err.Error(), `does not support native tool "unsupported"`) {
		t.Fatalf("unexpected unsupported tool error: %v", err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{SearchContextSize: "huge"}},
	}); err == nil || !strings.Contains(err.Error(), "invalid web search context size") {
		t.Fatalf("unexpected web search validation error: %v", err)
	}
	var nilTool *ai.WebSearchTool
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{nilTool},
	}); err == nil || !strings.Contains(err.Error(), "must not be nil") {
		t.Fatalf("unexpected nil native tool error: %v", err)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{ToolName: "unknown", ProviderName: "openai"},
	}}}, ai.ModelRequestParams{}); err != nil {
		t.Fatalf("unrelated native history should be omitted: %v", err)
	}
}

func TestResponsesWebSearchWithoutAction(t *testing.T) {
	model := newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"model":"gpt-5","output":[{"type":"web_search_call","id":"web","status":"failed"}]}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if call := response.Parts[0].(ai.NativeToolCallPart); string(call.Args) != `{}` {
		t.Fatalf("missing web search action was not normalized: %+v", call)
	}
}

func TestResponsesMultimodalInput(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}`))
	})
	result, err := ai.NewAgent[struct{}, string](model).RunParts(t.Context(), []ai.UserContent{
		ai.TextContent{Text: "describe"},
		ai.ImageURL{
			URL: "https://example.com/image.png", VendorMetadata: map[string]any{"detail": "low"},
		},
		ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
		ai.AudioURL{URL: "https://example.com/audio.mp3"},
		ai.DocumentURL{URL: "https://example.com/document.pdf"},
		ai.BinaryContent{Data: []byte("document"), MediaType: "application/pdf"},
		ai.UploadedFile{
			FileID: "file-image.png", ProviderName: "openai", VendorMetadata: map[string]any{"detail": "high"},
		},
		ai.UploadedFile{FileID: "file-document", ProviderName: "openai", MediaType: "application/pdf"},
		ai.UploadedFile{FileID: "file-image-auto", ProviderName: "openai", MediaType: "image/jpeg"},
	}, struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected multimodal result=%+v err=%v", result, err)
	}
	input := body["input"].([]any)
	content := input[0].(map[string]any)["content"].([]any)
	if len(content) != 9 || content[0].(map[string]any)["type"] != "input_text" ||
		content[0].(map[string]any)["text"] != "describe" ||
		content[1].(map[string]any)["image_url"] != "https://example.com/image.png" ||
		content[1].(map[string]any)["detail"] != "low" ||
		content[2].(map[string]any)["image_url"] != "data:image/png;base64,aW1hZ2U=" ||
		content[2].(map[string]any)["detail"] != "auto" ||
		content[3].(map[string]any)["file_url"] != "https://example.com/audio.mp3" ||
		content[4].(map[string]any)["file_url"] != "https://example.com/document.pdf" ||
		content[5].(map[string]any)["file_data"] != "data:application/pdf;base64,ZG9jdW1lbnQ=" ||
		content[5].(map[string]any)["filename"] != "filename.pdf" ||
		content[6].(map[string]any)["type"] != "input_image" ||
		content[6].(map[string]any)["file_id"] != "file-image.png" ||
		content[6].(map[string]any)["detail"] != "high" ||
		content[7].(map[string]any)["type"] != "input_file" ||
		content[7].(map[string]any)["file_id"] != "file-document" ||
		content[8].(map[string]any)["detail"] != "auto" {
		t.Fatalf("unexpected Responses multimodal content: %#v", content)
	}
}

func TestResponsesForcedFileURLs(t *testing.T) {
	fileServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/image":
			response.Header().Set("Content-Type", "image/png")
		case "/audio":
			response.Header().Set("Content-Type", "audio/mpeg")
		case "/document":
			response.Header().Set("Content-Type", "application/pdf")
		case "/unknown.bin":
			response.Header().Set("Content-Type", "application/unknown")
		default:
			response.Header().Set("Content-Type", "application/octet-stream")
		}
		_, _ = response.Write([]byte("file"))
	}))
	defer fileServer.Close()
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}`))
	})
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.ImageURL{URL: fileServer.URL + "/image", ForceDownload: ai.FileDownloadAllowLocal},
		ai.AudioURL{URL: fileServer.URL + "/audio", ForceDownload: ai.FileDownloadAllowLocal},
		ai.DocumentURL{URL: fileServer.URL + "/document", ForceDownload: ai.FileDownloadAllowLocal},
	}}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	content := body["input"].([]any)[0].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["image_url"] != "data:image/png;base64,ZmlsZQ==" ||
		content[1].(map[string]any)["file_data"] != "data:audio/mpeg;base64,ZmlsZQ==" ||
		content[1].(map[string]any)["filename"] != "filename.mp3" ||
		content[2].(map[string]any)["file_data"] != "data:application/pdf;base64,ZmlsZQ==" ||
		content[2].(map[string]any)["filename"] != "filename.pdf" {
		t.Fatalf("unexpected forced file content: %#v", content)
	}

	for name, item := range map[string]ai.UserContent{
		"image mode": ai.ImageURL{URL: "https://example.com/image.png", ForceDownload: "invalid"},
		"blocked image": ai.ImageURL{
			URL: fileServer.URL + "/image", ForceDownload: ai.FileDownloadSafe,
		},
		"audio mode": ai.AudioURL{URL: "https://example.com/audio.mp3", ForceDownload: "invalid"},
		"download media": ai.DocumentURL{
			URL: fileServer.URL + "/unknown.bin", ForceDownload: ai.FileDownloadAllowLocal,
		},
		"infer media": ai.DocumentURL{
			URL: fileServer.URL + "/unknown", ForceDownload: ai.FileDownloadAllowLocal,
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{item}},
			}}}, ai.ModelRequestParams{})
			if err == nil {
				t.Fatal("invalid forced file URL succeeded")
			}
		})
	}
}

func TestResponsesPromptCachePoints(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = response.Write([]byte(`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}`))
	}))
	defer server.Close()
	options := []openai.Option{openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client())}
	gpt := openai.NewResponsesModel("openai.gpt-5.6", options...)
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "cache me"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
	}}}}}
	if _, err := gpt.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	content := bodies[0]["input"].([]any)[0].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["prompt_cache_breakpoint"].(map[string]any)["mode"] != "explicit" {
		t.Fatalf("unexpected Responses cache breakpoint: %#v", content)
	}
	older := openai.NewResponsesModel("gpt-5.5", options...)
	if _, err := older.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	content = bodies[1]["input"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 1 || content[0].(map[string]any)["prompt_cache_breakpoint"] != nil {
		t.Fatalf("unsupported Responses cache point leaked: %#v", content)
	}
	for name, contents := range map[string][]ai.UserContent{
		"first":   {ai.CachePoint{}, ai.TextContent{Text: "later"}},
		"invalid": {ai.TextContent{Text: "first"}, ai.CachePoint{TTL: "1d"}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := gpt.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: contents},
			}}}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), "cache point") {
				t.Fatalf("unexpected Responses cache-point error: %v", err)
			}
		})
	}
}

func TestResponsesRejectsUnsupportedUserContent(t *testing.T) {
	text := ai.TextContent{Text: "pointer"}
	tests := []struct {
		name    string
		content ai.UserContent
		want    string
	}{
		{name: "pointer", content: &text, want: "unsupported Responses user content type *ai.TextContent"},
		{name: "audio binary", content: ai.BinaryContent{
			Data: []byte("audio"), MediaType: "audio/mpeg",
		}, want: `Responses does not support inline audio/mpeg input`},
		{name: "video binary", content: ai.BinaryContent{
			Data: []byte("video"), MediaType: "video/mp4",
		}, want: `Responses does not support inline video/mp4 input`},
		{name: "unsupported binary", content: ai.BinaryContent{
			Data: []byte("data"), MediaType: "application/json",
		}, want: `unsupported file media type "application/json"`},
		{name: "video URL", content: ai.VideoURL{
			URL: "https://example.com/video.mp4",
		}, want: `Responses does not support video URL input`},
		{name: "foreign uploaded file", content: ai.UploadedFile{
			FileID: "file", ProviderName: "anthropic", MediaType: "image/png",
		}, want: `uploaded file "file" belongs to provider "anthropic"`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newResponsesServer(t, func(http.ResponseWriter, *http.Request) {
				t.Fatal("request sent with unsupported content")
			})
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{test.content}},
			}}}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected content error: %v", err)
			}
		})
	}
}

func TestResponsesPhaseReplayUsesModelProfileAndOverride(t *testing.T) {
	tests := []struct {
		name         string
		modelName    string
		options      []openai.Option
		providerName string
		phase        string
		wantPhase    bool
	}{
		{name: "gpt 5.3 codex", modelName: "gpt-5.3-codex", phase: "commentary", wantPhase: true},
		{name: "gpt 5.4", modelName: "gpt-5.4", phase: "final_answer", wantPhase: true},
		{name: "gpt 5.5", modelName: "gpt-5.5-mini", phase: "commentary", wantPhase: true},
		{name: "gpt 5.6", modelName: "gpt-5.6-terra", phase: "final_answer", wantPhase: true},
		{name: "gpt 6 Astra", modelName: "gpt-6-astra", phase: "final_answer", wantPhase: true},
		{name: "Bedrock model ID", modelName: "openai.gpt-5.6-luna", phase: "final_answer", wantPhase: true},
		{name: "unsupported", modelName: "gpt-5", phase: "commentary"},
		{name: "enabled override", modelName: "gpt-5", options: []openai.Option{
			openai.WithResponsesPhaseSupport(true),
		}, phase: "commentary", wantPhase: true},
		{name: "disabled override", modelName: "gpt-5.4", options: []openai.Option{
			openai.WithResponsesPhaseSupport(false),
		}, phase: "commentary"},
		{name: "foreign provider", modelName: "gpt-5.4", providerName: "other", phase: "commentary"},
		{name: "unknown phase", modelName: "gpt-5.4", phase: "analysis"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var body map[string]any
			options := append([]openai.Option(nil), test.options...)
			options = append(options, openai.WithHTTPClient(&http.Client{Transport: compactionRoundTripFunc(func(request *http.Request) (*http.Response, error) {
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				return &http.Response{
					StatusCode: http.StatusOK, Header: http.Header{},
					Body: io.NopCloser(strings.NewReader(`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}`)),
				}, nil
			})}))
			model := openai.NewResponsesModel(test.modelName, options...)
			providerName := test.providerName
			if providerName == "" {
				providerName = "openai"
			}
			history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{
				Content: "earlier", ProviderName: providerName,
				ProviderDetails: map[string]any{"phase": test.phase},
			}}}}
			if _, err := model.Request(t.Context(), append(history, ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Content: "continue"},
			}}), ai.ModelRequestParams{}); err != nil {
				t.Fatal(err)
			}
			input := body["input"].([]any)
			_, hasPhase := input[0].(map[string]any)["phase"]
			if hasPhase != test.wantPhase {
				t.Fatalf("phase replay=%v, want %v: %#v", hasPhase, test.wantPhase, input[0])
			}
		})
	}
}

func TestResponsesTextMetadataWithoutLogprobs(t *testing.T) {
	model := newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{
			"status":"completed","output":[{"id":"message","type":"message","phase":"commentary","content":[
				{"type":"output_text","text":"annotated","annotations":[{"type":"url_citation"}]},
				{"type":"output_text","text":"phase only"}
			]}]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: rawAnnotationSettings(t)})
	if err != nil {
		t.Fatal(err)
	}
	first := response.Parts[0].(ai.TextPart)
	second := response.Parts[1].(ai.TextPart)
	if len(first.ProviderDetails["annotations"].([]map[string]any)) != 1 ||
		first.ProviderDetails["phase"] != "commentary" || second.ProviderDetails["phase"] != "commentary" {
		t.Fatalf("unexpected text metadata: first=%+v second=%+v", first, second)
	}

	withoutAnnotations, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	first = withoutAnnotations.Parts[0].(ai.TextPart)
	if _, exists := first.ProviderDetails["annotations"]; exists || first.ProviderDetails["phase"] != "commentary" {
		t.Fatalf("raw annotations were retained by default: %+v", first)
	}
}

func TestResponsesReasoningTextContent(t *testing.T) {
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{
			"id":"response","model":"openai.gpt-oss-20b","status":"completed","output":[{
				"id":"reasoning","type":"reasoning","encrypted_content":"signature",
				"content":[{"type":"ignored","text":"ignored"},{"type":"reasoning_text","text":"detail"}],
				"summary":[{"text":"summary"}]
			}]
		}`)
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Parts) != 2 || response.Parts[0].(ai.ThinkingPart).Content != "detail" ||
		response.Parts[0].(ai.ThinkingPart).Signature != "signature" ||
		response.Parts[1].(ai.ThinkingPart).Content != "summary" ||
		response.Parts[1].(ai.ThinkingPart).Signature != "" {
		t.Fatalf("unexpected reasoning parts: %#v", response.Parts)
	}
}

func TestResponsesTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotPath, gotCustom string
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotCustom = r.Header.Get("x-custom")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id": "response-1", "model": "gpt-5", "created_at": 1735689600.25,
			"status": "completed", "service_tier": "default",
			"output": [
				{"id": "reasoning-1", "type": "reasoning", "encrypted_content": "signature", "summary": [{"text": "thinking"}]},
				{"id": "message-1", "type": "message", "phase": "final_answer", "content": [{
					"type": "output_text", "text": "Hello!", "logprobs": [{"token": "Hello", "logprob": -0.1}],
					"annotations": [{"type": "url_citation", "url": "https://example.com"}]
				}]}
			],
			"usage": {
				"input_tokens": 12, "output_tokens": 5,
				"input_tokens_details": {"cached_tokens": 4},
				"output_tokens_details": {"reasoning_tokens": 2}
			}
		}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}
	temperature := 0.5
	logprobs := true
	topLogprobs := 3
	settings := rawAnnotationSettings(t)
	settings.Thinking = &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}
	settings.Temperature = &temperature
	settings.Logprobs = &logprobs
	settings.TopLogprobs = &topLogprobs
	settings.ServiceTier = ai.ServiceTierPriority
	settings.ExtraHeaders = map[string]string{"x-custom": "value"}
	settings.ExtraBody["store"] = true
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{
		Instructions: "be brief", AllowText: true, Settings: settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/responses" || gotCustom != "value" {
		t.Fatalf("unexpected path %q or custom header %q", gotPath, gotCustom)
	}
	if _, exists := gotBody["openai_include_raw_annotations"]; exists {
		t.Fatalf("local annotation setting entered provider body: %v", gotBody)
	}
	if gotBody["instructions"] != "be brief" || gotBody["temperature"] != nil ||
		gotBody["reasoning"].(map[string]any)["effort"] != "high" ||
		gotBody["top_logprobs"].(float64) != float64(topLogprobs) ||
		gotBody["include"].([]any)[0] != "message.output_text.logprobs" || gotBody["service_tier"] != "priority" ||
		gotBody["store"] != true {
		t.Fatalf("instructions or reasoning not sent: %v", gotBody)
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	thinking, ok := resp.Parts[0].(ai.ThinkingPart)
	if !ok || thinking.ID != "reasoning-1" || thinking.Signature != "signature" ||
		thinking.ProviderName != "openai" {
		t.Fatalf("reasoning metadata lost: %+v", resp.Parts)
	}
	text := resp.Parts[1].(ai.TextPart)
	if text.ID != "message-1" || text.ProviderName != "openai" ||
		len(text.ProviderDetails["logprobs"].([]map[string]any)) != 1 ||
		len(text.ProviderDetails["annotations"].([]map[string]any)) != 1 ||
		text.ProviderDetails["phase"] != "final_answer" {
		t.Fatalf("text metadata lost: %+v", text)
	}
	if resp.ProviderName != "openai" || resp.ProviderURL == "" || resp.ProviderResponseID != "response-1" ||
		resp.FinishReason != ai.FinishReasonStop || resp.State != ai.ModelResponseStateComplete ||
		resp.ProviderDetails["finish_reason"] != "completed" || resp.ProviderDetails["timestamp"] == nil ||
		resp.ProviderDetails["service_tier"] != "default" {
		t.Fatalf("unexpected response metadata %+v", resp)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 5 ||
		resp.Usage.CacheReadTokens != 4 || resp.Usage.ReasoningTokens != 2 ||
		resp.Usage.Details["reasoning_tokens"] != 2 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
}

func TestResponsesRejectInvalidThinkingLevel(t *testing.T) {
	model := openai.NewResponsesModel("gpt-5")
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: "extreme"},
	}})
	if err == nil || err.Error() != `openai: invalid thinking level "extreme"` {
		t.Fatalf("unexpected invalid thinking error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: "expedited",
	}})
	if err == nil || err.Error() != `openai: invalid service tier "expedited"` {
		t.Fatalf("unexpected invalid service tier error: %v", err)
	}
}

func TestResponsesPreservesEncryptedReasoningWithoutSummary(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model":"gpt-5","service_tier":"priority",
			"output":[{"id":"reasoning-1","type":"reasoning","encrypted_content":"signature"}]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	thinking := response.Parts[0].(ai.ThinkingPart)
	if thinking.Content != "" || thinking.ID != "reasoning-1" || thinking.Signature != "signature" ||
		thinking.ProviderName != "openai" || response.ProviderDetails["service_tier"] != "priority" {
		t.Fatalf("encrypted reasoning was not retained: %+v", thinking)
	}
}

func TestResponsesPendingStateMetadata(t *testing.T) {
	for name, test := range map[string]struct {
		status     string
		reason     string
		background bool
		want       ai.ModelResponseState
		wantFinish ai.FinishReason
	}{
		"foreground": {status: "queued", want: ai.ModelResponseStateIncomplete},
		"background": {status: "queued", background: true, want: ai.ModelResponseStateSuspended},
		"incomplete reason": {
			status: "incomplete", reason: "max_output_tokens",
			want: ai.ModelResponseStateComplete, wantFinish: ai.FinishReasonLength,
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
				details := ""
				if test.reason != "" {
					details = fmt.Sprintf(`,"incomplete_details":{"reason":%q}`, test.reason)
				}
				_, _ = fmt.Fprintf(w,
					`{"id":"pending","model":"gpt-5","status":%q,"background":%t%s}`,
					test.status, test.background, details,
				)
			})
			response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			wantRawReason := test.status
			if test.reason != "" {
				wantRawReason = test.reason
			}
			if response.State != test.want || response.FinishReason != test.wantFinish ||
				response.ProviderDetails["finish_reason"] != wantRawReason ||
				response.ProviderDetails["background"] != test.background && test.background {
				t.Fatalf("unexpected pending metadata: %+v", response)
			}
		})
	}
}

func TestResponsesAgentContinuesBackgroundResponse(t *testing.T) {
	var methods, paths []string
	var initialBody map[string]any
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		methods = append(methods, r.Method)
		paths = append(paths, r.URL.Path)
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing authorization header: %q", r.Header.Get("Authorization"))
		}
		if r.Method == http.MethodPost {
			if err := json.NewDecoder(r.Body).Decode(&initialBody); err != nil {
				t.Error(err)
			}
			_, _ = w.Write([]byte(`{
				"id":"job","model":"gpt-5","created_at":1735689600,
				"status":"queued","background":true,"usage":{"input_tokens":2}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"job","model":"gpt-5","created_at":1735689601,
			"status":"completed","background":true,
			"output":[{"id":"message","type":"message","content":[{
				"type":"output_text","text":"done","annotations":[{"type":"url_citation","url":"https://example.com"}]
			}]}],
			"usage":{"input_tokens":2,"output_tokens":1}
		}`))
	}, openai.WithBackgroundMode(true), openai.WithBackgroundPollInterval(0))
	result, err := ai.NewAgent[struct{}, string](
		model, ai.WithModelSettings(rawAnnotationSettings(t)),
	).Run(t.Context(), "go", struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected background result=%+v err=%v", result, err)
	}
	if !slices.Equal(methods, []string{http.MethodPost, http.MethodGet}) ||
		!slices.Equal(paths, []string{"/responses", "/responses/job"}) || initialBody["background"] != true {
		t.Fatalf("unexpected background requests methods=%v paths=%v body=%v", methods, paths, initialBody)
	}
	if usage := result.Usage(); usage.Requests != 1 || usage.InputTokens != 2 || usage.OutputTokens != 1 {
		t.Fatalf("background usage was double counted: %+v", usage)
	}
	response := result.NewMessages()[1].(ai.ModelResponse)
	text := response.Parts[0].(ai.TextPart)
	if response.Timestamp.IsZero() || len(text.ProviderDetails["annotations"].([]map[string]any)) != 1 {
		t.Fatalf("retrieved response metadata was lost: %+v", response)
	}
}

func TestResponsesBackgroundLifecycle(t *testing.T) {
	t.Run("delay", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBackgroundPollInterval(time.Millisecond))
		if delay := model.ContinuationDelay(ai.ModelResponse{
			State: ai.ModelResponseStateSuspended, ProviderDetails: map[string]any{"background": true},
		}); delay != time.Millisecond {
			t.Fatalf("unexpected background delay: %s", delay)
		}
		if delay := model.ContinuationDelay(ai.ModelResponse{State: ai.ModelResponseStateComplete}); delay != 0 {
			t.Fatalf("completed response had delay: %s", delay)
		}
	})

	t.Run("cancel", func(t *testing.T) {
		calls := 0
		model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
			calls++
			if r.Method != http.MethodPost || r.URL.Path != "/responses/job/cancel" {
				t.Fatalf("unexpected cancellation request %s %s", r.Method, r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"id":"job","status":"cancelled"}`))
		})
		if err := model.CancelSuspendedResponse(t.Context(), ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
		}); err != nil {
			t.Fatal(err)
		}
		if err := model.CancelSuspendedResponse(t.Context(), ai.ModelResponse{ProviderName: "other"}); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("unexpected cancellation calls: %d", calls)
		}
	})

	t.Run("cancel error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "cannot cancel", http.StatusConflict)
		})
		err := model.CancelSuspendedResponse(t.Context(), ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
		})
		var apiErr *openai.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusConflict {
			t.Fatalf("unexpected cancel error: %v", err)
		}
	})
}

func TestResponsesBackgroundRequestFailures(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true},
	}}
	t.Run("invalid retrieve URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve URL error")
		}
	})
	t.Run("retrieve transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve transport error")
		}
	})
	t.Run("retrieve read", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			_, _ = w.Write([]byte("short"))
		})
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve read error")
		}
	})
	t.Run("retrieve HTTP", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "missing", http.StatusNotFound)
		})
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve API error")
		}
	})

	response := ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
	}
	t.Run("invalid cancel URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if err := model.CancelSuspendedResponse(t.Context(), response); err == nil {
			t.Fatal("expected cancel URL error")
		}
	})
	t.Run("cancel transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if err := model.CancelSuspendedResponse(t.Context(), response); err == nil {
			t.Fatal("expected cancel transport error")
		}
	})
	t.Run("cancel read", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			_, _ = w.Write([]byte("short"))
		})
		if err := model.CancelSuspendedResponse(t.Context(), response); err == nil {
			t.Fatal("expected cancel read error")
		}
	})
}

func TestBackgroundPollIntervalRejectsNegativeValues(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != "openai: background poll interval must not be negative" {
			t.Fatalf("unexpected panic: %v", recovered)
		}
	}()
	_ = openai.WithBackgroundPollInterval(-time.Second)
}

func TestResponsesToolCallRoundTrip(t *testing.T) {
	var gotBody map[string]any
	first := true
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"model": "gpt-5",
				"output": [{"id": "function-1", "type": "function_call", "call_id": "c1", "name": "get_weather", "namespace": "weather", "arguments": "{\"city\":\"SF\"}"}],
				"usage": {"input_tokens": 20, "output_tokens": 8}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"model": "gpt-5",
			"output": [{"type": "message", "content": [{"type": "output_text", "text": "Sunny."}]}],
			"usage": {"input_tokens": 30, "output_tokens": 4}
		}`))
	})
	params := ai.ModelRequestParams{
		Tools:     []ai.ToolDefinition{{Name: "get_weather", Description: "d", Schema: map[string]any{"type": "object"}}},
		AllowText: true,
	}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "weather?"}}}}
	resp, err := model.Request(t.Context(), msgs, params)
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ToolCallID != "c1" || calls[0].ID != "function-1" ||
		calls[0].ProviderName != "openai" || calls[0].ProviderDetails["namespace"] != "weather" {
		t.Fatalf("unexpected calls %+v", calls)
	}
	msgs = append(msgs, *resp, ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny", ToolCallID: "c1"},
		ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"archive"}, ToolCallID: "c1"},
	}})
	if _, err := model.Request(t.Context(), msgs, params); err != nil {
		t.Fatal(err)
	}
	input := gotBody["input"].([]any)
	callItem := input[1].(map[string]any)
	if callItem["type"] != "function_call" || callItem["call_id"] != "c1" || callItem["id"] != "function-1" ||
		callItem["namespace"] != "weather" {
		t.Fatalf("function call not echoed: %v", callItem)
	}
	outputItem := input[2].(map[string]any)
	if outputItem["type"] != "function_call_output" || outputItem["output"] != "sunny" {
		t.Fatalf("function output not sent: %v", outputItem)
	}
	if len(gotBody["tools"].([]any)) != 1 {
		t.Fatalf("tools not sent: %v", gotBody["tools"])
	}
}

func TestResponsesNativeDeferredToolSearch(t *testing.T) {
	var gotBody map[string]any
	request := 0
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		request++
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if request == 1 {
			_, _ = w.Write([]byte(`{
				"id":"response-search","model":"gpt-5","status":"completed",
				"output":[{
					"id":"search-item","type":"tool_search_call","call_id":"search-call",
					"execution":"client","status":"completed","arguments":{"queries":["first"]}
				}],"usage":{}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"response-final","model":"gpt-5","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{}
		}`))
	})
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	search := ai.ToolDefinition{
		Name: ai.ToolSearchName, Description: "Find tools.", Schema: map[string]any{
			"type": "object", "properties": map[string]any{
				"queries": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			},
		},
		ToolKind: ai.ToolPartKindToolSearch, ToolSearchStrategy: ai.ToolSearchStrategyCustom,
	}
	first := ai.ToolDefinition{Name: "first", Schema: schema, DeferLoading: true}
	second := ai.ToolDefinition{Name: "second", Schema: schema, DeferLoading: true}
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{search}, DeferredTools: []ai.ToolDefinition{first, second}, AllowText: true,
	}
	if !model.SupportsToolAvailabilityDelta(params) || model.SupportsToolAvailabilityDelta(ai.ModelRequestParams{
		DeferredTools: []ai.ToolDefinition{first},
	}) {
		t.Fatal("unexpected deferred tool support")
	}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "find"}}}}
	response, err := model.Request(t.Context(), messages, params)
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.ToolCallPart)
	if call.ToolName != ai.ToolSearchName || call.ToolKind != ai.ToolPartKindToolSearch ||
		call.ToolCallID != "search-call" || call.ID != "search-item" || string(call.Args) != `{"queries":["first"]}` ||
		call.ProviderDetails["execution"] != "client" || call.ProviderDetails["status"] != "completed" {
		t.Fatalf("unexpected client tool-search call: %+v", call)
	}
	wireTools := gotBody["tools"].([]any)
	if len(wireTools) != 3 || wireTools[0].(map[string]any)["name"] != "first" ||
		wireTools[0].(map[string]any)["defer_loading"] != true ||
		wireTools[1].(map[string]any)["name"] != "second" ||
		wireTools[2].(map[string]any)["type"] != "tool_search" ||
		wireTools[2].(map[string]any)["execution"] != "client" || wireTools[2].(map[string]any)["name"] != nil {
		t.Fatalf("unexpected native deferred tools: %+v", wireTools)
	}

	messages = append(messages, *response,
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search-call", ToolKind: ai.ToolPartKindToolSearch,
				Content: ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "first"}}},
			},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"first"}, ToolCallID: "search-call"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "loader", ToolCallID: "loader", Args: []byte(`{}`),
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "loader", ToolCallID: "loader", Content: "loaded"},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"unknown", "second"}, ToolCallID: "loader"},
		}},
	)
	if _, err := model.Request(t.Context(), messages, params); err != nil {
		t.Fatal(err)
	}
	input := gotBody["input"].([]any)
	if len(input) != 6 {
		t.Fatalf("unexpected native search replay items: %+v", input)
	}
	searchCall := input[1].(map[string]any)
	searchOutput := input[2].(map[string]any)
	if searchCall["type"] != "tool_search_call" || searchCall["execution"] != "client" ||
		searchCall["arguments"].(map[string]any)["queries"].([]any)[0] != "first" ||
		searchOutput["type"] != "tool_search_output" || searchOutput["execution"] != "client" ||
		len(searchOutput["tools"].([]any)) != 1 || searchOutput["tools"].([]any)[0].(map[string]any)["name"] != "first" ||
		searchOutput["tools"].([]any)[0].(map[string]any)["defer_loading"] != nil {
		t.Fatalf("unexpected native search replay: call=%+v output=%+v", searchCall, searchOutput)
	}
	additional := input[5].(map[string]any)
	if additional["type"] != "additional_tools" || additional["role"] != "developer" ||
		len(additional["tools"].([]any)) != 1 || additional["tools"].([]any)[0].(map[string]any)["name"] != "second" {
		t.Fatalf("unexpected additional tools item: %+v", additional)
	}
}

func TestResponsesDeferredToolSupportCanBeDisabled(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","status":"completed","output":[],"usage":{}}`))
	}, openai.WithDeferredToolSupport(false))
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	}); err != nil {
		t.Fatal(err)
	}
	tools := gotBody["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["type"] != "function" ||
		tools[0].(map[string]any)["name"] != ai.ToolSearchName {
		t.Fatalf("deferred support override was ignored: %+v", tools)
	}
}

func TestResponsesNativeDeferredErrors(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"gpt-5","status":"completed","output":[],"usage":{}}`))
	})
	validSchema := map[string]any{"type": "object", "properties": map[string]any{}}
	invalidSchema := map[string]any{"type": "string"}
	search := ai.ToolDefinition{
		Name: ai.ToolSearchName, Schema: validSchema, ToolKind: ai.ToolPartKindToolSearch,
		ToolSearchStrategy: ai.ToolSearchStrategyCustom,
	}
	deferred := ai.ToolDefinition{Name: "hidden", Schema: validSchema, DeferLoading: true}
	searchReturn := func(content any) []ai.ModelMessage {
		return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			Content: content,
		}}}}
	}
	invalidDeferred := deferred
	invalidDeferred.Schema = invalidSchema
	invalidSearch := search
	invalidSearch.Schema = invalidSchema
	for name, test := range map[string]struct {
		messages []ai.ModelMessage
		search   ai.ToolDefinition
		deferred ai.ToolDefinition
	}{
		"deferred definition": {search: search, deferred: invalidDeferred},
		"search definition":   {search: invalidSearch, deferred: deferred},
		"search result marshal": {
			messages: searchReturn(make(chan int)), search: search, deferred: deferred,
		},
		"search result parse": {
			messages: searchReturn("bad"), search: search, deferred: deferred,
		},
		"search result definition": {
			messages: searchReturn(ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "hidden"}}}),
			search:   search, deferred: invalidDeferred,
		},
		"additional definition": {
			messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"hidden"}},
			}}},
			search: search, deferred: invalidDeferred,
		},
		"historical search arguments": {
			messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, Args: []byte(`{`),
			}}}},
			search: search, deferred: deferred,
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), test.messages, ai.ModelRequestParams{
				Tools: []ai.ToolDefinition{test.search}, DeferredTools: []ai.ToolDefinition{test.deferred},
			})
			if err == nil {
				t.Fatal("expected native deferred conversion error")
			}
		})
	}
}

func TestResponsesOutputToolAndRetries(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"function_call","call_id":"c1","name":"final_result","arguments":"{}"}],"usage":{}}`))
	})
	params := ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}},
	}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "sys"},
		ai.RetryPromptPart{Content: "bad", ToolCallID: "c0"},
		ai.RetryPromptPart{Content: "plain"},
	}}}
	if _, err := model.Request(t.Context(), msgs, params); err != nil {
		t.Fatal(err)
	}
	if gotBody["tool_choice"] != "required" {
		t.Fatalf("expected required tool choice, got %v", gotBody["tool_choice"])
	}
	input := gotBody["input"].([]any)
	if input[0].(map[string]any)["role"] != "system" {
		t.Fatalf("system part lost: %v", input[0])
	}
	if input[1].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("tool retry should be function output: %v", input[1])
	}
	if input[2].(map[string]any)["role"] != "user" {
		t.Fatalf("plain retry should be user: %v", input[2])
	}
}

func TestResponsesToolSearchResponseErrorsAndFallbacks(t *testing.T) {
	for name, item := range map[string]string{
		"invalid function arguments":      `{"type":"function_call","call_id":"call","name":"work","arguments":"{"}`,
		"invalid client search arguments": `{"type":"tool_search_call","call_id":"search","execution":"client","arguments":"{"}`,
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(w, `{"model":"gpt-5","status":"completed","output":[%s]}`, item)
			})
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected response conversion error")
			}
		})
	}

	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model":"gpt-5","status":"completed","output":[
				{"id":"fallback","type":"tool_search_call","execution":"client","arguments":null},
				{"type":"function_call","call_id":"empty","name":"work"}
			]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	search := response.Parts[0].(ai.ToolCallPart)
	function := response.Parts[1].(ai.ToolCallPart)
	if search.ToolCallID != "fallback" || string(search.Args) != `{}` || string(function.Args) != `{}` {
		t.Fatalf("missing argument fallbacks were not normalized: %+v", response.Parts)
	}
}

type responsesNativeResult struct {
	City string `json:"city"`
}

func TestResponsesNativeOutputAgent(t *testing.T) {
	model := newResponsesServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"model":"gpt-5","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"{\"city\":\"Paris\"}"}]}]}`))
	})
	result, err := ai.NewAgent[struct{}, responsesNativeResult](
		model, ai.WithOutputMode(ai.OutputModeNative),
	).Run(t.Context(), "capital", struct{}{})
	if err != nil || result.Output.City != "Paris" {
		t.Fatalf("unexpected native output result=%+v err=%v", result, err)
	}
}

func TestResponsesErrors(t *testing.T) {
	t.Run("native output", func(t *testing.T) {
		var requests []map[string]any
		model := newResponsesServer(t, func(w http.ResponseWriter, request *http.Request) {
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			requests = append(requests, body)
			_, _ = w.Write([]byte(`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"{}"}]}]}`))
		})
		params := ai.ModelRequestParams{
			OutputSchema: map[string]any{"type": "object", "additionalProperties": false},
			OutputMode:   ai.OutputModeNative,
		}
		if _, err := model.Request(t.Context(), nil, params); err != nil {
			t.Fatal(err)
		}
		format := requests[0]["text"].(map[string]any)["format"].(map[string]any)
		if format["type"] != "json_schema" || format["name"] != "final_result" || format["strict"] != true ||
			format["schema"].(map[string]any)["type"] != "object" {
			t.Fatalf("unexpected native output format: %#v", format)
		}
		params.OutputMode = ai.OutputModePrompted
		if _, err := model.Request(t.Context(), nil, params); err != nil {
			t.Fatalf("prompted output should not request native mode: %v", err)
		}
		if requests[1]["text"] != nil {
			t.Fatalf("prompted output sent a native format: %#v", requests[1])
		}
	})
	t.Run("invalid native output schema", func(t *testing.T) {
		model := newResponsesServer(t, func(http.ResponseWriter, *http.Request) {
			t.Fatal("request sent with invalid output schema")
		})
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			OutputSchema: map[string]any{
				"type": "object", "properties": map[string]any{"values": map[string]any{"type": "array"}},
			},
			OutputMode: ai.OutputModeNative,
		})
		if err == nil || !strings.Contains(err.Error(), "output schema") {
			t.Fatalf("unexpected native schema error: %v", err)
		}
	})

	t.Run("api error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad"))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("invalid json", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("not json")) })
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected parse error")
		}
	})
	t.Run("unknown message type", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		if _, err := model.Request(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown request part", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{nil}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unserializable tool return", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "t", Content: make(chan int)}}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unserializable tool schema", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "t", Schema: map[string]any{
			"type": "object", "properties": map[string]any{}, "required": []string{}, "bad": make(chan int),
		}}}}
		if _, err := model.Request(t.Context(), nil, params); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("transport error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("m", openai.WithBaseURL(server.URL))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := openai.NewResponsesModel("m", openai.WithBaseURL("http://[::1"))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "1000")
			_, _ = w.Write([]byte(`{"model`))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestResponsesModelName(t *testing.T) {
	if openai.NewResponsesModel("gpt-5").Name() != "gpt-5" {
		t.Fatal("unexpected name")
	}
}

func vcrResponsesModel(t *testing.T, name string) *openai.ResponsesModel {
	t.Helper()
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("OPENAI_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	r, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(i *cassette.Interaction) error {
			delete(i.Request.Headers, "Authorization")
			return nil
		}, recorder.AfterCaptureHook),
		recorder.WithMatcher(cassette.MatcherFunc(func(r *http.Request, i cassette.Request) bool {
			return r.Method == i.Method && r.URL.String() == i.URL
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})
	return openai.NewResponsesModel("gpt-4o-mini", openai.WithHTTPClient(r.GetDefaultClient()))
}

func TestRecordedResponsesRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrResponsesModel(t, "responses_run"),
		ai.WithInstructions("Answer with a single word."),
	)
	result, err := agent.Run(t.Context(), "What is the capital of Spain?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output == "" || result.Usage().TotalTokens() == 0 {
		t.Fatalf("unexpected result %+v", result)
	}
}

func TestRecordedResponsesToolRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrResponsesModel(t, "responses_tool_run"),
		ai.WithInstructions("Use the get_weather tool, then answer briefly."),
	)
	called := false
	ai.AddSimpleTool(agent, "get_weather", func(_ context.Context, args struct {
		City string `json:"city" jsonschema:"description=City name"`
	}) (string, error) {
		called = true
		return "sunny, 21C in " + args.City, nil
	}, ai.WithDescription("Get current weather for a city"))
	result, err := agent.Run(t.Context(), "What is the weather in Berlin?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if !called || result.Output == "" {
		t.Fatalf("called=%v output=%q", called, result.Output)
	}
}

func TestResponsesCompatibleReasoningContentHistory(t *testing.T) {
	var body map[string]any
	model := newResponsesServerWithOptions(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_, _ = response.Write([]byte(`{"id":"response","model":"compatible","status":"completed","output":[],"usage":{}}`))
	}, openai.WithChatCompatibility(openai.ChatCompatibility{ResponsesReasoningContent: true}))
	_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "private", ID: "rs_1", ProviderName: "openai"},
	}}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	item := body["input"].([]any)[0].(map[string]any)
	if item["content"].([]any)[0].(map[string]any)["text"] != "private" {
		t.Fatalf("reasoning content was not replayed: %#v", body)
	}
}

func TestResponsesAssistantHistoryWithThinking(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "hidden"},
		ai.ThinkingPart{ID: "reasoning-1", Signature: "signature", ProviderName: "openai"},
		ai.TextPart{Content: "previous"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	input := gotBody["input"].([]any)
	if len(input) != 2 || input[0].(map[string]any)["type"] != "reasoning" ||
		input[0].(map[string]any)["id"] != "reasoning-1" ||
		input[0].(map[string]any)["encrypted_content"] != "signature" ||
		input[1].(map[string]any)["role"] != "assistant" {
		t.Fatalf("provider reasoning metadata was not round-tripped: %v", input)
	}
}

func TestResponsesDoesNotReplaySyntheticChatReasoningID(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"model":"gpt-5","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	})
	history := []ai.ModelMessage{ai.ModelResponse{ProviderName: "openai", Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "thinking", ID: "reasoning", ProviderName: "openai"},
	}}}
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	input := gotBody["input"].([]any)
	if len(input) != 1 || input[0].(map[string]any)["id"] != nil ||
		input[0].(map[string]any)["content"] != "<think>\nthinking\n</think>" {
		t.Fatalf("synthetic reasoning ID reached Responses: %#v", input)
	}
	foreign := []ai.ModelMessage{ai.ModelResponse{ProviderName: "other", Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "private", ID: "reasoning", ProviderName: "other"},
		ai.TextPart{Content: "visible"},
	}}}
	if _, err := model.Request(t.Context(), foreign, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	input = gotBody["input"].([]any)
	if len(input) != 1 || input[0].(map[string]any)["content"] != "visible" {
		t.Fatalf("foreign reasoning was replayed: %#v", input)
	}
}

func TestResponsesStrictToolDefinition(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	})
	strict := true
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
		Name: "search", Schema: map[string]any{"type": "object"}, Strict: &strict,
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	tool := gotBody["tools"].([]any)[0].(map[string]any)
	if tool["strict"] != true {
		t.Fatalf("strict flag not sent: %v", tool)
	}
}

func TestResponsesParallelToolCallsSetting(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	})
	enabled := true
	params := ai.ModelRequestParams{
		AllowText: true,
		Tools:     []ai.ToolDefinition{{Name: "work", Schema: map[string]any{"type": "object"}}},
		Settings:  ai.ModelSettings{ParallelToolCalls: &enabled},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if gotBody["parallel_tool_calls"] != true {
		t.Fatalf("parallel setting not forwarded: %v", gotBody)
	}
}

func TestResponsesCompactionRoundTripTrimsHistory(t *testing.T) {
	var input []map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []map[string]any `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		input = body.Input
		_, _ = w.Write([]byte(`{
			"id":"response", "model":"gpt-5",
			"output":[
				{"id":"cmp-new","type":"compaction","encrypted_content":"new-encrypted"},
				{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}
			],
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	})
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "standing"}, ai.UserPromptPart{Content: "drop"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "drop older response"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "drop before boundary"},
			ai.CompactionPart{
				Content: "summary", ID: "cmp-old", ProviderName: "openai",
				ProviderDetails: map[string]any{"encrypted_content": "old-encrypted"},
			},
			ai.TextPart{Content: "keep after boundary"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "keep tail"}}},
	}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(input) != 4 || input[0]["role"] != "system" || input[0]["content"] != "standing" ||
		input[1]["type"] != "compaction" || input[1]["id"] != "cmp-old" ||
		input[1]["encrypted_content"] != "old-encrypted" || input[2]["content"] != "keep after boundary" ||
		input[3]["content"] != "keep tail" {
		t.Fatalf("unexpected compacted Responses input: %+v", input)
	}
	compaction, ok := response.Parts[0].(ai.CompactionPart)
	if !ok || compaction.ID != "cmp-new" || compaction.ProviderName != "openai" ||
		compaction.ProviderDetails["encrypted_content"] != "new-encrypted" ||
		response.ProviderDetails["compaction"] != true || response.Text() != "done" {
		t.Fatalf("unexpected Responses compaction: %+v", response.Parts)
	}
}

func TestResponsesIgnoresInvalidCompactionBoundaries(t *testing.T) {
	for name, compaction := range map[string]ai.CompactionPart{
		"foreign": {
			Content: "summary", ProviderName: "anthropic",
			ProviderDetails: map[string]any{"encrypted_content": "foreign"},
		},
		"missing encrypted content": {Content: "summary", ProviderName: "openai"},
	} {
		t.Run(name, func(t *testing.T) {
			var input []map[string]any
			model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Input []map[string]any `json:"input"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				input = body.Input
				_, _ = w.Write([]byte(`{
					"id":"response", "model":"gpt-5", "status":"completed",
					"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],
					"usage":{"input_tokens":1,"output_tokens":1}
				}`))
			})
			messages := []ai.ModelMessage{
				ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "keep"}}},
				ai.ModelResponse{Parts: []ai.ResponsePart{compaction, ai.TextPart{Content: "assistant"}}},
			}
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{AllowText: true}); err != nil {
				t.Fatal(err)
			}
			if len(input) != 2 || input[0]["content"] != "keep" || input[1]["content"] != "assistant" {
				t.Fatalf("invalid compaction changed input: %+v", input)
			}
		})
	}
}

func TestResponsesCompactionDoesNotDuplicatePlantedPrompt(t *testing.T) {
	var input []map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []map[string]any `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		input = body.Input
		_, _ = w.Write([]byte(`{
			"id":"response", "model":"gpt-5", "status":"completed",
			"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],
			"usage":{}
		}`))
	})
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "standing"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{
			ProviderName: "openai", ProviderDetails: map[string]any{
				"encrypted_content": "opaque", ai.StandingPromptPlantedKey: true,
			},
		}}},
	}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	if len(input) != 1 || input[0]["type"] != "compaction" {
		t.Fatalf("planted prompt was duplicated: %+v", input)
	}
}

func TestResponsesServerManagedToolSearch(t *testing.T) {
	request := 0
	var bodies []map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		request++
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request == 1 {
			_, _ = w.Write([]byte(`{
				"id":"response-search","model":"gpt-5.4","created_at":1735689600,"status":"completed",
				"output":[
					{"id":"tso-1","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"weather","description":"","parameters":{"type":"object"}}]},
					{"id":"ts-1","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["weather"]}},
					{"id":"fc-1","type":"function_call","call_id":"weather-1","name":"weather","namespace":"weather","arguments":{"city":"Paris"}}
				],"usage":{}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"response-final","model":"gpt-5.4","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"sunny"}]}],"usage":{}
		}`))
	})
	weather := ai.NewTool[struct{}, struct {
		City string `json:"city"`
	}, string]("weather", func(_ context.Context, _ *ai.RunContext[struct{}], args struct {
		City string `json:"city"`
	}) (string, error) {
		return args.City + ": sunny", nil
	}, ai.WithDeferredLoading())
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(weather), ai.ToolSearchConfig[struct{}]{}))
	result, err := agent.Run(t.Context(), "weather", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "sunny" || request != 2 {
		t.Fatalf("unexpected server-search run: output=%q requests=%d", result.Output, request)
	}
	messages := result.Messages()
	searchResponse := messages[1].(ai.ModelResponse)
	call := searchResponse.Parts[0].(ai.NativeToolCallPart)
	returned := searchResponse.Parts[1].(ai.NativeToolReturnPart)
	function := searchResponse.Parts[2].(ai.ToolCallPart)
	if call.ToolCallID != "ts-1" || string(call.Args) != `{"queries":["weather"]}` ||
		returned.ToolCallID != "ts-1" || returned.Timestamp.IsZero() ||
		returned.Content.(ai.ToolSearchResult).DiscoveredTools[0].Name != "weather" ||
		function.ToolName != "weather" {
		t.Fatalf("unexpected normalized server search: %+v", searchResponse.Parts)
	}
	tools := bodies[0]["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["name"] != "weather" ||
		tools[0].(map[string]any)["defer_loading"] != true ||
		tools[1].(map[string]any)["type"] != "tool_search" ||
		tools[1].(map[string]any)["execution"] != nil {
		t.Fatalf("unexpected server search tools: %+v", tools)
	}
	input := bodies[1]["input"].([]any)
	var replayCall, replayOutput map[string]any
	for _, raw := range input {
		item := raw.(map[string]any)
		switch item["type"] {
		case "tool_search_call":
			replayCall = item
		case "tool_search_output":
			replayOutput = item
		}
	}
	if replayCall == nil || replayOutput == nil || replayCall["call_id"] != nil || replayOutput["call_id"] != nil ||
		replayCall["execution"] != "server" || replayOutput["execution"] != "server" ||
		replayCall["id"] != "ts-1" || replayOutput["id"] != "tso-1" ||
		len(replayOutput["tools"].([]any)) != 1 {
		t.Fatalf("unexpected server search replay: call=%+v output=%+v", replayCall, replayOutput)
	}
}

func TestResponsesRejectsNamedToolSearchStrategies(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"gpt-5.4","status":"completed","output":[],"usage":{}}`))
	})
	if !model.SupportsToolSearchStrategy(ai.ToolSearchStrategyAuto) ||
		model.SupportsToolSearchStrategy(ai.ToolSearchStrategyBM25) ||
		model.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) {
		t.Fatal("unexpected OpenAI tool-search strategy support")
	}
	for _, strategy := range []ai.ToolSearchStrategy{ai.ToolSearchStrategyBM25, ai.ToolSearchStrategyRegex} {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			Tools: []ai.ToolDefinition{{
				Name: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, ToolSearchStrategy: strategy,
			}},
			DeferredTools: []ai.ToolDefinition{{Name: "hidden", DeferLoading: true}},
		})
		if err == nil || !strings.Contains(err.Error(), `tool search strategy "`+string(strategy)+`" is not supported`) {
			t.Fatalf("unexpected %s strategy error: %v", strategy, err)
		}
	}
}

func TestResponsesServerManagedToolSearchPairingEdges(t *testing.T) {
	t.Run("ambiguous null IDs", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"ts-a","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["a"]}},
					{"id":"tso-a","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"a"}]},
					{"id":"ts-b","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["b"]}},
					{"id":"tso-b","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"b"}]}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Parts) != 4 ||
			response.Parts[0].(ai.NativeToolCallPart).ToolCallID != "ts-a" ||
			response.Parts[1].(ai.NativeToolReturnPart).ToolCallID != "tso-a" ||
			response.Parts[2].(ai.NativeToolCallPart).ToolCallID != "ts-b" ||
			response.Parts[3].(ai.NativeToolReturnPart).ToolCallID != "tso-b" {
			t.Fatalf("ambiguous null IDs were guessed: %+v", response.Parts)
		}
	})
	t.Run("explicit output first", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"tso","type":"tool_search_output","call_id":"call","execution":"server","status":"in_progress","tools":[{"type":"function","name":"real"},{"type":"file_search"}]},
					{"id":"ts","type":"tool_search_call","call_id":"call","execution":"server","status":"incomplete","arguments":{}}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		call := response.Parts[0].(ai.NativeToolCallPart)
		returned := response.Parts[1].(ai.NativeToolReturnPart)
		matches := returned.Content.(ai.ToolSearchResult).DiscoveredTools
		if call.ToolCallID != "call" || returned.ToolCallID != "call" || len(matches) != 1 ||
			matches[0].Name != "real" || call.ProviderDetails["status"] != "incomplete" ||
			returned.ProviderDetails["status"] != "in_progress" {
			t.Fatalf("unexpected explicit pairing: %+v", response.Parts)
		}
	})
	t.Run("unknown execution and non-object arguments", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"server","type":"tool_search_call","execution":"server","arguments":[]},
					{"id":"future","type":"tool_search_call","execution":"future","arguments":{}}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Parts) != 1 || string(response.Parts[0].(ai.NativeToolCallPart).Args) != `[]` {
			t.Fatalf("unexpected search execution normalization: %+v", response.Parts)
		}
	})
	t.Run("unmatched and client output", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"client","type":"tool_search_output","execution":"client","status":"completed","tools":[]},
					{"id":"server","type":"tool_search_output","execution":"server","status":"completed","tools":[]}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Parts) != 1 || response.Parts[0].(ai.NativeToolReturnPart).ToolCallID != "server" {
			t.Fatalf("unexpected unmatched outputs: %+v", response.Parts)
		}
	})
}

func TestResponsesServerToolSearchReplayEdges(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	}
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"gpt-5.4","status":"completed","output":[],"usage":{}}`))
	})
	for name, part := range map[string]ai.ResponsePart{
		"malformed call": ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", Args: []byte(`{`),
		},
		"malformed return": ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, ProviderName: "openai",
			Content: make(chan int), ProviderDetails: map[string]any{"id": "output"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{part}}}, params)
			if err == nil {
				t.Fatal("expected native replay error")
			}
		})
	}

	var body map[string]any
	model = newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5.4","status":"completed","output":[],"usage":{}}`))
	})
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "fallback", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", ProviderDetails: map[string]any{"call_id": 42, "status": "future"},
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "ignored", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", Content: ai.ToolSearchResult{},
		},
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, ProviderName: "anthropic",
		},
		ai.NativeToolReturnPart{
			ToolName: "capability", ToolKind: ai.ToolPartKindCapabilityLoad, ProviderName: "openai",
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "explicit", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", Content: ai.ToolSearchResult{},
			ProviderDetails: map[string]any{
				"id": "output", "call_id": "explicit", "status": "in_progress",
			},
		},
	}}}
	if _, err := model.Request(t.Context(), history, params); err != nil {
		t.Fatal(err)
	}
	input := body["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("unexpected replay items: %+v", input)
	}
	call := input[0].(map[string]any)
	output := input[1].(map[string]any)
	if call["call_id"] != "fallback" || call["status"] != "completed" ||
		output["call_id"] != "explicit" || output["status"] != "in_progress" {
		t.Fatalf("unexpected replay defaults: call=%+v output=%+v", call, output)
	}
}

func TestResponsesReplaysForeignNativeSearchThroughClientProtocol(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"model":"gpt-5.4","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{}
		}`))
	})
	hidden := ai.NewSimpleTool[struct{}](
		"hidden", func(context.Context, struct{}) (string, error) { return "hidden", nil },
		ai.WithDeferredLoading(),
	)
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(hidden), ai.ToolSearchConfig[struct{}]{}))
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic", Args: []byte(`{"queries":["hidden"]}`),
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic",
			Content:      ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "hidden"}}},
		},
	}}}
	if _, err := agent.Run(t.Context(), "continue", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	input := body["input"].([]any)
	if input[0].(map[string]any)["type"] != "tool_search_call" ||
		input[0].(map[string]any)["execution"] != "client" ||
		input[1].(map[string]any)["type"] != "tool_search_output" ||
		input[1].(map[string]any)["execution"] != "client" ||
		len(input[1].(map[string]any)["tools"].([]any)) != 1 {
		t.Fatalf("unexpected foreign native search replay: %+v", input)
	}
	tools := body["tools"].([]any)
	if tools[len(tools)-1].(map[string]any)["type"] != "tool_search" ||
		tools[len(tools)-1].(map[string]any)["execution"] != nil {
		t.Fatalf("current automatic search did not remain server-managed: %+v", tools)
	}
}
