package ai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type localWebFetchModel struct {
	url      string
	requests [][]ai.ModelMessage
}

func (*localWebFetchModel) Name() string { return "local-web-fetch" }
func (*localWebFetchModel) SupportsNativeTool(ai.NativeTool) bool {
	return false
}
func (model *localWebFetchModel) Request(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.requests = append(model.requests, append([]ai.ModelMessage(nil), messages...))
	if len(model.requests) == 1 {
		arguments, _ := json.Marshal(ai.WebFetchArgs{URL: model.url})
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "web_fetch", ToolCallID: "fetch-1", Args: arguments,
		}}}, nil
	}
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func runLocalWebFetch(
	t *testing.T, rawURL string, config ai.LocalWebFetchConfig,
) (*ai.RunResult[string], *localWebFetchModel) {
	t.Helper()
	model := &localWebFetchModel{url: rawURL}
	local := ai.NewLocalWebFetchTool[struct{}](config)
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewWebFetchCapability(ai.WebFetchCapabilityConfig[struct{}]{
			Native: ai.WebFetchTool{}, Local: ai.NewFunctionToolset(local),
		}),
	))
	result, err := agent.Run(t.Context(), "fetch the URL", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	return result, model
}

func localWebFetchParts(t *testing.T, result *ai.RunResult[string]) (any, []ai.UserContent) {
	t.Helper()
	var returnValue any
	var content []ai.UserContent
	for _, message := range result.Messages() {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			switch part := part.(type) {
			case ai.ToolReturnPart:
				returnValue = part.Content
			case ai.UserPromptPart:
				if len(part.Contents) > 0 {
					content = append(content, part.Contents...)
				}
			}
		}
	}
	return returnValue, content
}

func TestLocalWebFetchTextAndHTML(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Accept") != "custom" || request.Header.Get("X-Test") != "configured" {
			t.Error("configured headers were not sent")
		}
		switch request.URL.Path {
		case "/html":
			response.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = response.Write([]byte(`<html><head><title> Go &amp; AI </title><script>bad()</script></head>` +
				`<body><h1>Hello</h1><a href="/docs">Docs</a><img src="secret.png"></body></html>`))
		case "/no-title":
			response.Header().Set("Content-Type", "application/xhtml+xml")
			_, _ = response.Write([]byte("<html><body><p>Text</p></body></html>"))
		case "/latin":
			response.Header().Set("Content-Type", "text/html; charset=iso-8859-1")
			_, _ = response.Write([]byte("<html><head><title>Caf\xe9</title></head><body>ol\xe9</body></html>"))
		case "/markdown":
			response.Header().Set("Content-Type", "text/markdown")
			_, _ = response.Write([]byte("# Heading\n\n\n\nBody"))
		case "/text-markdown":
			response.Header().Set("Content-Type", "text/x-markdown")
			_, _ = response.Write([]byte("# Alternate"))
		case "/plain":
			response.Header().Set("Content-Type", "text/plain")
			_, _ = response.Write([]byte("plain"))
		case "/xml":
			response.Header().Set("Content-Type", "application/xml")
			_, _ = response.Write([]byte("<value>xml</value>"))
		}
	}))
	t.Cleanup(server.Close)
	config := ai.LocalWebFetchConfig{
		AllowLocalURLs: true,
		Headers:        map[string]string{"Accept": "custom", "X-Test": "configured"},
	}
	for path, validate := range map[string]func(t *testing.T, result ai.WebFetchResult){
		"/html": func(t *testing.T, result ai.WebFetchResult) {
			if result.Title != "Go & AI" || !strings.Contains(result.Content, "# Hello") ||
				!strings.Contains(result.Content, server.URL+"/docs") || strings.Contains(result.Content, "bad()") ||
				strings.Contains(result.Content, "secret.png") {
				t.Fatalf("unexpected HTML result: %#v", result)
			}
		},
		"/no-title": func(t *testing.T, result ai.WebFetchResult) {
			if result.Title != "" || !strings.Contains(result.Content, "Text") {
				t.Fatalf("unexpected untitled HTML result: %#v", result)
			}
		},
		"/latin": func(t *testing.T, result ai.WebFetchResult) {
			if result.Title != "Café" || !strings.Contains(result.Content, "olé") {
				t.Fatalf("unexpected decoded HTML result: %#v", result)
			}
		},
		"/markdown": func(t *testing.T, result ai.WebFetchResult) {
			if result.Content != "# Heading\n\nBody" {
				t.Fatalf("unexpected Markdown result: %#v", result)
			}
		},
		"/text-markdown": func(t *testing.T, result ai.WebFetchResult) {
			if result.Content != "# Alternate" {
				t.Fatalf("unexpected alternate Markdown result: %#v", result)
			}
		},
		"/plain": func(t *testing.T, result ai.WebFetchResult) {
			if result.Content != "plain" {
				t.Fatalf("unexpected text result: %#v", result)
			}
		},
		"/xml": func(t *testing.T, result ai.WebFetchResult) {
			if result.Content != "<value>xml</value>" {
				t.Fatalf("unexpected XML result: %#v", result)
			}
		},
	} {
		t.Run(path, func(t *testing.T) {
			result, _ := runLocalWebFetch(t, server.URL+path, config)
			value, _ := localWebFetchParts(t, result)
			fetched, ok := value.(ai.WebFetchResult)
			if !ok || fetched.URL != server.URL+path {
				t.Fatalf("unexpected fetch value: %#v", value)
			}
			validate(t, fetched)
		})
	}
}

func TestLocalWebFetchJSONAndContentLimit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "application/problem+json")
		if request.URL.Path == "/invalid" {
			_, _ = response.Write([]byte("not json"))
			return
		}
		_, _ = response.Write([]byte(`{"a":"世界","z":2}`))
	}))
	t.Cleanup(server.Close)
	result, _ := runLocalWebFetch(t, server.URL+"/valid", ai.LocalWebFetchConfig{
		AllowLocalURLs: true, MaxContentLength: 26,
	})
	value, _ := localWebFetchParts(t, result)
	fetched := value.(ai.WebFetchResult)
	if !strings.HasSuffix(fetched.Content, "\n\n[Content truncated]") || !strings.Contains(fetched.Content, "世界") {
		t.Fatalf("JSON was not formatted and rune-truncated: %#v", fetched)
	}
	result, _ = runLocalWebFetch(t, server.URL+"/invalid", ai.LocalWebFetchConfig{
		AllowLocalURLs: true, MaxContentLength: 1, DisableContentLimit: true,
	})
	value, _ = localWebFetchParts(t, result)
	if fetched = value.(ai.WebFetchResult); fetched.Content != "not json" {
		t.Fatalf("invalid JSON was not preserved: %#v", fetched)
	}
}

func TestLocalWebFetchBinaryAndRetry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/octet" {
			response.Header().Set("Content-Type", "application/octet-stream")
			_, _ = response.Write([]byte("bytes"))
			return
		}
		response.Header().Set("Content-Type", "image/png")
		_, _ = response.Write([]byte("png"))
	}))
	t.Cleanup(server.Close)
	result, _ := runLocalWebFetch(t, server.URL, ai.LocalWebFetchConfig{AllowLocalURLs: true})
	value, content := localWebFetchParts(t, result)
	if value != "Fetched binary content from "+server.URL || len(content) != 1 {
		t.Fatalf("unexpected binary result: value=%#v content=%#v", value, content)
	}
	binary, ok := content[0].(ai.BinaryContent)
	if !ok || binary.MediaType != "image/png" || string(binary.Data) != "png" {
		t.Fatalf("unexpected binary content: %#v", content[0])
	}
	result, _ = runLocalWebFetch(t, server.URL+"/octet", ai.LocalWebFetchConfig{AllowLocalURLs: true})
	_, content = localWebFetchParts(t, result)
	binary, ok = content[0].(ai.BinaryContent)
	if !ok || binary.MediaType != "application/octet-stream" || string(binary.Data) != "bytes" {
		t.Fatalf("unexpected octet-stream content: %#v", content[0])
	}

	result, model := runLocalWebFetch(t, server.URL, ai.LocalWebFetchConfig{})
	value, _ = localWebFetchParts(t, result)
	if value != nil || len(model.requests) != 2 {
		t.Fatalf("failed fetch returned a value: %#v", value)
	}
	request := model.requests[1][len(model.requests[1])-1].(ai.ModelRequest)
	retry, ok := request.Parts[0].(ai.RetryPromptPart)
	if !ok || !strings.Contains(retry.Content, "Failed to fetch") || !strings.Contains(retry.Content, "private") {
		t.Fatalf("unexpected fetch retry: %#v", request.Parts)
	}
}

func TestLocalWebFetchConfigValidationAndDetachment(t *testing.T) {
	for name, config := range map[string]ai.LocalWebFetchConfig{
		"content":  {MaxContentLength: -1},
		"timeout":  {Timeout: -time.Second},
		"download": {MaxDownloadBytes: -1},
	} {
		t.Run(name, func(t *testing.T) {
			assertNativeOrLocalPanic(t, "must not be negative", func() {
				ai.NewLocalWebFetchTool[struct{}](config)
			})
		})
	}

	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Test") != "original" {
			t.Error("headers were not detached")
		}
		_, _ = response.Write([]byte("ok"))
	}))
	t.Cleanup(server.Close)
	config := ai.LocalWebFetchConfig{
		AllowLocalURLs: true, Timeout: time.Second, MaxDownloadBytes: 100,
		AllowedDomains: []string{"wrong.test"}, BlockedDomains: []string{"127.0.0.1"},
		Headers: map[string]string{"X-Test": "original"},
	}
	native := ai.WebFetchTool{
		AllowedDomains: []string{"127.0.0.1"}, BlockedDomains: []string{"example.com"},
	}
	capability := ai.NewWebFetchCapabilityWithLocal[struct{}](native, config)
	config.AllowedDomains[0] = "example.com"
	config.BlockedDomains[0] = "127.0.0.1"
	config.Headers["X-Test"] = "changed"
	native.AllowedDomains[0] = "example.com"
	native.BlockedDomains[0] = "127.0.0.1"
	model := &localWebFetchModel{url: server.URL}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "fetch", struct{}{}); err != nil {
		t.Fatal(err)
	}
}
