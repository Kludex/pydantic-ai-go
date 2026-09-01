package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type localWebSearchModel struct {
	query string
}

func (*localWebSearchModel) Name() string { return "local-web-search" }
func (*localWebSearchModel) SupportsNativeTool(ai.NativeTool) bool {
	return false
}
func (model *localWebSearchModel) Request(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	if len(messages) == 1 {
		arguments, _ := json.Marshal(ai.WebSearchArgs{Query: model.query})
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "duckduckgo_search", ToolCallID: "search-1", Args: arguments,
		}}}, nil
	}
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func runLocalWebSearch(
	t *testing.T, query string, config ai.LocalWebSearchConfig,
) *ai.RunResult[string] {
	t.Helper()
	model := &localWebSearchModel{query: query}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewWebSearchCapabilityWithDuckDuckGo[struct{}](ai.WebSearchTool{}, config),
	))
	result, err := agent.Run(t.Context(), "search the web", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func localWebSearchValue(t *testing.T, result *ai.RunResult[string]) any {
	t.Helper()
	for _, message := range result.Messages() {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			if returned, ok := part.(ai.ToolReturnPart); ok {
				return returned.Content
			}
		}
	}
	return nil
}

func localWebSearchRetry(t *testing.T, result *ai.RunResult[string]) ai.RetryPromptPart {
	t.Helper()
	for _, message := range result.Messages() {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			if retry, ok := part.(ai.RetryPromptPart); ok {
				return retry
			}
		}
	}
	t.Fatal("search did not return a retry prompt")
	return ai.RetryPromptPart{}
}

func TestRecordedLocalWebSearch(t *testing.T) {
	mode := recorder.ModeReplayOnly
	if os.Getenv("RECORD_DUCKDUCKGO") != "" {
		mode = recorder.ModeRecordOnce
	}
	recording, err := recorder.New(
		"testdata/duckduckgo_search",
		recorder.WithMode(mode),
		recorder.WithMatcher(cassette.DefaultMatcher),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := recording.Stop(); err != nil {
			t.Error(err)
		}
	})
	result := runLocalWebSearch(t, "pydantic ai", ai.LocalWebSearchConfig{
		HTTPClient: recording.GetDefaultClient(), MaxResults: 3,
	})
	results, ok := localWebSearchValue(t, result).([]ai.WebSearchResult)
	if !ok || len(results) != 3 || results[0].Title == "" || !strings.HasPrefix(results[0].URL, "http") {
		t.Fatalf("unexpected recorded DuckDuckGo results: %#v", results)
	}
}

func TestLocalWebSearchResults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.Header.Get("User-Agent") != "pydantic-ai-go web search" {
			t.Errorf("unexpected request: %s headers=%v", request.Method, request.Header)
		}
		if err := request.ParseForm(); err != nil || request.Form.Get("q") != "go agents" {
			t.Errorf("unexpected form: %v err=%v", request.Form, err)
		}
		_, _ = response.Write([]byte(`<html><body>
			<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc"> Go   docs </a><div class="result__snippet">Official   docs</div></div>
			<div class="result"><a class="result__a" href="/relative">Relative</a><div class="result__snippet">Second</div></div>
			<div class="result"><a class="result__a" href="%">Malformed</a></div>
			<div class="result"><span>No link</span></div>
			<div class="result"><a class="result__a" href="/empty"> </a></div>
		</body></html>`))
	}))
	t.Cleanup(server.Close)

	result := runLocalWebSearch(t, "go agents", ai.LocalWebSearchConfig{Endpoint: server.URL, MaxResults: 2})
	results, ok := localWebSearchValue(t, result).([]ai.WebSearchResult)
	if !ok || len(results) != 2 || results[0] != (ai.WebSearchResult{
		Title: "Go docs", URL: "https://go.dev/doc", Body: "Official docs",
	}) || results[1].Title != "Relative" || results[1].URL != server.URL+"/relative" {
		t.Fatalf("unexpected search results: %#v", results)
	}

	result = runLocalWebSearch(t, "go agents", ai.LocalWebSearchConfig{Endpoint: server.URL})
	results = localWebSearchValue(t, result).([]ai.WebSearchResult)
	if len(results) != 3 || results[2].URL != "%" {
		t.Fatalf("unlimited first-page results were not retained: %#v", results)
	}
}

type localSearchRoundTripper func(*http.Request) (*http.Response, error)

func (function localSearchRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type localSearchFailingBody struct{}

func (localSearchFailingBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (localSearchFailingBody) Close() error             { return nil }

func TestLocalWebSearchRetriesFailures(t *testing.T) {
	statusServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		http.Error(response, "unavailable", http.StatusServiceUnavailable)
	}))
	t.Cleanup(statusServer.Close)
	largeServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write(make([]byte, 5<<20+1))
	}))
	t.Cleanup(largeServer.Close)
	slowServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		time.Sleep(30 * time.Millisecond)
		_, _ = response.Write([]byte("late"))
	}))
	t.Cleanup(slowServer.Close)
	readClient := &http.Client{Transport: localSearchRoundTripper(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK, Status: "200 OK", Header: make(http.Header), Body: localSearchFailingBody{},
		}, nil
	})}
	closedServer := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closedURL := closedServer.URL
	closedServer.Close()

	for name, test := range map[string]struct {
		query    string
		config   ai.LocalWebSearchConfig
		contains string
	}{
		"empty query": {query: " ", config: ai.LocalWebSearchConfig{Endpoint: statusServer.URL}, contains: "must not be empty"},
		"status":      {query: "q", config: ai.LocalWebSearchConfig{Endpoint: statusServer.URL}, contains: "503 Service Unavailable"},
		"large":       {query: "q", config: ai.LocalWebSearchConfig{Endpoint: largeServer.URL}, contains: "exceeds"},
		"timeout": {query: "q", config: ai.LocalWebSearchConfig{
			Endpoint: slowServer.URL, Timeout: time.Millisecond,
		}, contains: "context deadline exceeded"},
		"read": {query: "q", config: ai.LocalWebSearchConfig{
			Endpoint: "https://search.test", HTTPClient: readClient,
		}, contains: "could not be read"},
		"connection": {query: "q", config: ai.LocalWebSearchConfig{
			Endpoint: closedURL,
		}, contains: "search failed"},
	} {
		t.Run(name, func(t *testing.T) {
			result := runLocalWebSearch(t, test.query, test.config)
			retry := localWebSearchRetry(t, result)
			if !strings.Contains(retry.Content, test.contains) {
				t.Fatalf("unexpected retry prompt: %#v", retry)
			}
		})
	}
}

func TestLocalWebSearchConfiguration(t *testing.T) {
	for name, config := range map[string]ai.LocalWebSearchConfig{
		"timeout":      {Timeout: -time.Second},
		"results":      {MaxResults: -1},
		"parse":        {Endpoint: "http://%"},
		"scheme":       {Endpoint: "file:///tmp/search"},
		"missing host": {Endpoint: "https:///search"},
	} {
		t.Run(name, func(t *testing.T) {
			assertNativeOrLocalPanic(t, "local web-search", func() {
				ai.NewLocalWebSearchTool[struct{}](config)
			})
		})
	}
	tool := ai.NewLocalWebSearchTool[struct{}](ai.LocalWebSearchConfig{})
	if tool.Definition().Name != "duckduckgo_search" {
		t.Fatalf("unexpected default search tool: %#v", tool.Definition())
	}
}
