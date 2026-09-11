package openaicodex_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaicodex"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type memorySource struct {
	mu          sync.Mutex
	credentials openaicodex.Credentials
	loads       int
	saves       []openaicodex.Credentials
	loadErr     error
	saveErr     error
	block       <-chan struct{}
}

func (source *memorySource) Load(ctx context.Context) (openaicodex.Credentials, error) {
	if source.block != nil {
		select {
		case <-ctx.Done():
			return openaicodex.Credentials{}, context.Cause(ctx)
		case <-source.block:
		}
	}
	source.mu.Lock()
	defer source.mu.Unlock()
	source.loads++
	return source.credentials, source.loadErr
}

func (source *memorySource) Save(_ context.Context, credentials openaicodex.Credentials) error {
	source.mu.Lock()
	defer source.mu.Unlock()
	source.credentials = credentials
	source.saves = append(source.saves, credentials)
	return source.saveErr
}

func (source *memorySource) replace(credentials openaicodex.Credentials) {
	source.mu.Lock()
	defer source.mu.Unlock()
	source.credentials = credentials
}

func (source *memorySource) failLoad(err error) {
	source.mu.Lock()
	defer source.mu.Unlock()
	source.loadErr = err
}

func credentials(access string) openaicodex.Credentials {
	return openaicodex.Credentials{AccessToken: access, RefreshToken: "refresh-old", AccountID: "account-old"}
}

func jwt(payload map[string]any) string {
	data, _ := json.Marshal(payload)
	return "header." + base64.RawURLEncoding.EncodeToString(data) + ".signature"
}

func response(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)),
	}
}

func codexSSE(text string) *http.Response {
	body := strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"message","type":"message"}}`,
		`data: {"type":"response.output_text.delta","item_id":"message",` +
			`"output_index":0,"content_index":0,"delta":` + fmt.Sprintf("%q", text) + `}`,
		`data: {"type":"response.completed","response":{"id":"response","model":"gpt-5.6-luna",` +
			`"status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":2}}}`,
		"",
	}, "\n\n")
	result := response(http.StatusOK, body)
	result.Header.Set("Content-Type", "text/event-stream")
	return result
}

func newModel(t *testing.T, transport http.RoundTripper, options ...openaicodex.Option) *openaicodex.Model {
	t.Helper()
	options = append([]openaicodex.Option{
		openaicodex.WithCredentials(credentials("access-old")),
		openaicodex.WithHTTPClient(&http.Client{Transport: transport}),
	}, options...)
	model, err := openaicodex.NewModel("gpt-5.6-luna", options...)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestModelWireDialectAndResponseWrapper(t *testing.T) {
	var requestBody map[string]any
	var requestHeaders http.Header
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://chatgpt.com/backend-api/codex/responses" {
			t.Fatalf("unexpected URL: %s", request.URL)
		}
		requestHeaders = request.Header.Clone()
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Fatal(err)
		}
		return codexSSE("hi there"), nil
	})
	temperature, topP := 0.2, 0.8
	parallel := true
	defaults := ai.ModelSettings{MaxTokens: 99, Temperature: &temperature, TopP: &topP}
	model := newModel(t, transport, openaicodex.WithDefaultSettings(defaults))
	messages := []ai.ModelMessage{
		ai.ModelResponse{ConversationID: "older"},
		ai.ModelRequest{ConversationID: "conversation", Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
	}
	result, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		Settings: ai.ModelSettings{
			ParallelToolCalls: &parallel,
			ExtraHeaders:      map[string]string{"Thread-Id": "child"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text() != "hi there" || result.ProviderName != "openai-codex" || result.ProviderURL == "" {
		t.Fatalf("unexpected response: %+v", result)
	}
	if requestBody["stream"] != true || requestBody["store"] != false ||
		requestBody["prompt_cache_key"] != "conversation" || requestBody["parallel_tool_calls"] != true {
		t.Fatalf("unexpected request body: %#v", requestBody)
	}
	for _, name := range []string{"max_output_tokens", "temperature", "top_p"} {
		if _, exists := requestBody[name]; exists {
			t.Fatalf("unsupported setting %q was sent: %#v", name, requestBody)
		}
	}
	if requestHeaders.Get("Authorization") != "Bearer access-old" ||
		requestHeaders.Get("chatgpt-account-id") != "account-old" || requestHeaders.Get("originator") != "pydantic-ai" ||
		requestHeaders.Get("session-id") != "conversation" || requestHeaders.Get("Thread-Id") != "child" ||
		requestHeaders.Get("x-client-request-id") != "conversation" {
		t.Fatalf("unexpected request headers: %v", requestHeaders)
	}
	if model.Name() != "gpt-5.6-luna" || model.ProviderName() != "openai-codex" ||
		model.ProviderURL() != "https://chatgpt.com/backend-api/codex" ||
		model.DefaultModelSettings().MaxTokens != 99 || !model.ModelProfile().SupportsImageOutput ||
		!model.SupportsNativeTool(ai.WebSearchTool{}) || model.SupportsNativeTool(testNativeTool{}) ||
		model.NativeToolSearchProvider() != "openai-codex" {
		t.Fatalf("unexpected model surface: %T", model)
	}
	current, err := model.Credentials()
	if err != nil || current.AccessToken != "access-old" {
		t.Fatalf("unexpected current credentials: %v %v", current, err)
	}
}

type testNativeTool struct{}

func (testNativeTool) Kind() string                   { return "test" }
func (testNativeTool) UniqueID() string               { return "test" }
func (testNativeTool) IsOptional() bool               { return false }
func (testNativeTool) CloneNativeTool() ai.NativeTool { return testNativeTool{} }

func TestStreamSettingsAndConversationFallbacks(t *testing.T) {
	var bodies []map[string]any
	var headers []http.Header
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, body)
		headers = append(headers, request.Header.Clone())
		return codexSSE("done"), nil
	})
	model := newModel(t, transport)
	explicit, err := (openai.Settings{PromptCacheKey: "explicit"}).Build()
	if err != nil {
		t.Fatal(err)
	}
	explicit.ExtraHeaders = map[string]string{
		"Session-Id": "session", "thread-id": "thread", "X-CLIENT-REQUEST-ID": "request",
	}
	stream, err := model.StreamRequest(t.Context(), []ai.ModelMessage{
		ai.ModelRequest{ConversationID: "request-conversation"},
		ai.ModelResponse{ConversationID: "response-conversation"},
	}, ai.ModelRequestParams{Settings: explicit})
	if err != nil {
		t.Fatal(err)
	}
	for _, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
	}
	stream, err = model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraBody: map[string]any{"prompt_cache_key": "wire"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
	}
	stream, err = model.StreamRequest(t.Context(), []ai.ModelMessage{
		ai.ModelRequest{ConversationID: "derived"},
	}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
	}
	if bodies[0]["prompt_cache_key"] != "explicit" || bodies[1]["prompt_cache_key"] != "wire" ||
		bodies[2]["prompt_cache_key"] != "derived" {
		t.Fatalf("unexpected cache keys: %#v", bodies)
	}
	if headers[0].Get("Session-Id") != "session" || headers[0].Get("thread-id") != "thread" ||
		headers[0].Get("X-CLIENT-REQUEST-ID") != "request" {
		t.Fatalf("explicit headers were replaced: %v", headers[0])
	}
	for _, name := range []string{"session-id", "thread-id", "x-client-request-id"} {
		if headers[1].Get(name) != "" {
			t.Fatalf("unexpected affinity header %q without a conversation", name)
		}
		if headers[2].Get(name) != "derived" {
			t.Fatalf("missing derived affinity header %q", name)
		}
	}
}

func TestModelErrorsAndUnsupportedOperations(t *testing.T) {
	valid := credentials("access")
	if _, err := openaicodex.NewModel(""); err == nil {
		t.Fatal("empty model name succeeded")
	}
	if _, err := openaicodex.NewModel(
		"model", openaicodex.WithHTTPClient(nil), openaicodex.WithCredentials(valid),
	); err == nil {
		t.Fatal("nil HTTP client succeeded")
	}
	if _, err := openaicodex.NewModel("model", openaicodex.WithCredentials(openaicodex.Credentials{})); err == nil {
		t.Fatal("invalid credentials succeeded")
	}
	source := &memorySource{credentials: valid}
	if _, err := openaicodex.NewModel(
		"model", openaicodex.WithCredentials(valid), openaicodex.WithCredentialSource(source),
	); err == nil {
		t.Fatal("mutually exclusive credentials succeeded")
	}
	model, err := openaicodex.NewModel("model", openaicodex.WithCredentialSource(source))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Credentials(); err == nil {
		t.Fatal("lazy credentials were available before first use")
	}
	if _, err := model.CountTokens(
		t.Context(), nil, ai.ModelRequestParams{},
	); !errors.Is(err, ai.ErrTokenCountingUnsupported) {
		t.Fatalf("unexpected count error: %v", err)
	}
	suspended := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai-codex", State: ai.ModelResponseStateSuspended,
	}}
	if _, err := model.Request(t.Context(), suspended, ai.ModelRequestParams{}); err == nil {
		t.Fatal("suspended request succeeded")
	}
	if _, err := model.StreamRequest(t.Context(), suspended, ai.ModelRequestParams{}); err == nil {
		t.Fatal("suspended stream succeeded")
	}
}

func TestCredentialRefreshSingleFlightAndPersistence(t *testing.T) {
	for _, proactive := range []bool{false, true} {
		t.Run(fmt.Sprintf("proactive=%v", proactive), func(t *testing.T) {
			access := "access-old"
			if proactive {
				access = jwt(map[string]any{"exp": time.Now().Add(-time.Minute).Unix()})
			}
			source := &memorySource{credentials: credentials(access)}
			var tokenCalls atomic.Int64
			var codexCalls atomic.Int64
			release := make(chan struct{})
			transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
				switch request.URL.Hostname() {
				case "auth.openai.com":
					tokenCalls.Add(1)
					<-release
					return response(http.StatusOK,
						`{"access_token":"access-new","refresh_token":"refresh-new","account_id":"account-new"}`,
					), nil
				case "chatgpt.com":
					codexCalls.Add(1)
					if request.Header.Get("Authorization") != "Bearer access-new" {
						return response(http.StatusUnauthorized, `{}`), nil
					}
					return codexSSE("done"), nil
				default:
					t.Fatalf("unexpected host: %s", request.URL.Hostname())
					return nil, nil
				}
			})
			model, err := openaicodex.NewModel("model",
				openaicodex.WithCredentialSource(source),
				openaicodex.WithHTTPClient(&http.Client{Transport: transport}),
			)
			if err != nil {
				t.Fatal(err)
			}
			var wait sync.WaitGroup
			errorsSeen := make(chan error, 5)
			wait.Add(5)
			for range 5 {
				go func() {
					defer wait.Done()
					_, requestErr := model.Request(t.Context(), nil, ai.ModelRequestParams{})
					errorsSeen <- requestErr
				}()
			}
			for tokenCalls.Load() == 0 {
				time.Sleep(time.Millisecond)
			}
			close(release)
			wait.Wait()
			close(errorsSeen)
			for requestErr := range errorsSeen {
				if requestErr != nil {
					t.Fatal(requestErr)
				}
			}
			if tokenCalls.Load() != 1 || len(source.saves) != 1 || source.saves[0].AccessToken != "access-new" {
				t.Fatalf("refresh was not single-flight: token=%d source=%+v", tokenCalls.Load(), source)
			}
			wantCodex := int64(5)
			if !proactive {
				wantCodex = 10
			}
			if codexCalls.Load() != wantCodex {
				t.Fatalf("unexpected Codex calls: %d", codexCalls.Load())
			}
		})
	}
}

func TestCredentialSourceAdoptsPeerRotation(t *testing.T) {
	source := &memorySource{credentials: credentials("access-old")}
	var tokenCalls int
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Hostname() == "auth.openai.com" {
			tokenCalls++
			return response(http.StatusInternalServerError, `{}`), nil
		}
		if request.Header.Get("Authorization") == "Bearer access-peer" {
			return codexSSE("peer"), nil
		}
		return response(http.StatusUnauthorized, `{}`), nil
	})
	model, err := openaicodex.NewModel("model", openaicodex.WithCredentialSource(source),
		openaicodex.WithHTTPClient(&http.Client{Transport: transport}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("initial stale credential unexpectedly succeeded")
	}
	source.replace(credentials("access-peer"))
	result, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || result.Text() != "peer" {
		t.Fatalf("peer rotation was not adopted: %+v %v", result, err)
	}
	if tokenCalls != 1 || len(source.saves) != 0 {
		t.Fatalf("unexpected peer refresh activity: token=%d saves=%d", tokenCalls, len(source.saves))
	}
}

func TestInflight401AdoptsCompletedRefresh(t *testing.T) {
	var mu sync.Mutex
	oldRequests := 0
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	var tokenCalls int
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Hostname() == "auth.openai.com" {
			mu.Lock()
			tokenCalls++
			mu.Unlock()
			return response(http.StatusOK,
				`{"access_token":"access-new","refresh_token":"refresh-new","account_id":"account-new"}`,
			), nil
		}
		if request.Header.Get("Authorization") == "Bearer access-new" {
			return codexSSE("done"), nil
		}
		mu.Lock()
		oldRequests++
		index := oldRequests
		mu.Unlock()
		if index == 1 {
			close(firstStarted)
			<-releaseFirst
		}
		return response(http.StatusUnauthorized, `{}`), nil
	})
	model := newModel(t, transport)
	first := make(chan error, 1)
	go func() {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		first <- err
	}()
	<-firstStarted
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	close(releaseFirst)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	if tokenCalls != 1 {
		t.Fatalf("in-flight 401 caused %d refreshes", tokenCalls)
	}
}

func TestCredentialSourceRefreshLoadErrors(t *testing.T) {
	for _, test := range []struct {
		name        string
		credentials openaicodex.Credentials
		err         error
	}{
		{name: "load", credentials: credentials("access"), err: errors.New("source unavailable")},
		{name: "invalid", credentials: openaicodex.Credentials{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			source := &memorySource{credentials: credentials("access")}
			requests := 0
			transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.URL.Hostname() == "auth.openai.com" {
					t.Fatal("token endpoint reached after source load failure")
				}
				requests++
				if requests == 1 {
					return codexSSE("first"), nil
				}
				return response(http.StatusUnauthorized, `{}`), nil
			})
			model, err := openaicodex.NewModel("model", openaicodex.WithCredentialSource(source),
				openaicodex.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
				t.Fatal(err)
			}
			source.replace(test.credentials)
			source.failLoad(test.err)
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
				t.Fatal("source refresh load failure succeeded")
			}
		})
	}
}

func TestPersistenceFailureKeepsRotatedCredentials(t *testing.T) {
	for _, stale := range []bool{false, true} {
		t.Run(fmt.Sprintf("stale=%v", stale), func(t *testing.T) {
			access := "access-old"
			if stale {
				access = jwt(map[string]any{"exp": time.Now().Add(-time.Minute).Unix()})
			}
			source := &memorySource{credentials: credentials(access), saveErr: errors.New("storage unavailable")}
			codexRequests := 0
			transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.URL.Hostname() == "auth.openai.com" {
					return response(http.StatusOK,
						`{"access_token":"access-new","refresh_token":"refresh-new","account_id":"account-new"}`,
					), nil
				}
				codexRequests++
				return response(http.StatusUnauthorized, `{}`), nil
			})
			model, err := openaicodex.NewModel("model", openaicodex.WithCredentialSource(source),
				openaicodex.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatal(err)
			}
			_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
			var persistence *openaicodex.CredentialsPersistenceError
			if !errors.As(err, &persistence) || !persistence.IsModelAPIError() ||
				!strings.Contains(persistence.Error(), "storage unavailable") || !errors.Is(persistence, source.saveErr) {
				t.Fatalf("unexpected persistence error: %v", err)
			}
			current, currentErr := model.Credentials()
			if currentErr != nil || current.AccessToken != "access-new" {
				t.Fatalf("rotated credentials were not retained: %v %v", current, currentErr)
			}
			wantRequests := 1
			if stale {
				wantRequests = 0
			}
			if codexRequests != wantRequests {
				t.Fatalf("unexpected requests before persistence failure: %d", codexRequests)
			}
		})
	}
}

func TestProactiveRefreshFailureFallsThrough(t *testing.T) {
	access := jwt(map[string]any{"exp": time.Now().Add(-time.Minute).Unix()})
	var tokenCalls int
	model := newModel(t, roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Hostname() == "auth.openai.com" {
			tokenCalls++
			return response(http.StatusServiceUnavailable, `{"error":"temporarily_unavailable"}`), nil
		}
		if request.Header.Get("Authorization") != "Bearer "+access {
			t.Fatalf("stale token was not retained: %v", request.Header)
		}
		return codexSSE("accepted"), nil
	}), openaicodex.WithCredentials(credentials(access)))
	result, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || result.Text() != "accepted" || tokenCalls != 1 {
		t.Fatalf("proactive refresh did not fall through: %+v err=%v calls=%d", result, err, tokenCalls)
	}
}

func TestUnauthorizedReplayRunsOnce(t *testing.T) {
	var tokenCalls, codexCalls int
	model := newModel(t, roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Hostname() == "auth.openai.com" {
			tokenCalls++
			return response(http.StatusOK,
				`{"access_token":"access-new","refresh_token":"refresh-new","account_id":"account-new"}`,
			), nil
		}
		codexCalls++
		return response(http.StatusUnauthorized, `{"error":"insufficient_quota"}`), nil
	}))
	for range 2 {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("unauthorized replay succeeded")
		}
	}
	if codexCalls != 4 || tokenCalls != 2 {
		t.Fatalf("unexpected retry counts: codex=%d token=%d", codexCalls, tokenCalls)
	}
}

func TestCredentialLeakProtectionAcrossRedirects(t *testing.T) {
	for _, location := range []string{
		"https://example.com/final", "http://chatgpt.com/final", "https://chatgpt.com/final",
	} {
		t.Run(location, func(t *testing.T) {
			var redirected http.Header
			transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.URL.Path != "/final" {
					status := http.StatusFound
					if strings.HasPrefix(location, "https://chatgpt.com/") {
						status = http.StatusTemporaryRedirect
					}
					redirect := response(status, "")
					redirect.Header.Set("Location", location)
					return redirect, nil
				}
				redirected = request.Header.Clone()
				return codexSSE("redirected"), nil
			})
			model := newModel(t, transport)
			result, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil || result.Text() != "redirected" {
				t.Fatalf("redirect request failed: %+v %v", result, err)
			}
			for _, name := range []string{"Authorization", "chatgpt-account-id", "originator"} {
				if strings.HasPrefix(location, "https://chatgpt.com/") {
					if redirected.Get(name) == "" {
						t.Fatalf("credential header %q was missing on the Codex host", name)
					}
				} else if redirected.Get(name) != "" {
					t.Fatalf("credential header %q leaked to %s", name, location)
				}
			}
		})
	}
}

func TestCredentialLoadCancellationAndFailure(t *testing.T) {
	blocked := make(chan struct{})
	source := &memorySource{credentials: credentials("access"), block: blocked}
	model, err := openaicodex.NewModel("model", openaicodex.WithCredentialSource(source),
		openaicodex.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return codexSSE("done"), nil
		})}))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 2)
	go func() {
		_, requestErr := model.Request(ctx, nil, ai.ModelRequestParams{})
		done <- requestErr
	}()
	go func() {
		_, requestErr := model.Request(ctx, nil, ai.ModelRequestParams{})
		done <- requestErr
	}()
	time.Sleep(10 * time.Millisecond)
	cancel()
	for range 2 {
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("unexpected canceled load: %v", err)
		}
	}
	close(blocked)

	loadFailure := errors.New("load failed")
	failedSource := &memorySource{credentials: credentials("access"), loadErr: loadFailure}
	failed, err := openaicodex.NewModel("model", openaicodex.WithCredentialSource(failedSource),
		openaicodex.WithHTTPClient(&http.Client{Transport: http.DefaultTransport}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := failed.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, loadFailure) {
		t.Fatalf("unexpected source load error: %v", err)
	}
	invalidSource := &memorySource{}
	invalid, err := openaicodex.NewModel("model", openaicodex.WithCredentialSource(invalidSource),
		openaicodex.WithHTTPClient(&http.Client{Transport: http.DefaultTransport}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := invalid.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("invalid loaded credentials succeeded")
	}
}

func TestRefreshFailureIsSharedThenRetried(t *testing.T) {
	var mu sync.Mutex
	tokenCalls := 0
	initialCalls := 0
	release := make(chan struct{})
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		mu.Lock()
		defer mu.Unlock()
		if request.URL.Hostname() == "auth.openai.com" {
			tokenCalls++
			if tokenCalls == 1 {
				return response(http.StatusServiceUnavailable, `{"error":"temporarily_unavailable"}`), nil
			}
			return response(http.StatusOK,
				`{"access_token":"access-new","refresh_token":"refresh-new","account_id":"account-new"}`,
			), nil
		}
		if request.Header.Get("Authorization") == "Bearer access-new" {
			return codexSSE("recovered"), nil
		}
		initialCalls++
		if initialCalls == 5 {
			close(release)
		}
		mu.Unlock()
		<-release
		mu.Lock()
		return response(http.StatusUnauthorized, `{}`), nil
	})
	model := newModel(t, transport)
	var wait sync.WaitGroup
	wait.Add(5)
	errs := make(chan error, 5)
	for range 5 {
		go func() {
			defer wait.Done()
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
			errs <- err
		}()
	}
	wait.Wait()
	close(errs)
	for err := range errs {
		var refresh *openaicodex.CredentialsRefreshError
		if !errors.As(err, &refresh) || !refresh.IsModelAPIError() || !strings.Contains(refresh.Error(), "503") {
			t.Fatalf("unexpected shared refresh error: %v", err)
		}
	}
	result, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || result.Text() != "recovered" || tokenCalls != 2 {
		t.Fatalf("later request did not recover: %+v err=%v calls=%d", result, err, tokenCalls)
	}
}

func TestMalformedJWTsUseThe401Authority(t *testing.T) {
	tokens := []string{
		"not-a-jwt", "header.%%.signature", "header..signature", jwt(map[string]any{}),
		jwt(map[string]any{"exp": "soon"}), jwt(map[string]any{"exp": 1e20}),
	}
	for _, token := range tokens {
		t.Run(token, func(t *testing.T) {
			var refreshes int
			model := newModel(t, roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.URL.Hostname() == "auth.openai.com" {
					refreshes++
					return response(http.StatusOK,
						`{"access_token":"new","refresh_token":"new-refresh","account_id":"new-account"}`,
					), nil
				}
				if request.Header.Get("Authorization") == "Bearer "+token {
					return response(http.StatusUnauthorized, `{}`), nil
				}
				return codexSSE("done"), nil
			}), openaicodex.WithCredentials(credentials(token)))
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
				t.Fatal(err)
			}
			if refreshes != 1 {
				t.Fatalf("JWT hint triggered %d refreshes", refreshes)
			}
		})
	}
}

func TestRequestTransportAndStreamErrors(t *testing.T) {
	model := newModel(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network down")
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("transport failure succeeded")
	}
	model = newModel(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
		return response(http.StatusOK, ""), nil
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "without response.completed") {
		t.Fatalf("unexpected empty stream error: %v", err)
	}
}

func TestHTTPClientRemainsCallerOwned(t *testing.T) {
	base := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if strings.HasSuffix(request.URL.Path, "/cancel") {
			if request.Body != nil {
				t.Fatal("cancel request unexpectedly had a body")
			}
			return response(http.StatusOK, `{}`), nil
		}
		return codexSSE("done"), nil
	})
	client := &http.Client{Transport: base}
	model, err := openaicodex.NewModel("model", openaicodex.WithCredentials(credentials("access")),
		openaicodex.WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := client.Transport.(roundTripFunc); !ok {
		t.Fatal("constructor replaced the caller's transport")
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if err := model.CancelSuspendedResponse(t.Context(), ai.ModelResponse{
		ProviderName: "openai-codex", ProviderResponseID: "response", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true},
	}); err != nil {
		t.Fatal(err)
	}
}
