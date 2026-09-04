package webchat_test

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/webchat"
)

func TestRemoteHTMLIsFetchedAndCached(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		_, _ = io.WriteString(response, "<title>remote official UI</title>")
	}))
	defer server.Close()
	config := webchat.Config{HTMLSource: server.URL, CacheDir: t.TempDir()}
	for range 2 {
		handler := htmlHandler(t, config)
		for range 2 {
			response := serve(handler, request(http.MethodGet, "/", ""))
			if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "remote official UI") {
				t.Fatalf("unexpected remote UI response: %d %s", response.Code, response.Body.String())
			}
		}
	}
	if requests.Load() != 1 {
		t.Fatalf("UI fetched %d times", requests.Load())
	}
}

func TestDefaultAndEmptyHTMLCache(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, "cache"))
	var requestedURL string
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requestedURL = request.URL.String()
		return htmlResponse(http.StatusOK, "<title>default official UI</title>"), nil
	})}
	handler, err := webchat.NewHandler(
		ai.NewAgent[struct{}, string](fakes.NewTestModel()), struct{}{}, webchat.Config{HTTPClient: client},
	)
	if err != nil {
		t.Fatal(err)
	}
	response := serve(handler, request(http.MethodGet, "/", ""))
	if response.Code != http.StatusOK || requestedURL != webchat.DefaultHTMLURL || webchat.ChatUIVersion != "2.1.0" ||
		!strings.Contains(webchat.OfflineHTMLURL, "@pydantic/ai-chat-ui@2.1.0/offline/index.html") {
		t.Fatalf("unexpected default UI: status=%d URL=%q body=%s", response.Code, requestedURL, response.Body.String())
	}

	cacheDir := t.TempDir()
	source := "https://ui.example/index.html"
	cacheFile := filepath.Join(cacheDir, remoteCacheName(source))
	if err := os.WriteFile(cacheFile, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return htmlResponse(http.StatusOK, "<title>refetched UI</title>"), nil
	})}
	response = serve(htmlHandler(t, webchat.Config{
		HTMLSource: source, CacheDir: cacheDir, HTTPClient: client,
	}), request(http.MethodGet, "/", ""))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "refetched UI") {
		t.Fatalf("empty cache was not replaced: %d %s", response.Code, response.Body.String())
	}
}

func TestLocalHTMLSources(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	path := filepath.Join(home, "ui.html")
	if err := os.WriteFile(path, []byte("<title>local official UI</title>"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, source := range []string{path, "~/ui.html"} {
		response := serve(htmlHandler(t, webchat.Config{HTMLSource: source}), request(http.MethodGet, "/", ""))
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "local official UI") {
			t.Fatalf("unexpected local UI for %q: %d %s", source, response.Code, response.Body.String())
		}
	}
	response := serve(htmlHandler(t, webchat.Config{HTMLSource: "~"}), request(http.MethodGet, "/", ""))
	if response.Code != http.StatusInternalServerError || !strings.Contains(response.Body.String(), "read UI HTML") {
		t.Fatalf("unexpected directory source response: %d %s", response.Code, response.Body.String())
	}
	missing := serve(htmlHandler(t, webchat.Config{HTMLSource: filepath.Join(home, "missing.html")}),
		request(http.MethodGet, "/", ""))
	if missing.Code != http.StatusInternalServerError || !strings.Contains(missing.Body.String(), "read UI HTML") {
		t.Fatalf("unexpected missing source response: %d %s", missing.Code, missing.Body.String())
	}
	emptyPath := filepath.Join(home, "empty.html")
	if err := os.WriteFile(emptyPath, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	empty := serve(htmlHandler(t, webchat.Config{HTMLSource: emptyPath}), request(http.MethodGet, "/", ""))
	if empty.Code != http.StatusInternalServerError || !strings.Contains(empty.Body.String(), "UI HTML is empty") {
		t.Fatalf("unexpected empty source response: %d %s", empty.Code, empty.Body.String())
	}
}

func TestHTMLDirectoryLookupErrors(t *testing.T) {
	for _, name := range []string{"HOME", "XDG_CACHE_HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"} {
		t.Setenv(name, "")
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	if _, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{}); err == nil ||
		!strings.Contains(err.Error(), "locate user cache") {
		t.Fatalf("unexpected cache location error: %v", err)
	}
	handler, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{HTMLSource: "~"})
	if err != nil {
		t.Fatal(err)
	}
	response := serve(handler, request(http.MethodGet, "/", ""))
	if response.Code != http.StatusInternalServerError || !strings.Contains(response.Body.String(), "locate home directory") {
		t.Fatalf("unexpected home location error: %d %s", response.Code, response.Body.String())
	}
}

func TestRemoteHTMLErrors(t *testing.T) {
	tests := []struct {
		name   string
		source string
		client *http.Client
		want   string
	}{
		{
			name: "invalid URL", source: "http://%",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, errors.New("unexpected request")
			})}, want: "prepare UI request",
		},
		{
			name: "transport", source: "https://ui.example/transport",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, errors.New("offline")
			})}, want: "fetch UI HTML",
		},
		{
			name: "status", source: "https://ui.example/status",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return htmlResponse(http.StatusBadGateway, "failed"), nil
			})}, want: "502 Bad Gateway",
		},
		{
			name: "body", source: "https://ui.example/body",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Body: errorReadCloser{}}, nil
			})}, want: "read UI response",
		},
		{
			name: "empty", source: "https://ui.example/empty",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return htmlResponse(http.StatusOK, ""), nil
			})}, want: "UI HTML response is empty",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := serve(htmlHandler(t, webchat.Config{
				HTMLSource: test.source, CacheDir: t.TempDir(), HTTPClient: test.client,
			}), request(http.MethodGet, "/", ""))
			if response.Code != http.StatusInternalServerError || !strings.Contains(response.Body.String(), test.want) {
				t.Fatalf("unexpected remote error: %d %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestHTMLCacheErrors(t *testing.T) {
	source := "https://ui.example/cache"
	t.Run("create directory", func(t *testing.T) {
		parent := filepath.Join(t.TempDir(), "file")
		if err := os.WriteFile(parent, []byte("file"), 0o600); err != nil {
			t.Fatal(err)
		}
		response := serve(htmlHandler(t, webchat.Config{
			HTMLSource: source, CacheDir: filepath.Join(parent, "cache"), HTTPClient: successfulHTMLClient(),
		}), request(http.MethodGet, "/", ""))
		assertHTMLFailure(t, response, "create UI cache")
	})
	t.Run("read", func(t *testing.T) {
		cacheDir := t.TempDir()
		if err := os.Mkdir(filepath.Join(cacheDir, remoteCacheName(source)), 0o700); err != nil {
			t.Fatal(err)
		}
		response := serve(htmlHandler(t, webchat.Config{
			HTMLSource: source, CacheDir: cacheDir, HTTPClient: successfulHTMLClient(),
		}), request(http.MethodGet, "/", ""))
		assertHTMLFailure(t, response, "read UI cache")
	})
	t.Run("write", func(t *testing.T) {
		cacheDir := t.TempDir()
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			if err := os.RemoveAll(cacheDir); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(cacheDir, []byte("file"), 0o600); err != nil {
				t.Fatal(err)
			}
			return htmlResponse(http.StatusOK, "<title>UI</title>"), nil
		})}
		response := serve(htmlHandler(t, webchat.Config{
			HTMLSource: source, CacheDir: cacheDir, HTTPClient: client,
		}), request(http.MethodGet, "/", ""))
		assertHTMLFailure(t, response, "write UI cache")
	})
	t.Run("replace", func(t *testing.T) {
		cacheDir := t.TempDir()
		cacheFile := filepath.Join(cacheDir, remoteCacheName(source))
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			if err := os.Mkdir(cacheFile, 0o700); err != nil {
				t.Fatal(err)
			}
			return htmlResponse(http.StatusOK, "<title>UI</title>"), nil
		})}
		response := serve(htmlHandler(t, webchat.Config{
			HTMLSource: source, CacheDir: cacheDir, HTTPClient: client,
		}), request(http.MethodGet, "/", ""))
		assertHTMLFailure(t, response, "write UI cache")
	})
}

func htmlHandler(t *testing.T, config webchat.Config) http.Handler {
	t.Helper()
	handler, err := webchat.NewHandler(ai.NewAgent[struct{}, string](fakes.NewTestModel()), struct{}{}, config)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func successfulHTMLClient() *http.Client {
	return &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return htmlResponse(http.StatusOK, "<title>UI</title>"), nil
	})}
}

func htmlResponse(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Status: fmt.Sprintf("%d %s", status, http.StatusText(status)),
		Body: io.NopCloser(strings.NewReader(body))}
}

func remoteCacheName(source string) string {
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(source)))
	return "url_" + digest[:16] + ".html"
}

func assertHTMLFailure(t *testing.T, response *httptest.ResponseRecorder, message string) {
	t.Helper()
	if response.Code != http.StatusInternalServerError || !strings.Contains(response.Body.String(), message) {
		t.Fatalf("unexpected cache error: %d %s", response.Code, response.Body.String())
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type errorReadCloser struct{}

func (errorReadCloser) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (errorReadCloser) Close() error             { return nil }
