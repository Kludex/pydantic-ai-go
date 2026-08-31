package google_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

type providerContextKey struct{}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func googleResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func consumeGoogleStream(t *testing.T, events iter.Seq2[ai.ModelStreamEvent, error]) []ai.ModelStreamEvent {
	t.Helper()
	var collected []ai.ModelStreamEvent
	for event, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		collected = append(collected, event)
	}
	return collected
}

func TestVertexRequestAndStreamRouting(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	var authorizations []string
	var requestTypes []string
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		mu.Lock()
		paths = append(paths, request.URL.Path)
		authorizations = append(authorizations, request.Header.Get("Authorization"))
		requestTypes = append(requestTypes, request.Header.Get("X-Vertex-AI-LLM-Shared-Request-Type"))
		bodies = append(bodies, body)
		mu.Unlock()
		if strings.Contains(request.URL.Path, ":streamGenerateContent") {
			response.Header().Set("Content-Type", "text/event-stream")
			payload := `{"modelVersion":"gemini-2.5-flash",` +
				`"candidates":[{"content":{"parts":[{"text":"streamed",` +
				`"thoughtSignature":"stream-signature"}]},"finishReason":"STOP"}]}`
			_, _ = fmt.Fprintln(response, "data: "+payload)
			return
		}
		_, _ = io.WriteString(response, `{
			"modelVersion":"gemini-2.5-flash",
			"candidates":[{"content":{"parts":[{"text":"done","thoughtSignature":"signature"}]},"finishReason":"STOP"}]
		}`)
	}))
	defer server.Close()

	tokenCalls := 0
	model, err := google.NewVertexModel("gemini-2.5-flash", google.VertexConfig{
		Project: "project-one", Location: "us-central1", Endpoint: server.URL, HTTPClient: server.Client(),
		TokenProvider: func(ctx context.Context) (string, error) {
			tokenCalls++
			if ctx.Value(providerContextKey{}) != "request" {
				t.Fatal("token provider did not receive the request context")
			}
			return "vertex-token", nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if model.Transport() != google.TransportVertexAI {
		t.Fatalf("unexpected transport %q", model.Transport())
	}
	ctx := context.WithValue(t.Context(), providerContextKey{}, "request")
	result, err := model.Request(ctx, nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: ai.ServiceTierFlex,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderName != "google-cloud" || !strings.Contains(result.ProviderURL, "/projects/project-one/") {
		t.Fatalf("unexpected Vertex attribution: %+v", result)
	}
	part := result.Parts[0].(ai.TextPart)
	if part.ProviderName != "google-cloud" || part.ProviderDetails["thought_signature"] != "signature" {
		t.Fatalf("unexpected Vertex part metadata: %+v", part)
	}
	if _, err := model.Request(ctx, []ai.ModelMessage{*result}, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraHeaders: map[string]string{"Authorization": "Bearer application-token"},
	}}); err != nil {
		t.Fatal(err)
	}

	events, err := model.StreamRequest(ctx, nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: ai.ServiceTierPriority,
	}})
	if err != nil {
		t.Fatal(err)
	}
	streamed := consumeGoogleStream(t, events)
	text := streamed[0].(ai.TextDeltaEvent)
	finish := streamed[len(streamed)-1].(ai.FinishEvent)
	if text.ProviderName != "google-cloud" || text.ProviderDetails["thought_signature"] != "stream-signature" ||
		finish.ProviderName != "google-cloud" || finish.ProviderURL != result.ProviderURL {
		t.Fatalf("unexpected Vertex stream metadata: %#v", streamed)
	}

	mu.Lock()
	defer mu.Unlock()
	wantPrefix := "/v1beta1/projects/project-one/locations/us-central1/publishers/google/models/gemini-2.5-flash:"
	if len(paths) != 3 || !strings.HasPrefix(paths[0], wantPrefix+"generateContent") ||
		!strings.HasPrefix(paths[1], wantPrefix+"generateContent") ||
		!strings.HasPrefix(paths[2], wantPrefix+"streamGenerateContent") {
		t.Fatalf("unexpected Vertex paths: %v", paths)
	}
	replayed := bodies[1]["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if replayed["thoughtSignature"] != "signature" {
		t.Fatalf("Vertex thought signature was not replayed: %v", replayed)
	}
	if authorizations[0] != "Bearer vertex-token" || authorizations[1] != "Bearer application-token" ||
		authorizations[2] != "Bearer vertex-token" || tokenCalls != 3 {
		t.Fatalf("unexpected Vertex authentication: headers=%v calls=%d", authorizations, tokenCalls)
	}
	if requestTypes[0] != "flex" || requestTypes[1] != "" || requestTypes[2] != "priority" {
		t.Fatalf("unexpected Vertex service tiers: %v", requestTypes)
	}
}

func TestVertexExpressEnvironmentFallback(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "legacy-express-key")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	var host, key string
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		host = request.URL.Host
		key = request.Header.Get("x-goog-api-key")
		return googleResponse(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`), nil
	})}
	model, err := google.NewVertexModel("gemini", google.VertexConfig{HTTPClient: client})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if host != "aiplatform.googleapis.com" || key != "legacy-express-key" {
		t.Fatalf("unexpected implicit Express route host=%q key=%q", host, key)
	}
}

func TestVertexExpressModeAndHeaderPrecedence(t *testing.T) {
	var key, authorization, requestType, path string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		key = request.Header.Get("x-goog-api-key")
		authorization = request.Header.Get("Authorization")
		requestType = request.Header.Get("X-Vertex-AI-LLM-Request-Type")
		path = request.URL.Path
		_, _ = io.WriteString(response, `{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`)
	}))
	defer server.Close()
	model, err := google.NewVertexModel("gemini", google.VertexConfig{
		APIKey: " express-key ", Endpoint: server.URL, HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: ai.ServiceTierDefault,
		ExtraHeaders: map[string]string{
			"Authorization":                "Bearer application-token",
			"X-Vertex-AI-LLM-Request-Type": "application",
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if key != "express-key" || authorization != "Bearer application-token" || requestType != "application" ||
		path != "/v1beta1/publishers/google/models/gemini:generateContent" {
		t.Fatalf("unexpected Express request: key=%q auth=%q tier=%q path=%q", key, authorization, requestType, path)
	}
}

func TestVertexLocationEndpoints(t *testing.T) {
	for _, test := range []struct {
		location string
		host     string
	}{
		{location: "global", host: "aiplatform.googleapis.com"},
		{location: "us", host: "aiplatform.us.rep.googleapis.com"},
		{location: "eu", host: "aiplatform.eu.rep.googleapis.com"},
		{location: "europe-west4", host: "europe-west4-aiplatform.googleapis.com"},
	} {
		t.Run(test.location, func(t *testing.T) {
			var host, path string
			client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				host, path = request.URL.Host, request.URL.EscapedPath()
				return googleResponse(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`), nil
			})}
			model, err := google.NewVertexModel("gemini", google.VertexConfig{
				Project: "my/project", Location: test.location, HTTPClient: client,
				TokenProvider: func(context.Context) (string, error) { return "token", nil },
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
				t.Fatal(err)
			}
			if host != test.host || !strings.Contains(path, "/projects/my%2Fproject/locations/"+test.location+"/") {
				t.Fatalf("unexpected route host=%q path=%q", host, path)
			}
		})
	}
}

func TestGoogleHistoryMatchingUsesTransport(t *testing.T) {
	for _, test := range []struct {
		name          string
		transport     google.Transport
		providerName  string
		historySource string
		wantSignature bool
	}{
		{
			name: "Vertex accepts cloud alias", transport: google.TransportVertexAI,
			providerName: "google", historySource: "google-cloud", wantSignature: true,
		},
		{
			name: "Vertex rejects Developer alias", transport: google.TransportVertexAI,
			providerName: "google", historySource: "google-gla",
		},
		{
			name: "Developer accepts API alias", transport: google.TransportGeminiAPI,
			providerName: "google-cloud", historySource: "google", wantSignature: true,
		},
		{
			name: "Developer rejects cloud alias", transport: google.TransportGeminiAPI,
			providerName: "google-cloud", historySource: "google-vertex",
		},
		{
			name: "custom rejects Google alias", transport: google.TransportGeminiAPI,
			providerName: "proxy", historySource: "google",
		},
		{
			name: "custom accepts itself", transport: google.TransportGeminiAPI,
			providerName: "proxy", historySource: "proxy", wantSignature: true,
		},
		{
			name: "legacy empty source", transport: google.TransportVertexAI,
			providerName: "google-cloud", wantSignature: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var signature any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				part := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0].(map[string]any)
				signature = part["thoughtSignature"]
				_, _ = io.WriteString(response, `{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`)
			}))
			defer server.Close()
			model := google.NewModel("gemini", google.WithProvider(google.ProviderConfig{
				Transport: test.transport, Name: test.providerName, BaseURL: server.URL,
				APIKey: "key", HTTPClient: server.Client(),
			}))
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.TextPart{
					Content: "previous", ProviderName: test.historySource,
					ProviderDetails: map[string]any{"thought_signature": "signature"},
				},
			}}}, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			if (signature == "signature") != test.wantSignature {
				t.Fatalf("unexpected replay signature: %v", signature)
			}
		})
	}
}

func TestGoogleProviderPreparationAndEnvironment(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "google-key")
	t.Setenv("GEMINI_API_KEY", "legacy-key")
	var key, prepared string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		key = request.Header.Get("x-goog-api-key")
		prepared = request.Header.Get("X-Prepared")
		_, _ = io.WriteString(response, `{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`)
	}))
	defer server.Close()
	model := google.NewModel("gemini", google.WithBaseURL(server.URL), google.WithHTTPClient(server.Client()))
	if model.Transport() != google.TransportGeminiAPI {
		t.Fatalf("unexpected default transport %q", model.Transport())
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if key != "google-key" {
		t.Fatalf("GOOGLE_API_KEY did not take precedence: %q", key)
	}

	model = google.NewModel("gemini", google.WithProvider(google.ProviderConfig{
		Transport: google.TransportGeminiAPI, Name: "proxy", BaseURL: server.URL, APIKey: "provider-key",
		HTTPClient: server.Client(), PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Prepared", "yes")
			return nil
		},
	}))
	result, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if key != "provider-key" || prepared != "yes" || result.ProviderName != "proxy" {
		t.Fatalf("unexpected provider request key=%q prepared=%q result=%+v", key, prepared, result)
	}

	vertexModel, err := google.NewVertexModel("gemini", google.VertexConfig{
		Project: "project", TokenProvider: func(context.Context) (string, error) { return "token", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := vertexModel.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: "expedited",
	}}); err == nil || err.Error() != `google: invalid service tier "expedited"` {
		t.Fatalf("unexpected Vertex service tier error: %v", err)
	}

	prepareError := errors.New("credentials unavailable")
	model = google.NewModel("gemini", google.WithProvider(google.ProviderConfig{
		Transport: google.TransportGeminiAPI, Name: "proxy", BaseURL: server.URL,
		PrepareRequest: func(*http.Request) error { return prepareError },
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, prepareError) ||
		err.Error() != "google: prepare request: credentials unavailable" {
		t.Fatalf("unexpected preparation error: %v", err)
	}
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, prepareError) {
		t.Fatalf("unexpected stream preparation error: %v", err)
	}
}

func TestVertexConfigurationAndTokenErrors(t *testing.T) {
	func() {
		defer func() {
			if value := recover(); value != `google: invalid transport "teleport"` {
				t.Fatalf("unexpected transport panic: %v", value)
			}
		}()
		google.WithProvider(google.ProviderConfig{Transport: "teleport"})
	}()
	model := google.NewModel("gemini", google.WithProvider(google.ProviderConfig{}))
	if model.Transport() != google.TransportGeminiAPI {
		t.Fatalf("empty provider configuration changed transport: %q", model.Transport())
	}

	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
	for _, test := range []struct {
		name   string
		config google.VertexConfig
		want   string
	}{
		{
			name: "authentication conflict",
			config: google.VertexConfig{
				APIKey: "key", TokenProvider: func(context.Context) (string, error) { return "token", nil },
			},
			want: "API key and token provider cannot both be set",
		},
		{name: "invalid endpoint", config: google.VertexConfig{APIKey: "key", Endpoint: "://bad"}, want: "endpoint"},
		{name: "invalid location", config: google.VertexConfig{APIKey: "key", Location: "bad/location"}, want: "location"},
		{
			name:   "missing project",
			config: google.VertexConfig{TokenProvider: func(context.Context) (string, error) { return "token", nil }},
			want:   "project is required",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := google.NewVertexModel("gemini", test.config); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected configuration error: %v", err)
			}
		})
	}

	for _, test := range []struct {
		name     string
		provider google.TokenProvider
		want     string
	}{
		{name: "provider error", provider: func(context.Context) (string, error) {
			return "", errors.New("token failed")
		}, want: "token failed"},
		{name: "empty token", provider: func(context.Context) (string, error) { return "", nil }, want: "empty token"},
	} {
		t.Run(test.name, func(t *testing.T) {
			model, err := google.NewVertexModel("gemini", google.VertexConfig{
				Project: "project", TokenProvider: test.provider,
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected token error: %v", err)
			}
		})
	}
}

func TestVertexApplicationDefaultCredentials(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	tokenFailure := false
	tokenServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := request.ParseForm(); err != nil {
			t.Error(err)
		}
		if tokenFailure {
			http.Error(response, "token unavailable", http.StatusInternalServerError)
			return
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(response, `{"access_token":"adc-token","token_type":"Bearer","expires_in":3600}`)
	}))
	defer tokenServer.Close()
	credentials := fmt.Sprintf(`{
		"type":"authorized_user",
		"client_id":"client",
		"client_secret":"secret",
		"refresh_token":"refresh",
		"token_uri":%q
	}`, tokenServer.URL)
	path := t.TempDir() + "/credentials.json"
	if err := os.WriteFile(path, []byte(credentials), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", path)

	var authorization string
	apiServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		authorization = request.Header.Get("Authorization")
		_, _ = io.WriteString(response, `{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`)
	}))
	defer apiServer.Close()
	model, err := google.NewVertexModel("gemini", google.VertexConfig{
		Project: "project", Endpoint: apiServer.URL, HTTPClient: apiServer.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if authorization != "Bearer adc-token" {
		t.Fatalf("unexpected ADC authorization %q", authorization)
	}

	tokenFailure = true
	model, err = google.NewVertexModel("gemini", google.VertexConfig{
		Project: "project", Endpoint: apiServer.URL, HTTPClient: apiServer.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "cannot fetch token") {
		t.Fatalf("unexpected ADC token error: %v", err)
	}

	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", path+"-missing")
	model, err = google.NewVertexModel("gemini", google.VertexConfig{Project: "project"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "credentials") {
		t.Fatalf("unexpected ADC discovery error: %v", err)
	}
}
