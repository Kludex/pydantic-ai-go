package azure_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/azure"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestAzureV1Model(t *testing.T) {
	t.Setenv("OPENAI_API_VERSION", "")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/openai/v1/chat/completions" || r.URL.RawQuery != "" {
			t.Errorf("unexpected request URL: %s", r.URL.String())
		}
		if authorization := r.Header.Get("Authorization"); authorization != "Bearer key" {
			t.Errorf("unexpected authorization: %q", authorization)
		}
		if apiKey := r.Header.Get("api-key"); apiKey != "" {
			t.Errorf("unexpected API key header: %q", apiKey)
		}
		_, _ = w.Write([]byte(`{
			"id":"response","model":"deployment","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]
		}`))
	}))
	defer server.Close()

	model, err := azure.NewModel("deployment", azure.Config{
		Endpoint: server.URL + "/openai/v1/", APIKey: "key", HTTPClient: server.Client(),
	}, openai.WithDefaultSettings(ai.ModelSettings{MaxTokens: 10}))
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "azure" || response.ProviderURL != server.URL+"/openai/v1" {
		t.Fatalf("unexpected provider: %+v", response)
	}
	if model.DefaultModelSettings().MaxTokens != 10 {
		t.Fatal("additional OpenAI options were not applied")
	}
	_, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{
			ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"},
		}},
	}}}, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "azure: Chat Completions does not support document input") {
		t.Fatalf("unexpected Azure document input error: %v", err)
	}
}

func TestAzureLegacyResponsesModelFromEnvironment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/openai/responses" || r.URL.Query().Get("api-version") != "2025-04-01" {
			t.Errorf("unexpected request URL: %s", r.URL.String())
		}
		if apiKey := r.Header.Get("api-key"); apiKey != "key" {
			t.Errorf("unexpected API key header: %q", apiKey)
		}
		if authorization := r.Header.Get("Authorization"); authorization != "" {
			t.Errorf("unexpected authorization: %q", authorization)
		}
		_, _ = w.Write([]byte(`{
			"id":"response","model":"deployment","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}
		}`))
	}))
	defer server.Close()
	t.Setenv("AZURE_OPENAI_ENDPOINT", server.URL)
	t.Setenv("AZURE_OPENAI_API_KEY", "key")
	t.Setenv("OPENAI_API_VERSION", "2025-04-01")

	model, err := azure.NewResponsesModel("deployment", azure.Config{HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "azure" || model.NativeToolSearchProvider() != "" {
		t.Fatalf("unexpected Azure Responses profile: response=%+v native=%q", response, model.NativeToolSearchProvider())
	}
}

func TestAzureFoundryServerlessEndpoint(t *testing.T) {
	t.Setenv("OPENAI_API_VERSION", "")
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://weather.eastus.models.ai.azure.com/v1/chat/completions" {
			t.Errorf("unexpected request URL: %s", request.URL)
		}
		return jsonResponse(`{
			"model":"deployment","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]
		}`), nil
	})}
	model, err := azure.NewModel("deployment", azure.Config{
		Endpoint: "https://weather.eastus.models.ai.azure.com", APIKey: "key", HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
}

func TestAzureTokenProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if authorization := r.Header.Get("Authorization"); authorization != "Bearer entra-token" {
			t.Errorf("unexpected authorization: %q", authorization)
		}
		if apiKey := r.Header.Get("api-key"); apiKey != "" {
			t.Errorf("unexpected API key header: %q", apiKey)
		}
		_, _ = w.Write([]byte(`{
			"model":"deployment","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]
		}`))
	}))
	defer server.Close()

	model, err := azure.NewModel("deployment", azure.Config{
		Endpoint: server.URL, APIVersion: "2025-04-01", HTTPClient: server.Client(),
		TokenProvider: func(context.Context) (string, error) { return "entra-token", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
}

func TestAzureTokenProviderFailures(t *testing.T) {
	tests := map[string]azure.TokenProvider{
		"error": func(context.Context) (string, error) { return "", errors.New("expired") },
		"empty": func(context.Context) (string, error) { return "", nil },
	}
	for name, tokenProvider := range tests {
		t.Run(name, func(t *testing.T) {
			model, err := azure.NewModel("deployment", azure.Config{
				Endpoint: "https://resource.openai.azure.com", APIVersion: "2025-04-01",
				TokenProvider: tokenProvider,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), "get Azure token") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestAzureConfigurationValidation(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	t.Setenv("AZURE_OPENAI_API_KEY", "")
	t.Setenv("OPENAI_API_VERSION", "")
	tests := map[string]struct {
		deployment string
		config     azure.Config
		contains   string
	}{
		"deployment": {config: azure.Config{}, contains: "deployment must not be empty"},
		"endpoint": {
			deployment: "deployment", config: azure.Config{Endpoint: "://bad", APIKey: "key"},
			contains: "endpoint must be an absolute URL",
		},
		"credentials": {
			deployment: "deployment", config: azure.Config{Endpoint: "https://resource.openai.azure.com"},
			contains: "API key or token provider is required",
		},
		"credential conflict": {
			deployment: "deployment", config: azure.Config{
				Endpoint: "https://resource.openai.azure.com", APIKey: "key",
				TokenProvider: func(context.Context) (string, error) { return "token", nil },
			},
			contains: "cannot both be set",
		},
		"legacy version": {
			deployment: "deployment", config: azure.Config{
				Endpoint: "https://resource.openai.azure.com", APIKey: "key",
			},
			contains: "API version is required",
		},
		"v1 version": {
			deployment: "deployment", config: azure.Config{
				Endpoint: "https://resource.openai.azure.com/openai/v1", APIKey: "key", APIVersion: "old",
			},
			contains: "API version is not supported",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := azure.NewModel(test.deployment, test.config)
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestAzureV1RejectsEnvironmentAPIVersion(t *testing.T) {
	t.Setenv("OPENAI_API_VERSION", "old")
	_, err := azure.NewResponsesModel("deployment", azure.Config{
		Endpoint: "https://resource.openai.azure.com/openai/v1", APIKey: "key",
	})
	if err == nil || !strings.Contains(err.Error(), "API version is not supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func jsonResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
