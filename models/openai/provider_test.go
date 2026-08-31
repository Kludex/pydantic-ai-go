package openai_test

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestOpenAICompatibleProvider(t *testing.T) {
	headers := http.Header{"X-Provider": {"configured"}}
	query := url.Values{"region": {"west"}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.URL.Query().Get("region") != "west" {
			t.Errorf("unexpected request URL: %s", r.URL.String())
		}
		if authorization := r.Header.Get("Authorization"); authorization != "" {
			t.Errorf("unexpected authorization: %q", authorization)
		}
		if value := r.Header.Get("X-Provider"); value != "per-request" {
			t.Errorf("unexpected provider header: %q", value)
		}
		if value := r.Header.Get("X-Dynamic"); value != "token" {
			t.Errorf("unexpected dynamic header: %q", value)
		}
		_, _ = w.Write([]byte(`{
			"id":"response-id","model":"compatible-model","created":1,
			"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	model := openai.NewModel("compatible-model", openai.WithProvider(openai.ProviderConfig{
		Name: "local", BaseURL: server.URL + "/v1/", HTTPClient: server.Client(),
		Headers: headers, Query: query,
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "token")
			return nil
		},
	}))
	headers.Set("X-Provider", "mutated")
	query.Set("region", "mutated")

	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{ExtraHeaders: map[string]string{"X-Provider": "per-request"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "local" || response.ProviderURL != server.URL+"/v1" {
		t.Fatalf("unexpected provider identity: %+v", response)
	}
}

func TestOpenAICompatibleResponsesProviderIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"response-id","model":"compatible-model","created_at":1,"status":"completed",
			"output":[
				{"type":"reasoning","id":"reasoning-id","encrypted_content":"signature"},
				{"type":"message","id":"message-id","content":[{"type":"output_text","text":"hello"}]},
				{"type":"function_call","id":"item-id","call_id":"call-id","name":"tool","arguments":"{}"},
				{"type":"compaction","id":"compaction-id","encrypted_content":"opaque"},
				{"type":"tool_search_call","id":"search-call","call_id":"search","execution":"server","arguments":{}},
				{"type":"tool_search_output","id":"search-output","call_id":"search","execution":"server","tools":[]}
			],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	model := openai.NewResponsesModel("compatible-model", openai.WithProvider(openai.ProviderConfig{
		Name: "compatible", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "compatible" || model.NativeToolSearchProvider() != "compatible" {
		t.Fatalf("unexpected provider identity: %+v", response)
	}
	for _, part := range response.Parts {
		switch part := part.(type) {
		case ai.TextPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected text provider: %+v", part)
			}
		case ai.ThinkingPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected thinking provider: %+v", part)
			}
		case ai.ToolCallPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected tool provider: %+v", part)
			}
		case ai.NativeToolCallPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected native call provider: %+v", part)
			}
		case ai.NativeToolReturnPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected native return provider: %+v", part)
			}
		case ai.CompactionPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected compaction provider: %+v", part)
			}
		}
	}
}

func TestOpenAICompatibleResponsesStreamingIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message\",\"delta\":\"hi\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"response\",\"model\":\"model\",\"status\":\"completed\",\"output\":[{\"id\":\"message\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}],\"usage\":{}}}\n\n"))
	}))
	defer server.Close()

	model := openai.NewResponsesModel("model", openai.WithProvider(openai.ProviderConfig{
		Name: "compatible", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var text ai.TextDeltaEvent
	var finish ai.FinishEvent
	for event, streamErr := range events {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			text = event
		case ai.FinishEvent:
			finish = event
		}
	}
	if text.ProviderName != "compatible" || finish.ProviderName != "compatible" {
		t.Fatalf("unexpected streaming provider identity: text=%+v finish=%+v", text, finish)
	}
	if len(finish.Parts) != 1 || finish.Parts[0].(ai.TextPart).ProviderName != "compatible" {
		t.Fatalf("unexpected snapshot provider identity: %+v", finish.Parts)
	}
}

func TestOpenAICompatibleResponsesCanDisableNativeHistory(t *testing.T) {
	model := openai.NewResponsesModel("model", openai.WithDeferredToolSupport(false))
	if provider := model.NativeToolSearchProvider(); provider != "" {
		t.Fatalf("unexpected native history provider: %q", provider)
	}
}

func TestOpenAIProviderPreparationError(t *testing.T) {
	provider := openai.ProviderConfig{
		Name: "local", BaseURL: "http://localhost", PrepareRequest: func(*http.Request) error {
			return errors.New("credentials unavailable")
		},
	}
	chat := openai.NewModel("model", openai.WithProvider(provider))
	responses := openai.NewResponsesModel("model", openai.WithProvider(provider))
	suspended := ai.ModelResponse{
		ProviderName: "local", ProviderResponseID: "response",
		ProviderDetails: map[string]any{"background": true}, State: ai.ModelResponseStateSuspended,
	}
	streamSuspended := suspended
	streamSuspended.ProviderDetails = map[string]any{"background": true, "sequence_number": 1}
	tests := map[string]func() error{
		"chat": func() error {
			_, err := chat.Request(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"chat stream": func() error {
			_, err := chat.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"responses": func() error {
			_, err := responses.Request(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"responses stream": func() error {
			_, err := responses.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"compaction": func() error {
			_, err := responses.CompactMessages(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"retrieve": func() error {
			_, err := responses.Request(t.Context(), []ai.ModelMessage{suspended}, ai.ModelRequestParams{})
			return err
		},
		"retrieve stream": func() error {
			_, err := responses.StreamRequest(
				t.Context(), []ai.ModelMessage{streamSuspended}, ai.ModelRequestParams{},
			)
			return err
		},
		"cancel": func() error { return responses.CancelSuspendedResponse(t.Context(), suspended) },
	}
	for name, operation := range tests {
		t.Run(name, func(t *testing.T) {
			err := operation()
			if err == nil || !strings.Contains(err.Error(), "openai: prepare request: credentials unavailable") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestOpenAIProviderValidation(t *testing.T) {
	tests := map[string]openai.ProviderConfig{
		"name":     {BaseURL: "https://example.com/v1"},
		"base URL": {Name: "compatible"},
	}
	for name, provider := range tests {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			_ = openai.WithProvider(provider)
		})
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	_ = openai.WithProviderName("")
}

func TestOpenAIBaseURLFromEnvironment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"model":"model","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]
		}`))
	}))
	defer server.Close()
	t.Setenv("OPENAI_BASE_URL", server.URL+"/v1/")
	t.Setenv("OPENAI_API_KEY", "")

	model := openai.NewModel("model", openai.WithHTTPClient(server.Client()), openai.WithProviderName("proxy"))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "proxy" {
		t.Fatalf("unexpected provider name: %q", response.ProviderName)
	}
}
