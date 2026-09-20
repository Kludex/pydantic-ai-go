package typesafe_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/infer"
	"github.com/Kludex/pydantic-ai-go/ai/models/typesafe"
)

type decision struct {
	Harmful bool   `json:"harmful" jsonschema_description:"Is this harmful?"`
	Intent  string `json:"intent" jsonschema:"enum=run,enum=reject" jsonschema_description:"What should happen?"`
}

func TestModel(t *testing.T) {
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, incoming *http.Request) {
		if incoming.Header.Get("Authorization") != "Bearer secret" || incoming.Header.Get("X-Test") != "yes" {
			t.Errorf("unexpected headers: %v", incoming.Header)
		}
		if err := json.NewDecoder(incoming.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{"answers":{"harmful":{"type":"noul","noul":0.9},"intent":{"type":"choice","choice":"reject","confidence":0.8,"probabilities":{"run":0.2,"reject":0.8}}},"model":"jev-1.13.0","usage":{"input_tokens":10,"output_tokens":2}}`))
	}))
	defer server.Close()
	threshold := .75
	settings, err := (typesafe.Settings{BooleanThreshold: &threshold}).Build()
	if err != nil {
		t.Fatal(err)
	}
	settings.ExtraHeaders = map[string]string{"X-Test": "yes"}
	model := typesafe.NewModel("jev-latest", typesafe.WithBaseURL(server.URL), typesafe.WithAPIKey("secret"), typesafe.WithHTTPClient(server.Client()), typesafe.WithDefaultSettings(settings))
	agent := ai.NewAgent[struct{}, decision](model, ai.WithInstructions("Classify the command."))
	result, err := agent.Run(t.Context(), "delete everything", struct{}{})
	if err != nil || !result.Output.Harmful || result.Output.Intent != "reject" || result.Usage().InputTokens != 10 {
		t.Fatalf("unexpected result=%+v err=%v", result, err)
	}
	if request["model"] != "jev-latest" || model.ProviderName() != "typesafe" || model.ProviderURL() != server.URL ||
		model.ContextWindow() != 64_000 || model.DefaultModelSettings().ExtraHeaders["X-Test"] != "yes" {
		t.Fatalf("unexpected model or request: %#v", request)
	}
	profile := model.ModelProfile()
	if profile.SupportsTextOutput == nil || *profile.SupportsTextOutput || profile.ContextWindow != 64_000 {
		t.Fatalf("unexpected profile: %+v", profile)
	}
	resolved, err := infer.Model("typesafe:jev-preview", infer.WithProvider("typesafe", func(name string) (ai.Model, error) {
		return typesafe.NewModel(name, typesafe.WithAPIKey("x")), nil
	}))
	if err != nil || resolved.Name() != "jev-preview" {
		t.Fatalf("unexpected inference: %v %v", resolved, err)
	}
}

func TestModelErrors(t *testing.T) {
	model := typesafe.NewModel("jev-latest", typesafe.WithAPIKey("x"))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected text output error")
	}
	bad := -1.0
	if _, err := (typesafe.Settings{BooleanThreshold: &bad}).Build(); err == nil {
		t.Fatal("expected threshold error")
	}
	settings := ai.ModelSettings{ExtraBody: map[string]any{"typesafe_settings": "bad"}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err == nil {
		t.Fatal("expected settings error")
	}

	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		http.Error(response, "failed", http.StatusTooManyRequests)
	}))
	defer server.Close()
	model = typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "x"}}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Description: "Choose", Schema: map[string]any{"type": "object", "properties": map[string]any{"ok": map[string]any{"type": "boolean", "description": "Is it ok?"}}}}})
	var apiError *typesafe.APIError
	if !errors.As(err, &apiError) || !apiError.IsModelAPIError() || !strings.Contains(apiError.Error(), "429") {
		t.Fatalf("unexpected API error: %v", err)
	}
	proposed := &typesafe.ToolCallProposed{ModelName: "jev", ToolName: "tool", Probability: .7}
	if !proposed.IsModelAPIError() || !strings.Contains(proposed.Error(), "tool") {
		t.Fatal(proposed)
	}
}

func TestProviderValidation(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	typesafe.NewModel("jev", typesafe.WithProvider(typesafe.ProviderConfig{}))
}

func TestSettingsReserved(t *testing.T) {
	_, err := (typesafe.Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{"typesafe_settings": true}}}).Build()
	if err == nil {
		t.Fatal("expected reserved setting error")
	}
	client := &http.Client{}
	provider := typesafe.NewProviderConfig()
	provider.Name = "custom"
	provider.BaseURL = "https://example.com"
	provider.HTTPClient = client
	provider.Headers = http.Header{"X": {"y"}}
	provider.PrepareRequest = func(*http.Request) error { return context.Canceled }
	clone := provider.Clone()
	clone.Headers.Set("X", "z")
	if provider.Headers.Get("X") != "y" {
		t.Fatal("provider clone retained headers")
	}
	model := typesafe.NewModel("jev", typesafe.WithProvider(provider))
	_, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "x"}}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: map[string]any{"type": "boolean", "description": "ok?"}}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected prepare error: %v", err)
	}
}
