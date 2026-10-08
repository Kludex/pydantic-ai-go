package systemone_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/infer"
	"github.com/Kludex/pydantic-ai-go/ai/models/systemone"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type ticket struct {
	Urgent bool   `json:"urgent" jsonschema_description:"Does this need a reply within the hour?"`
	Area   string `json:"area" jsonschema:"enum=billing,enum=bug" jsonschema_description:"Which team owns it?"`
}

const ticketReply = `{"model":"laya-rl-agent","answers":{"urgent":{"type":"noul","noul":0.91,"answer_confidence":0.91,"action":{"act_probability":1}},"area":{"type":"choice","choice":"billing","confidence":0.88,"probabilities":{"billing":0.94,"bug":0.06}}},"usage":{"input_tokens":42,"output_tokens":0},"routing":{"reason":"English"}}`

func newModel(t *testing.T, options ...systemone.Option) *systemone.Model {
	t.Helper()
	model, err := systemone.NewModel("clm-latest", options...)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func floatPointer(value float64) *float64 { return &value }

func prompt() []ai.ModelMessage {
	return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "Charged twice."}}}}
}

func boolParams() ai.ModelRequestParams {
	return ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: map[string]any{"type": "boolean", "description": "Is this urgent?"}}}
}

func serverModel(t *testing.T, reply string) *systemone.Model {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		_, _ = writer.Write([]byte(reply))
	}))
	t.Cleanup(server.Close)
	return newModel(t, systemone.WithBaseURL(server.URL), systemone.WithHTTPClient(server.Client()))
}

func TestAgentAndConfiguration(t *testing.T) {
	for _, suffix := range []string{"", "/v1/"} {
		t.Run(suffix, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/v1/systemone" || request.Method != http.MethodPost || request.Header.Get("Authorization") != "Bearer override" || len(request.Header.Values("Authorization")) != 1 || request.Header.Get("X-Team") != "support" || request.Header.Get("X-Prepared") != "yes" {
					t.Errorf("unexpected request: %s %s %v", request.Method, request.URL, request.Header)
				}
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = writer.Write([]byte(ticketReply))
			}))
			defer server.Close()
			t.Setenv("SYSTEM_ONE_BASE_URL", server.URL+suffix)
			t.Setenv("SYSTEM_ONE_API_KEY", "secret")
			provider, err := systemone.NewProviderConfig()
			if err != nil || provider.Name != "system-one" || provider.APIKey != "secret" {
				t.Fatalf("provider=%+v error=%v", provider, err)
			}
			provider.HTTPClient = server.Client()
			provider.Headers = http.Header{"X-Team": {"wrong"}}
			provider.PrepareRequest = func(request *http.Request) error { request.Header.Set("X-Prepared", "yes"); return nil }
			clone := provider.Clone()
			clone.Headers.Set("X-Team", "other")
			if provider.Headers.Get("X-Team") != "wrong" {
				t.Fatal("headers were not detached")
			}
			temperature, threshold := .5, .95
			settings, err := (systemone.Settings{Common: ai.ModelSettings{Temperature: &temperature, RequestTimeout: time.Second, ExtraHeaders: map[string]string{"authorization": "Bearer override", "X-Team": "support"}, ExtraBody: map[string]any{"trace": true}}, BooleanThreshold: &threshold}).Build()
			if err != nil {
				t.Fatal(err)
			}
			threshold = 0
			model := newModel(t, systemone.WithProvider(provider), systemone.WithDefaultSettings(settings), systemone.WithProfile(systemone.Profile{ContextWindow: 8000}))
			provider.Headers.Set("X-Team", "mutated")
			settings.ExtraHeaders["X-Team"] = "mutated"
			defaults := model.DefaultModelSettings()
			defaults.ExtraHeaders["X-Team"] = "also mutated"
			profile := model.ModelProfile()
			if model.Name() != "clm-latest" || model.ProviderName() != "system-one" || model.ProviderURL() != strings.TrimRight(server.URL+suffix, "/") || model.ContextWindow() != 8000 || profile.DefaultOutputMode != ai.OutputModeTool || profile.SupportsTextOutput == nil || *profile.SupportsTextOutput || profile.ContextWindow != 8000 {
				t.Fatalf("unexpected identity or profile: %+v", profile)
			}
			result, err := ai.NewAgent[struct{}, ticket](model, ai.WithInstructions("Triage a support ticket.")).Run(t.Context(), "Charged twice.", struct{}{})
			if err != nil || result.Output.Urgent || result.Output.Area != "billing" || result.Usage().InputTokens != 42 {
				t.Fatalf("result=%+v error=%v", result, err)
			}
			if body["state"] != "Charged twice." || body["model"] != "clm-latest" || body["temperature"] != .5 || body["trace"] != true || body["system_one_settings"] != nil {
				t.Fatalf("body=%v", body)
			}
			questions := body["questions"].(map[string]any)
			if questions["urgent"].(map[string]any)["type"] != "noul" || questions["area"].(map[string]any)["type"] != "choice" {
				t.Fatalf("questions=%v", questions)
			}
		})
	}
}

func TestEnvironmentAndValidation(t *testing.T) {
	t.Setenv("SYSTEM_ONE_BASE_URL", "")
	t.Setenv("SYSTEM_ONE_API_KEY", "")
	if _, err := systemone.NewProviderConfig(); err == nil {
		t.Fatal("expected missing URL")
	}
	if _, err := infer.Model("system-one:clm-latest"); err == nil {
		t.Fatal("expected missing URL")
	}
	for _, options := range [][]systemone.Option{
		{systemone.WithBaseURL(":")}, {systemone.WithBaseURL("ftp://example.com")}, {systemone.WithBaseURL("https://example.com?x=1")}, {systemone.WithBaseURL("https://example.com#fragment")},
		{systemone.WithProvider(systemone.ProviderConfig{BaseURL: "https://example.com"})},
		{systemone.WithBaseURL("https://example.com"), systemone.WithProfile(systemone.Profile{MaxChoiceOptions: -1})},
	} {
		if _, err := systemone.NewModel("clm", options...); err == nil {
			t.Fatal("expected constructor error")
		}
	}
	if _, err := systemone.NewModel("", systemone.WithBaseURL("https://example.com")); err == nil {
		t.Fatal("expected empty model name error")
	}
	for _, threshold := range []float64{-1, 2, math.NaN(), math.Inf(1)} {
		if _, err := (systemone.Settings{BooleanThreshold: &threshold}).Build(); err == nil {
			t.Fatal("expected threshold error")
		}
	}
	if _, err := (systemone.Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{"system_one_settings": true}}}).Build(); err == nil {
		t.Fatal("expected reserved setting error")
	}
	model := newModel(t, systemone.WithBaseURL("https://example.com"), systemone.WithAPIKey(""))
	for _, settings := range []ai.ModelSettings{
		{ExtraBody: map[string]any{"system_one_settings": "bad"}}, {RequestTimeout: -time.Second}, {Temperature: floatPointer(math.NaN())},
	} {
		params := boolParams()
		params.Settings = settings
		if _, err := model.Request(t.Context(), prompt(), params); err == nil {
			t.Fatal("expected settings error")
		}
	}
	if _, err := model.Request(t.Context(), prompt(), ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected text output error")
	}
	if model.ContextWindow() != 0 {
		t.Fatal("default context window must be unknown")
	}
}

func TestDefaultSettingsOverrideAndSpan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer from-header" {
			t.Errorf("headers=%v", request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["temperature"] != .7 {
			t.Errorf("body=%v", body)
		}
		_, _ = writer.Write([]byte(`{"model":"clm-1","answers":{"urgent":{"type":"noul","noul":0.7}},"usage":{}}`))
	}))
	defer server.Close()
	defaults, err := (systemone.Settings{Common: ai.ModelSettings{Temperature: floatPointer(.2)}, BooleanThreshold: floatPointer(.9)}).Build()
	if err != nil {
		t.Fatal(err)
	}
	settings, err := (systemone.Settings{Common: ai.ModelSettings{Temperature: floatPointer(.7)}, BooleanThreshold: floatPointer(.6)}).Build()
	if err != nil {
		t.Fatal(err)
	}
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer func() { _ = tracerProvider.Shutdown(context.Background()) }()
	model := ai.NewInstrumentedModel(newModel(t, systemone.WithProvider(systemone.ProviderConfig{Name: "system-one", BaseURL: server.URL, APIKey: "secret", HTTPClient: server.Client(), Headers: http.Header{"authorization": {"Bearer from-header"}}}), systemone.WithDefaultSettings(defaults)), ai.WithInstrumentationTracerProvider(tracerProvider))
	result, err := ai.NewAgent[struct{}, struct {
		Urgent bool `json:"urgent"`
	}](model, ai.WithInstructions("Urgent?")).Run(t.Context(), "Charged twice.", struct{}{}, ai.WithRunModelSettings(settings))
	if err != nil || !result.Output.Urgent {
		t.Fatalf("result=%v error=%v", result, err)
	}
	spans := exporter.GetSpans()
	found := false
	for _, span := range spans {
		if span.Name == "decide clm-latest" {
			found = true
		}
	}
	if !found {
		t.Fatal("missing decision span")
	}
	var handoff systemone.DecisionHandOff = &systemone.UnsureRoute{}
	if !handoff.IsModelAPIError() || errors.Is(handoff, context.Canceled) {
		t.Fatal("unexpected handoff")
	}
}
