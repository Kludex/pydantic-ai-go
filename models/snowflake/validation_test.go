package snowflake_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/snowflake"
)

func TestConfigurationFailures(t *testing.T) {
	for _, account := range []string{"", "attacker.example/path"} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("expected account panic for %q", account)
				}
			}()
			snowflake.NewModel("model", snowflake.WithAccount(account))
		}()
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected base URL panic")
			}
		}()
		snowflake.NewModel("model", snowflake.WithBaseURL(""))
	}()
	t.Setenv("SNOWFLAKE_ACCOUNT", "")
	t.Setenv("SNOWFLAKE_TOKEN", "")
	t.Setenv("SNOWFLAKE_BASE_URL", "")
	model := snowflake.NewModel("model")
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected account error")
	}
	if _, err := snowflake.NewProviderConfig(); err == nil {
		t.Fatal("expected provider error")
	}
	t.Setenv("SNOWFLAKE_ACCOUNT", "account")
	if _, err := snowflake.NewProviderConfig(); err == nil {
		t.Fatal("expected token error")
	}
}

func TestSettingsFailures(t *testing.T) {
	for _, reasoning := range []snowflake.Reasoning{
		{},
		{Effort: snowflake.ReasoningEffort("bad")},
		{MaxTokens: -1},
		{Effort: snowflake.ReasoningEffortLow, MaxTokens: 1},
	} {
		if _, err := (snowflake.Settings{Reasoning: &reasoning}).Build(); err == nil {
			t.Fatalf("expected reasoning error: %#v", reasoning)
		}
	}
	reasoning := snowflake.Reasoning{Effort: snowflake.ReasoningEffortLow}
	for _, common := range []ai.ModelSettings{
		{ExtraBody: map[string]any{"snowflake_reasoning": reasoning}},
		{ExtraBody: map[string]any{"reasoning": map[string]any{"effort": "low"}}},
	} {
		if _, err := (snowflake.Settings{Common: common, Reasoning: &reasoning}).Build(); err == nil {
			t.Fatal("expected setting conflict")
		}
	}
	model, requests := snowflakeValidationModel(t, "claude-sonnet-4-6")
	for _, value := range []any{(*snowflake.Reasoning)(nil), "low", snowflake.Reasoning{}} {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"snowflake_reasoning": value},
		}})
		if err == nil {
			t.Fatalf("expected extracted reasoning error for %#v", value)
		}
	}
	settings, err := (snowflake.Settings{Reasoning: &reasoning}).Build()
	if err != nil {
		t.Fatal(err)
	}
	other, _ := snowflakeValidationModel(t, "openai-gpt-5")
	if _, err := other.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err == nil {
		t.Fatal("expected Claude-only reasoning error")
	}
	conflicting := ai.ModelSettings{ExtraBody: map[string]any{
		"snowflake_reasoning": reasoning, "reasoning": map[string]any{"effort": "high"},
	}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: conflicting}); err == nil {
		t.Fatal("expected portable reasoning conflict")
	}
	validPointer := &snowflake.Reasoning{Effort: snowflake.ReasoningEffortMedium}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraBody: map[string]any{"snowflake_reasoning": validPointer},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if *requests != 1 {
		t.Fatalf("unexpected requests: %d", *requests)
	}
}

func TestUnsupportedFamilyFeatures(t *testing.T) {
	model, requests := snowflakeValidationModel(t, "mistral-large")
	for _, params := range []ai.ModelRequestParams{
		{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}},
		{Tools: []ai.ToolDefinition{{Name: "tool"}}},
		{OutputTool: &ai.ToolDefinition{Name: "result"}},
		{OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"}},
	} {
		if _, err := model.Request(t.Context(), nil, params); err == nil {
			t.Fatalf("expected unsupported feature error for %#v", params)
		}
	}
	if *requests != 0 {
		t.Fatalf("unexpected requests: %d", *requests)
	}
}

func TestReasoningLevels(t *testing.T) {
	model, requests := snowflakeValidationModel(t, "claude-sonnet-4-6")
	levels := []struct {
		level  ai.ThinkingLevel
		effort string
	}{
		{level: ai.ThinkingLevelDisabled},
		{level: ai.ThinkingLevelEnabled, effort: "medium"},
		{level: ai.ThinkingLevelMinimal, effort: "low"},
		{level: ai.ThinkingLevelLow, effort: "low"},
		{level: ai.ThinkingLevelMedium, effort: "medium"},
		{level: ai.ThinkingLevelHigh, effort: "high"},
		{level: ai.ThinkingLevelXHigh, effort: "high"},
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	for _, test := range levels {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: test.level},
		}})
		if err != nil {
			t.Fatal(err)
		}
	}
	budget := 300
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{TokenBudget: &budget},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, thinking := range []*ai.ThinkingSettings{
		{IncludeThoughts: boolPointer(true)},
		{TokenBudget: intPointer(-1)},
	} {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: thinking,
		}}); err == nil {
			t.Fatalf("expected thinking error for %#v", thinking)
		}
	}
	other, _ := snowflakeValidationModel(t, "openai-gpt-5")
	if _, err := other.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{TokenBudget: &budget},
	}}); err == nil {
		t.Fatal("expected non-Claude token budget error")
	}
	if *requests != len(levels)+2 {
		t.Fatalf("unexpected requests: %d", *requests)
	}
}

func boolPointer(value bool) *bool { return &value }
func intPointer(value int) *int    { return &value }

func snowflakeValidationModel(t *testing.T, name string) (*snowflake.Model, *int) {
	t.Helper()
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		var body map[string]any
		_ = json.NewDecoder(request.Body).Decode(&body)
		_, _ = io.WriteString(response, `{"model":"model","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(server.Close)
	return snowflake.NewModel(
		name, snowflake.WithBaseURL(server.URL), snowflake.WithToken("token"), snowflake.WithHTTPClient(server.Client()),
	), &requests
}
