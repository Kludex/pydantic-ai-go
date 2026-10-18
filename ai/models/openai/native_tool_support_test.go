package openai_test

import (
	"errors"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestModelsRejectUnpreparedSpeech(t *testing.T) {
	models := []ai.Model{openai.NewModel("gpt-5"), openai.NewResponsesModel("gpt-5")}
	histories := [][]ai.ModelMessage{
		{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}}}},
		{ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant}}}},
	}
	for _, model := range models {
		for _, history := range histories {
			if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); !errors.Is(
				err, ai.ErrUnpreparedSpeech,
			) {
				t.Fatalf("%T accepted unprepared speech: %v", model, err)
			}
		}
	}
}

func TestNativeToolSupport(t *testing.T) {
	search := openai.NewModel("gpt-4o-search-preview")
	if !search.SupportsNativeTool(ai.WebSearchTool{}) || search.SupportsNativeTool(ai.CodeExecutionTool{}) ||
		search.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected Chat native-tool support")
	}

	compatible := openai.NewModel("gateway", openai.WithChatCompatibility(openai.ChatCompatibility{
		NativeToolFunc: func(ai.NativeTool) (openai.ChatNativeTool, bool, error) {
			return openai.ChatNativeTool{Type: "custom"}, true, nil
		},
	}))
	if !compatible.SupportsNativeTool(ai.CodeExecutionTool{}) {
		t.Fatal("compatibility native tool was not supported")
	}

	responses := openai.NewResponsesModel("gpt-5")
	if profile := responses.ModelProfile(); !profile.SupportsImageOutput ||
		profile.DefaultOutputMode != ai.OutputModeTool {
		t.Fatalf("unexpected Responses profile: %#v", profile)
	}
	if !responses.SupportsNativeTool(ai.WebSearchTool{}) || !responses.SupportsNativeTool(ai.MCPServerTool{
		ID: "server", URL: "https://example.com/mcp",
	}) || responses.SupportsNativeTool(ai.MemoryTool{}) || responses.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected Responses native-tool support")
	}

	cached := false
	for _, restricted := range []ai.WebSearchTool{
		{AllowedDomains: []string{"go.dev"}}, {BlockedDomains: []string{}}, {MaxUses: 1}, {ExternalWebAccess: &cached},
	} {
		if search.SupportsNativeTool(restricted) {
			t.Fatalf("Chat supported restricted web search %#v", restricted)
		}
	}
	if !responses.SupportsNativeTool(ai.WebSearchTool{
		AllowedDomains: []string{"go.dev"}, BlockedDomains: []string{"spam.example"}, ExternalWebAccess: &cached,
	}) || responses.SupportsNativeTool(ai.WebSearchTool{MaxUses: 1}) {
		t.Fatal("unexpected Responses restricted web search support")
	}
	xai := openai.NewResponsesModel("grok-4.3", openai.WithProvider(openai.ProviderConfig{Name: "xai", BaseURL: "https://example.invalid/v1"}))
	if xai.SupportsNativeTool(ai.WebSearchTool{ExternalWebAccess: &cached}) {
		t.Fatal("xAI Responses supported cached-only web search")
	}
}
