package openai_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

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
	if !responses.SupportsNativeTool(ai.WebSearchTool{}) || !responses.SupportsNativeTool(ai.MCPServerTool{
		ID: "server", URL: "https://example.com/mcp",
	}) || responses.SupportsNativeTool(ai.MemoryTool{}) || responses.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected Responses native-tool support")
	}
}
