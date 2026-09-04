package openrouter_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openrouter"
)

func TestNativeToolSupport(t *testing.T) {
	model := openrouter.NewModel("openai/gpt-5")
	advisor := ai.AdvisorTool{Model: "anthropic/claude-opus-4.8"}
	if !model.SupportsNativeTool(ai.WebSearchTool{}) || !model.SupportsNativeTool(advisor) ||
		model.SupportsNativeTool(ai.CodeExecutionTool{}) || model.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected OpenRouter native-tool support")
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{(*ai.WebSearchTool)(nil)},
	}); err == nil {
		t.Fatal("expected invalid native tool to fail before transport")
	}
}
