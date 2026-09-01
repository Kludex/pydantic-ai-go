package anthropic_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func TestNativeToolSupport(t *testing.T) {
	model := anthropic.NewModel("claude-sonnet-4-5")
	if !model.SupportsNativeTool(ai.WebFetchTool{}) || !model.SupportsNativeTool(ai.CodeExecutionTool{}) ||
		model.SupportsNativeTool(ai.ImageGenerationTool{}) ||
		model.SupportsNativeTool(ai.AdvisorTool{Model: "claude-opus-4-8"}) ||
		model.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected Anthropic native-tool support")
	}
	advisor := anthropic.NewModel("claude-opus-4-8")
	if !advisor.SupportsNativeTool(ai.AdvisorTool{Model: "claude-opus-4-8"}) {
		t.Fatal("advisor model did not report advisor support")
	}
}
