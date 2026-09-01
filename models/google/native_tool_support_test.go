package google_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func TestNativeToolSupport(t *testing.T) {
	model := google.NewModel("gemini-3-pro")
	if !model.SupportsNativeTool(ai.WebSearchTool{}) || !model.SupportsNativeTool(ai.FileSearchTool{
		FileStoreIDs: []string{"store"},
	}) || model.SupportsNativeTool(ai.ImageGenerationTool{}) || model.SupportsNativeTool(ai.MemoryTool{}) ||
		model.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected Google native-tool support")
	}
	image := google.NewModel("gemini-3-pro-image-preview")
	if !image.SupportsNativeTool(ai.ImageGenerationTool{}) {
		t.Fatal("image model did not report image-generation support")
	}
}
