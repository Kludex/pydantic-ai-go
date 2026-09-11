package fakes_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/fakes"
)

func TestModelRecordsLastCall(t *testing.T) {
	dimensions := images.Dimensions{Width: 512, Height: 512}
	model := fakes.NewModel(
		fakes.WithName("image-test"), fakes.WithProviderName("provider"),
		fakes.WithDefaultSettings(images.Settings{Dimensions: &dimensions}),
	)
	result, err := model.Generate(t.Context(), "two words", []images.Input{
		ai.BinaryContent{Data: []byte("reference"), MediaType: "image/png"},
	}, images.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "image-test" || model.ProviderName() != "provider" || model.ProviderURL() != "" ||
		model.DefaultSettings().Dimensions.Width != 512 || result.ModelName != "image-test" ||
		result.ProviderName != "provider" || result.Usage.InputTokens != 2 || result.Image().MediaType != "image/png" {
		t.Fatalf("unexpected result: %#v", result)
	}
	inputs, settings, ok := model.LastCall()
	if !ok || len(inputs) != 1 || settings.Dimensions.Width != 512 {
		t.Fatalf("unexpected last call: %#v %#v %v", inputs, settings, ok)
	}
	if _, _, ok := fakes.NewModel().LastCall(); ok {
		t.Fatal("fresh model reported a call")
	}
	if _, err := model.Generate(t.Context(), " ", nil, images.Settings{}); err == nil {
		t.Fatal("empty prompt accepted")
	}
	if result, err := model.Generate(t.Context(), "", nil, images.Settings{}); err == nil || result != nil {
		t.Fatalf("unexpected empty result: %#v %v", result, err)
	}
}
