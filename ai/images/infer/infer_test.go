package infer_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/images/infer"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

func TestModel(t *testing.T) {
	for name, provider := range map[string]string{
		"openai:gpt-image-2": "openai", "google:gemini-3.1-flash-image": "google",
		"xai:grok-imagine-image": "xai",
	} {
		model, err := infer.Model(name)
		if err != nil || model.ProviderName() != provider {
			t.Fatalf("unexpected model for %s: %#v %v", name, model, err)
		}
	}
	custom := fakes.NewModel()
	model, err := infer.Model("custom:image", infer.WithProvider("custom", func(name string) (images.Model, error) {
		if name != "image" {
			t.Fatalf("unexpected name: %s", name)
		}
		return custom, nil
	}))
	if err != nil || model != custom {
		t.Fatalf("unexpected custom model: %#v %v", model, err)
	}
	model, err = infer.Model("google-cloud:image", infer.WithProvider("google-cloud", func(string) (images.Model, error) {
		return custom, nil
	}))
	if err != nil || model != custom {
		t.Fatalf("custom Google Cloud resolver was ignored: %#v %v", model, err)
	}
	model, err = infer.Model("google-cloud:gemini-image", infer.WithVertexConfig(modelgoogle.VertexConfig{
		APIKey: "key", Endpoint: "https://example.com",
	}))
	if err != nil || model.ProviderName() != "google-cloud" {
		t.Fatalf("unexpected Vertex model: %#v %v", model, err)
	}
	resolveErr := errors.New("resolve")
	if _, err := infer.Model("custom:image", infer.WithProvider("custom", func(string) (images.Model, error) {
		return nil, resolveErr
	})); !errors.Is(err, resolveErr) {
		t.Fatalf("unexpected resolver error: %v", err)
	}
	for _, name := range []string{"missing", ":model", "provider:", "other:model", "openai:dall-e-2"} {
		if _, err := infer.Model(name); err == nil {
			t.Fatalf("invalid name %q accepted", name)
		}
	}
	for _, resolver := range []infer.Resolver{
		func(string) (images.Model, error) { return nil, nil },
		func(string) (images.Model, error) { var model *fakeNil; return model, nil },
	} {
		if _, err := infer.Model("nil:model", infer.WithProvider("nil", resolver)); err == nil ||
			!strings.Contains(err.Error(), "nil model") {
			t.Fatalf("unexpected nil error: %v", err)
		}
	}
	for _, function := range []func(){
		func() { infer.WithProvider("", func(string) (images.Model, error) { return custom, nil }) },
		func() { infer.WithProvider("custom", nil) },
	} {
		if capturePanic(function) == nil {
			t.Fatal("invalid provider did not panic")
		}
	}
}

type fakeNil struct{}

func (*fakeNil) Generate(context.Context, string, []images.Input, images.Settings) (*images.Result, error) {
	return nil, nil
}
func (*fakeNil) Name() string                     { return "" }
func (*fakeNil) ProviderName() string             { return "" }
func (*fakeNil) ProviderURL() string              { return "" }
func (*fakeNil) DefaultSettings() images.Settings { return images.Settings{} }

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
