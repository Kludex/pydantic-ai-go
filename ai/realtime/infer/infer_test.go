package infer_test

import (
	"context"
	"errors"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/infer"
)

type customModel struct{}

func (customModel) Name() string              { return "custom" }
func (customModel) ProviderName() string      { return "custom" }
func (customModel) Profile() realtime.Profile { return realtime.DefaultProfile() }
func (customModel) Connect(context.Context, realtime.ConnectParams) (realtime.Connection, error) {
	return nil, errors.New("unused")
}

func TestProviderOptionValidation(t *testing.T) {
	for _, function := range []func(){
		func() { infer.WithProvider("", func(string) (realtime.Model, error) { return customModel{}, nil }) },
		func() { infer.WithProvider("custom", nil) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			function()
		}()
	}
}

func TestModel(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
	t.Setenv("AZURE_OPENAI_API_KEY", "key")
	for _, name := range []string{
		"azure:gpt-realtime", "openai:gpt-realtime", "xai:grok-voice-latest",
		"google:gemini-live", "google-cloud:gemini-live",
	} {
		model, err := infer.Model(name)
		if err != nil || model.Name() == "" {
			t.Fatalf("infer %q: model=%v err=%v", name, model, err)
		}
	}
	for _, name := range []string{"missing", "openai:", "future:model"} {
		if _, err := infer.Model(name); err == nil {
			t.Fatalf("expected inference error for %q", name)
		}
	}

	model, err := infer.Model("tenant:voice", infer.WithProvider("tenant", func(name string) (realtime.Model, error) {
		if name != "voice" {
			t.Fatalf("unexpected model name %q", name)
		}
		return customModel{}, nil
	}))
	if err != nil || model.Name() != "custom" {
		t.Fatalf("custom resolver: model=%v err=%v", model, err)
	}
	resolverError := errors.New("resolver failed")
	if _, err := infer.Model("tenant:voice", infer.WithProvider("tenant", func(string) (realtime.Model, error) {
		return nil, resolverError
	})); !errors.Is(err, resolverError) {
		t.Fatalf("unexpected resolver error: %v", err)
	}
	for _, resolver := range []infer.Resolver{
		func(string) (realtime.Model, error) { return nil, nil },
		func(string) (realtime.Model, error) {
			var model *customModel
			return model, nil
		},
	} {
		if _, err := infer.Model("tenant:voice", infer.WithProvider("tenant", resolver)); err == nil {
			t.Fatal("expected nil model error")
		}
	}
}
