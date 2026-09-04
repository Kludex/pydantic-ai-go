package infer_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/infer"
)

type valueModel struct{}

func (valueModel) Name() string { return "value" }
func (valueModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{}, nil
}

func TestModels(t *testing.T) {
	tests := []struct {
		name     string
		provider string
	}{
		{"openai:gpt-5-mini", "openai"},
		{"openai-responses:gpt-5-mini", "openai"},
		{"anthropic:claude-sonnet-4-5", "anthropic"},
		{"google:gemini-2.5-flash", "google"},
		{"bedrock:us.amazon.nova-lite-v1:0", "bedrock"},
		{"cerebras:gpt-oss-120b", "cerebras"},
		{"cohere:command-r7b-12-2024", "cohere"},
		{"crusoe:openai/gpt-oss-120b", "crusoe"},
		{"groq:openai/gpt-oss-20b", "groq"},
		{"huggingface:Qwen/Qwen3-32B", "huggingface"},
		{"ollama:qwen3", "ollama"},
		{"openrouter:anthropic/claude-sonnet-4.6", "openrouter"},
		{"zai:glm-5.3-flash", "zai"},
	}
	for _, test := range tests {
		model, err := infer.Model(test.name)
		if err != nil {
			t.Fatalf("name=%q: %v", test.name, err)
		}
		identity, ok := model.(ai.ModelProviderIdentity)
		if !ok || identity.ProviderName() != test.provider {
			t.Fatalf("name=%q: unexpected identity %T %#v", test.name, model, identity)
		}
	}
}

func TestCustomProvider(t *testing.T) {
	model, err := infer.Model("custom:model", infer.WithProvider("custom", func(name string) (ai.Model, error) {
		if name != "model" {
			t.Fatalf("unexpected name: %q", name)
		}
		return valueModel{}, nil
	}))
	if err != nil || model.Name() != "value" {
		t.Fatalf("unexpected model: %T %v", model, err)
	}
	failure := errors.New("resolver failed")
	_, err = infer.Model("custom:model", infer.WithProvider("custom", func(string) (ai.Model, error) {
		return nil, failure
	}))
	if !errors.Is(err, failure) {
		t.Fatalf("unexpected resolver error: %v", err)
	}
	for _, resolver := range []infer.Resolver{
		func(string) (ai.Model, error) { return nil, nil },
		func(string) (ai.Model, error) {
			var model *valueModel
			return model, nil
		},
	} {
		_, err = infer.Model("custom:model", infer.WithProvider("custom", resolver))
		if err == nil || !strings.Contains(err.Error(), "returned a nil model") {
			t.Fatalf("unexpected nil error: %v", err)
		}
	}
}

func TestErrors(t *testing.T) {
	assertPanic(t, func() { infer.WithProvider("", func(string) (ai.Model, error) { return valueModel{}, nil }) })
	assertPanic(t, func() { infer.WithProvider("custom", nil) })
	for _, test := range []struct {
		name  string
		match string
	}{
		{name: "model", match: "provider prefix"},
		{name: ":model", match: "non-empty"},
		{name: "openai:", match: "non-empty"},
		{name: "unknown:model", match: "unknown provider"},
	} {
		_, err := infer.Model(test.name)
		if err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("name=%q: unexpected error %v", test.name, err)
		}
	}
}

func assertPanic(t *testing.T, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	function()
}
