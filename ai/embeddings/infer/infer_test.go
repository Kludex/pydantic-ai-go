package infer_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/bedrock"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/cohere"
	embeddinggoogle "github.com/Kludex/pydantic-ai-go/ai/embeddings/google"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/infer"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/ollama"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/openai"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/voyageai"
	modelazure "github.com/Kludex/pydantic-ai-go/ai/models/azure"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

func TestBuiltInModels(t *testing.T) {
	tests := []struct {
		name         string
		typeCheck    func(embeddings.Model) bool
		providerName string
		modelName    string
	}{
		{
			name: "bedrock:amazon.titan-embed-text-v2:0", providerName: "bedrock", modelName: "amazon.titan-embed-text-v2:0",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*bedrock.Model); return ok },
		},
		{
			name: "openai:text-embedding-3-small", providerName: "openai", modelName: "text-embedding-3-small",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*openai.Model); return ok },
		},
		{
			name: "ollama:nomic-embed-text", providerName: "ollama", modelName: "nomic-embed-text",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*ollama.Model); return ok },
		},
		{
			name: "openrouter:qwen/qwen3-embedding-8b", providerName: "openrouter", modelName: "qwen/qwen3-embedding-8b",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*openai.Model); return ok },
		},
		{
			name: "zai:embedding-3", providerName: "zai", modelName: "embedding-3",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*openai.Model); return ok },
		},
		{
			name: "cohere:embed-v4.0", providerName: "cohere", modelName: "embed-v4.0",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*cohere.Model); return ok },
		},
		{
			name: "google:gemini-embedding-2", providerName: "google", modelName: "gemini-embedding-2",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*embeddinggoogle.Model); return ok },
		},
		{
			name: "voyageai:voyage-4", providerName: "voyageai", modelName: "voyage-4",
			typeCheck: func(model embeddings.Model) bool { _, ok := model.(*voyageai.Model); return ok },
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model, err := infer.Model(test.name)
			if err != nil || !test.typeCheck(model) || model.Name() != test.modelName ||
				model.ProviderName() != test.providerName {
				t.Fatalf("unexpected inferred model: %T %#v %v", model, model, err)
			}
		})
	}
}

func TestOpenAICompatibleModels(t *testing.T) {
	for _, name := range []string{
		"ALIBABA_BASE_URL", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY", "CEREBRAS_BASE_URL", "CEREBRAS_API_KEY",
		"CRUSOE_BASE_URL", "CRUSOE_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "FIREWORKS_BASE_URL",
		"FIREWORKS_API_KEY", "GITHUB_MODELS_BASE_URL", "GITHUB_API_KEY", "HEROKU_INFERENCE_URL",
		"HEROKU_INFERENCE_KEY", "LITELLM_BASE_URL", "LITELLM_API_KEY", "MOONSHOTAI_BASE_URL", "MOONSHOTAI_API_KEY",
		"NEBIUS_BASE_URL", "NEBIUS_API_KEY", "OVHCLOUD_BASE_URL", "OVHCLOUD_API_KEY", "SAMBANOVA_BASE_URL",
		"SAMBANOVA_API_KEY", "SNOWFLAKE_BASE_URL", "SNOWFLAKE_TOKEN", "TOGETHER_BASE_URL", "TOGETHER_API_KEY",
		"VERCEL_AI_GATEWAY_BASE_URL", "VERCEL_AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN",
	} {
		t.Setenv(name, "")
	}
	tests := map[string]string{
		"alibaba":    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
		"cerebras":   "https://api.cerebras.ai/v1",
		"crusoe":     "https://api.inference.crusoecloud.com/v1",
		"deepseek":   "https://api.deepseek.com",
		"fireworks":  "https://api.fireworks.ai/inference/v1",
		"github":     "https://models.github.ai/inference",
		"heroku":     "https://us.inference.heroku.com/v1",
		"moonshotai": "https://api.moonshot.ai/v1",
		"nebius":     "https://api.studio.nebius.com/v1",
		"ovhcloud":   "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
		"sambanova":  "https://api.sambanova.ai/v1",
		"together":   "https://api.together.xyz/v1",
		"vercel":     "https://ai-gateway.vercel.sh/v1",
	}
	for providerName, expectedURL := range tests {
		t.Run(providerName, func(t *testing.T) {
			model, err := infer.Model(providerName + ":embedding-model")
			if err != nil || model.Name() != "embedding-model" || model.ProviderName() != providerName ||
				model.ProviderURL() != expectedURL {
				t.Fatalf("unexpected compatible model: %T %#v %v", model, model, err)
			}
		})
	}
	for _, providerName := range []string{"litellm", "snowflake"} {
		_, err := infer.Model(providerName + ":embedding-model")
		if err == nil || !strings.Contains(err.Error(), "requires") {
			t.Fatalf("unexpected unconfigured %s error: %v", providerName, err)
		}
	}
	t.Setenv("LITELLM_BASE_URL", "https://litellm.example/v1")
	t.Setenv("LITELLM_API_KEY", "key")
	model, err := infer.Model("litellm:embedding-model")
	if err != nil || model.ProviderURL() != "https://litellm.example/v1" {
		t.Fatalf("unexpected configured LiteLLM model: %#v %v", model, err)
	}
	t.Setenv("ALIBABA_BASE_URL", "https://alibaba.example/v1")
	model, err = infer.Model("alibaba:embedding-model")
	if err != nil || model.ProviderURL() != "https://alibaba.example/v1" {
		t.Fatalf("unexpected configured Alibaba model: %#v %v", model, err)
	}
}

func TestAzureModel(t *testing.T) {
	t.Setenv("OPENAI_API_VERSION", "")
	model, err := infer.Model("azure:embedding-deployment", infer.WithAzureConfig(modelazure.Config{
		Endpoint: "https://resource.openai.azure.com/openai/v1", APIKey: "key",
	}))
	if err != nil {
		t.Fatal(err)
	}
	openAIModel, ok := model.(*openai.Model)
	if !ok || openAIModel.ProviderName() != "azure" ||
		openAIModel.ProviderURL() != "https://resource.openai.azure.com/openai/v1" {
		t.Fatalf("unexpected Azure model: %T %#v", model, model)
	}
}

func TestGoogleCloudModel(t *testing.T) {
	model, err := infer.Model(
		"google-cloud:gemini-embedding-001",
		infer.WithVertexConfig(modelgoogle.VertexConfig{
			APIKey: "key", Endpoint: "https://vertex.example",
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	googleModel, ok := model.(*embeddinggoogle.Model)
	if !ok || googleModel.ProviderName() != "google-cloud" ||
		googleModel.ProviderURL() != "https://vertex.example/v1beta1/publishers/google" ||
		googleModel.Transport() != modelgoogle.TransportVertexAI {
		t.Fatalf("unexpected Google Cloud model: %T %#v", model, model)
	}
}

func TestCustomProvider(t *testing.T) {
	model := &testModel{name: "custom-model"}
	resolved, err := infer.Model("custom:selected", infer.WithProvider("custom", func(name string) (embeddings.Model, error) {
		if name != "selected" {
			t.Fatalf("unexpected local name: %q", name)
		}
		return model, nil
	}))
	if err != nil || resolved != model {
		t.Fatalf("unexpected custom model: %#v %v", resolved, err)
	}

	resolverErr := errors.New("resolver failed")
	_, err = infer.Model("custom:selected", infer.WithProvider("custom", func(string) (embeddings.Model, error) {
		return nil, resolverErr
	}))
	if !errors.Is(err, resolverErr) {
		t.Fatalf("unexpected resolver error: %v", err)
	}
	for _, returned := range []embeddings.Model{nil, (*testModel)(nil)} {
		_, err := infer.Model("custom:selected", infer.WithProvider("custom", func(string) (embeddings.Model, error) {
			return returned, nil
		}))
		if err == nil || !strings.Contains(err.Error(), "returned a nil model") {
			t.Fatalf("unexpected nil model error: %v", err)
		}
	}
	value, err := infer.Model("custom:value", infer.WithProvider("custom", func(string) (embeddings.Model, error) {
		return testModel{name: "value"}, nil
	}))
	if err != nil || value.Name() != "value" {
		t.Fatalf("unexpected value model: %#v %v", value, err)
	}
}

func TestErrors(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	t.Setenv("AZURE_OPENAI_API_KEY", "")
	if _, err := infer.Model("azure:model"); err == nil || !strings.Contains(err.Error(), "endpoint") {
		t.Fatalf("unexpected Azure inference error: %v", err)
	}
	for _, test := range []struct {
		name  string
		match string
	}{
		{name: "model", match: "must include a provider prefix"},
		{name: ":model", match: "non-empty provider and model names"},
		{name: "openai:", match: "non-empty provider and model names"},
		{name: "unknown:model", match: `unknown provider "unknown"`},
	} {
		if _, err := infer.Model(test.name); err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("unexpected error for %q: %v", test.name, err)
		}
	}
	for _, function := range []func(){
		func() { infer.WithProvider("", func(string) (embeddings.Model, error) { return nil, nil }) },
		func() { infer.WithProvider("custom", nil) },
	} {
		if panicValue := capturePanic(function); panicValue == nil {
			t.Fatal("invalid provider registration did not panic")
		}
	}
}

func TestConcurrentInference(t *testing.T) {
	var group sync.WaitGroup
	failures := make(chan error, 20)
	for range 20 {
		group.Go(func() {
			model, err := infer.Model("openai:text-embedding-3-small")
			if err == nil && model.Name() != "text-embedding-3-small" {
				err = errors.New("wrong model")
			}
			failures <- err
		})
	}
	group.Wait()
	close(failures)
	for err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
}

type testModel struct{ name string }

func (testModel) Embed(
	context.Context, []string, embeddings.InputType, embeddings.Settings,
) (*embeddings.Result, error) {
	return &embeddings.Result{}, nil
}
func (model testModel) Name() string   { return model.name }
func (testModel) ProviderName() string { return "custom" }
func (testModel) ProviderURL() string  { return "https://example.com" }
func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
