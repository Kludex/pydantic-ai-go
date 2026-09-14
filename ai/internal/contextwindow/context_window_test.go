package contextwindow_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type unidentifiedModel struct {
	name       string
	providerID string
}

func (model unidentifiedModel) Name() string { return model.name }

func (model unidentifiedModel) ProviderName() string { return model.providerID }

func (unidentifiedModel) ProviderURL() string { return "" }

func (unidentifiedModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{}, nil
}

func TestContextWindowLookupThroughModels(t *testing.T) {
	models := []struct {
		model ai.Model
		want  int
	}{
		{model: openai.NewModel("gpt-5"), want: 400_000},
		{model: openai.NewModel("gpt-5", openai.WithBaseURL("https://example.com/v1")), want: 400_000},
		{model: openai.NewModel("unknown-model"), want: 0},
		{model: unidentifiedModel{name: "unidentified"}, want: 0},
	}
	for _, test := range models {
		if got := ai.WrapModel(test.model).ContextWindow(); got != test.want {
			t.Fatalf("unexpected %s context window: got %d want %d", test.model.Name(), got, test.want)
		}
	}
}

func TestContextWindowLookupForEveryMetadataProvider(t *testing.T) {
	models := []struct {
		providerID string
		model      string
		want       int
	}{
		{providerID: "anthropic", model: "claude-2", want: 200_000},
		{providerID: "avian", model: "deepseek/deepseek-v3.2", want: 163_000},
		{providerID: "aws", model: "amazon.nova-2-sonic-v1:0", want: 1_000_000},
		{providerID: "azure", model: "mai-ds-r1:free", want: 163_840},
		{providerID: "baseten", model: "deepseek-ai/DeepSeek-V4-Flash-0731", want: 1_048_000},
		{providerID: "cerebras", model: "gemma-4-31b", want: 131_072},
		{providerID: "cloudflare", model: "@cf/google/gemma-4-26b-a4b-it", want: 256_000},
		{providerID: "cohere", model: "c4ai-aya-expanse-32b", want: 128_000},
		{providerID: "cursor", model: "composer-2.5", want: 200_000},
		{providerID: "deepseek", model: "deepseek-chat", want: 64_000},
		{providerID: "doubleword", model: "Qwen/Qwen3.8-27B-FP8", want: 262_144},
		{providerID: "fireworks", model: "accounts/fireworks/models/deepseek-r1-0528", want: 160_000},
		{providerID: "google", model: "claude-3-5-haiku", want: 200_000},
		{providerID: "groq", model: "deepseek-r1-distill-llama-70b", want: 131_072},
		{providerID: "huggingface_fireworks-ai", model: "meta-llama/Llama-3.3-70B-Instruct", want: 131_072},
		{providerID: "huggingface_groq", model: "Qwen/Qwen3-32B", want: 131_072},
		{providerID: "huggingface_hyperbolic", model: "Qwen/Qwen2.5-VL-72B-Instruct", want: 32_768},
		{providerID: "huggingface_nebius", model: "NousResearch/Hermes-4-405B", want: 131_072},
		{providerID: "huggingface_novita", model: "MiniMaxAI/MiniMax-M1-80k", want: 1_000_000},
		{providerID: "huggingface_nscale", model: "Qwen/QwQ-32B", want: 131_072},
		{providerID: "huggingface_ovhcloud", model: "Qwen/Qwen2.5-VL-72B-Instruct", want: 32_768},
		{providerID: "huggingface_sambanova", model: "Qwen/Qwen3-32B", want: 32_768},
		{providerID: "huggingface_together", model: "EssentialAI/rnj-1-instruct", want: 32_768},
		{providerID: "minimax", model: "M2-her", want: 64_000},
		{providerID: "mistral", model: "codestral-latest", want: 256_000},
		{providerID: "modal", model: "moonshotai/Kimi-K3", want: 1_048_576},
		{providerID: "moonshotai", model: "kimi-k2-0711-preview", want: 131_072},
		{providerID: "novita", model: "Sao10K/L3-8B-Stheno-v3.2", want: 8_192},
		{providerID: "openai", model: "chatgpt-4o-latest", want: 128_000},
		{providerID: "openrouter", model: "aion-labs/aion-2.0", want: 131_072},
		{providerID: "ovhcloud", model: "DeepSeek-R1-Distill-Llama-70B", want: 131_072},
		{providerID: "perplexity", model: "sonar", want: 128_000},
		{providerID: "quicksilverpro", model: "claude-fable-5", want: 1_000_000},
		{providerID: "x-ai", model: "grok-2-1212", want: 32_768},
		{providerID: "zai", model: "GLM-5.2", want: 1_000_000},
		{providerID: "zhipuai", model: "GLM-4-Air", want: 128_000},
	}
	for _, test := range models {
		model := unidentifiedModel{name: test.model, providerID: test.providerID}
		if got := ai.WrapModel(model).ContextWindow(); got != test.want {
			t.Fatalf("unexpected %s/%s context window: got %d want %d", test.providerID, test.model, got, test.want)
		}
	}
}
