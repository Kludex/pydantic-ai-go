package contextwindow_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type unidentifiedModel struct{}

func (unidentifiedModel) Name() string { return "unidentified" }

func (unidentifiedModel) ProviderName() string { return "" }

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
		{model: unidentifiedModel{}, want: 0},
	}
	for _, test := range models {
		if got := ai.WrapModel(test.model).ContextWindow(); got != test.want {
			t.Fatalf("unexpected %s context window: got %d want %d", test.model.Name(), got, test.want)
		}
	}
}
