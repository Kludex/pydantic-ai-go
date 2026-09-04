package fakes_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/fakes"
)

func TestModel(t *testing.T) {
	dimensions := 3
	defaults := embeddings.Settings{
		Dimensions: &dimensions, ExtraBody: map[string]any{"items": []any{"original"}},
	}
	model := fakes.NewModel(
		fakes.WithName("fake-model"), fakes.WithProviderName("fake-provider"),
		fakes.WithDimensions(2), fakes.WithDefaultSettings(defaults),
	)
	*defaults.Dimensions = 99
	defaults.ExtraBody["items"].([]any)[0] = "mutated"
	result, err := model.Embed(
		context.Background(), []string{"hello, world", ""}, embeddings.InputTypeDocument, embeddings.Settings{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "fake-model" || model.ProviderName() != "fake-provider" || model.ProviderURL() != "" {
		t.Fatalf("unexpected identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 1, 1}, {1, 1, 1}}) ||
		result.Usage.Requests != 1 || result.Usage.InputTokens != 2 || result.ProviderResponseID != "test-1" {
		t.Fatalf("unexpected result: %#v", result)
	}
	last, ok := model.LastSettings()
	if !ok || *last.Dimensions != 3 || last.ExtraBody["items"].([]any)[0] != "original" {
		t.Fatalf("unexpected settings: %#v %v", last, ok)
	}
	*last.Dimensions = 8
	lastAgain, _ := model.LastSettings()
	if *lastAgain.Dimensions != 3 {
		t.Fatal("LastSettings exposed model state")
	}
	if maximum, known, err := model.MaxInputTokens(context.Background()); err != nil || !known || maximum != 1024 {
		t.Fatalf("unexpected maximum: %d %v %v", maximum, known, err)
	}
	if count, err := model.CountTokens(context.Background(), "one: two. three"); err != nil || count != 3 {
		t.Fatalf("unexpected count: %d %v", count, err)
	}
}

func TestModelDimensionOptionAndInitialSettings(t *testing.T) {
	model := fakes.NewModel(fakes.WithDimensions(2))
	if settings, ok := model.LastSettings(); ok || !reflect.DeepEqual(settings, embeddings.Settings{}) {
		t.Fatalf("unexpected initial settings: %#v %v", settings, ok)
	}
	zero := 0
	if _, err := model.Embed(
		context.Background(), []string{"one"}, embeddings.InputTypeQuery,
		embeddings.Settings{Dimensions: &zero},
	); err == nil {
		t.Fatal("invalid dimensions unexpectedly succeeded")
	}
	result, err := model.Embed(context.Background(), []string{"one"}, embeddings.InputTypeQuery, embeddings.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 1}}) {
		t.Fatalf("unexpected dimensions: %#v", result.Embeddings)
	}
}
