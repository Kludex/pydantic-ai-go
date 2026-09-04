package voyageai

import (
	"context"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

func TestSettings(t *testing.T) {
	common := embeddings.Settings{ExtraBody: map[string]any{"nested": []any{"original"}}}
	built, err := (Settings{Common: common, InputType: InputTypeNone}).Build()
	if err != nil {
		t.Fatal(err)
	}
	common.ExtraBody["nested"].([]any)[0] = "changed"
	if built.ExtraBody[inputTypeKey] != InputTypeNone || built.ExtraBody["nested"].([]any)[0] != "original" {
		t.Fatalf("settings were not detached: %#v", built)
	}
	for _, inputType := range []InputType{InputTypeQuery, InputTypeDocument, InputTypeNone} {
		if _, err := (Settings{InputType: inputType}).Build(); err != nil {
			t.Fatalf("valid input type %q failed: %v", inputType, err)
		}
	}
	if empty, err := (Settings{}).Build(); err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v %v", empty, err)
	}
	zero := 0
	if _, err := (Settings{Common: embeddings.Settings{Dimensions: &zero}}).Build(); err == nil ||
		!strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("unexpected dimensions error: %v", err)
	}
	if _, err := (Settings{InputType: InputType("bad")}).Build(); err == nil ||
		!strings.Contains(err.Error(), "invalid input type") {
		t.Fatalf("unexpected input type error: %v", err)
	}
	if _, err := (Settings{
		Common: embeddings.Settings{ExtraBody: map[string]any{inputTypeKey: "query"}}, InputType: InputTypeQuery,
	}).Build(); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected conflict error: %v", err)
	}
}

func TestGenericInputTypeSettings(t *testing.T) {
	for _, test := range []struct {
		value any
		match string
	}{
		{value: 1, match: "must be an input type"},
		{value: "bad", match: "invalid input type"},
	} {
		model := NewModel("voyage-4")
		if _, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{
			ExtraBody: map[string]any{inputTypeKey: test.value},
		}); err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("unexpected error for %#v: %v", test.value, err)
		}
	}

	model := inspectingModel(t, func(body map[string]any) {
		if body["input_type"] != "document" {
			t.Errorf("unexpected body: %#v", body)
		}
	})
	if _, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{inputTypeKey: "document"},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestDefaultInputTypeMerge(t *testing.T) {
	defaults, err := (Settings{InputType: InputTypeDocument}).Build()
	if err != nil {
		t.Fatal(err)
	}
	override := embeddings.Settings{ExtraBody: map[string]any{"per_call": true}}
	model := inspectingModel(t, func(body map[string]any) {
		if body["input_type"] != "document" || body["per_call"] != true {
			t.Errorf("unexpected body: %#v", body)
		}
	})
	WithDefaultSettings(defaults)(model)
	if _, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, override); err != nil {
		t.Fatal(err)
	}

	override, err = (Settings{InputType: InputTypeQuery}).Build()
	if err != nil {
		t.Fatal(err)
	}
	overridden := inspectingModel(t, func(body map[string]any) {
		if body["input_type"] != "query" {
			t.Errorf("unexpected overridden body: %#v", body)
		}
	})
	WithDefaultSettings(defaults)(overridden)
	if _, err := overridden.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, override); err != nil {
		t.Fatal(err)
	}
}
