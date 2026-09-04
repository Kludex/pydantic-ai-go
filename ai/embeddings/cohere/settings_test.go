package cohere

import (
	"context"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

func TestSettingsBuild(t *testing.T) {
	maxTokens := 12
	common := embeddings.Settings{ExtraBody: map[string]any{"nested": []any{"original"}}}
	built, err := (Settings{
		Common: common, InputType: InputTypeImage, MaxTokens: &maxTokens, Truncate: TruncationNone,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	maxTokens = 99
	common.ExtraBody["nested"].([]any)[0] = "changed"
	if built.ExtraBody[inputTypeKey] != InputTypeImage || built.ExtraBody[maxTokensKey] != 12 ||
		built.ExtraBody[truncateKey] != TruncationNone || built.ExtraBody["nested"].([]any)[0] != "original" {
		t.Fatalf("settings were not detached: %#v", built)
	}
	if empty, err := (Settings{}).Build(); err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v %v", empty, err)
	}
	for _, inputType := range []InputType{
		InputTypeSearchQuery, InputTypeSearchDocument, InputTypeClassification, InputTypeClustering, InputTypeImage,
	} {
		if _, err := (Settings{InputType: inputType}).Build(); err != nil {
			t.Fatalf("valid input type %q failed: %v", inputType, err)
		}
	}
	for _, truncation := range []Truncation{TruncationNone, TruncationEnd, TruncationStart} {
		if _, err := (Settings{Truncate: truncation}).Build(); err != nil {
			t.Fatalf("valid truncation %q failed: %v", truncation, err)
		}
	}
}

func TestSettingsBuildErrors(t *testing.T) {
	zero := 0
	negative := -1
	tests := []struct {
		name     string
		settings Settings
		match    string
	}{
		{name: "dimensions", settings: Settings{Common: embeddings.Settings{Dimensions: &zero}}, match: "dimensions"},
		{name: "input type", settings: Settings{InputType: InputType("bad")}, match: "invalid input type"},
		{name: "max tokens", settings: Settings{MaxTokens: &negative}, match: "max tokens"},
		{name: "truncate", settings: Settings{Truncate: Truncation("bad")}, match: "invalid truncation"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := test.settings.Build(); err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
	for _, key := range []string{inputTypeKey, maxTokensKey, truncateKey} {
		settings := Settings{Common: embeddings.Settings{ExtraBody: map[string]any{key: "existing"}}}
		switch key {
		case inputTypeKey:
			settings.InputType = InputTypeSearchQuery
		case maxTokensKey:
			settings.MaxTokens = intPointer(1)
		case truncateKey:
			settings.Truncate = TruncationEnd
		}
		if _, err := settings.Build(); err == nil || !strings.Contains(err.Error(), "conflicts") {
			t.Fatalf("unexpected conflict for %s: %v", key, err)
		}
	}
}

func TestGenericProviderSettings(t *testing.T) {
	tests := []struct {
		name  string
		key   string
		value any
		match string
	}{
		{name: "input type value", key: inputTypeKey, value: 1, match: "must be an input type"},
		{name: "input type string", key: inputTypeKey, value: "bad", match: "invalid input type"},
		{name: "max tokens type", key: maxTokensKey, value: "1", match: "positive integer"},
		{name: "max tokens value", key: maxTokensKey, value: 0, match: "positive integer"},
		{name: "truncation value", key: truncateKey, value: 1, match: "must be a truncation"},
		{name: "truncation string", key: truncateKey, value: "bad", match: "invalid truncation"},
	}
	model := NewModel("embed-v4.0")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{
				ExtraBody: map[string]any{test.key: test.value},
			})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	valid := inspectingModel(t, func(body map[string]any) {
		if body["input_type"] != "clustering" || body["truncate"] != "START" || body["max_tokens"] != float64(8) {
			t.Errorf("unexpected body: %#v", body)
		}
	})
	if _, err := valid.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{inputTypeKey: "clustering", truncateKey: "START", maxTokensKey: 8},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestProviderTruncationOverridesPortableSetting(t *testing.T) {
	settings, err := (Settings{
		Common:   embeddings.Settings{Truncate: boolPointer(true)},
		Truncate: TruncationNone,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := inspectingModel(t, func(body map[string]any) {
		if body["truncate"] != "NONE" {
			t.Errorf("unexpected truncation: %#v", body)
		}
	})
	if _, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, settings); err != nil {
		t.Fatal(err)
	}
}

func intPointer(value int) *int { return &value }
