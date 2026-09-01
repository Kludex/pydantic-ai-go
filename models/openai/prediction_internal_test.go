package openai

import (
	"encoding/json"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestPredictionExtractionValidation(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  string
	}{
		{name: "type", value: "content", want: "must use Prediction"},
		{name: "JSON", value: json.RawMessage(`{`), want: "decode prediction"},
		{name: "payload", value: json.RawMessage(`{"type":"other","content":"text"}`), want: "invalid prediction"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := extractPredictionSettings(ai.ModelSettings{ExtraBody: map[string]any{
				predictionSetting: test.value,
			}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	malformed := ai.ModelSettings{ExtraBody: map[string]any{predictionSetting: "content"}}
	for _, model := range []ai.Model{NewModel("model"), NewResponsesModel("model")} {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: malformed})
		if err == nil || !strings.Contains(err.Error(), "must use Prediction") {
			t.Fatalf("unexpected malformed request error from %T: %v", model, err)
		}
	}

	settings := ai.ModelSettings{ExtraBody: map[string]any{
		predictionSetting: json.RawMessage(`{"type":"content","content":"text"}`),
		"custom":          true,
	}}
	cleaned, prediction, err := extractPredictionSettings(settings)
	if err != nil {
		t.Fatal(err)
	}
	if prediction == nil || prediction.Type != "content" || prediction.Content != "text" ||
		cleaned.ExtraBody["custom"] != true {
		t.Fatalf("unexpected extraction: %#v %#v", cleaned, prediction)
	}
}
