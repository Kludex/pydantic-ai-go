package openai_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestOpenAIChatPrediction(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if body["stream"] == true {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response,
				"data: {\"model\":\"gpt-4.1\",\"choices\":["+
					"{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n"+
					"data: [DONE]\n\n",
			)
			return
		}
		_, _ = io.WriteString(response, `{
			"model":"gpt-4.1","choices":[{"message":{"content":"done"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	contentPrediction := &openai.Prediction{Content: "package main\n"}
	contentSettings, err := (openai.Settings{Prediction: contentPrediction}).Build()
	if err != nil {
		t.Fatal(err)
	}
	contentPrediction.Content = "changed"

	parts := []openai.PredictionContentPart{
		{Text: "package main\n"},
		{Text: "func main() {}\n", PromptCacheBreakpoint: true},
	}
	partSettings, err := (openai.Settings{
		Common:     ai.ModelSettings{ExtraBody: map[string]any{"custom": true}},
		Prediction: &openai.Prediction{ContentParts: parts},
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	parts[0].Text = "changed"

	model := openai.NewModel("gpt-4.1", openai.WithProvider(openai.ProviderConfig{
		Name: "openai", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	for _, settings := range []ai.ModelSettings{contentSettings, partSettings} {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := collect(t, model, ai.ModelRequestParams{Settings: contentSettings}); err != nil {
		t.Fatal(err)
	}

	content := bodies[0]["prediction"].(map[string]any)
	if content["type"] != "content" || content["content"] != "package main\n" ||
		bodies[2]["prediction"].(map[string]any)["content"] != "package main\n" {
		t.Fatalf("unexpected content prediction: static=%#v stream=%#v", content, bodies[2])
	}
	prediction := bodies[1]["prediction"].(map[string]any)
	predictionParts := prediction["content"].([]any)
	first := predictionParts[0].(map[string]any)
	second := predictionParts[1].(map[string]any)
	breakpoint := second["prompt_cache_breakpoint"].(map[string]any)
	if prediction["type"] != "content" || first["type"] != "text" || first["text"] != "package main\n" ||
		first["prompt_cache_breakpoint"] != nil || second["text"] != "func main() {}\n" ||
		breakpoint["mode"] != "explicit" || bodies[1]["custom"] != true {
		t.Fatalf("unexpected part prediction: %#v", bodies[1])
	}
}

func TestOpenAIPredictionValidation(t *testing.T) {
	tests := []struct {
		name     string
		settings openai.Settings
		want     string
	}{
		{name: "empty", settings: openai.Settings{Prediction: &openai.Prediction{}}, want: "must not be empty"},
		{name: "both", settings: openai.Settings{Prediction: &openai.Prediction{
			Content: "text", ContentParts: []openai.PredictionContentPart{{Text: "part"}},
		}}, want: "either content or content parts"},
		{name: "wire conflict", settings: openai.Settings{
			Common:     ai.ModelSettings{ExtraBody: map[string]any{"prediction": map[string]any{}}},
			Prediction: &openai.Prediction{Content: "text"},
		}, want: `extra body field "prediction" conflicts`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	settings, err := (openai.Settings{Prediction: &openai.Prediction{Content: "text"}}).Build()
	if err != nil {
		t.Fatal(err)
	}
	responses := openai.NewResponsesModel("gpt-5")
	_, err = responses.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings})
	if err == nil || !strings.Contains(err.Error(), "only supported by Chat Completions") {
		t.Fatalf("unexpected Responses prediction error: %v", err)
	}
}
