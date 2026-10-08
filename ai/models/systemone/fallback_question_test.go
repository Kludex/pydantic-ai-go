package systemone_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/systemone"
)

func TestGenericQuestionFallback(t *testing.T) {
	for _, test := range []struct {
		name   string
		schema map[string]any
		answer string
	}{
		{name: "choice", schema: map[string]any{"type": "string", "enum": []any{"billing", "bug"}}, answer: `{"type":"choice","choice":"bug","confidence":1,"probabilities":{"billing":0,"bug":1}}`},
		{name: "rubric", schema: map[string]any{"type": "integer", "anyOf": []any{map[string]any{"const": 0, "description": "Calm"}, map[string]any{"const": 1, "description": "Angry"}}}, answer: `{"type":"score","score":1,"confidence":1,"probabilities":{"0":0,"1":1}}`},
		{name: "described bool", schema: map[string]any{"type": "boolean", "anyOf": []any{map[string]any{"const": true, "description": "Spam"}, map[string]any{"const": false, "description": "Not spam"}}}, answer: `{"type":"noul","noul":0.9}`},
	} {
		for _, mode := range []string{"default", "optional", "instructed"} {
			t.Run(test.name+"/"+mode, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body struct {
						Questions map[string]map[string]any `json:"questions"`
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					want := any("Which of these applies?")
					if mode == "optional" {
						want = nil
					}
					if mode == "instructed" {
						want = "Classify the input."
					}
					if got := body.Questions["response"]["instructions"]; got != want {
						t.Errorf("instructions=%v want=%v", got, want)
					}
					_, _ = w.Write([]byte(`{"model":"clm","answers":{"response":` + test.answer + `},"usage":{}}`))
				}))
				defer server.Close()
				required := false
				var options []systemone.Option
				if mode == "optional" {
					options = append(options, systemone.WithProfile(systemone.Profile{RequiresInstructions: &required}))
					required = true
				}
				options = append(options, systemone.WithBaseURL(server.URL))
				model := newModel(t, options...)
				params := ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: test.schema}}
				if mode == "instructed" {
					params.Instructions = "Classify the input."
				}
				if _, err := model.Request(t.Context(), prompt(), params); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
	model := newModel(t, systemone.WithBaseURL("http://127.0.0.1:1"))
	for _, schema := range []map[string]any{{"type": "boolean"}, {"type": "number", "minimum": 0, "maximum": 1}} {
		if _, err := model.Request(t.Context(), prompt(), ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: schema}}); err == nil {
			t.Fatal("generic fallback supplied meaning for an undescribed scalar")
		}
	}
}
