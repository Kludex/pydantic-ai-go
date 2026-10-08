package systemone_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/systemone"
)

func TestDecisionHistory(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if _, exists := body["state"].(map[string]any); !exists {
			t.Errorf("missing history: %v", body)
		}
		_, _ = writer.Write([]byte(`{"model":"clm","answers":{"response":{"type":"noul","noul":0.9}},"usage":{}}`))
	}))
	defer server.Close()
	model := newModel(t, systemone.WithBaseURL(server.URL))
	for _, messages := range [][]ai.ModelMessage{
		{ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "History"}, ai.ToolAvailabilityDeltaPart{}}}},
		{ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "Before"}, ai.UserPromptPart{Content: "judge"}, ai.ToolReturnPart{ToolName: "a", Content: "done"}}}, ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{Content: "Summary"}, ai.CompactionPart{}}}},
	} {
		if _, err := model.Request(t.Context(), messages, boolParams()); err != nil {
			t.Fatal(err)
		}
	}
	for _, messages := range [][]ai.ModelMessage{
		{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{ai.ImageURL{URL: "https://example.com/image"}}}}}, prompt()[0]},
		{prompt()[0], ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{}}}},
	} {
		if _, err := model.Request(t.Context(), messages, boolParams()); err == nil {
			t.Fatal("expected unsupported file error")
		}
	}
}

func TestNestedDefinitionContext(t *testing.T) {
	for _, title := range []string{"", "Judgement"} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			var body struct {
				Questions map[string]struct {
					Instructions map[string]any `json:"instructions"`
				} `json:"questions"`
			}
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if body.Questions["nested.ok"].Instructions["context"] == nil {
				t.Error("missing nested definition context")
			}
			_, _ = writer.Write([]byte(`{"model":"clm","answers":{"nested.ok":{"type":"noul","noul":0.9}},"usage":{}}`))
		}))
		model := newModel(t, systemone.WithBaseURL(server.URL))
		params := fieldParams("nested", map[string]any{"$ref": "#/$defs/Nested"})
		params.OutputTool.Schema["title"] = "Decision"
		params.OutputTool.Schema["$defs"] = map[string]any{"Nested": map[string]any{"type": "object", "title": title, "description": "Nested judgement.", "properties": map[string]any{"ok": map[string]any{"type": "boolean", "description": "Okay?"}}}}
		response, err := model.Request(t.Context(), prompt(), params)
		server.Close()
		if err != nil || string(response.Parts[0].(ai.ToolCallPart).Args) != `{"nested":{"ok":true}}` {
			t.Fatalf("response=%+v error=%v", response, err)
		}
	}
}

func TestCollidingRoutesAndQuestionNames(t *testing.T) {
	for _, tools := range [][]ai.ToolDefinition{
		{{Name: "a", Schema: map[string]any{"type": "object"}}, {Name: "a", Schema: map[string]any{"type": "object"}}},
		{{Name: "a", Schema: fieldParams("b", map[string]any{"type": "object", "properties": map[string]any{"c": map[string]any{"type": "boolean", "description": "Okay?"}}}).OutputTool.Schema}, {Name: "a.b", Schema: fieldParams("c", map[string]any{"type": "boolean", "description": "Okay?"}).OutputTool.Schema}},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			var body struct {
				Questions map[string]struct {
					Type     string         `json:"type"`
					Criteria map[string]any `json:"criteria"`
				} `json:"questions"`
			}
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			answers := map[string]any{}
			for name, question := range body.Questions {
				if question.Type == "noul" {
					answers[name] = map[string]any{"type": "noul", "noul": .9}
					continue
				}
				probabilities := map[string]float64{}
				for label := range question.Criteria {
					probabilities[label] = 0
				}
				probabilities["a"] = 1
				answers[name] = map[string]any{"type": "choice", "choice": "a", "confidence": 1, "probabilities": probabilities}
			}
			if err := json.NewEncoder(writer).Encode(map[string]any{"model": "clm", "answers": answers, "usage": map[string]int{}}); err != nil {
				t.Error(err)
			}
		}))
		model := newModel(t, systemone.WithBaseURL(server.URL))
		params := ai.ModelRequestParams{Tools: tools}
		if _, err := model.Request(t.Context(), prompt(), params); err != nil {
			t.Fatal(err)
		}
		if tools[1].Name == "a" {
			params.OutputTool = &ai.ToolDefinition{Name: "a", Schema: map[string]any{"type": "object"}}
			if _, err := model.Request(t.Context(), prompt(), params); err != nil {
				t.Fatal(err)
			}
		}
		server.Close()
	}
	if _, err := newModel(t, systemone.WithBaseURL("https://example.com")).Request(t.Context(), prompt(), ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: map[string]any{"type": "object"}}}); err == nil {
		t.Fatal("expected no typed questions")
	}
}

func TestSecondDecisionFailure(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 2 {
			writer.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = writer.Write([]byte(`{"model":"clm","answers":{"route":{"type":"choice","choice":"act","confidence":0.9,"probabilities":{"out":0.1,"act":0.9}},"out.response":{"type":"noul","noul":0.9}},"usage":{}}`))
	}))
	defer server.Close()
	params := boolParams()
	params.Tools = []ai.ToolDefinition{{Name: "act", Schema: fieldParams("ok", map[string]any{"type": "boolean", "description": strings.Repeat("Proceed? ", 2000)}).OutputTool.Schema}}
	_, err := newModel(t, systemone.WithBaseURL(server.URL)).Request(t.Context(), prompt(), params)
	var apiError *systemone.APIError
	if calls != 2 || !errors.As(err, &apiError) || !strings.Contains(err.Error(), "failed to fill") {
		t.Fatalf("calls=%d error=%v", calls, err)
	}
}
