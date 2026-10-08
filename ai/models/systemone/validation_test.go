package systemone_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/systemone"
)

func fieldParams(name string, property map[string]any) ai.ModelRequestParams {
	return ai.ModelRequestParams{Instructions: "Judge this.", OutputTool: &ai.ToolDefinition{Name: "out", Schema: map[string]any{"type": "object", "properties": map[string]any{name: property}}}}
}

func replyJSON(t *testing.T, name string, answer map[string]any) string {
	t.Helper()
	data, err := json.Marshal(map[string]any{"model": "clm-1", "answers": map[string]any{name: answer}, "usage": map[string]int{"input_tokens": 42}})
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func scoreProperty(levels int) map[string]any {
	options := make([]any, levels)
	for level := range options {
		options[level] = map[string]any{"const": level, "description": fmt.Sprintf("Level %d", level)}
	}
	return map[string]any{"type": "integer", "anyOf": options}
}

func TestInvalidResponses(t *testing.T) {
	choice := map[string]any{"type": "choice", "choice": "billing", "confidence": .8, "probabilities": map[string]float64{"billing": .9, "bug": .1}}
	score := map[string]any{"type": "score", "score": 1.7, "confidence": .55, "probabilities": map[string]float64{"0": .05, "1": .2, "2": .75}, "legend": map[string]string{"0": "Calm", "1": "Frustrated", "2": "Angry"}}
	for _, group := range []struct {
		name     string
		property map[string]any
		answer   map[string]any
		changes  []map[string]any
	}{
		{"urgent", map[string]any{"type": "boolean"}, map[string]any{"type": "noul", "noul": .9}, []map[string]any{{"noul": 1.2}, {"noul": -.1}, {"noul": nil}, {"type": "choice"}}},
		{"area", map[string]any{"enum": []any{"billing", "bug"}}, choice, []map[string]any{
			{"type": "noul"}, {"choice": "other"}, {"choice": nil}, {"confidence": 2}, {"confidence": nil},
			{"probabilities": map[string]any{"billing": -1, "bug": 2}}, {"probabilities": map[string]any{"billing": 1}},
			{"probabilities": map[string]any{"billing": .8, "bug": .1}}, {"probabilities": map[string]any{"billing": nil, "bug": 1}},
			{"probabilities": map[string]any{"other": .9, "bug": .1}},
		}},
		{"score", scoreProperty(3), score, []map[string]any{
			{"score": 3}, {"score": nil}, {"score": 0}, {"score": .51, "probabilities": map[string]float64{"0": .51, "1": .49, "2": 0}},
			{"score": .51, "probabilities": map[string]float64{"0": .5101, "1": .4899, "2": 0}}, {"confidence": -.1},
			{"probabilities": map[string]float64{"0": .05, "1": .2}}, {"probabilities": map[string]float64{"0": .05, "1": -.2, "2": 1.15}},
			{"probabilities": map[string]float64{"0": .05, "1": .2, "2": .7}}, {"probabilities": map[string]float64{"0": .0501, "1": .2001, "2": .7398}},
			{"legend": map[string]string{"0": "Calm", "1": "Frustrated"}}, {"legend": map[string]string{"0": "Calm", "1": "Frustrated", "3": "Angry"}},
		}},
	} {
		for index, change := range group.changes {
			t.Run(fmt.Sprintf("%s/%d", group.name, index), func(t *testing.T) {
				answer := maps.Clone(group.answer)
				for key, value := range change {
					if value == nil {
						delete(answer, key)
					} else {
						answer[key] = value
					}
				}
				_, err := serverModel(t, replyJSON(t, group.name, answer)).Request(t.Context(), prompt(), fieldParams(group.name, group.property))
				var unexpected *ai.UnexpectedModelBehaviorError
				if !errors.As(err, &unexpected) {
					t.Fatalf("expected invalid response, got %v", err)
				}
			})
		}
	}
	for _, reply := range []string{
		`not JSON`, `null`, `{}`, `{"model":"clm","answers":{},"usage":{}}`,
		`{"model":"clm","answers":{"response":{"type":"noul","noul":0.5},"extra":{"type":"noul","noul":0.5}},"usage":{}}`,
		`{"model":"clm","answers":{"different":{"type":"noul","noul":0.5}},"usage":{}}`,
		`{"model":"clm","answers":{"response":{"type":"noul","noul":0.5}},"usage":{"input_tokens":-1}}`,
		`{"model":"clm","answers":{"response":{"type":"noul","noul":0.5}},"usage":{"output_tokens":-1}}`,
		`{"model":"clm","answers":{"response":{"type":"noul","noul":1e999}},"usage":{}}`,
		`{"model":"clm","answers":{"response":{"type":"noul","noul":0.5}},"usage":{"input_tokens":null}}`,
		`{"model":"clm","answers":{"response":{"type":"noul","noul":0.5}},"usage":{"output_tokens":"bad"}}`,
	} {
		_, err := serverModel(t, reply).Request(t.Context(), prompt(), boolParams())
		var unexpected *ai.UnexpectedModelBehaviorError
		if !errors.As(err, &unexpected) {
			t.Fatalf("reply=%s error=%v", reply, err)
		}
	}
}

func TestRoundedAnswers(t *testing.T) {
	for _, probabilities := range []map[string]float64{{"0": .51, "1": .49, "2": 0}, {"0": .51, "1": .49, "2": .0001}} {
		answer := map[string]any{"type": "score", "score": .5, "confidence": .5, "probabilities": probabilities}
		response, err := serverModel(t, replyJSON(t, "score", answer)).Request(t.Context(), prompt(), fieldParams("score", scoreProperty(3)))
		if err != nil || string(response.Parts[0].(ai.ToolCallPart).Args) != `{"score":1}` {
			t.Fatalf("response=%+v error=%v", response, err)
		}
		if response.ProviderDetails["scores"].(map[string]float64)["score"] != .5 {
			t.Fatal("raw score was not preserved")
		}
	}
	answer := map[string]any{"type": "choice", "choice": "billing", "confidence": .8, "probabilities": map[string]float64{"billing": .6, "bug": .39}}
	response, err := serverModel(t, replyJSON(t, "area", answer)).Request(t.Context(), prompt(), fieldParams("area", map[string]any{"enum": []any{"billing", "bug"}}))
	if err != nil || response.ProviderDetails["probabilities"].(map[string]map[string]float64)["area"]["bug"] != .39 {
		t.Fatalf("response=%+v error=%v", response, err)
	}
}

func TestEmptyChoiceLabel(t *testing.T) {
	answer := map[string]any{"type": "choice", "choice": "", "confidence": 1, "probabilities": map[string]float64{"": 1, "other": 0}}
	response, err := serverModel(t, replyJSON(t, "label", answer)).Request(t.Context(), prompt(), fieldParams("label", map[string]any{"enum": []any{"", "other"}}))
	if err != nil || string(response.Parts[0].(ai.ToolCallPart).Args) != `{"label":""}` {
		t.Fatalf("response=%+v error=%v", response, err)
	}
}

func TestProfileLimits(t *testing.T) {
	for _, limit := range []int{0, 10} {
		t.Run(strconv.Itoa(limit), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				var body struct {
					Questions map[string]struct {
						Type string `json:"type"`
					} `json:"questions"`
				}
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				probabilities := map[string]float64{}
				for level := range 11 {
					probabilities[strconv.Itoa(level)] = 0
				}
				probabilities["3"] = 1
				answer := map[string]any{"type": "score", "score": 3, "confidence": .5, "probabilities": probabilities}
				if limit == 10 {
					answer["type"], answer["choice"] = "choice", "3"
				}
				if body.Questions["score"].Type != answer["type"] {
					t.Errorf("questions=%v", body.Questions)
				}
				_, _ = writer.Write([]byte(replyJSON(t, "score", answer)))
			}))
			defer server.Close()
			model := newModel(t, systemone.WithBaseURL(server.URL), systemone.WithProfile(systemone.Profile{MaxScoreLevels: limit}))
			response, err := model.Request(t.Context(), prompt(), fieldParams("score", scoreProperty(11)))
			if err != nil || string(response.Parts[0].(ai.ToolCallPart).Args) != `{"score":3}` {
				t.Fatalf("response=%+v error=%v", response, err)
			}
		})
	}
	model := newModel(t, systemone.WithBaseURL("https://example.com"), systemone.WithProfile(systemone.Profile{MaxChoiceOptions: 1}))
	if _, err := model.Request(t.Context(), prompt(), fieldParams("area", map[string]any{"enum": []any{"billing", "bug"}})); err == nil {
		t.Fatal("expected choice cap error before network")
	}
	params := boolParams()
	params.Tools = []ai.ToolDefinition{{Name: "a", Schema: map[string]any{"type": "object"}}, {Name: "b", Schema: map[string]any{"type": "object"}}}
	if _, err := model.Request(t.Context(), prompt(), params); err == nil {
		t.Fatal("expected route cap error before network")
	}
}

func TestExtraBodyQuestionOverrides(t *testing.T) {
	for _, questions := range []any{
		[]any{}, nil, map[string]any{"response": map[string]any{"type": "unknown"}},
		map[string]any{"response": map[string]any{"type": "choice", "criteria": []any{}}},
		map[string]any{"response": map[string]any{"type": "score", "criteria": map[string]any{}}},
		map[string]any{"response": map[string]any{"type": "noul", "criteria": []any{}}},
	} {
		params := boolParams()
		params.Settings.ExtraBody = map[string]any{"questions": questions}
		if _, err := newModel(t, systemone.WithBaseURL("https://example.com")).Request(t.Context(), prompt(), params); err == nil {
			t.Fatal("expected invalid question override")
		}
	}
	model := serverModel(t, `{"model":"clm-1","answers":{"response":{"type":"noul","noul":0.8},"extra":{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":1,"b":0}},"score":{"type":"score","score":1,"confidence":1,"probabilities":{"0":0,"1":1}}},"usage":{}}`)
	params := boolParams()
	params.Settings.ExtraBody = map[string]any{"questions": map[string]any{
		"response": map[string]any{"type": "noul", "criteria": map[string]any{"true": "Yes", "false": "No"}},
		"extra":    map[string]any{"type": "choice", "criteria": map[string]any{"a": nil, "b": nil}},
		"score":    map[string]any{"type": "score", "criteria": []any{"Low", "High"}},
	}}
	response, err := model.Request(t.Context(), prompt(), params)
	if err != nil || string(response.Parts[0].(ai.ToolCallPart).Args) != "true" {
		t.Fatalf("response=%+v error=%v", response, err)
	}
	params.Settings.ExtraBody["questions"] = map[string]any{"response": map[string]any{"type": "noul"}}
	if _, err := model.Request(t.Context(), prompt(), params); err == nil {
		t.Fatal("expected answer-name mismatch against actual questions")
	}
}
