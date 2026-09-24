package typesafe_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/typesafe"
)

func TestCompositeSchemaAndHistory(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["state"].(map[string]any)["prompt"] != "latest" {
			t.Errorf("unexpected state: %#v", body["state"])
		}
		_, _ = response.Write([]byte(`{"answers":{
			"nested.flag":{"type":"noul","noul":0.1},
			"tags.a":{"type":"noul","noul":0.8},"tags.b":{"type":"noul","noul":0.2},
			"applies.a":{"type":"noul","noul":0.9},"applies.b":{"type":"noul","noul":0.1},
			"optional":{"type":"choice","choice":"none","confidence":0.7,"probabilities":{"a":0.2,"b":0.1,"none":0.7}},
			"probability":{"type":"noul","noul":0.75},"percentage":{"type":"noul","noul":0.42},
			"literal_bool":{"type":"noul","noul":0.8},
			"rubric":{"type":"score","score":1.6,"confidence":0.9,"probabilities":{"0":0.1,"1":0.3,"2":0.6}}
		},"model":"jev-1","usage":{}}`))
	}))
	defer server.Close()
	schema := map[string]any{"type": "object", "$defs": map[string]any{
		"areas": map[string]any{"type": "string", "enum": []any{"a", "b"}},
	}, "properties": map[string]any{
		"nested":       map[string]any{"type": "object", "properties": map[string]any{"flag": map[string]any{"type": "boolean", "description": "Flag?"}}},
		"tags":         map[string]any{"type": "array", "description": "Tags?", "items": map[string]any{"type": "string", "enum": []any{"a", "b"}}},
		"applies":      map[string]any{"type": "object", "description": "Applies?", "additionalProperties": map[string]any{"type": "boolean"}, "propertyNames": map[string]any{"$ref": "#/$defs/areas"}},
		"optional":     map[string]any{"description": "Optional?", "anyOf": []any{map[string]any{"type": "string", "enum": []any{"a", "b"}}, map[string]any{"type": "null", "description": "Neither"}}},
		"probability":  map[string]any{"type": "number", "minimum": float64(0), "maximum": float64(1), "description": "Probability?"},
		"percentage":   map[string]any{"type": "number", "minimum": float64(0), "maximum": float64(100), "description": "Percentage?"},
		"literal_bool": map[string]any{"type": "boolean", "enum": []any{true, false}, "description": "Enabled?"},
		"rubric": map[string]any{"anyOf": []any{
			map[string]any{"const": float64(0), "description": "bad"}, map[string]any{"const": float64(1), "description": "ok"}, map[string]any{"const": float64(2), "description": "good"},
		}},
	}}
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "system"}, ai.UserPromptPart{Content: "old"}, ai.ToolReturnPart{ToolName: "tool", Content: "done"}, ai.RetryPromptPart{Content: "retry"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "answer"}, ai.ToolCallPart{ToolName: "tool", Args: []byte(`{}`)}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "latest"}, ai.CachePoint{}}}}},
	}
	response, err := typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client())).Request(
		t.Context(), messages, ai.ModelRequestParams{Instructions: "judge", OutputTool: &ai.ToolDefinition{Name: "out", Description: "Output", Schema: schema}},
	)
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.ToolCallPart)
	var output map[string]any
	if err := json.Unmarshal(call.Args, &output); err != nil {
		t.Fatal(err)
	}
	applies := output["applies"].(map[string]any)
	if output["rubric"] != float64(2) || output["percentage"] != float64(42) || output["literal_bool"] != true ||
		applies["a"] != true || applies["b"] != false || !strings.Contains(string(call.Args), `"tags":["a"]`) ||
		response.ProviderDetails["scores"] == nil {
		t.Fatalf("unexpected response: %+v %s", response, call.Args)
	}
}

func TestDescribedBooleanAndWholeNumberChoices(t *testing.T) {
	tests := []struct {
		name       string
		schema     map[string]any
		answer     string
		want       any
		wantType   string
		wantLabels map[string]any
	}{
		{
			name: "described boolean",
			schema: map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{
				"anyOf": []any{
					map[string]any{"const": true, "description": "Money was returned."},
					map[string]any{"const": false, "description": "No refund was issued."},
				},
			}}},
			answer: `{"answers":{"value":{"type":"noul","noul":0.9}},"model":"jev","usage":{}}`,
			want:   true, wantType: "noul",
			wantLabels: map[string]any{"true": "Money was returned.", "false": "No refund was issued."},
		},
		{
			name: "integer choice",
			schema: map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{
				"enum": []any{float64(200), float64(404), float64(500)}, "description": "Which status?",
			}}},
			answer: `{"answers":{"value":{"type":"choice","choice":"404","confidence":0.9}},"model":"jev","usage":{}}`,
			want:   float64(404), wantType: "choice",
			wantLabels: map[string]any{"200": nil, "404": nil, "500": nil},
		},
		{
			name: "colliding labels",
			schema: map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{
				"enum": []any{"1", float64(1)}, "description": "Which value?",
			}}},
			answer: `{"answers":{"value":{"type":"choice","choice":"1 (number)","confidence":0.9}},"model":"jev","usage":{}}`,
			want:   float64(1), wantType: "choice",
			wantLabels: map[string]any{"1": nil, "1 (number)": nil},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				question := body["questions"].(map[string]any)["value"].(map[string]any)
				if question["type"] != test.wantType || !reflect.DeepEqual(question["criteria"], test.wantLabels) {
					t.Fatalf("unexpected question: %#v", question)
				}
				_, _ = response.Write([]byte(test.answer))
			}))
			defer server.Close()
			response, err := typesafe.NewModel(
				"jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()),
			).Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Content: "judge"},
			}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: test.schema}})
			if err != nil {
				t.Fatal(err)
			}
			var arguments map[string]any
			if err := json.Unmarshal(response.Parts[0].(ai.ToolCallPart).Args, &arguments); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(arguments["value"], test.want) {
				t.Fatalf("got %#v, want %#v", arguments["value"], test.want)
			}
		})
	}
}

func TestOptionalDescriptionsAndDefaults(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		questions := body["questions"].(map[string]any)
		if requests == 1 {
			criteria := questions["team"].(map[string]any)["criteria"].(map[string]any)
			if criteria["none"] != "Nobody owns it." {
				t.Fatalf("unexpected none description: %#v", criteria)
			}
			_, _ = response.Write([]byte(`{"answers":{"team":{"type":"choice","choice":"none","confidence":1}},"model":"jev","usage":{}}`))
			return
		}
		_, _ = response.Write([]byte(`{"answers":{"nested.team":{"type":"choice","choice":"none","confidence":1}},"model":"jev","usage":{}}`))
	}))
	defer server.Close()
	model := typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	optional := map[string]any{
		"description": "Which team?", "default": "shipping", "anyOf": []any{
			map[string]any{"type": "string", "enum": []any{"billing", "shipping"}},
			map[string]any{"type": "null", "description": "Nobody owns it."},
		},
	}
	response, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "judge"},
	}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: map[string]any{
		"type": "object", "properties": map[string]any{"team": optional},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Parts[0].(ai.ToolCallPart).Args) != `{}` {
		t.Fatalf("defaulted field was not omitted: %s", response.Parts[0].(ai.ToolCallPart).Args)
	}
	response, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "judge"},
	}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: map[string]any{
		"type": "object", "properties": map[string]any{"nested": map[string]any{
			"type": "object", "default": map[string]any{"team": "shipping"},
			"properties": map[string]any{"team": optional},
		}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if string(response.Parts[0].(ai.ToolCallPart).Args) != `{}` {
		t.Fatalf("defaulted nested object was not omitted: %s", response.Parts[0].(ai.ToolCallPart).Args)
	}
}

func TestToolRouting(t *testing.T) {
	calls := 0
	var requests []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		calls++
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		requests = append(requests, body)
		if calls == 1 {
			_, _ = response.Write([]byte(`{"answers":{"result":{"type":"noul","noul":0.2},"tool":{"type":"choice","choice":"act","confidence":0.9,"probabilities":{"result":0.1,"act":0.9}}},"model":"jev","usage":{}}`))
		} else {
			_, _ = response.Write([]byte(`{"answers":{"ok":{"type":"noul","noul":0.9}},"model":"jev","usage":{}}`))
		}
	}))
	defer server.Close()
	model := typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	response, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "run it"}}}}, ai.ModelRequestParams{
		Instructions: "decide", OutputTool: &ai.ToolDefinition{Name: "result", Description: "Answer", Schema: map[string]any{"type": "object", "properties": map[string]any{"result": map[string]any{"type": "boolean", "description": "Done?"}}}},
		Tools: []ai.ToolDefinition{{Name: "act", Description: "Act", Schema: map[string]any{"type": "object", "properties": map[string]any{"ok": map[string]any{"type": "boolean", "description": "Proceed?"}}}}},
	})
	toolDetails, _ := response.ProviderDetails["tool"].(map[string]any)
	secondQuestions := requests[1]["questions"].(map[string]any)
	okInstructions := secondQuestions["ok"].(map[string]any)["instructions"].(map[string]any)
	if err != nil || calls != 2 || response.Parts[0].(ai.ToolCallPart).ToolName != "act" || response.Usage.Requests != 2 ||
		response.ProviderDetails["requests"] != 2 || toolDetails["choice"] != "act" ||
		okInstructions["chosen"] != "act" {
		t.Fatalf("unexpected route: %+v calls=%d requests=%#v err=%v", response, calls, requests, err)
	}
}

func TestReturnedToolIsNotOfferedAgain(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		questions := body["questions"].(map[string]any)
		if _, exists := questions["tool"]; exists {
			t.Fatalf("returned tool was offered again: %#v", questions)
		}
		history := body["state"].(map[string]any)["history"].([]any)
		toolCall := history[1].(map[string]any)["tool_call"].(map[string]any)
		if toolCall["args"].(map[string]any) == nil {
			t.Fatalf("tool arguments were not preserved as JSON: %#v", toolCall)
		}
		_, _ = response.Write([]byte(`{"answers":{"done":{"type":"noul","noul":0.9}},"model":"jev","usage":{}}`))
	}))
	defer server.Close()
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "run it"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "act", Args: []byte(`{}`)}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "act", Content: "done"}}},
	}
	response, err := typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client())).Request(
		t.Context(), messages, ai.ModelRequestParams{
			OutputTool: &ai.ToolDefinition{Name: "result", Schema: map[string]any{"type": "object", "properties": map[string]any{
				"done": map[string]any{"type": "boolean", "description": "Done?"},
			}}},
			Tools: []ai.ToolDefinition{{Name: "act", Schema: map[string]any{"type": "object"}}},
		},
	)
	if err != nil || response.Parts[0].(ai.ToolCallPart).ToolName != "result" {
		t.Fatalf("unexpected response: %+v err=%v", response, err)
	}
}

type outputFunctionInput struct {
	Team   string `json:"team" jsonschema:"enum=billing,enum=legal"`
	Urgent bool   `json:"urgent" jsonschema_description:"Is this urgent?"`
}

func TestOutputFunctionArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = response.Write([]byte(`{"answers":{"team":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"billing":0.8,"legal":0.2}},"urgent":{"type":"noul","noul":0.9}},"model":"jev","usage":{}}`))
	}))
	defer server.Close()
	output := ai.NewOutputFunction("route", func(
		_ context.Context, _ *ai.RunContext[struct{}], input outputFunctionInput,
	) (string, error) {
		return input.Team, nil
	})
	result, err := ai.NewOutputFunctionAgent(
		typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client())),
		output,
		ai.WithInstructions("Route the ticket."),
	).Run(t.Context(), "I was charged twice.", struct{}{})
	if err != nil || result.Output != "billing" {
		t.Fatalf("unexpected result: %+v err=%v", result, err)
	}
}

func TestChoicesDescriptionNamesOutputRoute(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		questions := body["questions"].(map[string]any)
		criteria := questions["tool"].(map[string]any)["criteria"].(map[string]any)
		if criteria["intent"] != "Triage the ticket." {
			t.Fatalf("unexpected route description: %#v", criteria)
		}
		_, _ = response.Write([]byte(`{"answers":{"response":{"type":"choice","choice":"urgent","confidence":0.9,"probabilities":{"urgent":0.9,"normal":0.1}},"tool":{"type":"choice","choice":"intent","confidence":0.9,"probabilities":{"intent":0.9,"act":0.1}}},"model":"jev","usage":{}}`))
	}))
	defer server.Close()
	model := typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	agent := ai.NewChoicesAgent[struct{}](model, ai.NewStringChoices("intent", "Triage the ticket.", "urgent", "normal"))
	ai.AddSimpleTool(agent, "act", func(context.Context, struct{}) (string, error) { return "done", nil }, ai.WithDescription("Act now."))
	result, err := agent.Run(t.Context(), "help", struct{}{})
	if err != nil || result.Output != "urgent" {
		t.Fatalf("unexpected result: %+v err=%v", result, err)
	}
}

func TestUnsupportedInputs(t *testing.T) {
	model := typesafe.NewModel("jev", typesafe.WithBaseURL(":"))
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"}}}}}}
	_, err := model.Request(t.Context(), messages, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: map[string]any{"type": "boolean", "description": "ok?"}}})
	if err == nil || !strings.Contains(err.Error(), "files") {
		t.Fatalf("unexpected file error: %v", err)
	}
	for _, schema := range []map[string]any{
		{"type": "object", "properties": map[string]any{"bad.name": map[string]any{"type": "boolean"}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "string"}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "array", "items": map[string]any{"type": "string", "enum": []any{"one"}}}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "array", "prefixItems": []any{map[string]any{"type": "string"}}}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "array", "maxItems": float64(1), "items": map[string]any{"type": "string", "enum": []any{"one", "two"}}}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "number", "minimum": float64(0), "maximum": float64(100), "multipleOf": float64(10)}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "object", "maxProperties": float64(1), "additionalProperties": map[string]any{"type": "boolean"}, "propertyNames": map[string]any{"enum": []any{"one", "two"}}}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}, "propertyNames": map[string]any{"enum": []any{"one", "two"}}}}},
		{"type": "object", "properties": map[string]any{"bad": map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "boolean"}}}},
		{"$ref": "#/$defs/thread", "$defs": map[string]any{"thread": map[string]any{"type": "object", "properties": map[string]any{"parent": map[string]any{"$ref": "#/$defs/thread"}}}}},
	} {
		_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "x"}}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: schema}})
		if err == nil || !strings.Contains(err.Error(), "field") {
			t.Fatalf("expected schema error for %#v, got %v", schema, err)
		}
	}
}
