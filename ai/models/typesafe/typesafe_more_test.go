package typesafe_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
			"optional":{"type":"choice","choice":"none","confidence":0.7,"probabilities":{"a":0.2,"b":0.1,"none":0.7}},
			"probability":{"type":"noul","noul":0.75},
			"rubric":{"type":"score","score":1.6,"confidence":0.9,"probabilities":{"0":0.1,"1":0.3,"2":0.6}}
		},"model":"jev-1","usage":{}}`))
	}))
	defer server.Close()
	schema := map[string]any{"type": "object", "properties": map[string]any{
		"nested":      map[string]any{"type": "object", "properties": map[string]any{"flag": map[string]any{"type": "boolean", "description": "Flag?"}}},
		"tags":        map[string]any{"type": "array", "description": "Tags?", "items": map[string]any{"type": "string", "enum": []any{"a", "b"}}},
		"optional":    map[string]any{"description": "Optional?", "anyOf": []any{map[string]any{"type": "string", "enum": []any{"a", "b"}}, map[string]any{"type": "null"}}},
		"probability": map[string]any{"type": "number", "minimum": float64(0), "maximum": float64(1), "description": "Probability?"},
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
	if !strings.Contains(string(call.Args), `"rubric":2`) || !strings.Contains(string(call.Args), `"tags":["a"]`) || response.ProviderDetails["scores"] == nil {
		t.Fatalf("unexpected response: %+v %s", response, call.Args)
	}
}

func TestToolRouting(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		calls++
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
	if err != nil || calls != 2 || response.Parts[0].(ai.ToolCallPart).ToolName != "act" || response.Usage.Requests != 2 {
		t.Fatalf("unexpected route: %+v calls=%d err=%v", response, calls, err)
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
	} {
		_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "x"}}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: schema}})
		if err == nil {
			t.Fatalf("expected schema error for %#v", schema)
		}
	}
}
