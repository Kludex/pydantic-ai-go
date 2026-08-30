package anthropic_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func TestAnthropicStrictToolProfile(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		extra     []anthropic.Option
		strict    bool
	}{
		{name: "supported", modelName: "claude-sonnet-4-5", strict: true},
		{name: "unsupported", modelName: "claude-sonnet-4-0"},
		{name: "disabled override", modelName: "claude-sonnet-4-5", extra: []anthropic.Option{anthropic.WithStrictToolSupport(false)}},
		{name: "enabled override", modelName: "model-alias", extra: []anthropic.Option{anthropic.WithStrictToolSupport(true)}, strict: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var body map[string]any
			model := newNamedAnthropicServer(t, test.modelName, anthropicRecorder(t, &body), test.extra...)
			strict := true
			params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
				Name: "work", Schema: map[string]any{"type": "object"}, Strict: &strict,
			}}}
			if _, err := model.Request(t.Context(), nil, params); err != nil {
				t.Fatal(err)
			}
			tool := body["tools"].([]any)[0].(map[string]any)
			_, gotStrict := tool["strict"]
			if gotStrict != test.strict {
				t.Fatalf("expected strict=%v, got %v", test.strict, tool)
			}
			schema := tool["input_schema"].(map[string]any)
			if schema["additionalProperties"] != false || schema["properties"] == nil {
				t.Fatalf("explicit strict schema was not transformed: %v", schema)
			}
		})
	}
}

func TestAnthropicStrictIsOptIn(t *testing.T) {
	for name, strict := range map[string]*bool{"default": nil, "disabled": boolPointer(false)} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			model := newServer(t, anthropicRecorder(t, &body))
			params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
				Name: "work", Strict: strict,
				Schema: map[string]any{"$schema": "draft", "title": "Input", "type": "object", "additionalProperties": true},
			}}}
			if _, err := model.Request(t.Context(), nil, params); err != nil {
				t.Fatal(err)
			}
			tool := body["tools"].([]any)[0].(map[string]any)
			if _, ok := tool["strict"]; ok {
				t.Fatalf("strict must be opt-in: %v", tool)
			}
			schema := tool["input_schema"].(map[string]any)
			if schema["$schema"] != nil || schema["title"] != nil || schema["additionalProperties"] != true {
				t.Fatalf("non-strict schema changed unexpectedly: %v", schema)
			}
		})
	}
}

func TestAnthropicStrictSchemaTransform(t *testing.T) {
	var body map[string]any
	model := newServer(t, anthropicRecorder(t, &body))
	strict := true
	params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
		Name: "work", Strict: &strict,
		Schema: map[string]any{
			"type": "object",
			"$defs": map[string]any{
				"Named": map[string]any{"title": "Named", "type": "string", "maxLength": 5},
			},
			"properties": map[string]any{
				"email": map[string]any{"type": "string", "format": "email", "enum": []string{"a@example.com"}},
				"state": map[string]any{"type": "string", "enum": []any{"ready"}},
				"uri": map[string]any{
					"type": "string", "format": "custom", "minLength": 2, "description": "Address",
				},
				"zero":  map[string]any{"type": "array", "minItems": 0, "items": map[string]any{"type": "integer"}},
				"one":   map[string]any{"type": "array", "minItems": int64(1), "items": map[string]any{"type": "number"}},
				"two":   map[string]any{"type": "array", "minItems": float64(2), "items": map[string]any{"type": "boolean"}},
				"other": map[string]any{"type": "array", "minItems": "many", "items": map[string]any{"type": "null"}},
				"any": map[string]any{"anyOf": []any{
					map[string]any{"type": "string"}, map[string]any{"type": "integer"},
				}},
				"oneOf": map[string]any{"oneOf": []any{map[string]any{"type": "boolean"}}},
				"all":   map[string]any{"allOf": []any{map[string]any{"type": "number"}}},
				"ref": map[string]any{
					"$defs": map[string]any{"Local": map[string]any{"type": "integer"}},
					"$ref":  "#/$defs/Local", "description": "discarded sibling",
				},
				"extra": map[string]any{"type": "integer", "minimum": 1},
			},
			"required":             []string{"email"},
			"additionalProperties": map[string]any{"type": "string"},
		},
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	tool := body["tools"].([]any)[0].(map[string]any)
	schema := tool["input_schema"].(map[string]any)
	if schema["additionalProperties"] != false {
		t.Fatalf("object was not closed: %v", schema)
	}
	definition := schema["$defs"].(map[string]any)["Named"].(map[string]any)
	if definition["title"] != nil || definition["description"] != "{maxLength: 5}" {
		t.Fatalf("definition was not transformed: %v", definition)
	}
	properties := schema["properties"].(map[string]any)
	email := properties["email"].(map[string]any)
	if email["format"] != "email" || email["enum"] == nil {
		t.Fatalf("supported string fields were lost: %v", email)
	}
	uri := properties["uri"].(map[string]any)
	if uri["format"] != nil || uri["description"] != "Address\n\n{format: custom, minLength: 2}" {
		t.Fatalf("unsupported constraints were not moved: %v", uri)
	}
	if properties["zero"].(map[string]any)["minItems"] != float64(0) ||
		properties["one"].(map[string]any)["minItems"] != float64(1) {
		t.Fatalf("supported minItems were lost: %v", properties)
	}
	if !strings.Contains(properties["two"].(map[string]any)["description"].(string), "minItems: 2") ||
		!strings.Contains(properties["other"].(map[string]any)["description"].(string), "minItems: many") {
		t.Fatalf("unsupported minItems were not described: %v", properties)
	}
	if properties["oneOf"].(map[string]any)["anyOf"] == nil || properties["all"].(map[string]any)["allOf"] == nil {
		t.Fatalf("variants were not transformed: %v", properties)
	}
	ref := properties["ref"].(map[string]any)
	if ref["$ref"] != "#/$defs/Local" || ref["description"] != nil || ref["$defs"] == nil {
		t.Fatalf("reference siblings were not discarded: %v", ref)
	}
	if properties["extra"].(map[string]any)["description"] != "{minimum: 1}" {
		t.Fatalf("extra constraint was not described: %v", properties["extra"])
	}
}

func TestAnthropicStrictSchemaErrors(t *testing.T) {
	strict := true
	tests := []struct {
		name   string
		schema map[string]any
		output bool
	}{
		{name: "definition value", schema: map[string]any{"type": "object", "$defs": map[string]any{"Bad": true}}},
		{name: "definition schema", schema: map[string]any{"type": "object", "$defs": map[string]any{"Bad": map[string]any{}}}},
		{name: "missing type", schema: map[string]any{}},
		{name: "any variant value", schema: map[string]any{"anyOf": []any{true}}},
		{name: "one variant schema", schema: map[string]any{"oneOf": []any{map[string]any{}}}},
		{name: "all variant schema", schema: map[string]any{"allOf": []any{map[string]any{}}}},
		{name: "property value", schema: map[string]any{"type": "object", "properties": map[string]any{"bad": true}}},
		{name: "property schema", schema: map[string]any{"type": "object", "properties": map[string]any{"bad": map[string]any{}}}},
		{name: "array items value", schema: map[string]any{"type": "array", "items": true}},
		{name: "array items schema", schema: map[string]any{"type": "array", "items": map[string]any{}}},
		{name: "unsupported type", schema: map[string]any{"type": "funky"}},
		{name: "output tool", schema: map[string]any{}, output: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newServer(t, anthropicRecorder(t, nil))
			tool := ai.ToolDefinition{Name: "bad", Schema: test.schema, Strict: &strict}
			params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{tool}}
			if test.output {
				params.Tools = nil
				params.OutputTool = &tool
			}
			if _, err := model.Request(t.Context(), nil, params); err == nil {
				t.Fatal("expected schema error")
			}
		})
	}
}

func anthropicRecorder(t *testing.T, body *map[string]any) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if body != nil {
			if err := json.NewDecoder(r.Body).Decode(body); err != nil {
				t.Error(err)
			}
		}
		_, _ = w.Write([]byte(`{"model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}
}

func newNamedAnthropicServer(
	t *testing.T, name string, handler http.HandlerFunc, extra ...anthropic.Option,
) *anthropic.Model {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	opts := []anthropic.Option{
		anthropic.WithAPIKey("test-key"), anthropic.WithBaseURL(server.URL), anthropic.WithHTTPClient(server.Client()),
	}
	return anthropic.NewModel(name, append(opts, extra...)...)
}

func boolPointer(value bool) *bool { return &value }
