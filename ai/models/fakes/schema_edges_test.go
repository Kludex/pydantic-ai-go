package fakes_test

import (
	"encoding/json"
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestGeneratedSchemaReferencesAndTuples(t *testing.T) {
	for _, test := range []struct {
		name string
		node any
		want any
	}{
		{name: "draft-7 definition", node: map[string]any{"$ref": "#/definitions/value"}, want: "shared"},
		{name: "escaped property", node: map[string]any{"$ref": "#/properties/a~1b~0c"}, want: "escaped"},
		{name: "array pointer", node: map[string]any{"$ref": "#/oneOf/0"}, want: true},
		{name: "root recursion", node: map[string]any{"$ref": "#"}, want: map[string]any{
			"value": map[string]any{}, "a/b~c": "escaped",
		}},
		{name: "external reference", node: map[string]any{"$ref": "https://example.com/schema"}},
		{name: "missing reference", node: map[string]any{"$ref": "#/missing"}},
		{name: "pointer through scalar", node: map[string]any{"$ref": "#/type/child"}},
		{name: "noncanonical index", node: map[string]any{"$ref": "#/oneOf/00"}},
		{name: "negative index", node: map[string]any{"$ref": "#/oneOf/-1"}},
		{name: "out-of-range index", node: map[string]any{"$ref": "#/oneOf/1"}},
		{name: "nonnumeric index", node: map[string]any{"$ref": "#/oneOf/no"}},
		{name: "empty enum", node: map[string]any{"enum": []any{}, "type": "string"}, want: "a"},
		{name: "empty string enum", node: map[string]any{"enum": []string{}, "type": "string"}, want: "a"},
		{name: "boolean alternatives", node: map[string]any{"oneOf": []any{true}}, want: "a"},
		{name: "tuple maximum", node: map[string]any{
			"type": "array", "prefixItems": []any{true, false}, "maxItems": 1.0,
		}, want: []any{"a"}},
		{name: "tuple minimum", node: map[string]any{
			"type": "array", "items": []any{true}, "minItems": 2.0, "maxItems": 1,
		}, want: []any{"a", "a"}},
		{name: "extra items", node: map[string]any{
			"type": "array", "items": true, "minItems": 2,
		}, want: []any{"a", "a"}},
		{name: "unknown limit", node: map[string]any{
			"type": "array", "items": true, "maxItems": "unknown",
		}, want: []any{"a"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			schema := map[string]any{
				"type": "object", "properties": map[string]any{
					"value": test.node, "a/b~c": map[string]any{"const": "escaped"},
				},
				"definitions": map[string]any{"value": map[string]any{"const": "shared"}},
				"oneOf":       []any{map[string]any{"const": true}},
			}
			if test.name == "root recursion" {
				delete(schema, "oneOf")
			}
			response, err := fakes.NewTestModel().Request(t.Context(), nil, ai.ModelRequestParams{
				OutputTool: &ai.ToolDefinition{Name: "result", Schema: schema},
			})
			if err != nil {
				t.Fatal(err)
			}
			var args map[string]any
			if err := json.Unmarshal(response.ToolCalls()[0].Args, &args); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(args["value"], test.want) {
				t.Fatalf("generated value = %#v, want %#v", args["value"], test.want)
			}
		})
	}
}
