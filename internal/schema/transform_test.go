package schema_test

import (
	"reflect"
	"testing"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

func TestTransformVisitsOnlySchemaNodes(t *testing.T) {
	source := map[string]any{
		"marker": "root",
		"properties": map[string]any{
			"title": map[string]any{"type": "string", "marker": "property"},
		},
		"$defs":                 map[string]any{"Named": map[string]any{"type": "integer", "marker": "definition"}},
		"definitions":           map[string]any{"Legacy": map[string]any{"type": "number", "marker": "legacy"}},
		"dependentSchemas":      map[string]any{"kind": map[string]any{"type": "object", "marker": "dependent"}},
		"patternProperties":     map[string]any{"^x": map[string]any{"type": "boolean", "marker": "pattern"}},
		"allOf":                 []any{map[string]any{"marker": "all"}, true},
		"anyOf":                 []any{map[string]any{"marker": "any"}},
		"oneOf":                 []any{map[string]any{"marker": "one"}},
		"prefixItems":           []any{map[string]any{"marker": "prefix"}},
		"items":                 map[string]any{"marker": "items"},
		"additionalProperties":  map[string]any{"marker": "additional"},
		"contains":              map[string]any{"marker": "contains"},
		"contentSchema":         map[string]any{"marker": "content"},
		"else":                  map[string]any{"marker": "else"},
		"if":                    map[string]any{"marker": "if"},
		"not":                   map[string]any{"marker": "not"},
		"propertyNames":         map[string]any{"marker": "names"},
		"then":                  map[string]any{"marker": "then"},
		"unevaluatedProperties": map[string]any{"marker": "unevaluated"},
	}
	var visited []string
	result := schema.Transform(source, func(node map[string]any) {
		if marker, ok := node["marker"].(string); ok {
			visited = append(visited, marker)
		}
		delete(node, "marker")
	})
	want := []string{
		"root", "additional", "contains", "content", "else", "if", "items", "not", "names", "then",
		"unevaluated", "all", "any", "one", "prefix", "definition", "legacy", "dependent", "pattern", "property",
	}
	if !reflect.DeepEqual(visited, want) {
		t.Fatalf("unexpected traversal\nwant: %v\ngot:  %v", want, visited)
	}
	properties := result["properties"].(map[string]any)
	if _, ok := properties["title"]; !ok {
		t.Fatal("property name was treated as a schema keyword")
	}
	if _, ok := source["marker"]; !ok {
		t.Fatal("source schema was mutated")
	}
	result["allOf"].([]any)[0].(map[string]any)["new"] = true
	if _, ok := source["allOf"].([]any)[0].(map[string]any)["new"]; ok {
		t.Fatal("nested source value was mutated")
	}
}
