package schema_test

import (
	"reflect"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/internal/schema"
)

func TestInlineDefinitions(t *testing.T) {
	source := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"$ref": "#/$defs/Value", "description": "At this site"},
			"union": map[string]any{"anyOf": []any{
				map[string]any{"$ref": "#/$defs/Value"}, map[string]any{"type": "integer"},
			}},
		},
		"$defs": map[string]any{
			"Value": map[string]any{"type": "string", "description": "Shared"},
		},
	}
	transformed := schema.InlineDefinitions(source)
	if transformed["$defs"] != nil || source["$defs"] == nil {
		t.Fatalf("definitions were not detached and removed: transformed=%#v source=%#v", transformed, source)
	}
	value := transformed["properties"].(map[string]any)["value"].(map[string]any)
	if value["type"] != "string" || value["description"] != "At this site" || value["$ref"] != nil {
		t.Fatalf("definition siblings were not preserved: %#v", value)
	}
	union := transformed["properties"].(map[string]any)["union"].(map[string]any)["anyOf"].([]any)
	if union[0].(map[string]any)["type"] != "string" {
		t.Fatalf("definition in an array was not inlined: %#v", union)
	}

	recursive := map[string]any{
		"$ref": "#/$defs/Node",
		"$defs": map[string]any{"Node": map[string]any{
			"type": "object", "properties": map[string]any{"next": map[string]any{"$ref": "#/$defs/Node"}},
		}},
	}
	retained := schema.InlineDefinitions(recursive)
	if retained["$defs"] == nil || retained["$ref"] != "#/$defs/Node" {
		t.Fatalf("recursive definitions were unsafely inlined: %#v", retained)
	}
	mutual := map[string]any{
		"$ref": "#/$defs/A~1B",
		"$defs": map[string]any{
			"A/B": map[string]any{"anyOf": []any{map[string]any{"$ref": "#/$defs/C"}}},
			"C":   map[string]any{"properties": map[string]any{"back": map[string]any{"$ref": "#/$defs/A~1B"}}},
		},
	}
	if retained := schema.InlineDefinitions(mutual); retained["$defs"] == nil {
		t.Fatalf("escaped mutual recursion was unsafely inlined: %#v", retained)
	}
	missing := map[string]any{
		"$ref":  "#/$defs/A",
		"$defs": map[string]any{"A": map[string]any{"$ref": "#/$defs/Missing"}},
	}
	if retained := schema.InlineDefinitions(missing); retained["$ref"] != "#/$defs/Missing" {
		t.Fatalf("missing definition reference changed: %#v", retained)
	}
	plain := map[string]any{"type": "object"}
	if !reflect.DeepEqual(schema.InlineDefinitions(plain), plain) {
		t.Fatal("schema without definitions changed")
	}
}
