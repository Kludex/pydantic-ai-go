package openrouter

import (
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestGoogleSchemaTransform(t *testing.T) {
	source := map[string]any{
		"$schema": "draft", "title": "Input", "type": "object", "discriminator": map[string]any{},
		"examples": []any{}, "exclusiveMaximum": 10, "exclusiveMinimum": 1,
		"properties": map[string]any{
			"status": map[string]any{"const": 7},
			"flags":  map[string]any{"enum": []any{true, false, nil, 3}},
			"category": map[string]any{"oneOf": []any{
				map[string]any{"type": "string"}, map[string]any{"type": "integer"},
			}},
			"email": map[string]any{"type": "string", "format": "email", "description": "User email"},
			"date":  map[string]any{"type": "string", "format": "date"},
			"name": map[string]any{"anyOf": []any{
				map[string]any{"type": "string"}, map[string]any{"type": "null"},
			}},
			"boolean_union": map[string]any{"anyOf": []any{true, map[string]any{"type": "null"}}},
		},
	}
	transformed := transformGoogleSchema(source)
	if source["title"] != "Input" {
		t.Fatal("Google transformation mutated its input")
	}
	for _, key := range []string{"$schema", "title", "discriminator", "examples", "exclusiveMaximum", "exclusiveMinimum"} {
		if transformed[key] != nil {
			t.Fatalf("unsupported field %q remains: %#v", key, transformed)
		}
	}
	properties := transformed["properties"].(map[string]any)
	if !reflect.DeepEqual(properties["status"], map[string]any{"type": "string", "enum": []any{"7"}}) {
		t.Fatalf("const was not converted: %#v", properties["status"])
	}
	if !reflect.DeepEqual(properties["flags"], map[string]any{
		"type": "string", "enum": []any{"True", "False", "None", "3"},
	}) {
		t.Fatalf("enum values were not stringified compatibly: %#v", properties["flags"])
	}
	category := properties["category"].(map[string]any)
	if category["oneOf"] != nil || category["anyOf"] == nil {
		t.Fatalf("oneOf was not converted: %#v", category)
	}
	if properties["email"].(map[string]any)["description"] != "User email (format: email)" ||
		properties["date"].(map[string]any)["description"] != "Format: date" {
		t.Fatalf("formats were not moved to descriptions: %#v", properties)
	}
	name := properties["name"].(map[string]any)
	if name["type"] != "string" || name["nullable"] != true || name["anyOf"] != nil {
		t.Fatalf("nullable union was not simplified: %#v", name)
	}
}

func TestInlineSchemaDefinitions(t *testing.T) {
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
	transformed := inlineSchemaDefinitions(source)
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
	retained := inlineSchemaDefinitions(recursive)
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
	if retained := inlineSchemaDefinitions(mutual); retained["$defs"] == nil {
		t.Fatalf("escaped mutual recursion was unsafely inlined: %#v", retained)
	}
	missing := map[string]any{
		"$ref":  "#/$defs/A",
		"$defs": map[string]any{"A": map[string]any{"$ref": "#/$defs/Missing"}},
	}
	if retained := inlineSchemaDefinitions(missing); retained["$ref"] != "#/$defs/Missing" {
		t.Fatalf("missing definition reference changed: %#v", retained)
	}
	plain := map[string]any{"type": "object"}
	if !reflect.DeepEqual(inlineSchemaDefinitions(plain), plain) {
		t.Fatal("schema without definitions changed")
	}
}

func TestTransformOpenRouterSchemas(t *testing.T) {
	schema := map[string]any{
		"type": "object", "$defs": map[string]any{"Value": map[string]any{"type": "string"}},
		"properties": map[string]any{"value": map[string]any{"$ref": "#/$defs/Value"}},
	}
	params := ai.ModelRequestParams{
		Tools:         []ai.ToolDefinition{{Name: "tool", Schema: schema, ReturnSchema: schema}},
		DeferredTools: []ai.ToolDefinition{{Name: "deferred", Schema: schema, ReturnSchema: schema}},
		OutputTool:    &ai.ToolDefinition{Name: "output", Schema: schema, ReturnSchema: schema},
		OutputSchema:  schema,
	}
	transformed := transformSchemas(params, "qwen")
	for _, definition := range append(transformed.Tools, transformed.DeferredTools...) {
		if definition.Schema["$defs"] != nil || definition.ReturnSchema["$defs"] != nil {
			t.Fatalf("tool definitions were not inlined: %#v", definition)
		}
	}
	if transformed.OutputTool.Schema["$defs"] != nil || transformed.OutputTool.ReturnSchema["$defs"] != nil ||
		transformed.OutputSchema["$defs"] != nil || params.Tools[0].Schema["$defs"] == nil {
		t.Fatalf("output schemas were not transformed defensively: %#v", transformed)
	}
	if untouched := transformSchemas(params, "unknown"); !reflect.DeepEqual(untouched, params) {
		t.Fatalf("unknown downstream schema changed: %#v", untouched)
	}
	if transformed := transformToolDefinitions(nil, inlineSchemaDefinitions); transformed != nil {
		t.Fatalf("nil tool definitions became non-nil: %#v", transformed)
	}
}
