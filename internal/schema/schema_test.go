package schema_test

import (
	"reflect"
	"testing"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

type nested struct {
	Value float32 `json:"value"`
}

type everything struct {
	Name     string            `json:"name" jsonschema:"description=A name"`
	Count    int               `json:"count"`
	Unsigned uint8             `json:"unsigned"`
	Ratio    float64           `json:"ratio,omitempty"`
	Enabled  bool              `json:"enabled"`
	Tags     []string          `json:"tags"`
	Pair     [2]int            `json:"pair"`
	Labels   map[string]string `json:"labels"`
	Child    nested            `json:"child"`
	Ptr      *nested           `json:"ptr,omitzero"`
	Anything any               `json:"anything"`
	Unit     string            `json:"unit" jsonschema:"enum=celsius,enum=fahrenheit"`
	Renamed  string            `json:",omitempty"`
	Skipped  string            `json:"-"`
	ignored  string            //nolint:unused // exercises the unexported-field branch
}

func TestForCoversAllTypes(t *testing.T) {
	s, err := schema.For(reflect.TypeFor[*everything]())
	if err != nil {
		t.Fatal(err)
	}
	properties := s["properties"].(map[string]any)
	expectType := map[string]string{
		"name": "string", "count": "integer", "unsigned": "integer", "ratio": "number",
		"enabled": "boolean", "tags": "array", "pair": "array", "labels": "object",
		"child": "object", "ptr": "object",
	}
	for name, want := range expectType {
		p := properties[name].(map[string]any)
		if p["type"] != want {
			t.Fatalf("field %s: expected type %s, got %v", name, want, p["type"])
		}
	}
	if properties["name"].(map[string]any)["description"] != "A name" {
		t.Fatal("description tag not applied")
	}
	if enum := properties["unit"].(map[string]any)["enum"].([]string); len(enum) != 2 || enum[0] != "celsius" {
		t.Fatalf("unexpected enum %v", enum)
	}
	if _, ok := properties["Skipped"]; ok {
		t.Fatal("json:\"-\" field should be skipped")
	}
	if _, ok := properties["Renamed"]; !ok {
		t.Fatal("field with empty json name should use the Go name")
	}
	if _, ok := properties["ignored"]; ok {
		t.Fatal("unexported field should be skipped")
	}
	required := s["required"].([]string)
	for _, name := range required {
		if name == "ratio" || name == "ptr" || name == "Renamed" {
			t.Fatalf("omitempty field %s should not be required", name)
		}
	}
	if s["additionalProperties"] != false {
		t.Fatal("expected additionalProperties false")
	}
}

func TestForTypeSupportsScalarRoots(t *testing.T) {
	value, err := schema.ForType(reflect.TypeFor[string]())
	if err != nil {
		t.Fatal(err)
	}
	if value["type"] != "string" {
		t.Fatalf("unexpected scalar schema: %+v", value)
	}
}

func TestForSupportsRecursiveStructs(t *testing.T) {
	type node struct {
		Name     string           `json:"name"`
		Children []*node          `json:"children,omitempty"`
		Lookup   map[string]*node `json:"lookup,omitempty"`
	}
	type tree struct {
		Root *node `json:"root/node"`
	}
	value, err := schema.For(reflect.TypeFor[tree]())
	if err != nil {
		t.Fatal(err)
	}
	root := value["properties"].(map[string]any)["root/node"].(map[string]any)
	properties := root["properties"].(map[string]any)
	children := properties["children"].(map[string]any)["items"].(map[string]any)
	lookup := properties["lookup"].(map[string]any)["additionalProperties"].(map[string]any)
	const reference = "#/properties/root~1node"
	if children["$ref"] != reference || lookup["$ref"] != reference {
		t.Fatalf("unexpected recursive references: children=%+v lookup=%+v", children, lookup)
	}
}

func TestForTypeSupportsRecursiveRoot(t *testing.T) {
	type node struct {
		Next *node `json:"next,omitempty"`
	}
	value, err := schema.ForType(reflect.TypeFor[node]())
	if err != nil {
		t.Fatal(err)
	}
	reference := value["properties"].(map[string]any)["next"].(map[string]any)["$ref"]
	if reference != "#" {
		t.Fatalf("unexpected root reference: %v", reference)
	}
}

func TestForRejectsNonStructs(t *testing.T) {
	if _, err := schema.For(reflect.TypeFor[string]()); err == nil {
		t.Fatal("expected error for non-struct")
	}
}

func TestForRejectsUnsupportedFields(t *testing.T) {
	type badChan struct {
		C chan int `json:"c"`
	}
	if _, err := schema.For(reflect.TypeFor[badChan]()); err == nil {
		t.Fatal("expected error for chan field")
	}
	type badMap struct {
		M map[int]string `json:"m"`
	}
	if _, err := schema.For(reflect.TypeFor[badMap]()); err == nil {
		t.Fatal("expected error for non-string map key")
	}
	type badSlice struct {
		S []chan int `json:"s"`
	}
	if _, err := schema.For(reflect.TypeFor[badSlice]()); err == nil {
		t.Fatal("expected error for slice of unsupported type")
	}
	type badNested struct {
		N struct {
			C chan int `json:"c"`
		} `json:"n"`
	}
	if _, err := schema.For(reflect.TypeFor[badNested]()); err == nil {
		t.Fatal("expected error for nested unsupported type")
	}
	type badMapValue struct {
		M map[string]chan int `json:"m"`
	}
	if _, err := schema.For(reflect.TypeFor[badMapValue]()); err == nil {
		t.Fatal("expected error for map of unsupported values")
	}
}
