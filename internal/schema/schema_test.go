package schema_test

import (
	"encoding/json"
	"reflect"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

type nested struct {
	Value float32 `json:"value"`
}

type textValue int

func (textValue) MarshalText() ([]byte, error) { return []byte("value"), nil }

type jsonValue struct{}

func (jsonValue) MarshalJSON() ([]byte, error) { return []byte(`{"custom":true}`), nil }

type embeddedConflictA struct {
	Value string
}

type embeddedConflictB struct {
	Value int
}

type embeddedTaggedConflict struct {
	Value bool `json:"Value"`
}

type hiddenEmbedded struct {
	Visible string `json:"visible"`
}

type recursiveEmbedded struct {
	*recursiveEmbedded
	Value string `json:"value"`
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
		if name == "ptr" {
			p = nonNullSchema(p)
		}
		if p["type"] != want {
			t.Fatalf("field %s: expected type %s, got %v", name, want, p["type"])
		}
	}
	if properties["name"].(map[string]any)["description"] != "A name" {
		t.Fatal("description tag not applied")
	}
	if enum := properties["unit"].(map[string]any)["enum"].([]any); len(enum) != 2 || enum[0] != "celsius" {
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
	root := nonNullSchema(value["properties"].(map[string]any)["root/node"].(map[string]any))
	properties := root["properties"].(map[string]any)
	children := nonNullSchema(properties["children"].(map[string]any)["items"].(map[string]any))
	lookup := nonNullSchema(properties["lookup"].(map[string]any)["additionalProperties"].(map[string]any))
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
	reference := nonNullSchema(value["properties"].(map[string]any)["next"].(map[string]any))["$ref"]
	if reference != "#" {
		t.Fatalf("unexpected root reference: %v", reference)
	}
}

func TestForSupportsJSONRepresentationsAndTags(t *testing.T) {
	type Embedded struct {
		Embedded string `json:"embedded"`
	}
	type OptionalEmbedded struct {
		Optional string `json:"optional"`
	}
	type value struct {
		Embedded
		*OptionalEmbedded
		Time       time.Time            `json:"time"`
		Raw        json.RawMessage      `json:"raw"`
		Data       []byte               `json:"data"`
		Text       textValue            `json:"text"`
		Code       int                  `json:"code" jsonschema:"title=Code,minimum=1,maximum=9,multipleOf=2,default=2,example=4"`
		Name       string               `json:"name" jsonschema:"format=email,pattern=^[a-z]+$,minLength=1,maxLength=20,readOnly=true"`
		List       []string             `json:"list" jsonschema:"minItems=1,maxItems=3"`
		Forced     string               `json:"forced,omitempty" jsonschema:"required,A required value"`
		Flags      []string             `json:"flags" jsonschema:"uniqueItems,minContains=1,maxContains=2"`
		Choice     any                  `json:"choice" jsonschema:"oneOf=[{\"type\":\"string\"},{\"type\":\"null\"}],examples=[\"one\",\"two\"],const=\"one\""`
		Object     map[string]any       `json:"object" jsonschema:"additionalProperties=false,deprecated,contentMediaType=application/json"`
		Scalar     string               `json:"scalar" jsonschema:"examples=single"`
		Escaped    string               `json:"escaped" jsonschema:"const=\"a\\\"b\""`
		StringInt  int                  `json:"string_int,string"`
		StringPtr  *bool                `json:"string_ptr,string,omitempty"`
		FixedBytes [2]byte              `json:"fixed_bytes"`
		IntMap     map[int]string       `json:"int_map"`
		TextMap    map[textValue]string `json:"text_map"`
		Custom     jsonValue            `json:"custom"`
		Number     json.Number          `json:"number"`
		Pointer    uintptr              `json:"pointer"`
		PointerMap map[uintptr]string   `json:"pointer_map"`
	}
	result, err := schema.For(reflect.TypeFor[value]())
	if err != nil {
		t.Fatal(err)
	}
	properties := result["properties"].(map[string]any)
	if properties["time"].(map[string]any)["format"] != "date-time" ||
		len(properties["raw"].(map[string]any)) != 0 ||
		properties["data"].(map[string]any)["contentEncoding"] != "base64" ||
		properties["text"].(map[string]any)["type"] != "string" ||
		properties["code"].(map[string]any)["minimum"] != float64(1) ||
		properties["code"].(map[string]any)["default"] != float64(2) ||
		properties["name"].(map[string]any)["readOnly"] != true ||
		properties["list"].(map[string]any)["maxItems"] != 3 ||
		properties["forced"].(map[string]any)["description"] != "A required value" ||
		properties["flags"].(map[string]any)["uniqueItems"] != true ||
		len(properties["choice"].(map[string]any)["oneOf"].([]any)) != 2 ||
		properties["choice"].(map[string]any)["const"] != "one" ||
		properties["object"].(map[string]any)["additionalProperties"] != false ||
		properties["object"].(map[string]any)["deprecated"] != true ||
		properties["scalar"].(map[string]any)["examples"].([]any)[0] != "single" ||
		properties["escaped"].(map[string]any)["const"] != `a"b` ||
		properties["string_int"].(map[string]any)["type"] != "string" ||
		nonNullSchema(properties["string_ptr"].(map[string]any))["type"] != "string" ||
		properties["fixed_bytes"].(map[string]any)["minItems"] != 2 ||
		properties["fixed_bytes"].(map[string]any)["items"].(map[string]any)["type"] != "integer" ||
		properties["int_map"].(map[string]any)["type"] != "object" ||
		properties["text_map"].(map[string]any)["type"] != "object" ||
		len(properties["custom"].(map[string]any)) != 0 ||
		properties["number"].(map[string]any)["type"] != "number" ||
		properties["pointer"].(map[string]any)["type"] != "integer" ||
		properties["pointer_map"].(map[string]any)["type"] != "object" {
		t.Fatalf("unexpected reflected schema: %#v", result)
	}
	if _, ok := properties["embedded"]; !ok {
		t.Fatal("embedded field was not promoted")
	}
	if _, ok := properties["optional"]; !ok {
		t.Fatal("pointer-embedded field was not promoted")
	}
	required := result["required"].([]string)
	for _, name := range required {
		if name == "optional" {
			t.Fatal("pointer-embedded field was required")
		}
	}
	if !slices.Contains(required, "forced") {
		t.Fatal("required annotation did not override omitempty")
	}

	for _, tag := range []string{
		"minimum=nope", "minLength=-1", "readOnly=nope", "oneOf=nope", "unknown=value",
		"description=known,unknown", `oneOf=[{"type":"string"}`, `description="unterminated`, "description=x,]",
	} {
		type invalid struct {
			Value int `json:"value"`
		}
		field, _ := reflect.TypeFor[invalid]().FieldByName("Value")
		field.Tag = reflect.StructTag(`json:"value" jsonschema:` + strconv.Quote(tag))
		if _, err := schema.For(reflect.StructOf([]reflect.StructField{field})); err == nil {
			t.Fatalf("invalid tag %q was accepted", tag)
		}
	}
}

func nonNullSchema(value map[string]any) map[string]any {
	return value["anyOf"].([]any)[0].(map[string]any)
}

func TestReflectedSchemaMatchesEmbeddedJSONFields(t *testing.T) {
	t.Run("shallow field wins", func(t *testing.T) {
		type value struct {
			embeddedConflictA
			Value bool
		}
		result, err := schema.For(reflect.TypeFor[value]())
		if err != nil {
			t.Fatal(err)
		}
		property := result["properties"].(map[string]any)["Value"].(map[string]any)
		if property["type"] != "boolean" {
			t.Fatalf("shallow field did not win: %#v", result)
		}
	})
	t.Run("equal fields conflict", func(t *testing.T) {
		type value struct {
			embeddedConflictA
			embeddedConflictB
		}
		result, err := schema.For(reflect.TypeFor[value]())
		if err != nil {
			t.Fatal(err)
		}
		if len(result["properties"].(map[string]any)) != 0 {
			t.Fatalf("conflicting fields were retained: %#v", result)
		}
	})
	t.Run("tagged field wins", func(t *testing.T) {
		type untaggedConflict struct {
			Value string
		}
		type value struct {
			untaggedConflict
			embeddedTaggedConflict
		}
		result, err := schema.For(reflect.TypeFor[value]())
		if err != nil {
			t.Fatal(err)
		}
		properties := result["properties"].(map[string]any)
		if properties["Value"].(map[string]any)["type"] != "boolean" {
			t.Fatalf("tagged field did not win: %#v", result)
		}
	})
	t.Run("duplicate embedded type conflicts", func(t *testing.T) {
		type first struct{ embeddedConflictA }
		type second struct{ embeddedConflictA }
		type value struct {
			first
			second
		}
		result, err := schema.For(reflect.TypeFor[value]())
		if err != nil {
			t.Fatal(err)
		}
		if len(result["properties"].(map[string]any)) != 0 {
			t.Fatalf("duplicate embedded fields were retained: %#v", result)
		}
	})
	t.Run("pointer and hidden embedding", func(t *testing.T) {
		var recursive recursiveEmbedded
		_ = recursive.recursiveEmbedded
		type value struct {
			*hiddenEmbedded
			*recursiveEmbedded
		}
		result, err := schema.For(reflect.TypeFor[value]())
		if err != nil {
			t.Fatal(err)
		}
		properties := result["properties"].(map[string]any)
		if properties["visible"].(map[string]any)["type"] != "string" ||
			properties["value"].(map[string]any)["type"] != "string" {
			t.Fatalf("embedded fields were not promoted: %#v", result)
		}
		if _, exists := result["required"]; exists {
			t.Fatalf("pointer-embedded fields were required: %#v", result)
		}
	})
	t.Run("named embedding", func(t *testing.T) {
		type value struct {
			embeddedConflictA `json:"nested"`
		}
		result, err := schema.For(reflect.TypeFor[value]())
		if err != nil {
			t.Fatal(err)
		}
		if result["properties"].(map[string]any)["nested"].(map[string]any)["type"] != "object" {
			t.Fatalf("named embedded field was promoted: %#v", result)
		}
	})
	t.Run("invalid tag name", func(t *testing.T) {
		typeWithInvalidTag := reflect.StructOf([]reflect.StructField{{
			Name: "Value", Type: reflect.TypeFor[string](), Tag: `json:"bad\\name"`,
		}})
		result, err := schema.For(typeWithInvalidTag)
		if err != nil {
			t.Fatal(err)
		}
		if _, exists := result["properties"].(map[string]any)["Value"]; !exists {
			t.Fatalf("invalid JSON tag name was accepted: %#v", result)
		}
	})
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
		M map[float64]string `json:"m"`
	}
	if _, err := schema.For(reflect.TypeFor[badMap]()); err == nil {
		t.Fatal("expected error for unsupported map key")
	}
	for _, fieldType := range []reflect.Type{reflect.TypeFor[[]string](), reflect.TypeFor[*[]string]()} {
		typeWithStringOption := reflect.StructOf([]reflect.StructField{{
			Name: "Value", Type: fieldType, Tag: `json:"value,string"`,
		}})
		if _, err := schema.For(typeWithStringOption); err == nil {
			t.Fatalf("expected error for unsupported %s json string option", fieldType)
		}
	}
	type badSlice struct {
		S []chan int `json:"s"`
	}
	if _, err := schema.For(reflect.TypeFor[badSlice]()); err == nil {
		t.Fatal("expected error for slice of unsupported type")
	}
	type badPointer struct {
		C *chan int `json:"c"`
	}
	if _, err := schema.For(reflect.TypeFor[badPointer]()); err == nil {
		t.Fatal("expected error for pointer to unsupported type")
	}
	type BadEmbedded struct {
		C chan int `json:"c"`
	}
	type badEmbedding struct {
		BadEmbedded
	}
	if _, err := schema.For(reflect.TypeFor[badEmbedding]()); err == nil {
		t.Fatal("expected error for unsupported embedded field")
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
