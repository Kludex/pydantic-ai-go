package ai_test

import (
	"errors"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

type xmlRecord struct {
	Name    string    `json:"name"`
	Created time.Time `json:"created"`
	Plain   string
	Ignored string `json:"-"`
	private string
}

type xmlText string

func (value xmlText) MarshalText() ([]byte, error) {
	return []byte(strings.ToUpper(string(value))), nil
}

type failingXMLText struct{}

func (failingXMLText) MarshalText() ([]byte, error) { return nil, errors.New("broken text") }

func TestFormatAsXMLStructuredValues(t *testing.T) {
	record := xmlRecord{
		Name: "Ada", Created: time.Date(2026, 1, 2, 3, 4, 5, 6, time.UTC), Ignored: "hidden", private: "secret",
	}
	formatted, err := ai.FormatAsXML(map[string]any{
		"active": true,
		"bytes":  []byte("A&B"),
		"count":  int8(-2),
		"delay":  2 * time.Second,
		"empty":  nil,
		"items":  []any{"one", uint16(2), 1.5},
		"label":  xmlText("hello"),
		"record": record,
	}, ai.WithXMLRoot("data"))
	if err != nil {
		t.Fatal(err)
	}
	for _, fragment := range []string{
		"<data>", "<active>true</active>", "<bytes>A&amp;B</bytes>", "<count>-2</count>",
		"<delay>2s</delay>", "<empty>null</empty>", "<item>one</item>", "<item>2</item>",
		"<item>1.5</item>", "<label>HELLO</label>", "<record>", "<name>Ada</name>",
		"<created>2026-01-02T03:04:05.000000006Z</created>", "<Plain></Plain>",
	} {
		if !strings.Contains(formatted, fragment) {
			t.Fatalf("missing %q in %s", fragment, formatted)
		}
	}
	if strings.Contains(formatted, "hidden") || strings.Contains(formatted, "secret") {
		t.Fatalf("unexpected ignored field in %s", formatted)
	}
}

func TestFormatAsXMLRootlessAndOptions(t *testing.T) {
	formatted, err := ai.FormatAsXML(
		[]xmlRecord{{Name: "Ada"}, {Name: "Grace"}},
		ai.WithXMLItemTag("person"), ai.WithXMLNull("nil"), ai.WithXMLIndent(""),
	)
	if err != nil {
		t.Fatal(err)
	}
	if formatted != "<xmlRecord><name>Ada</name><created>0001-01-01T00:00:00Z</created><Plain></Plain></xmlRecord>"+
		"<xmlRecord><name>Grace</name><created>0001-01-01T00:00:00Z</created><Plain></Plain></xmlRecord>" {
		t.Fatalf("unexpected XML: %s", formatted)
	}

	formatted, err = ai.FormatAsXML((*string)(nil), ai.WithXMLNull("nil"), ai.WithXMLIndent(""))
	if err != nil || formatted != "<item>nil</item>" {
		t.Fatalf("unexpected nil XML %q: %v", formatted, err)
	}
	formatted, err = ai.FormatAsXML(nil, ai.WithXMLIndent(""))
	if err != nil || formatted != "<item>null</item>" {
		t.Fatalf("unexpected untyped nil XML %q: %v", formatted, err)
	}
	formatted, err = ai.FormatAsXML([]any{nil}, ai.WithXMLIndent(""))
	if err != nil || formatted != "<item>null</item>" {
		t.Fatalf("unexpected nil item XML %q: %v", formatted, err)
	}
	formatted, err = ai.FormatAsXML([2]int{1, 2}, ai.WithXMLItemTag("number"), ai.WithXMLIndent(""))
	if err != nil || formatted != "<number>1</number><number>2</number>" {
		t.Fatalf("unexpected array XML %q: %v", formatted, err)
	}
	formatted, err = ai.FormatAsXML([]int{1, 2})
	if err != nil || formatted != "<item>1</item>\n<item>2</item>" {
		t.Fatalf("unexpected pretty rootless XML %q: %v", formatted, err)
	}
	formatted, err = ai.FormatAsXML(map[int]string{2: "two", 1: "one"}, ai.WithXMLRoot("numbers"), ai.WithXMLIndent(""))
	if err != nil || formatted != "<numbers><1>one</1><2>two</2></numbers>" {
		t.Fatalf("unexpected numeric-key XML %q: %v", formatted, err)
	}
	formatted, err = ai.FormatAsXML(map[uint]string{2: "two"}, ai.WithXMLRoot("numbers"), ai.WithXMLIndent(""))
	if err != nil || formatted != "<numbers><2>two</2></numbers>" {
		t.Fatalf("unexpected unsigned-key XML %q: %v", formatted, err)
	}
	formatted, err = ai.FormatAsXML(map[string]string{}, ai.WithXMLIndent(""))
	if err != nil || formatted != "" {
		t.Fatalf("unexpected empty XML %q: %v", formatted, err)
	}
}

func TestFormatAsXMLRejectsUnsupportedValues(t *testing.T) {
	for name, value := range map[string]any{
		"unsupported value": make(chan int),
		"unsupported key":   map[bool]string{true: "yes"},
		"invalid key":       map[string]string{"bad key": "value"},
		"text error":        failingXMLText{},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ai.FormatAsXML(value); err == nil {
				t.Fatal("expected error")
			}
		})
	}
	if _, err := ai.FormatAsXML("value", nil); err == nil {
		t.Fatal("expected nil-option error")
	}
	for name, option := range map[string]ai.XMLFormatOption{
		"root": ai.WithXMLRoot("bad tag"),
		"item": ai.WithXMLItemTag(""),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ai.FormatAsXML("value", option); err == nil {
				t.Fatal("expected tag error")
			}
		})
	}

	type invalidField struct {
		Value string `json:"bad tag"`
	}
	if _, err := ai.FormatAsXML(invalidField{}); err == nil {
		t.Fatal("expected field-tag error")
	}
}

func TestFormatAsXMLRejectsCycles(t *testing.T) {
	type node struct {
		Next *node `json:"next"`
	}
	pointerCycle := &node{}
	pointerCycle.Next = pointerCycle
	mapCycle := map[string]any{}
	mapCycle["self"] = mapCycle
	sliceCycle := make([]any, 1)
	sliceCycle[0] = sliceCycle
	for name, value := range map[string]any{
		"pointer": pointerCycle,
		"map":     mapCycle,
		"slice":   sliceCycle,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ai.FormatAsXML(value); err == nil || !strings.Contains(err.Error(), "cyclic") {
				t.Fatalf("expected cycle error, got %v", err)
			}
		})
	}
}
