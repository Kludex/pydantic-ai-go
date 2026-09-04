package ai

import (
	"bytes"
	"encoding"
	"encoding/xml"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"
	"time"
)

// XMLFormatOption configures FormatAsXML.
type XMLFormatOption func(*xmlFormatConfig) error

type xmlFormatConfig struct {
	rootTag string
	itemTag string
	null    string
	indent  string
}

// WithXMLRoot wraps the formatted value in tag. An empty tag removes the wrapper.
func WithXMLRoot(tag string) XMLFormatOption {
	return func(config *xmlFormatConfig) error {
		if tag != "" {
			if err := validateXMLTag(tag); err != nil {
				return err
			}
		}
		config.rootTag = tag
		return nil
	}
}

// WithXMLItemTag sets the tag used for slice and array items.
func WithXMLItemTag(tag string) XMLFormatOption {
	return func(config *xmlFormatConfig) error {
		if err := validateXMLTag(tag); err != nil {
			return err
		}
		config.itemTag = tag
		return nil
	}
}

// WithXMLNull sets the text used for nil values.
func WithXMLNull(value string) XMLFormatOption {
	return func(config *xmlFormatConfig) error {
		config.null = value
		return nil
	}
}

// WithXMLIndent sets one indentation level. An empty value emits compact XML.
func WithXMLIndent(indent string) XMLFormatOption {
	return func(config *xmlFormatConfig) error {
		config.indent = indent
		return nil
	}
}

// FormatAsXML formats scalars, maps, slices, arrays, and structs as prompt-friendly XML.
// Struct fields use their json names. Maps are ordered by their formatted keys.
func FormatAsXML(value any, options ...XMLFormatOption) (string, error) {
	config := xmlFormatConfig{itemTag: "item", null: "null", indent: "  "}
	for _, option := range options {
		if option == nil {
			return "", fmt.Errorf("ai: XML format option must not be nil")
		}
		if err := option(&config); err != nil {
			return "", err
		}
	}

	formatter := xmlFormatter{config: config, active: map[visit]bool{}}
	node, err := formatter.node(reflect.ValueOf(value), config.itemTag, true)
	if err != nil {
		return "", err
	}
	if config.rootTag != "" {
		node.tag = config.rootTag
		return encodeXMLNodes([]xmlPromptNode{node}, config.indent), nil
	}
	if node.composite {
		return encodeXMLNodes(node.children, config.indent), nil
	}
	return encodeXMLNodes([]xmlPromptNode{node}, config.indent), nil
}

type xmlPromptNode struct {
	tag       string
	text      string
	children  []xmlPromptNode
	composite bool
}

type visit struct {
	typeName reflect.Type
	pointer  uintptr
}

type xmlFormatter struct {
	config xmlFormatConfig
	active map[visit]bool
}

func (formatter *xmlFormatter) node(value reflect.Value, tag string, inferStructName bool) (xmlPromptNode, error) {
	for value.IsValid() && (value.Kind() == reflect.Interface || value.Kind() == reflect.Pointer) {
		if value.IsNil() {
			return xmlPromptNode{tag: tag, text: formatter.config.null}, nil
		}
		if value.Kind() == reflect.Pointer {
			current := visit{typeName: value.Type(), pointer: value.Pointer()}
			if formatter.active[current] {
				return xmlPromptNode{}, fmt.Errorf("ai: cannot format cyclic value of type %s as XML", value.Type())
			}
			formatter.active[current] = true
			defer delete(formatter.active, current)
		}
		value = value.Elem()
	}
	if !value.IsValid() {
		return xmlPromptNode{tag: tag, text: formatter.config.null}, nil
	}
	if text, ok, err := xmlScalar(value); ok || err != nil {
		return xmlPromptNode{tag: tag, text: text}, err
	}
	if value.Kind() == reflect.Map || value.Kind() == reflect.Slice {
		current := visit{typeName: value.Type(), pointer: uintptr(value.UnsafePointer())}
		if current.pointer != 0 {
			if formatter.active[current] {
				return xmlPromptNode{}, fmt.Errorf("ai: cannot format cyclic value of type %s as XML", value.Type())
			}
			formatter.active[current] = true
			defer delete(formatter.active, current)
		}
	}

	node := xmlPromptNode{tag: tag, composite: true}
	switch value.Kind() {
	case reflect.Map:
		entries, err := formatter.mapEntries(value)
		if err != nil {
			return xmlPromptNode{}, err
		}
		for _, entry := range entries {
			child, err := formatter.node(entry.value, entry.key, false)
			if err != nil {
				return xmlPromptNode{}, err
			}
			node.children = append(node.children, child)
		}
	case reflect.Slice, reflect.Array:
		for index := range value.Len() {
			childTag := formatter.config.itemTag
			item := value.Index(index)
			unwrapped := item
			for unwrapped.IsValid() && (unwrapped.Kind() == reflect.Interface || unwrapped.Kind() == reflect.Pointer) {
				if unwrapped.IsNil() {
					break
				}
				unwrapped = unwrapped.Elem()
			}
			if unwrapped.IsValid() && unwrapped.Kind() == reflect.Struct && unwrapped.Type() != reflect.TypeFor[time.Time]() {
				childTag = unwrapped.Type().Name()
			}
			child, err := formatter.node(item, childTag, true)
			if err != nil {
				return xmlPromptNode{}, err
			}
			node.children = append(node.children, child)
		}
	case reflect.Struct:
		if inferStructName && value.Type().Name() != "" {
			node.tag = value.Type().Name()
		}
		for index := range value.NumField() {
			fieldType := value.Type().Field(index)
			if !fieldType.IsExported() {
				continue
			}
			name := strings.Split(fieldType.Tag.Get("json"), ",")[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = fieldType.Name
			}
			if err := validateXMLTag(name); err != nil {
				return xmlPromptNode{}, fmt.Errorf("ai: struct field %s: %w", fieldType.Name, err)
			}
			child, err := formatter.node(value.Field(index), name, false)
			if err != nil {
				return xmlPromptNode{}, err
			}
			node.children = append(node.children, child)
		}
	default:
		return xmlPromptNode{}, fmt.Errorf("ai: unsupported XML value type %s", value.Type())
	}
	return node, nil
}

type xmlMapEntry struct {
	key   string
	value reflect.Value
}

func (formatter *xmlFormatter) mapEntries(value reflect.Value) ([]xmlMapEntry, error) {
	entries := make([]xmlMapEntry, 0, value.Len())
	iterator := value.MapRange()
	for iterator.Next() {
		key := iterator.Key()
		var text string
		switch key.Kind() {
		case reflect.String:
			text = key.String()
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			text = fmt.Sprint(key.Int())
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			text = fmt.Sprint(key.Uint())
		default:
			return nil, fmt.Errorf("ai: unsupported XML map key type %s", key.Type())
		}
		if key.Kind() == reflect.String {
			if err := validateXMLTag(text); err != nil {
				return nil, fmt.Errorf("ai: map key %q: %w", text, err)
			}
		}
		entries = append(entries, xmlMapEntry{key: text, value: iterator.Value()})
	}
	sort.Slice(entries, func(left, right int) bool { return entries[left].key < entries[right].key })
	return entries, nil
}

func xmlScalar(value reflect.Value) (string, bool, error) {
	if value.CanInterface() {
		if instant, ok := value.Interface().(time.Time); ok {
			return instant.Format(time.RFC3339Nano), true, nil
		}
		if duration, ok := value.Interface().(time.Duration); ok {
			return duration.String(), true, nil
		}
		if marshaler, ok := value.Interface().(encoding.TextMarshaler); ok {
			text, err := marshaler.MarshalText()
			if err != nil {
				return "", true, fmt.Errorf("ai: marshal %s as XML text: %w", value.Type(), err)
			}
			return string(text), true, nil
		}
	}
	switch value.Kind() {
	case reflect.String:
		return value.String(), true, nil
	case reflect.Bool:
		return fmt.Sprint(value.Bool()), true, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return fmt.Sprint(value.Int()), true, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return fmt.Sprint(value.Uint()), true, nil
	case reflect.Float32, reflect.Float64:
		return fmt.Sprint(value.Interface()), true, nil
	case reflect.Slice:
		if value.Type().Elem().Kind() == reflect.Uint8 {
			return string(value.Bytes()), true, nil
		}
	}
	return "", false, nil
}

func validateXMLTag(tag string) error {
	if tag == "" {
		return fmt.Errorf("ai: XML tag must not be empty")
	}
	decoder := xml.NewDecoder(strings.NewReader("<" + tag + "></" + tag + ">"))
	for {
		if _, err := decoder.Token(); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("ai: invalid XML tag %q: %w", tag, err)
		}
	}
}

func encodeXMLNodes(nodes []xmlPromptNode, indent string) string {
	var output bytes.Buffer
	for index, node := range nodes {
		if index != 0 && indent != "" {
			output.WriteByte('\n')
		}
		encodeXMLNode(&output, node, indent, 0)
	}
	return output.String()
}

func encodeXMLNode(output *bytes.Buffer, node xmlPromptNode, indent string, depth int) {
	output.WriteByte('<')
	output.WriteString(node.tag)
	output.WriteByte('>')
	if len(node.children) != 0 {
		if indent != "" {
			output.WriteByte('\n')
		}
		for index, child := range node.children {
			if indent != "" {
				output.WriteString(strings.Repeat(indent, depth+1))
			}
			encodeXMLNode(output, child, indent, depth+1)
			if indent != "" && index != len(node.children)-1 {
				output.WriteByte('\n')
			}
		}
		if indent != "" {
			output.WriteByte('\n')
			output.WriteString(strings.Repeat(indent, depth))
		}
	} else {
		_ = xml.EscapeText(output, []byte(node.text))
	}
	output.WriteString("</")
	output.WriteString(node.tag)
	output.WriteByte('>')
}
