package ai

import "encoding/json"

// UnmarshalJSON accepts the current native-fallback field and its upstream legacy aliases.
func (definition *ToolDefinition) UnmarshalJSON(data []byte) error {
	var wire struct {
		Name               string         `json:"name"`
		Description        string         `json:"description"`
		Schema             map[string]any `json:"parameters_json_schema"`
		NativeFallbackFor  *string        `json:"unless_native"`
		PreferNative       *string        `json:"prefer_native"`
		PreferBuiltin      *string        `json:"prefer_builtin"`
		NativeCompanionFor string         `json:"with_native"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	fallback := ""
	switch {
	case wire.NativeFallbackFor != nil:
		fallback = *wire.NativeFallbackFor
	case wire.PreferNative != nil:
		fallback = *wire.PreferNative
	case wire.PreferBuiltin != nil:
		fallback = *wire.PreferBuiltin
	}
	*definition = ToolDefinition{
		Name: wire.Name, Description: wire.Description, Schema: wire.Schema,
		NativeFallbackFor: fallback, NativeCompanionFor: wire.NativeCompanionFor,
	}
	return nil
}
