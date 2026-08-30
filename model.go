package ai

import "context"

// Model is the provider contract. A provider package implements Model;
// the agent calls Request once per loop iteration.
//
// Generics never cross this boundary: providers deal only in messages
// and schemas, which keeps adding a provider trivial.
type Model interface {
	Request(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error)

	// Name identifies the model (e.g. "gpt-5") for tracing and results.
	Name() string
}

// ModelRequestParams carries everything a provider needs beyond the messages.
type ModelRequestParams struct {
	Instructions string
	Tools        []ToolDefinition
	// OutputTool, when non-nil, is the tool the model must call to
	// produce the final structured output.
	OutputTool *ToolDefinition
	// OutputSchema, when non-nil, asks the provider for native JSON-mode
	// output conforming to the schema. Set instead of OutputTool when the
	// agent uses OutputModeNative.
	OutputSchema map[string]any
	// AllowText reports whether plain text is an acceptable final output.
	AllowText bool
	Settings  ModelSettings
}

// ModelSettings tunes a model request. The zero value uses provider defaults.
type ModelSettings struct {
	MaxTokens     int
	Temperature   *float64
	TopP          *float64
	Seed          *int
	StopSequences []string
}

// ToolDefinition describes a tool to the model.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"parameters_json_schema"`
}
