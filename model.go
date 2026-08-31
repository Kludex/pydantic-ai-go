package ai

import (
	"context"
	"reflect"
	"slices"
	"time"
)

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

// ModelDefaultSettings is implemented by models with request-setting defaults.
// Agent, capability, and run settings override these values field by field.
type ModelDefaultSettings interface {
	DefaultModelSettings() ModelSettings
}

func modelName(model Model) string {
	if modelIsNil(model) {
		return ""
	}
	return model.Name()
}

func modelIsNil(model Model) bool {
	if model == nil {
		return true
	}
	value := reflect.ValueOf(model)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

// ModelSelection selects either a concrete model or an ID resolved by a
// registered ModelIDResolverFunc. The zero value makes no selection.
type ModelSelection struct {
	Model Model
	ID    string
}

// ModelSelectionContext is a read-only snapshot passed to a model selector.
// Step starts at 1. Model is the lower-precedence model on the first step and
// the model used by the previous request thereafter.
type ModelSelectionContext[Deps any] struct {
	Deps     Deps
	Model    Model
	ModelID  string
	Step     int
	Messages []ModelMessage
	Usage    Usage
}

// ModelSelectorFunc selects a model before one logical model request.
type ModelSelectorFunc[Deps any] func(
	ctx context.Context, selection ModelSelectionContext[Deps],
) (ModelSelection, error)

// ModelResolutionContext is passed when resolving an application model ID.
type ModelResolutionContext[Deps any] struct {
	Deps Deps
}

// ModelIDResolverFunc resolves an application model ID. Return nil, nil to
// let the next resolver try. Resolved IDs are cached for one agent run.
type ModelIDResolverFunc[Deps any] func(
	ctx context.Context, resolution ModelResolutionContext[Deps], modelID string,
) (Model, error)

// ModelRequestParams carries everything a provider needs beyond the messages.
type ModelRequestParams struct {
	// Instructions is the joined instruction text for providers that do not
	// need origin metadata.
	Instructions string
	// InstructionParts preserves static and dynamic instruction boundaries.
	// Providers may use these boundaries for prompt caching.
	InstructionParts []InstructionPart
	Tools            []ToolDefinition
	// DeferredTools contains prepared definitions hidden by the agent. Providers
	// with native deferral may advertise them without making them executable.
	DeferredTools []ToolDefinition
	// OutputTool, when non-nil, is the tool the model must call to
	// produce the final structured output.
	OutputTool *ToolDefinition
	// OutputSchema validates structured text output. Providers enforce it
	// natively unless OutputMode is OutputModePrompted.
	OutputSchema map[string]any
	// OutputMode identifies how structured output is requested.
	OutputMode OutputMode
	// OutputPrompt contains provider-neutral prompted-output instructions.
	OutputPrompt string
	// AllowText reports whether plain text is an acceptable final output.
	AllowText bool
	Settings  ModelSettings
}

// InstructionPart is one model instruction block.
type InstructionPart struct {
	Content string
	Dynamic bool
}

// ModelSettings tunes a model request. The zero value uses provider defaults.
type ModelSettings struct {
	MaxTokens         int
	RequestTimeout    time.Duration
	Temperature       *float64
	TopP              *float64
	Seed              *int
	StopSequences     []string
	ParallelToolCalls *bool
}

// Clone returns settings detached from pointer and slice fields.
func (s ModelSettings) Clone() ModelSettings {
	s.Temperature = clonePointer(s.Temperature)
	s.TopP = clonePointer(s.TopP)
	s.Seed = clonePointer(s.Seed)
	s.StopSequences = slices.Clone(s.StopSequences)
	s.ParallelToolCalls = clonePointer(s.ParallelToolCalls)
	return s
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func mergeModelSettings(base ModelSettings, override *ModelSettings) ModelSettings {
	base = base.Clone()
	if override == nil {
		return base
	}
	if override.MaxTokens != 0 {
		base.MaxTokens = override.MaxTokens
	}
	if override.RequestTimeout != 0 {
		base.RequestTimeout = override.RequestTimeout
	}
	if override.Temperature != nil {
		base.Temperature = clonePointer(override.Temperature)
	}
	if override.TopP != nil {
		base.TopP = clonePointer(override.TopP)
	}
	if override.Seed != nil {
		base.Seed = clonePointer(override.Seed)
	}
	if override.StopSequences != nil {
		base.StopSequences = slices.Clone(override.StopSequences)
	}
	if override.ParallelToolCalls != nil {
		base.ParallelToolCalls = clonePointer(override.ParallelToolCalls)
	}
	return base
}

// OutputToolConfig customizes tool-based structured output. Empty Name and
// Description fields use the defaults.
type OutputToolConfig struct {
	Name        string
	Description string
	Sequential  bool
	Strict      *bool
	// MaxRetries overrides the run's output retry budget for this output tool.
	MaxRetries *int
}

// ToolDefinition describes a tool to the model.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"parameters_json_schema"`
	// ReturnSchema describes the tool value for discovery and dynamic clients.
	// It is local metadata and is not sent to providers.
	ReturnSchema map[string]any `json:"-"`
	// Sequential makes this tool an execution barrier. Calls before it
	// finish first; the tool then runs alone; later calls start afterward.
	Sequential bool `json:"-"`
	// Strict asks the provider to constrain generated arguments to Schema.
	// Nil uses the provider default; true forces strict mode; false disables it.
	Strict *bool `json:"-"`
	// Metadata is available to preparation and filtering hooks but is not sent
	// to the model.
	Metadata map[string]any `json:"-"`
	// ToolsetID identifies the dynamic toolset that owns this definition. It is
	// local lifecycle metadata and is not sent to providers.
	ToolsetID string `json:"-"`
	// DeferLoading hides the tool until a rich tool return reveals its name.
	// Native providers may advertise its schema without making it executable.
	DeferLoading bool `json:"-"`
	// ToolKind identifies framework-managed typed tool calls and returns.
	ToolKind   ToolPartKind `json:"-"`
	maxRetries *int
	timeout    time.Duration
}
