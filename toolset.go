package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// Toolset lists a dynamic collection of reusable tools for one model step.
// Implementations may use RunContext to vary availability between steps.
type Toolset[Deps any] interface {
	Tools(ctx context.Context, rc *RunContext[Deps]) ([]Tool[Deps], error)
}

// ToolsetIDProvider optionally gives a toolset a stable application ID. The
// ID is copied onto its tool definitions for durable execution and tracing.
type ToolsetIDProvider interface {
	ToolsetID() string
}

// ToolsetRunProvider optionally returns an isolated toolset for one run. It
// is called once before the toolset is opened.
type ToolsetRunProvider[Deps any] interface {
	ForRun(ctx context.Context, rc *RunContext[Deps]) (Toolset[Deps], error)
}

// ToolsetStepProvider optionally replaces a toolset before one model request.
// A provider returning a different open resource manages that transition.
type ToolsetStepProvider[Deps any] interface {
	ForRunStep(ctx context.Context, rc *RunContext[Deps]) (Toolset[Deps], error)
}

// ToolsetCloseFunc releases resources acquired by ToolsetOpener. It is called
// exactly once, including when a later toolset fails to open.
type ToolsetCloseFunc func(ctx context.Context) error

// ToolsetOpener optionally acquires run-scoped resources and returns the
// toolset used for the run plus its cleanup function.
type ToolsetOpener[Deps any] interface {
	OpenToolset(
		ctx context.Context, rc *RunContext[Deps],
	) (Toolset[Deps], ToolsetCloseFunc, error)
}

// ToolsetInstructionsProvider optionally contributes instructions before each
// model request. Plain toolsets only need to implement Toolset.
type ToolsetInstructionsProvider[Deps any] interface {
	ToolsetInstructions(ctx context.Context, rc *RunContext[Deps]) ([]InstructionPart, error)
}

type toolsetInstructionRelay interface {
	relaysToolsetInstructions()
}

// ToolFilterFunc decides whether one tool remains available for a model step.
type ToolFilterFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps], definition ToolDefinition,
) (bool, error)

// ToolsetApprovalFunc dynamically requests approval for one validated call.
// Return nil to execute immediately.
type ToolsetApprovalFunc[Deps any] func(
	ctx context.Context,
	rc *RunContext[Deps],
	definition ToolDefinition,
	args json.RawMessage,
) (*ToolApprovalRequest, error)

// NewFunctionToolset creates a toolset from reusable function tools.
func NewFunctionToolset[Deps any](tools ...Tool[Deps]) Toolset[Deps] {
	cloned := cloneTools(tools)
	return functionToolset[Deps]{tools: cloned}
}

// CombineToolsets combines toolsets in order. Duplicate names fail when the
// tools are resolved for a model step.
func CombineToolsets[Deps any](toolsets ...Toolset[Deps]) Toolset[Deps] {
	return combinedToolset[Deps]{toolsets: append([]Toolset[Deps](nil), toolsets...)}
}

// FilterToolset keeps tools accepted by filter on each model step.
func FilterToolset[Deps any](toolset Toolset[Deps], filter ToolFilterFunc[Deps]) Toolset[Deps] {
	return filteredToolset[Deps]{toolset: toolset, filter: filter}
}

// PrefixToolset adds prefix and an underscore to every exposed tool name.
// The wrapped callback still sees the original name in RunContext.ToolName.
func PrefixToolset[Deps any](toolset Toolset[Deps], prefix string) Toolset[Deps] {
	return prefixedToolset[Deps]{toolset: toolset, prefix: prefix}
}

// RenameToolset renames selected tools using an original-name to new-name map.
// Unlisted tools keep their names.
func RenameToolset[Deps any](toolset Toolset[Deps], names map[string]string) Toolset[Deps] {
	cloned := make(map[string]string, len(names))
	for original, renamed := range names {
		cloned[original] = renamed
	}
	return renamedToolset[Deps]{toolset: toolset, names: cloned}
}

// PrepareToolset filters or modifies detached definitions on each model step.
// The callback cannot add or rename tools.
func PrepareToolset[Deps any](toolset Toolset[Deps], prepare ToolsPrepareFunc[Deps]) Toolset[Deps] {
	return preparedToolset[Deps]{toolset: toolset, prepare: prepare}
}

// DeferLoadingToolset hides all wrapped tools, or the selected names, until
// another tool reveals them through ToolReturn.Tools.
func DeferLoadingToolset[Deps any](toolset Toolset[Deps], names ...string) Toolset[Deps] {
	var selected map[string]struct{}
	if len(names) > 0 {
		selected = make(map[string]struct{}, len(names))
		for _, name := range names {
			selected[name] = struct{}{}
		}
	}
	return deferredToolset[Deps]{toolset: toolset, names: selected}
}

// RequireApprovalToolset requires approval for every wrapped tool, or only
// the selected original names when names are provided.
func RequireApprovalToolset[Deps any](toolset Toolset[Deps], names ...string) Toolset[Deps] {
	var selected map[string]struct{}
	if len(names) > 0 {
		selected = make(map[string]struct{}, len(names))
		for _, name := range names {
			selected[name] = struct{}{}
		}
	}
	return approvalRequiredToolset[Deps]{toolset: toolset, names: selected}
}

// RequireApprovalToolsetWhen evaluates approval for each validated call.
func RequireApprovalToolsetWhen[Deps any](
	toolset Toolset[Deps], check ToolsetApprovalFunc[Deps],
) Toolset[Deps] {
	if check == nil {
		panic("ai: toolset approval function must not be nil")
	}
	return approvalRequiredToolset[Deps]{toolset: toolset, check: check}
}

// SetToolsetMetadata merges metadata onto every tool. New values take precedence.
func SetToolsetMetadata[Deps any](toolset Toolset[Deps], metadata map[string]any) Toolset[Deps] {
	return metadataToolset[Deps]{toolset: toolset, metadata: cloneSchemaMap(metadata)}
}

// WithToolsetMaxRetries sets the default retry budget for tools that do not
// have an explicit WithToolMaxRetries option.
func WithToolsetMaxRetries[Deps any](toolset Toolset[Deps], retries int) Toolset[Deps] {
	if retries < 0 {
		panic(fmt.Sprintf("ai: toolset max retries must be non-negative, got %d", retries))
	}
	return defaultedToolset[Deps]{toolset: toolset, maxRetries: &retries}
}

// WithToolsetTimeout sets the default timeout for tools without an explicit
// WithToolTimeout option.
func WithToolsetTimeout[Deps any](toolset Toolset[Deps], timeout time.Duration) Toolset[Deps] {
	if timeout <= 0 {
		panic(fmt.Sprintf("ai: toolset timeout must be positive, got %s", timeout))
	}
	return defaultedToolset[Deps]{toolset: toolset, timeout: timeout}
}

type functionToolset[Deps any] struct {
	tools []Tool[Deps]
}

func (t functionToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools := cloneTools(t.tools)
	for index, tool := range tools {
		if tool.entry.prepare == nil {
			continue
		}
		definition, err := tool.entry.prepare(ctx, rc, cloneToolDefinition(tool.entry.def))
		if err != nil {
			return nil, fmt.Errorf("prepare tool %q: %w", tool.entry.def.Name, err)
		}
		if definition == nil {
			tools[index].entry.def.Name = ""
			continue
		}
		originalName := tool.entry.def.Name
		tool.entry.def = cloneToolDefinition(*definition)
		tool.entry.prepare = nil
		if tool.entry.def.Name != originalName {
			preparedName := tool.entry.def.Name
			tool.entry.def.Name = originalName
			tool = routeToolName(tool, preparedName)
		}
		tools[index] = tool
	}
	kept := tools[:0]
	for _, tool := range tools {
		if tool.entry.def.Name != "" {
			kept = append(kept, tool)
		}
	}
	return kept, nil
}

type combinedToolset[Deps any] struct {
	toolsets []Toolset[Deps]
}

func (combinedToolset[Deps]) relaysToolsetInstructions() {}

func (t combinedToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	var combined []Tool[Deps]
	seen := map[string]struct{}{}
	for _, toolset := range t.toolsets {
		tools, err := resolveToolsetTools(ctx, rc, toolset)
		if err != nil {
			return nil, err
		}
		for _, tool := range tools {
			name := tool.entry.def.Name
			if _, exists := seen[name]; exists {
				return nil, fmt.Errorf("duplicate toolset tool name %q", name)
			}
			seen[name] = struct{}{}
			combined = append(combined, cloneTool(tool))
		}
	}
	return combined, nil
}

func (t combinedToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	var instructions []InstructionPart
	owners := make(map[string]int)
	for index, toolset := range t.toolsets {
		parts, err := resolveToolsetInstructions(ctx, rc, toolset)
		if err != nil {
			return nil, err
		}
		if err := recordToolsetInstructionOwners(owners, index, parts); err != nil {
			return nil, err
		}
		instructions = append(instructions, parts...)
	}
	return instructions, nil
}

type filteredToolset[Deps any] struct {
	toolset Toolset[Deps]
	filter  ToolFilterFunc[Deps]
}

func (filteredToolset[Deps]) relaysToolsetInstructions() {}

func (t filteredToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	filtered := make([]Tool[Deps], 0, len(tools))
	for _, tool := range tools {
		keep, err := t.filter(ctx, rc, cloneToolDefinition(tool.entry.def))
		if err != nil {
			return nil, err
		}
		if keep {
			filtered = append(filtered, cloneTool(tool))
		}
	}
	return filtered, nil
}

func (t filteredToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

type prefixedToolset[Deps any] struct {
	toolset Toolset[Deps]
	prefix  string
}

func (prefixedToolset[Deps]) relaysToolsetInstructions() {}

func (t prefixedToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	for index := range tools {
		tools[index] = routeToolName(tools[index], t.prefix+"_"+tools[index].entry.def.Name)
	}
	return tools, nil
}

func (t prefixedToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

type renamedToolset[Deps any] struct {
	toolset Toolset[Deps]
	names   map[string]string
}

func (renamedToolset[Deps]) relaysToolsetInstructions() {}

func (t renamedToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]struct{}, len(tools))
	for index, tool := range tools {
		name := tool.entry.def.Name
		if renamed, ok := t.names[name]; ok {
			name = renamed
			tools[index] = routeToolName(tool, renamed)
		}
		if name == "" {
			return nil, fmt.Errorf("renamed tool name must not be empty")
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("renamed tool name %q conflicts with another tool", name)
		}
		seen[name] = struct{}{}
	}
	return tools, nil
}

func (t renamedToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

type preparedToolset[Deps any] struct {
	toolset Toolset[Deps]
	prepare ToolsPrepareFunc[Deps]
}

func (preparedToolset[Deps]) relaysToolsetInstructions() {}

func (t preparedToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	definitions := make([]ToolDefinition, len(tools))
	byName := make(map[string]Tool[Deps], len(tools))
	for index, tool := range tools {
		definitions[index] = cloneToolDefinition(tool.entry.def)
		byName[tool.entry.def.Name] = tool
	}
	prepared, err := t.prepare(ctx, rc, definitions)
	if err != nil {
		return nil, err
	}
	result := make([]Tool[Deps], 0, len(prepared))
	seen := make(map[string]struct{}, len(prepared))
	for _, definition := range prepared {
		tool, exists := byName[definition.Name]
		if !exists {
			return nil, fmt.Errorf("prepare toolset returned added or renamed tool %q", definition.Name)
		}
		if _, duplicate := seen[definition.Name]; duplicate {
			return nil, fmt.Errorf("prepare toolset returned duplicate tool %q", definition.Name)
		}
		seen[definition.Name] = struct{}{}
		tool.entry.def = cloneToolDefinition(definition)
		result = append(result, tool)
	}
	return result, nil
}

func (t preparedToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

type defaultedToolset[Deps any] struct {
	toolset    Toolset[Deps]
	maxRetries *int
	timeout    time.Duration
}

func (defaultedToolset[Deps]) relaysToolsetInstructions() {}

func (t defaultedToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	for index, tool := range tools {
		if t.maxRetries != nil && tool.entry.def.maxRetries == nil {
			retries := *t.maxRetries
			tool.entry.def.maxRetries = &retries
		}
		if t.timeout > 0 && tool.entry.def.timeout == 0 {
			tool.entry.def.timeout = t.timeout
		}
		tools[index] = tool
	}
	return tools, nil
}

func (t defaultedToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

type deferredToolset[Deps any] struct {
	toolset Toolset[Deps]
	names   map[string]struct{}
}

func (deferredToolset[Deps]) relaysToolsetInstructions() {}

func (t deferredToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	for index := range tools {
		if _, selected := t.names[tools[index].entry.def.Name]; t.names == nil || selected {
			tools[index].entry.def.DeferLoading = true
		}
	}
	return tools, nil
}

func (t deferredToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

type approvalRequiredToolset[Deps any] struct {
	toolset Toolset[Deps]
	names   map[string]struct{}
	check   ToolsetApprovalFunc[Deps]
}

func (approvalRequiredToolset[Deps]) relaysToolsetInstructions() {}

func (t approvalRequiredToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	for index := range tools {
		if _, selected := t.names[tools[index].entry.def.Name]; t.names != nil && !selected {
			continue
		}
		if t.check == nil {
			tools[index].entry.def.RequiresApproval = true
			continue
		}
		tool := tools[index]
		original := tool.entry.execute
		definition := cloneToolDefinition(tool.entry.def)
		tool.entry.def.DynamicApproval = true
		tool.entry.execute = func(
			ctx context.Context, rc *RunContext[Deps], args any,
		) (any, error) {
			if !rc.ToolCallApproved {
				rawArgs, err := json.Marshal(args)
				if err != nil {
					return nil, fmt.Errorf("marshal validated arguments: %w", err)
				}
				request, err := t.check(ctx, rc, cloneToolDefinition(definition), rawArgs)
				if err != nil {
					return nil, err
				}
				if request != nil {
					return RequestToolApproval(request.Metadata), nil
				}
			}
			return original(ctx, rc, args)
		}
		tools[index] = tool
	}
	return tools, nil
}

func (t approvalRequiredToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

type metadataToolset[Deps any] struct {
	toolset  Toolset[Deps]
	metadata map[string]any
}

func (metadataToolset[Deps]) relaysToolsetInstructions() {}

func (t metadataToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	for index, tool := range tools {
		metadata := cloneSchemaMap(tool.entry.def.Metadata)
		if metadata == nil {
			metadata = map[string]any{}
		}
		for key, value := range t.metadata {
			metadata[key] = cloneSchemaValue(value)
		}
		tool.entry.def.Metadata = metadata
		tools[index] = tool
	}
	return tools, nil
}

func (t metadataToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

func cloneTools[Deps any](tools []Tool[Deps]) []Tool[Deps] {
	cloned := make([]Tool[Deps], len(tools))
	for index, tool := range tools {
		cloned[index] = cloneTool(tool)
	}
	return cloned
}

func cloneTool[Deps any](tool Tool[Deps]) Tool[Deps] {
	tool.entry.def = cloneToolDefinition(tool.entry.def)
	return tool
}

func routeToolName[Deps any](tool Tool[Deps], name string) Tool[Deps] {
	originalName := tool.entry.def.Name
	validate := tool.entry.validate
	execute := tool.entry.execute
	tool.entry.def = cloneToolDefinition(tool.entry.def)
	tool.entry.def.Name = name
	tool.entry.validate = func(
		ctx context.Context, rc *RunContext[Deps], rawArgs json.RawMessage,
	) (any, error) {
		routed := *rc
		routed.ToolName = originalName
		return validate(ctx, &routed, rawArgs)
	}
	tool.entry.execute = func(ctx context.Context, rc *RunContext[Deps], args any) (any, error) {
		routed := *rc
		routed.ToolName = originalName
		return execute(ctx, &routed, args)
	}
	return tool
}

func resolveToolsetInstructions[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) ([]InstructionPart, error) {
	provider, ok := toolset.(ToolsetInstructionsProvider[Deps])
	if !ok {
		return nil, nil
	}
	parts, err := provider.ToolsetInstructions(ctx, rc)
	if err != nil {
		return nil, err
	}
	var source *InstructionSource
	if identified, ok := toolset.(ToolsetIDProvider); ok {
		if id := identified.ToolsetID(); id != "" {
			source = &InstructionSource{Kind: InstructionSourceToolset, ID: id}
		}
	}
	if source == nil {
		if _, relay := toolset.(toolsetInstructionRelay); !relay {
			parts = cloneInstructionParts(parts)
			for index := range parts {
				parts[index].ID = nil
			}
		}
	}
	qualified, err := qualifyInstructionParts(parts, source)
	if err != nil {
		return nil, err
	}
	return qualified, nil
}

func resolveToolsetTools[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) ([]Tool[Deps], error) {
	tools, err := toolset.Tools(ctx, rc)
	if err != nil {
		return nil, err
	}
	provider, ok := toolset.(ToolsetIDProvider)
	if !ok {
		return tools, nil
	}
	id := provider.ToolsetID()
	if id == "" {
		return tools, nil
	}
	for index := range tools {
		if tools[index].entry.def.ToolsetID == "" {
			tools[index].entry.def.ToolsetID = id
		}
	}
	return tools, nil
}
