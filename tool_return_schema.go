package ai

import (
	"context"
	"encoding/json"
	"fmt"
)

// ToolReturnSchemaSelector selects tool definitions using a detached snapshot.
// Implementations may inspect run metadata and must be safe for concurrent runs.
type ToolReturnSchemaSelector func(
	ctx context.Context, runInfo *RunInfo, definition ToolDefinition,
) (bool, error)

// IncludeToolReturnSchemas enables model-facing return schemas. A nil Select
// enables every tool. Explicit per-tool inclusion or omission takes precedence.
type IncludeToolReturnSchemas struct {
	Select ToolReturnSchemaSelector
}

// Setup implements Capability.
func (IncludeToolReturnSchemas) Setup(*CapabilityRegistry) error { return nil }

// BeforeModelRequest enables selected schemas on fresh request definitions.
func (capability IncludeToolReturnSchemas) BeforeModelRequest(
	ctx context.Context, runInfo *RunInfo, request ModelRequestContext,
) (ModelRequestContext, error) {
	selected := make(map[string]bool)
	resolve := func(definition ToolDefinition) (ToolDefinition, error) {
		if definition.IncludeReturnSchema != nil {
			return definition, nil
		}
		included, exists := selected[definition.Name]
		if !exists {
			included = true
			if capability.Select != nil {
				var err error
				included, err = capability.Select(ctx, runInfo, cloneToolDefinition(definition))
				if err != nil {
					return ToolDefinition{}, err
				}
			}
			selected[definition.Name] = included
		}
		definition.IncludeReturnSchema = clonePointer(&included)
		return definition, nil
	}
	var err error
	for index := range request.Params.Tools {
		request.Params.Tools[index], err = resolve(request.Params.Tools[index])
		if err != nil {
			return ModelRequestContext{}, err
		}
	}
	for index := range request.Params.DeferredTools {
		request.Params.DeferredTools[index], err = resolve(request.Params.DeferredTools[index])
		if err != nil {
			return ModelRequestContext{}, err
		}
	}
	return request, nil
}

// PrepareToolReturnSchema resolves a definition for a provider. Providers with
// native support retain ReturnSchema. Other providers receive it as formatted
// description text.
func PrepareToolReturnSchema(definition ToolDefinition, native bool) (ToolDefinition, error) {
	definition = cloneToolDefinition(definition)
	if definition.IncludeReturnSchema == nil || !*definition.IncludeReturnSchema || len(definition.ReturnSchema) == 0 {
		definition.ReturnSchema = nil
		return definition, nil
	}
	if native {
		return definition, nil
	}
	encoded, err := json.MarshalIndent(definition.ReturnSchema, "", "  ")
	if err != nil {
		return ToolDefinition{}, fmt.Errorf("ai: marshal return schema for tool %q: %w", definition.Name, err)
	}
	if definition.Description != "" {
		definition.Description += "\n\n"
	}
	definition.Description += "Return schema:\n\n" + string(encoded)
	definition.ReturnSchema = nil
	return definition, nil
}
