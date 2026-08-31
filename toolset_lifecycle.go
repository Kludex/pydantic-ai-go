package ai

import (
	"context"
	"fmt"
	"reflect"
)

func toolsetForRun[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) (Toolset[Deps], error) {
	if provider, ok := toolset.(ToolsetRunProvider[Deps]); ok {
		resolved, err := provider.ForRun(ctx, rc)
		return validToolsetReplacement(resolved, err, "ForRun")
	}
	if combined, ok := toolset.(combinedToolset[Deps]); ok {
		resolved := make([]Toolset[Deps], len(combined.toolsets))
		for index, child := range combined.toolsets {
			var err error
			resolved[index], err = toolsetForRun(ctx, rc, child)
			if err != nil {
				return nil, err
			}
		}
		combined.toolsets = resolved
		return combined, nil
	}
	if resolved, wrapped, err := wrappedToolsetForRun(ctx, rc, toolset); wrapped || err != nil {
		return resolved, err
	}
	return toolset, nil
}

func wrappedToolsetForRun[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) (Toolset[Deps], bool, error) {
	switch wrapper := toolset.(type) {
	case filteredToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case prefixedToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case renamedToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case preparedToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case defaultedToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case metadataToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case deferredToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case approvalRequiredToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case toolSearchToolset[Deps]:
		resolved, err := toolsetForRun(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	default:
		return nil, false, nil
	}
}

func toolsetForRunStep[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) (Toolset[Deps], error) {
	if provider, ok := toolset.(ToolsetStepProvider[Deps]); ok {
		resolved, err := provider.ForRunStep(ctx, rc)
		return validToolsetReplacement(resolved, err, "ForRunStep")
	}
	if combined, ok := toolset.(combinedToolset[Deps]); ok {
		resolved := make([]Toolset[Deps], len(combined.toolsets))
		for index, child := range combined.toolsets {
			var err error
			resolved[index], err = toolsetForRunStep(ctx, rc, child)
			if err != nil {
				return nil, err
			}
		}
		combined.toolsets = resolved
		return combined, nil
	}
	if resolved, wrapped, err := wrappedToolsetForRunStep(ctx, rc, toolset); wrapped || err != nil {
		return resolved, err
	}
	return toolset, nil
}

func wrappedToolsetForRunStep[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) (Toolset[Deps], bool, error) {
	switch wrapper := toolset.(type) {
	case filteredToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case prefixedToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case renamedToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case preparedToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case defaultedToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case metadataToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case deferredToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case approvalRequiredToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	case toolSearchToolset[Deps]:
		resolved, err := toolsetForRunStep(ctx, rc, wrapper.toolset)
		wrapper.toolset = resolved
		return wrapper, true, err
	default:
		return nil, false, nil
	}
}

func validToolsetReplacement[Deps any](
	toolset Toolset[Deps], err error, method string,
) (Toolset[Deps], error) {
	if err != nil {
		return nil, err
	}
	if toolsetIsNil(toolset) {
		return nil, fmt.Errorf("toolset %s returned nil", method)
	}
	return toolset, nil
}

func toolsetIsNil[Deps any](toolset Toolset[Deps]) bool {
	if toolset == nil {
		return true
	}
	value := reflect.ValueOf(toolset)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
