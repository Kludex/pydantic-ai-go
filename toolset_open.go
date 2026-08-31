package ai

import (
	"context"
	"errors"
)

func openToolset[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) (Toolset[Deps], ToolsetCloseFunc, error) {
	if opener, ok := toolset.(ToolsetOpener[Deps]); ok {
		opened, closeFunc, err := opener.OpenToolset(ctx, rc)
		opened, err = validToolsetReplacement(opened, err, "OpenToolset")
		return opened, closeFunc, err
	}
	if combined, ok := toolset.(combinedToolset[Deps]); ok {
		opened := make([]Toolset[Deps], len(combined.toolsets))
		closers := make([]ToolsetCloseFunc, 0, len(combined.toolsets))
		for index, child := range combined.toolsets {
			var closeFunc ToolsetCloseFunc
			var err error
			opened[index], closeFunc, err = openToolset(ctx, rc, child)
			if err != nil {
				return nil, nil, errors.Join(err, closeToolsetFuncs(ctx, closers))
			}
			if closeFunc != nil {
				closers = append(closers, closeFunc)
			}
		}
		combined.toolsets = opened
		return combined, func(ctx context.Context) error {
			return closeToolsetFuncs(ctx, closers)
		}, nil
	}
	if opened, closeFunc, wrapped, err := openWrappedToolset(ctx, rc, toolset); wrapped || err != nil {
		return opened, closeFunc, err
	}
	return toolset, nil, nil
}

func openWrappedToolset[Deps any](
	ctx context.Context, rc *RunContext[Deps], toolset Toolset[Deps],
) (Toolset[Deps], ToolsetCloseFunc, bool, error) {
	var wrapped Toolset[Deps]
	var replace func(Toolset[Deps]) Toolset[Deps]
	switch wrapper := toolset.(type) {
	case filteredToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case prefixedToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case renamedToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case preparedToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case defaultedToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case metadataToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case deferredToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case approvalRequiredToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	case toolSearchToolset[Deps]:
		wrapped = wrapper.toolset
		replace = func(toolset Toolset[Deps]) Toolset[Deps] {
			wrapper.toolset = toolset
			return wrapper
		}
	default:
		return nil, nil, false, nil
	}
	opened, closeFunc, err := openToolset(ctx, rc, wrapped)
	if err != nil {
		return nil, nil, true, err
	}
	return replace(opened), closeFunc, true, nil
}

func closeToolsetFuncs(ctx context.Context, closers []ToolsetCloseFunc) error {
	var errs []error
	for index := len(closers) - 1; index >= 0; index-- {
		if err := closers[index](ctx); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
