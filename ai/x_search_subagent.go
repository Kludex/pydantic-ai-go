package ai

import (
	"context"
	"errors"
)

const defaultXSearchInstructions = "Search X/Twitter based on the user query. Return a comprehensive summary of the results."

// XSearchArgs is the model-facing input for the default local X-search tool.
type XSearchArgs struct {
	// Query describes the posts or topics to find.
	Query string `json:"query" jsonschema_description:"The X/Twitter search query."`
}

// XSearchFallbackModelFunc resolves the X-search subagent model for one tool call.
// It may run concurrently and must return a concurrency-safe model.
type XSearchFallbackModelFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps],
) (Model, error)

// XSearchSubagentConfig configures the default local X-search subagent.
type XSearchSubagentConfig[Deps any] struct {
	// Model is the fixed X-search subagent model.
	Model Model
	// ResolveModel selects an X-search model for each call.
	ResolveModel XSearchFallbackModelFunc[Deps]
	// Native configures the subagent's provider-hosted X-search tool.
	Native XSearchTool
	// ResolveNative resolves X-search settings from the outer tool-call context.
	ResolveNative XSearchFunc[Deps]
	// Instructions overrides the default X-search prompt.
	Instructions string
}

// NewXSearchSubagentTool creates an x_search tool backed by a native X-search subagent.
func NewXSearchSubagentTool[Deps any](config XSearchSubagentConfig[Deps]) Tool[Deps] {
	if modelIsNil(config.Model) == (config.ResolveModel == nil) {
		panic("ai: X-search subagent requires exactly one model or model resolver")
	}
	native := cloneNativeTool(config.Native).(XSearchTool)
	instructions := config.Instructions
	if instructions == "" {
		instructions = defaultXSearchInstructions
	}
	return NewTool("x_search", func(
		ctx context.Context, rc *RunContext[Deps], args XSearchArgs,
	) (string, error) {
		model := config.Model
		if config.ResolveModel != nil {
			var err error
			model, err = config.ResolveModel(ctx, rc.clone())
			if err != nil {
				return "", err
			}
			if modelIsNil(model) {
				return "", errors.New("ai: X-search model resolver returned nil")
			}
		}
		resolvedNative := native
		if config.ResolveNative != nil {
			var err error
			resolvedNative, err = config.ResolveNative(ctx, rc.clone())
			if err != nil {
				return "", err
			}
		}
		agent := NewAgent[struct{}, string](
			model,
			WithInstructions(instructions),
			WithCapabilities(NewXSearchCapability(XSearchCapabilityConfig[struct{}]{Native: resolvedNative})),
		)
		result, err := agent.Run(ctx, args.Query, struct{}{})
		if err != nil {
			var unexpected *UnexpectedModelBehaviorError
			if errors.As(err, &unexpected) || errors.Is(err, ErrMaxRetriesExceeded) {
				return "", Retryf("%v", err)
			}
			return "", err
		}
		if result.Output == "" {
			return "", Retryf("X-search subagent returned no summary")
		}
		return result.Output, nil
	}, WithDescription("Search X/Twitter for posts and content based on the given query."))
}

// NewXSearchCapabilityWithFallback creates native-first X search with a subagent fallback.
// The fallback subagent enforces the same handle, date, media-understanding, and output settings.
func NewXSearchCapabilityWithFallback[Deps any](config XSearchSubagentConfig[Deps]) *NativeOrLocalTool[Deps] {
	native := cloneNativeTool(config.Native).(XSearchTool)
	capability := NewNativeOrLocalTool(
		native,
		NewXSearchSubagentTool(config),
	)
	capability.registration.rebuildLocal = func(native NativeTool) Toolset[Deps] {
		updated := config
		updated.Native = native.(XSearchTool)
		return NewFunctionToolset(NewXSearchSubagentTool(updated))
	}
	return capability
}

// NewDynamicXSearchCapabilityWithFallback creates dynamic native-first X search
// with a subagent that resolves the same settings when used.
func NewDynamicXSearchCapabilityWithFallback[Deps any](
	resolve XSearchFunc[Deps], config XSearchSubagentConfig[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if resolve == nil {
		panic("ai: dynamic X-search resolver must not be nil")
	}
	config.ResolveNative = resolve
	return NewDynamicNativeOrLocalToolset(
		"x_search",
		func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error) { return resolve(ctx, rc) },
		NewFunctionToolset(NewXSearchSubagentTool(config)),
		options...,
	)
}
