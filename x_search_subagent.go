package ai

import (
	"context"
	"errors"
)

const defaultXSearchInstructions = "Search X/Twitter based on the user query. Return a comprehensive summary of the results."

// XSearchArgs is the model-facing input for the default local X-search tool.
type XSearchArgs struct {
	Query string `json:"query" jsonschema_description:"The X/Twitter search query."`
}

// XSearchFallbackModelFunc resolves the X-search subagent model for one tool call.
// It may run concurrently and must return a concurrency-safe model.
type XSearchFallbackModelFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps],
) (Model, error)

// XSearchSubagentConfig configures the default local X-search subagent.
type XSearchSubagentConfig[Deps any] struct {
	Model        Model
	ResolveModel XSearchFallbackModelFunc[Deps]
	Native       XSearchTool
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
		agent := NewAgent[struct{}, string](
			model,
			WithInstructions(instructions),
			WithCapabilities(NewXSearchCapability(XSearchCapabilityConfig[struct{}]{Native: native})),
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
	return NewNativeOrLocalTool(
		native,
		NewXSearchSubagentTool(config),
	)
}
