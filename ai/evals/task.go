package evals

import (
	"context"
	"errors"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	pydanticevals "github.com/Kludex/pydantic-evals-go"
)

// ErrDeferredRun reports that an evaluation case paused for approval or
// external tool execution. Evaluation tasks must produce a final output.
var ErrDeferredRun = errors.New("ai/evals: agent run is deferred")

// AgentInput is one prepared agent invocation for an evaluation case.
type AgentInput[Deps any] struct {
	// Prompt is the text or multimodal input for the case.
	Prompt ai.UserPromptPart
	// Deps supplies the agent's typed run dependencies.
	Deps Deps
	// Options applies case-specific run configuration.
	Options []ai.RunOption
}

// InputMapper converts one dataset input into a detached agent invocation. It
// must be safe for concurrent calls when the dataset runs cases concurrently.
type InputMapper[Input, Deps any] func(
	ctx context.Context, input Input,
) (AgentInput[Deps], error)

// NewTask adapts an agent to a Pydantic Evals task. Successful tasks return the
// agent output and record usage, run identity, model identity, and application
// metadata on the evaluation case.
func NewTask[Input, Deps, Output any](
	agent *ai.Agent[Deps, Output], mapper InputMapper[Input, Deps],
) pydanticevals.TaskFunc[Input, Output] {
	if agent == nil {
		panic("ai/evals: agent must not be nil")
	}
	if mapper == nil {
		panic("ai/evals: input mapper must not be nil")
	}
	return func(ctx context.Context, input Input) (Output, error) {
		prepared, err := mapper(ctx, input)
		if err != nil {
			var zero Output
			return zero, fmt.Errorf("ai/evals: map task input: %w", err)
		}
		if prepared.Prompt.Content != "" && len(prepared.Prompt.Contents) > 0 {
			var zero Output
			return zero, errors.New("ai/evals: prompt cannot contain both text Content and multimodal Contents")
		}
		var result *ai.RunResult[Output]
		if len(prepared.Prompt.Contents) > 0 {
			result, err = agent.RunParts(ctx, prepared.Prompt.Contents, prepared.Deps, prepared.Options...)
		} else {
			result, err = agent.Run(ctx, prepared.Prompt.Content, prepared.Deps, prepared.Options...)
		}
		if err != nil {
			var zero Output
			return zero, err
		}
		recordResult(ctx, result)
		if deferred := result.Deferred(); deferred != nil {
			var zero Output
			return zero, fmt.Errorf(
				"%w: %d tool calls need completion",
				ErrDeferredRun, len(deferred.Calls)+len(deferred.Approvals),
			)
		}
		return result.Output, nil
	}
}

// NewTextTask adapts string dataset inputs to agent text prompts. Deps and
// options are reused across concurrent cases and must be safe to share.
func NewTextTask[Deps, Output any](
	agent *ai.Agent[Deps, Output], deps Deps, options ...ai.RunOption,
) pydanticevals.TaskFunc[string, Output] {
	return NewTask(agent, func(_ context.Context, input string) (AgentInput[Deps], error) {
		return AgentInput[Deps]{
			Prompt: ai.UserPromptPart{Content: input}, Deps: deps,
			Options: append([]ai.RunOption(nil), options...),
		}, nil
	})
}

func recordResult[Output any](ctx context.Context, result *ai.RunResult[Output]) {
	usage := result.Usage()
	pydanticevals.SetAttribute(ctx, "pydantic_ai.usage", usage)
	recordMetric(ctx, "pydantic_ai.requests", usage.Requests)
	recordMetric(ctx, "pydantic_ai.tool_calls", usage.ToolCalls)
	recordMetric(ctx, "pydantic_ai.input_tokens", usage.InputTokens)
	recordMetric(ctx, "pydantic_ai.output_tokens", usage.OutputTokens)
	recordMetric(ctx, "pydantic_ai.total_tokens", usage.TotalTokens())
	recordMetric(ctx, "pydantic_ai.cache_write_tokens", usage.CacheWriteTokens)
	recordMetric(ctx, "pydantic_ai.cache_read_tokens", usage.CacheReadTokens)
	recordMetric(ctx, "pydantic_ai.input_audio_tokens", usage.InputAudioTokens)
	recordMetric(ctx, "pydantic_ai.cache_audio_read_tokens", usage.CacheAudioReadTokens)
	recordMetric(ctx, "pydantic_ai.output_audio_tokens", usage.OutputAudioTokens)
	recordMetric(ctx, "pydantic_ai.reasoning_tokens", usage.ReasoningTokens)
	recordMetric(ctx, "pydantic_ai.accepted_prediction_tokens", usage.AcceptedPredictionTokens)
	recordMetric(ctx, "pydantic_ai.rejected_prediction_tokens", usage.RejectedPredictionTokens)
	if usage.CostUSD != nil {
		pydanticevals.IncrementMetric(ctx, "pydantic_ai.cost_usd", *usage.CostUSD)
	}
	for name, value := range usage.Details {
		recordMetric(ctx, "pydantic_ai.usage.details."+name, value)
	}
	if metadata := result.Metadata(); len(metadata) > 0 {
		pydanticevals.SetAttribute(ctx, "pydantic_ai.metadata", metadata)
	}
	messages := result.Messages()
	response := messages[len(messages)-1].(ai.ModelResponse)
	recordResponseAttributes(ctx, response)
}

func recordMetric(ctx context.Context, name string, value int) {
	if value != 0 {
		pydanticevals.IncrementMetric(ctx, name, float64(value))
	}
}

func recordResponseAttributes(ctx context.Context, response ai.ModelResponse) {
	recordIdentityAttributes(ctx, response.RunID, response.ConversationID)
	for name, value := range map[string]string{
		"pydantic_ai.model_name":           response.ModelName,
		"pydantic_ai.provider_name":        response.ProviderName,
		"pydantic_ai.provider_response_id": response.ProviderResponseID,
	} {
		if value != "" {
			pydanticevals.SetAttribute(ctx, name, value)
		}
	}
}

func recordIdentityAttributes(ctx context.Context, runID, conversationID string) {
	if runID != "" {
		pydanticevals.SetAttribute(ctx, "pydantic_ai.run_id", runID)
	}
	if conversationID != "" {
		pydanticevals.SetAttribute(ctx, "pydantic_ai.conversation_id", conversationID)
	}
}
