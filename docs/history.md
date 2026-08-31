# Message history

## Trim history by input tokens

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "My deployment region is eu-west-1."},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "I will remember that region."},
		}},
	}

	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5-mini"),
		ai.WithCapabilities(ai.TokenHistoryTrimmer{
			MaxInputTokens:     8_000,
			MinimumRecentTurns: 2,
		}),
	)
	result, err := agent.Run(
		context.Background(),
		"Which region should I deploy to?",
		struct{}{},
		ai.WithMessageHistory(history),
	)
	if errors.Is(err, ai.ErrHistoryTokenLimitExceeded) {
		var limitErr *ai.HistoryTokenLimitError
		if errors.As(err, &limitErr) {
			fmt.Printf("protected history needs %d tokens\n", limitErr.Usage.InputTokens)
		}
		return
	}
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`TokenHistoryTrimmer` counts the complete prospective request. The count includes instructions, tools, and output schemas. It removes the oldest complete user turns with a binary search until the request fits `MaxInputTokens`.

The current run's turn is always protected. `MinimumRecentTurns` defaults to one. Keeping whole turns prevents a function-tool call from being separated from its return value.

By itself, trimming changes only the request snapshot. `Result.Messages()` still contains the complete durable history. You can persist that history and apply a different limit on the next run.

A preceding model-request hook can explicitly set `ModelRequestContext.ReplaceHistory`. In that composition, the final processed snapshot becomes durable.

`ErrHistoryTokenLimitExceeded` means the protected turns and request configuration exceed the limit. Inspect `HistoryTokenLimitError.Usage` for the smallest count the trimmer could send.

> [!NOTE]
> Token-aware trimming calls the selected model's token-counting endpoint more than once when trimming is needed. This adds latency and may consume a separate provider rate limit.

The selected model must implement `TokenCountingModel`. OpenAI Responses, Anthropic Messages, Gemini Developer API, and Vertex AI provide token counting. See [Usage limits and token counting](usage.md) for endpoint details and unsupported-model handling.

## Summarize old turns

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	summaryAgent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5-mini"),
		ai.WithInstructions("Preserve decisions, constraints, and unresolved work."),
	)
	summarizer := ai.HistorySummarizer{
		MinimumRecentTurns: 3,
		Summarize: func(
			ctx context.Context,
			_ *ai.RunInfo,
			messages []ai.ModelMessage,
		) (ai.HistorySummary, error) {
			result, err := summaryAgent.Run(
				ctx,
				"Summarize the preceding conversation.",
				struct{}{},
				ai.WithMessageHistory(messages),
			)
			if err != nil {
				return ai.HistorySummary{}, err
			}
			return ai.HistorySummary{Content: result.Output, Usage: result.Usage()}, nil
		},
	}

	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5"),
		ai.WithCapabilities(summarizer),
	)
	result, err := agent.Run(context.Background(), "Continue the design review.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`HistorySummarizer` sends only complete old user turns to your callback. It replaces them with one portable assistant text response. The replacement is durable, so a tool loop reuses the summary instead of paying to generate it again.

Return the summary model's `Usage`. The outer run adds that usage before its primary request and applies `UsageLimits` immediately. This prevents a hidden summarization request from escaping your application budget.

The callback receives detached messages and can use any provider or summarization service. It must be safe when concurrent runs use the same agent.

## Write a custom history processor

```go
package main

import (
	"context"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	keepLatest := ai.HistoryProcessor(func(
		_ context.Context,
		_ *ai.RunInfo,
		messages []ai.ModelMessage,
	) ([]ai.ModelMessage, error) {
		if len(messages) <= 5 {
			return messages, nil
		}
		return messages[len(messages)-5:], nil
	})

	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5-mini"),
		ai.WithCapabilities(keepLatest),
	)
	if _, err := agent.Run(context.Background(), "Continue.", struct{}{}); err != nil {
		log.Fatal(err)
	}
}
```

A `HistoryProcessor` receives detached messages before each request. Processors compose in capability order. Their results do not replace durable history unless another model-request hook sets `ModelRequestContext.ReplaceHistory`.

A message-count slice can split tool calls from their results. Use `TokenHistoryTrimmer` when you need automatic turn-safe boundaries.
