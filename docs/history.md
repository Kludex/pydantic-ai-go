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

Trimming changes only the request snapshot. `Result.Messages()` still contains the complete durable history. You can persist that history and apply a different limit on the next run.

`ErrHistoryTokenLimitExceeded` means the protected turns and request configuration exceed the limit. Inspect `HistoryTokenLimitError.Usage` for the smallest count the trimmer could send.

> [!NOTE]
> Token-aware trimming calls the selected model's token-counting endpoint more than once when trimming is needed. This adds latency and may consume a separate provider rate limit.

The selected model must implement `TokenCountingModel`. OpenAI Responses, Anthropic Messages, Gemini Developer API, and Vertex AI provide token counting. See [Usage limits and token counting](usage.md) for endpoint details and unsupported-model handling.

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
