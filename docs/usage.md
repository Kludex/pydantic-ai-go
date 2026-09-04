# Usage limits and token counting

Use `UsageLimits` to stop a run before it exceeds an application budget.

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	costLimit := 0.05
	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5-mini"),
		ai.WithUsageLimits(ai.UsageLimits{
			RequestLimit:              5,
			InputTokenLimit:           20_000,
			OutputTokenLimit:          2_000,
			TotalTokenLimit:           22_000,
			PerRequestInputTokenLimit: 8_000,
			CountTokensBeforeRequest:  true,
			CostLimitUSD:              &costLimit,
		}),
	)

	result, err := agent.Run(context.Background(), "Explain structured concurrency.", struct{}{})
	if errors.Is(err, ai.ErrUsageLimitExceeded) {
		fmt.Println("The run reached its budget.")
		return
	}
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("tokens=%d cost=%v\n", result.Usage().TotalTokens(), result.Usage().CostUSD)
}
```

`RequestLimit` is checked before generation. A limit of one permits exactly one model request.

Token and cost limits are checked after each response by default. The provider has already processed that request when a post-response limit fails.

## Count before generation

Set `CountTokensBeforeRequest` to call the selected model's token-counting endpoint before generation. This prevents an oversized request from being sent.

The bundled implementations are:

| Model | Endpoint |
| --- | --- |
| OpenAI Responses | `/responses/input_tokens` |
| Anthropic Messages | `/messages/count_tokens` |
| Gemini Developer API and Vertex AI | `:countTokens` |

OpenAI Chat Completions and Z.AI do not expose compatible token-counting endpoints. They return `ErrTokenCountingUnsupported` when pre-request counting is enabled.

Pre-request counting uses the final request after model hooks and capability middleware. It includes prepared tools, output schemas, instructions, and the request-only history view. The count is a projection. It is not added to `Result.Usage()`.

The projected input cost is priced with `genai-prices` when pricing data is available. This lets `CostLimitUSD` reject a request whose input alone exceeds the remaining budget. The final response still determines authoritative run cost.

> [!NOTE]
> Token counting adds one provider operation. It adds latency and may be rate-limited separately. Keep it disabled when post-response limits are sufficient.

## Count a request directly

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func main() {
	model := google.NewModel("gemini-2.5-flash")
	usage, err := ai.CountModelTokens(
		context.Background(),
		model,
		[]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "Explain structured concurrency."},
		}}},
		ai.ModelRequestParams{Instructions: "Answer concisely."},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(usage.InputTokens)
}
```

`CountModelTokens` validates settings and passes detached messages, schemas, tools, headers, and nested settings to the provider. `RequestTimeout` covers the complete count operation.

Use `errors.Is(err, ai.ErrTokenCountingUnsupported)` when a model may not support counting.

## Per-request and cumulative limits

`PerRequestInputTokenLimit` caps one context window. `InputTokenLimit` accumulates input tokens over the run. A tool loop can stay below the per-request limit while exceeding the cumulative limit.

Without pre-request counting, `PerRequestInputTokenLimit` uses the input count reported by each response. A suspended continuation is checked conservatively against its combined input usage.

All zero numeric limits are disabled. `ToolCallLimit` and `CostLimitUSD` use pointers so you can enforce a zero limit explicitly.
