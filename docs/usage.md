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
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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

OpenAI Chat Completions, OpenAI Codex subscription models, and Z.AI do not expose compatible token-counting endpoints. They return `ErrTokenCountingUnsupported` when pre-request counting is enabled.

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
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
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

## Inspect context-window usage

```go
package main

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func main() {
	profile := openai.NewModel("gpt-5").ModelProfile()
	fmt.Println(profile.ContextWindow)
}
```

Bundled provider models resolve `ModelProfile.ContextWindow` from the `genai-prices` v0.1.6 snapshot. The lookup uses the model's provider identity, so a model name from another provider does not supply a limit. A zero value means the provider or model is unknown, or its metadata does not specify a limit. An explicit `ai.NewProfiledModel` profile always wins, including an explicit zero.

`RunContext.ContextWindowUsed` and `RunInfo.ContextWindowUsed` divide the latest response token count by this window. A fallback model uses the smallest known candidate window. Both methods return `known=false` when either value is unavailable.

## Per-request and cumulative limits

`PerRequestInputTokenLimit` caps one context window. `InputTokenLimit` accumulates input tokens over the run. A tool loop can stay below the per-request limit while exceeding the cumulative limit.

Without pre-request counting, `PerRequestInputTokenLimit` uses the input count reported by each response. A suspended continuation is checked conservatively against its combined input usage.

All zero numeric limits are disabled. `ToolCallLimit` and `CostLimitUSD` use pointers so you can enforce a zero limit explicitly.

## Keep pricing data current

```go
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	updater := ai.UpdatePricesInBackground(ctx, ai.PriceUpdateConfig{})
	defer updater.Stop()

	<-ctx.Done()
}
```

`UpdatePricesInBackground` downloads the current `genai-prices` data immediately. It then refreshes the data every hour. Downloads do not block your caller. `ModelResponse.Price`, automatic cost calculation, image pricing, and embedding pricing use the latest valid snapshot.

A failed download leaves the last valid snapshot in use. The updater accepts only HTTP 200 responses and limits response bodies to 8 MiB by default. Set `PriceUpdateConfig.OnError` when you need to report failures. The library does not log failures.

Set `PriceUpdateConfig.HTTPClient`, `URL`, `Interval`, or `MaxBodyBytes` to control downloads. A supplied HTTP client remains yours. The updater does not mutate it or close its idle connections.

Call `Stop` during shutdown. It is safe to call more than once. Identical configurations without an error callback share one download worker. The worker remains active until every subscribing updater stops or its context is canceled.
