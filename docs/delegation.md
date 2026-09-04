# Agent delegation

Call a specialist agent from a parent tool. Return the specialist's usage with its result so the parent reports and enforces the combined work.

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type Deps struct {
	CustomerID string
}

type ResearchArgs struct {
	Question string `json:"question"`
}

func main() {
	researcher := ai.NewAgent[Deps, string](
		openai.NewModel("gpt-5-mini"),
		ai.WithAgentName("researcher"),
		ai.WithInstructions("Research the question. Return a concise factual answer."),
	)
	coordinator := ai.NewAgent[Deps, string](
		openai.NewModel("gpt-5"),
		ai.WithAgentName("coordinator"),
		ai.WithInstructions("Delegate research questions, then explain the result to the user."),
		ai.WithUsageLimits(ai.UsageLimits{RequestLimit: 8, TotalTokenLimit: 20_000}),
	)
	ai.AddTool(coordinator, "research", func(
		ctx context.Context,
		rc *ai.RunContext[Deps],
		args ResearchArgs,
	) (ai.ToolReturn, error) {
		result, err := researcher.Run(ctx, args.Question, rc.Deps)
		if err != nil {
			return ai.ToolReturn{}, err
		}
		return ai.ToolReturn{
			ReturnValue: result.Output,
			Usage:       result.Usage(),
		}, nil
	}, ai.WithDescription("Research a question with the specialist agent"))

	result, err := coordinator.Run(context.Background(), "How does HTTP conditional caching work?", Deps{
		CustomerID: "customer-123",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
	fmt.Printf("requests: %d, tokens: %d\n", result.Usage().Requests, result.Usage().TotalTokens())
}
```

`ToolReturn.Usage` adds every nested request, token category, cost, usage-detail key, and tool call to the parent run. The parent checks the combined usage before its next model request. This explicit return channel also works when a durable executor serializes and replays the tool result.

Return a detached `Usage` value. `RunResult.Usage()` already returns one. The parent clones usage details and costs instead of retaining data owned by the child result.

## Dependencies

Pass shared clients and immutable configuration through the parent's dependencies. A specialist can use the same `Deps` type, a field from it, or a new value built by the tool. Do not create a new network client for every delegated call.

The parent and specialist can use different output types. `ToolReturn.ReturnValue` accepts the specialist's typed output and sends it through the parent tool-result path.

## Limits

Set limits on both agents when the specialist must stop independently. Parent limits include usage returned by completed specialist calls, but they cannot interrupt work that the specialist has already performed. Give the specialist its own tighter limits when you must bound that work before it starts.

Concurrent delegated tools add usage safely. The final total includes each specialist once. Streamed runs publish the updated total after each delegated tool finishes.

## Errors and cancellation

Pass the tool's `context.Context` into the specialist. A parent deadline or cancellation then stops the entire active call tree. The specialist still owns its derived run cancellation, so cancelling only that run does not directly cancel its parent.

Returning an ordinary specialist error aborts the parent run. Convert a recoverable specialist failure to `ToolFailedf` when the parent model should receive the failure and choose another action. Preserve `ErrUsageLimitExceeded` when a limit must remain terminal.

Name both agents with `WithAgentName`. Instrumentation then distinguishes parent and specialist spans without changing the delegation API.
