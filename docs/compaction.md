# Compact model history

## Compact automatically

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	model := openai.NewResponsesModel("gpt-5.4")
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithCapabilities(openai.NewCompaction(
			openai.WithCompactionMessageCountThreshold(20),
		)),
	)

	result, err := agent.Run(
		context.Background(),
		"Summarize the decisions so far.",
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Stateless compaction calls OpenAI's `/responses/compact` endpoint before the primary request when the history exceeds the threshold. It replaces older durable history with one opaque `CompactionPart`. The current user turn remains outside the compacted window.

Compaction is a side request. Its tokens and cost count toward the run's usage limits before the primary request starts. This prevents compaction from silently exceeding an application budget.

Use `openai.WithCompactionTrigger` when a message count is not enough. The trigger receives a detached history snapshot and may run concurrently across agent runs.

## Use provider-managed compaction

OpenAI Responses can ask the server to compact when its input reaches a token threshold:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5.4"),
		ai.WithCapabilities(openai.NewCompaction(
			openai.WithCompactionTokenThreshold(100_000),
		)),
	)
	result, err := agent.Run(context.Background(), "Continue the analysis.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

The zero-value OpenAI compaction mode is stateful unless you configure a message-count threshold or trigger. Stateful mode sends `context_management` with the normal response request. It does not make a separate `/responses/compact` call.

Anthropic exposes the same capability shape with provider-specific controls:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-6"),
		ai.WithCapabilities(anthropic.NewCompaction(
			anthropic.WithCompactionTokenThreshold(100_000),
			anthropic.WithCompactionInstructions("Keep decisions and unresolved questions."),
		)),
	)
	result, err := agent.Run(context.Background(), "Continue the analysis.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Anthropic requires a threshold of at least 50,000 tokens. Its default is 150,000 tokens.

## Call a compaction endpoint directly

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "Compare the two deployment plans."},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "Plan A favors simplicity. Plan B favors isolation."},
		}},
	}

	response, err := ai.CompactModelMessages(
		context.Background(),
		openai.NewResponsesModel("gpt-5.4"),
		history,
		ai.ModelRequestParams{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Compacted with %d input tokens.\n", response.Usage.InputTokens)
}
```

`CompactModelMessages` accepts any `ModelCompactor`. It clones messages, settings, schemas, and the returned response. It also applies `ModelSettings.RequestTimeout`.

The helper returns `ErrCompactionUnsupported` when the selected model has no explicit compaction endpoint. Provider-managed compaction does not implement this direct operation because it runs as part of ordinary generation.

## Trace compaction

`NewInstrumentation` and `NewInstrumentedModel` create a `compact <model>` client span for every explicit compaction request. The span uses `gen_ai.operation.name=compact`. It includes detached input and output message attributes under the same privacy controls as ordinary model requests.

Automatic stateless compaction is a child of the agent run span. It remains separate from the following `chat <model>` span, so you can attribute latency, tokens, and cost to the correct request.
