# Evaluate an agent

```go
package main

import (
	"context"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aievals "github.com/Kludex/pydantic-ai-go/evals"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	pydanticevals "github.com/Kludex/pydantic-evals-go"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5-mini"),
		ai.WithInstructions("Answer with only the city name."),
	)
	task := aievals.NewTextTask(agent, struct{}{})

	suite := pydanticevals.For[string, string, any]()
	dataset := suite.Dataset(
		"capitals",
		suite.Case("What is the capital of France?").Name("france").Expect("Paris"),
	).With(suite.EqualsExpected())

	report, err := dataset.Evaluate(
		context.Background(),
		task,
		pydanticevals.Config{Name: "capital-agent"},
	)
	if err != nil {
		log.Fatal(err)
	}
	report.Print(pydanticevals.RenderOptions{
		IncludeInput: true, IncludeOutput: true, IncludeAverages: true,
	})
}
```

`NewTextTask` turns each string case input into an agent prompt. The task returns the typed agent output, so your evaluators use the same output type as the agent.

Pydantic Evals runs cases concurrently by default. Reused dependencies and run options must be safe for concurrent calls. Agent configuration remains detached between cases.

## Recorded run data

The adapter records agent usage as evaluation metrics:

| Metric | Value |
| --- | --- |
| `pydantic_ai.requests` | Model generation requests |
| `pydantic_ai.tool_calls` | Successful local tool calls |
| `pydantic_ai.input_tokens` | Input tokens |
| `pydantic_ai.output_tokens` | Output tokens |
| `pydantic_ai.total_tokens` | Input plus output tokens |
| `pydantic_ai.cost_usd` | Cost when pricing is available |
| `pydantic_ai.usage.details.<name>` | Provider-specific usage counters |

Cache, audio, reasoning, and prediction token metrics use their `Usage` field names with the `pydantic_ai.` prefix.

The adapter also records the complete detached `Usage` value as `pydantic_ai.usage`. This preserves the difference between unavailable cost and a known zero cost even though zero-valued report metrics are omitted.

Run ID, conversation ID, model name, provider name, provider response ID, and application metadata are recorded as case attributes. Configure `NewInstrumentation` on the agent when you also need the complete nested run, request, and tool span tree.

## Map typed inputs

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aievals "github.com/Kludex/pydantic-ai-go/evals"
	"github.com/Kludex/pydantic-ai-go/models/google"
	pydanticevals "github.com/Kludex/pydantic-evals-go"
)

type Input struct {
	Question string
	ImageURL string
}

type Deps struct {
	Locale string
}

func main() {
	agent := ai.NewAgent[Deps, string](google.NewModel("gemini-2.5-flash"))
	task := aievals.NewTask(agent, func(
		_ context.Context,
		input Input,
	) (aievals.AgentInput[Deps], error) {
		return aievals.AgentInput[Deps]{
			Prompt: ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: input.Question},
				ai.ImageURL{URL: input.ImageURL},
			}},
			Deps: Deps{Locale: "en-GB"},
		}, nil
	})

	suite := pydanticevals.For[Input, string, any]()
	dataset := suite.Dataset("diagrams", suite.Case(Input{
		Question: "What colour is the status indicator?",
		ImageURL: "https://example.com/status.png",
	}))
	report, err := dataset.Evaluate(context.Background(), task)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(report.Cases[0].Output)
}
```

`NewTask` accepts any dataset input type. Your `InputMapper` returns the prompt, typed dependencies, and optional per-case `RunOption` values. The mapper must be safe for concurrent calls.

A prompt can use `Content` for text or `Contents` for multimodal input, but not both.

> [!WARNING]
> A deferred agent run has no final output to evaluate. The adapter returns `ErrDeferredRun` when a tool still needs approval or external execution. Resolve deferred tools in a task-specific wrapper when deferred behavior is part of the evaluation.
