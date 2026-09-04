# Instructions

Give each instruction block a stable address when your application needs to edit it:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func main() {
	model := fakes.NewFunctionModel(func(
		_ context.Context,
		_ []ai.ModelMessage,
		params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		fmt.Println(params.Instructions)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "Rome"},
		}}, nil
	})

	rewritePersona := ai.BeforeModelRequestFunc(func(
		_ context.Context,
		_ *ai.RunInfo,
		request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		for index, part := range request.Params.InstructionParts {
			if part.ID != nil && part.ID.String() == "agent:persona" {
				part.Content = "Answer as a careful geographer."
				request.Params.InstructionParts[index] = part
			}
		}
		return request, nil
	})

	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithInstructionParts(
			ai.InstructionPart{Content: "Answer briefly.", Name: "persona"},
			ai.InstructionPart{Content: "Use verified facts."},
		),
		ai.WithCapabilities(rewritePersona),
	)
	result, err := agent.Run(context.Background(), "What is Italy's capital?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

The model receives:

```text
Answer as a careful geographer.

Use verified facts.
Rome
```

`InstructionParts` is the source of truth when a `BeforeModelRequestHook` changes it. The agent joins the edited parts before calling the model and stores the same rendered text in message history.

## Stable IDs

You declare `InstructionPart.Name`. The agent qualifies it with its source:

| Source | ID |
| --- | --- |
| Agent instructions | `agent` |
| Named agent block | `agent:<name>` |
| Identified toolset | `toolset:<toolset-id>` |
| Named toolset block | `toolset:<toolset-id>:<name>` |
| Identified capability | `capability:<capability-id>` |
| Named capability block | `capability:<capability-id>:<name>` |

Implement `ToolsetIDProvider` or `CapabilityIDProvider` to identify a reusable source. Use `AddNamedInstructionsFunc` to identify dynamic agent instructions. Per-run instructions keep their declared name but have no stable ID because their lifetime is one run.

IDs survive JSON serialization. Unknown future source namespaces decode as unaddressable instruction parts instead of making the entire payload fail.

Do not rename a published source ID or instruction name. Applications can persist configuration against the complete ID.

## Prompt caching

Set `InstructionPart.Dynamic` when a block can change between model requests. The agent keeps static blocks first and preserves source order within the static and dynamic groups. Providers can place their prompt-cache boundary between those groups.

Plain dynamic instruction functions are dynamic automatically. Literal `InstructionPart` values are static unless you set `Dynamic: true`.

## Validation

The colon character is reserved as the ID delimiter. Toolset IDs, capability IDs, and instruction names cannot contain `:`. The instruction name `agent` is also reserved.

Two capabilities or toolsets that contribute instructions cannot share the same source ID. The run fails before sending an ambiguous instruction address to the model.
