# Output

Use a Go struct when your agent has one result shape:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type City struct {
	Name    string `json:"name"`
	Country string `json:"country"`
}

func main() {
	agent := ai.NewAgent[struct{}, City](openai.NewModel("gpt-5-mini"))
	result, err := agent.Run(context.Background(), "Return information about Paris.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s, %s\n", result.Output.Name, result.Output.Country)
}
```

The agent reflects a Draft 2020-12 JSON Schema. It validates the raw response before decoding it into `City`.

Reflection follows `json` names and optional fields. It promotes anonymous embedded structs. Pointers accept `null`. `time.Time`, `json.Number`, text and JSON marshalers, `[]byte`, fixed arrays, JSON-compatible map keys, `json:",string"` fields, and `json.RawMessage` use their encoded JSON representations. Embedded fields follow `encoding/json` depth, tag, and conflict rules. Use `jsonschema` entries such as `description=...`, `enum=...`, `minimum=...`, `maxLength=...`, `format=...`, or `pattern=...` for additional constraints. Bare text becomes a description, and `required` overrides `omitempty`. JSON-valued entries such as `oneOf=[{"type":"string"},{"type":"null"}]` may contain commas.

All Draft 2020-12 keywords are available as annotations. Use `schema={...}` to merge keywords that reflection cannot infer. Use `schemaOverride={...}` to replace the reflected field schema completely. These two forms also preserve provider extensions and schemas copied from Pydantic.

## Multiple output alternatives

Use `UnionOutput` when the model can return one of several Go types:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type Answer interface {
	answer()
}

type City struct {
	Name       string `json:"name"`
	Population int    `json:"population"`
}

func (City) answer() {}

type Refusal struct {
	Reason string `json:"reason"`
}

func (Refusal) answer() {}

func main() {
	output := ai.NewUnionOutput(
		ai.NewOutputAlternative("city", func(city City) Answer { return city }),
		ai.NewOutputAlternative("refusal", func(refusal Refusal) Answer { return refusal }),
	)
	agent := ai.NewUnionAgent[struct{}](openai.NewModel("gpt-5-mini"), output)
	result, err := agent.Run(context.Background(), "Return information about Paris.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}

	switch answer := result.Output.(type) {
	case City:
		fmt.Printf("%s: %d\n", answer.Name, answer.Population)
	case Refusal:
		fmt.Println(answer.Reason)
	}
}
```

The model returns one portable envelope:

```json
{
  "result": {
    "kind": "city",
    "data": {
      "name": "Paris",
      "population": 2102650
    }
  }
}
```

The envelope is the same for tool, native, and prompted output. This avoids provider-specific unions and gives the discriminator a stable meaning in persisted prompts.

`NewUnionOutput` requires at least two unique kinds. Treat each kind as persisted API data. Do not rename it after you have stored message history or prompts that reference it.

Use `NewOutputAlternativeFunc` when conversion can fail or request a model retry. Use `NewRawOutputAlternative` when a type needs a hand-written schema or decoder. `UnionOutput.Schema` returns a detached schema for inspection. `UnionOutput.Decode` decodes an envelope outside an agent.

## Output functions

Use an output function when the model's schema should differ from your final result:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type CityInput struct {
	Name    string `json:"name"`
	Country string `json:"country"`
}

func main() {
	output := ai.NewOutputFunction("city_label", func(
		ctx context.Context,
		rc *ai.RunContext[struct{}],
		city CityInput,
	) (string, error) {
		if city.Country == "" {
			return "", ai.Retryf("include the country")
		}
		return strings.ToUpper(city.Name + ", " + city.Country), nil
	})
	agent := ai.NewOutputFunctionAgent(openai.NewModel("gpt-5-mini"), output)
	result, err := agent.Run(context.Background(), "Return information about Paris.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

The model produces `CityInput`. The agent validates that value before calling your function. The function returns the agent's final `string` output. Its name becomes the default output-tool name, and you can override it with `WithOutputTool`.

Return `Retryf` to send feedback to the model. Other errors stop the run unless an output-processing capability recovers them. Output validators run after the function and receive the final value.

Structured output functions run for tool, native, and prompted output. They also process partial structured values returned by `StreamedRun.Outputs`. Your function can therefore run concurrently when you use exhaustive output processing or concurrent stream consumers. Synchronize shared mutable state.

### Transform plain text

Use `NewTextOutputFunction` when the model should return text without a JSON Schema:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func main() {
	output := ai.NewTextOutputFunction("words", func(
		ctx context.Context,
		rc *ai.RunContext[struct{}],
		text string,
	) ([]string, error) {
		words := strings.Fields(text)
		if len(words) < 2 {
			return nil, ai.Retryf("return at least two words")
		}
		return words, nil
	})
	agent := ai.NewTextOutputFunctionAgent(openai.NewModel("gpt-5-mini"), output)
	result, err := agent.Run(context.Background(), "Name a city and country.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

The function name is used in telemetry. It does not register a model-facing tool. The callback receives partial text when you consume `StreamedRun.Outputs`.

## Image output

Use `NewImageOutputAgent` when the generated image itself is the result:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func main() {
	agent := ai.NewImageOutputAgent[struct{}](
		openai.NewResponsesModel("gpt-5.4"),
		ai.WithCapabilities(ai.NewImageGenerationCapability(
			ai.ImageGenerationCapabilityConfig[struct{}]{
				Native: ai.ImageGenerationTool{Quality: ai.ImageGenerationQualityHigh},
			},
		)),
	)
	result, err := agent.Run(context.Background(), "Paint a watercolor Go gopher.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s: %d bytes\n", result.Output.MediaType, len(result.Output.Data))
}
```

The agent returns the first `FilePart` whose media type starts with `image/`. The `BinaryContent` result is detached from message history. A response without an image requests another model response and consumes the output retry budget.

Image output uses output-processing hooks and validators, but skips JSON validation hooks because there is nothing to decode. `OutputHookContext.Mode` is `OutputHookModeImage`. With `EndStrategyEarly`, an image skips ordinary function calls emitted in the same response. Graceful and exhaustive strategies finish those calls first.

The selected model must set `ModelProfile.SupportsImageOutput`. OpenAI Responses and supported Google image models provide this profile automatically. Custom models can use `NewProfiledModel`.

## Output modes

`OutputModeAuto` is the default for an agent. It asks the selected model's `ModelProfile` which structured-output mode to use. The standard profile selects tool output because it works across providers.

Require a mode when your application depends on its wire behavior:

```go
package main

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type City struct {
	Name string `json:"name"`
}

func main() {
	agent := ai.NewAgent[struct{}, City](
		openai.NewModel("gpt-5-mini"),
		ai.WithOutputMode(ai.OutputModeNative),
	)
	fmt.Printf("%T\n", agent)
}
```

- `OutputModeTool` asks the model to call a final-result tool.
- `OutputModeNative` uses the provider's native JSON Schema feature. OpenAI Chat Completions, OpenAI Responses, Anthropic, Gemini, and Bedrock support it on compatible models.
- `OutputModePrompted` puts the schema in the instructions and validates returned JSON.
- `OutputModeAuto` resolves the mode after model selection.

An explicit run mode overrides the agent and model profile. Use `WithRunOutputMode` for one run.

## Validation and retries

Add a semantic validator after schema validation and Go decoding:

```go
package main

import (
	"context"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type City struct {
	Name       string `json:"name"`
	Population int    `json:"population"`
}

func main() {
	agent := ai.NewAgent[struct{}, City](openai.NewModel("gpt-5-mini"))
	agent.AddOutputValidator(func(
		ctx context.Context,
		rc *ai.RunContext[struct{}],
		city City,
	) error {
		if city.Population <= 0 {
			return ai.Retryf("population must be positive")
		}
		return nil
	})
	if _, err := agent.Run(context.Background(), "Return information about Paris.", struct{}{}); err != nil {
		log.Fatal(err)
	}
}
```

A retry is sent back to the model and consumes the output retry budget. Configure that budget with `WithRetryLimits` or `WithRunRetryLimits`.
