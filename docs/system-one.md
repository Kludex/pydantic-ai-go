# System One decision models

```go
package main

import (
    "context"
    "fmt"
    "time"

    "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/models/systemone"
)

type Ticket struct {
    Urgent bool   `json:"urgent" jsonschema_description:"Does this need a reply within the hour?"`
    Area   string `json:"area" jsonschema:"enum=billing,enum=bug,enum=account"`
}

func main() {
    threshold := 0.75
    settings, err := (systemone.Settings{
        Common:           ai.ModelSettings{RequestTimeout: 10 * time.Second},
        BooleanThreshold: &threshold,
    }).Build()
    if err != nil {
        panic(err)
    }
    model, err := systemone.NewModel("nimble",
        systemone.WithBaseURL("http://localhost:11434"),
        systemone.WithDefaultSettings(settings),
        systemone.WithProfile(systemone.Profile{MaxChoiceOptions: 26, MaxScoreLevels: 26}),
    )
    if err != nil {
        panic(err)
    }
    agent := ai.NewAgent[struct{}, Ticket](model, ai.WithInstructions("Triage this support ticket."))
    result, err := agent.Run(context.Background(), "Our checkout has returned 500 errors since 9am.", struct{}{})
    if err != nil {
        panic(err)
    }
    fmt.Printf("%+v\n", result.Output)
}
```

You use `ai/models/systemone` for decision models such as CLM, Laya, and Ollama's Nimble. They answer typed questions about text instead of generating text. Put questions in field descriptions or agent instructions. Your prompt contains the text to judge.

For Ollama, pull the model before running the example. Decision models need a server that supports `POST /v1/systemone`, not just Chat Completions.

## Endpoint and credentials

You pass `WithBaseURL` or set `SYSTEM_ONE_BASE_URL`. There is no default server. The URL can end in `/v1`; both forms send requests to `/v1/systemone`. Set `SYSTEM_ONE_API_KEY` or pass `WithAPIKey` when the server requires a bearer token. Local servers can omit it.

You can use `infer.Model("system-one:clm-latest")` with the same environment variables. `NewProviderConfig` reads those variables into a reusable `ProviderConfig`. `WithProvider` replaces environment configuration. Options apply in order. Headers and settings are copied; an injected `HTTPClient` remains caller-owned. `PrepareRequest` runs after provider headers and before `ExtraHeaders`, so request headers take precedence over credentials.

## Settings and limits

| Setting | Behavior |
| --- | --- |
| `Settings.BooleanThreshold` | Defaults to `0.5`, rounding a yes probability into a boolean |
| `Settings.RouteThreshold` | Defaults to `0`, taking every selected route; raise it to hand off uncertain routes |
| `Common.Temperature` | Sent when set; whether it changes probabilities depends on the model |
| `Common.RequestTimeout` | Bounds the request, including a second request to fill a selected route |
| `Common.ExtraHeaders` | Replaces request headers, including `Authorization` |
| `Common.ExtraBody` | Overrides wire fields; a `questions` override is validated against the answers actually returned |
| `Profile.MaxChoiceOptions` | Refuses a pick-one field or route question over the limit before sending |
| `Profile.MaxScoreLevels` | Asks larger rubrics as pick-one questions instead of scores |
| `Profile.ContextWindow` | Reports a token limit to history processors |
| `Profile.RequiresInstructions` | Nil defaults to true; false permits option-described questions without instructions |

You call `Settings.Build` before passing thresholds to an agent. Both thresholds must be finite and between `0` and `1`. Other portable generation settings are ignored. Model defaults are overridden by agent and run settings using the usual `ai.ModelSettings` rules; supplying `ExtraBody` replaces the whole defaults block.

Profile limits default to zero, meaning unknown or uncapped. Unlike TypeSafe's Jev, this provider does not assume 255 choices or 10 rubric levels. You set limits for the model behind your URL. A limit unknown to the client is enforced by the server.

## Questions without instructions

```go
package main

import (
    "context"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/models/systemone"
)

func main() {
    model, err := systemone.NewModel("nimble", systemone.WithBaseURL("http://localhost:11434"))
    if err != nil {
        panic(err)
    }
    response, err := model.Request(context.Background(), []ai.ModelMessage{
        ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "Charged twice."}}},
    }, ai.ModelRequestParams{
        OutputTool: &ai.ToolDefinition{
            Name: "classify",
            Schema: map[string]any{"type": "string", "enum": []any{"billing", "bug"}},
        },
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(string(response.Parts[0].(ai.ToolCallPart).Args))
}
```

The System One API requires instructions on every question. A bare choice, rubric, or boolean with described answers can carry its meaning in the options. With no other instructions, the provider sends `Which of these applies?`. It leaves existing instructions unchanged.

Set `Profile.RequiresInstructions` to a pointer to `false` for a compatible server that accepts questions without instructions. This does not make undescribed booleans or numbers meaningful. They still need an actual question.

## Output and errors

You use the same [typed question shapes as TypeSafe](providers.md#typesafe-jev). Rubrics round halves up and preserve the raw score. `ProviderDetails` includes confidence, probabilities, scores, and routing details. History and tool results become the state being judged. Text generation, files, and free-form output fields are unsupported.

You can inspect `APIError` with `errors.As`. It retains the status, requested model name, raw body, and response headers. Transport failures use `ai.ModelTransportError`. Malformed answers use `ai.UnexpectedModelBehaviorError`, including invalid probabilities, negative token usage, mismatched question names, and inconsistent rubric scores. Independently rounded probabilities and scores are accepted when their rounding intervals agree.

A selected route with unsupported fields returns `UnfillableRoute`. A route below your threshold returns `UnsureRoute`. Both implement `DecisionHandOff` and work with `ai.NewFallbackModel`. Calibrate thresholds on your own labelled data; probabilities from different models are not interchangeable.
