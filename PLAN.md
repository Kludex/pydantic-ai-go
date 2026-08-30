# pydantic-ai-go - Project Plan

An idiomatic Go library for the LLM agent loop. Typed outputs via generics, tool calling, model-agnostic providers, OTel-native tracing. Companion to [pydantic-evals-go](https://github.com/Kludex/pydantic-evals-go).

Module: `github.com/Kludex/pydantic-ai-go`, package `ai`. Go 1.25.

## Design Principles

- **No graph layer.** The agent run is a plain loop: call model, execute tool calls, repeat until final output. PydanticAI's graph exists for history and durability reasons that do not apply here. A middleware-shaped loop gives the same extension points without exposing nodes.
- **Generics replace runtime validation.** `Agent[Deps, Output]` carries types end to end. Generics never cross the `Model` or `Capability` boundaries - those stay untyped so providers and capabilities are reusable across agents.
- **Idiomatic Go over API parity.** `context.Context` first everywhere, explicit errors, small structural interfaces, `http.Handler`-style middleware, `New`/`NewX` constructors, usable zero values where possible.
- **Two shapes per extension point, not a signature matrix.** PydanticAI's flexible signatures are a Python affordance. In Go, explicit variants (`AddTool` / `AddSimpleTool`) beat reflection-based signature sniffing.

## Package Layout

Flat core package, providers and capabilities as separate import paths so consumers only pull the dependencies they use.

```
github.com/Kludex/pydantic-ai-go        // package ai: Agent, Model, messages, tools, loop, usage, tracing
├── internal/schema/                     // JSON schema reflection (private, free to churn)
├── models/openai/                       // package openai: implements ai.Model
├── models/anthropic/
├── models/google/
├── models/fakes/                        // package fakes: TestModel, FunctionModel for users' tests
├── capabilities/mcp/                    // v0.4+
└── examples/
```

Rule: the core package owns everything a user touches on every run. Providers are separate because you import only what you use. Implementation details live under `internal/`.

## Core API

### Messages and the Model interface

`Model` is the provider contract - the one interface a provider package implements. The agent calls it once per loop iteration. It is untyped: providers deal only in messages and schemas.

```go
type Message struct {
    Role  Role          // user, assistant, tool
    Parts []Part        // TextPart | ToolCallPart | ToolReturnPart | RetryPromptPart | ThinkingPart | ...
}

type ModelRequestParams struct {
    Instructions string
    Tools        []ToolDefinition   // name, description, JSON schema
    OutputSchema *OutputDefinition  // nil for plain-text output
    Settings     ModelSettings      // temperature, max tokens, ... (zero value = provider defaults)
}

type ModelResponse struct {
    Parts []ResponsePart
    Usage Usage
}

type Model interface {
    Request(ctx context.Context, msgs []Message, params ModelRequestParams) (*ModelResponse, error)
}

// Streaming is a separate optional interface (v0.2), discovered by type assertion:
type StreamingModel interface {
    Model
    StreamRequest(ctx context.Context, msgs []Message, params ModelRequestParams) (ResponseStream, error)
}
```

Message part types mirror PydanticAI's so serialized histories interoperate with pydantic-evals-go and PydanticAI tooling.

### Agent

```go
type Agent[Deps, Output any] struct { ... }

func NewAgent[Deps, Output any](model Model, opts ...Option) *Agent[Deps, Output]

func (a *Agent[Deps, Output]) Run(ctx context.Context, prompt string, deps Deps, opts ...RunOption) (*RunResult[Output], error)

type RunResult[Output any] struct {
    Output Output
    // Usage() Usage
    // Messages() []Message      // full history including the new turn
    // NewMessages() []Message   // just this run's messages
}
```

Consumer view:

```go
model := openai.NewModel("gpt-5")   // NewModel, not New: provider packages will grow more constructors
agent := ai.NewAgent[MyDeps, Weather](model,
    ai.WithInstructions("You are a weather assistant."),
)
ai.AddTool(agent, "get_weather", getWeather,
    ai.WithDescription("Get current weather for a city"),
)

result, err := agent.Run(ctx, "Weather in SF?", deps)
result.Output // Weather, unmarshalled and validated
```

Options at construction: `WithInstructions`, `WithInstructionsFunc`, `WithModelSettings`, `WithUsageLimits`, `WithMaxToolRetries`, `WithOutputMode`, later `WithCapabilities`. Run options: `WithMessageHistory`, per-run settings overrides.

Multimodal input arrives later as `RunParts(ctx, []UserPart{...}, deps)` without breaking the string signature.

### RunContext

Run-scoped data passed to tools and dynamic hooks. Pure data - it does not embed or wrap `context.Context`; `ctx` stays the sole cancellation carrier and is always the first parameter.

```go
type RunContext[Deps any] struct {
    Deps       Deps
    Retry      int      // retries of the current tool
    RunID      string
    ToolCallID string   // empty outside tool execution
}
// Methods: Usage() Usage, Messages() []Message
```

One signature convention across every extension point:

```go
tool:            func(ctx context.Context, rc *ai.RunContext[D], args A) (R, error)
instructions:    func(ctx context.Context, rc *ai.RunContext[D]) (string, error)
output validator: func(ctx context.Context, rc *ai.RunContext[D], out O) error
```

### Tools

Go methods cannot have type parameters, so typed registration is a package-level generic function - inference means users never write type arguments.

```go
type WeatherArgs struct {
    City string `json:"city" jsonschema:"description=City name"`
    Unit string `json:"unit,omitempty" jsonschema:"enum=celsius,enum=fahrenheit"`
}

func getWeather(ctx context.Context, rc *ai.RunContext[MyDeps], args WeatherArgs) (string, error) {
    if !known(args.City) {
        return "", ai.Retryf("unknown city %q, use the full English name", args.City)
    }
    return rc.Deps.Client.Get(ctx, args.City, args.Unit)
}

ai.AddTool(agent, "get_weather", getWeather)   // schema reflected from WeatherArgs
ai.AddSimpleTool(agent, "now", nowFn)          // func(ctx, args A) (R, error) - no deps
```

- Schema reflected from the args struct (`json` + `jsonschema` tags).
- Argument unmarshal/validation failure is sent back to the model as a retry prompt, not a run failure. `ai.Retryf(...)` does the same from inside a tool; any other error aborts the run. Per-tool retry cap via `WithMaxRetries(n)`.
- Registration panics after the agent's first run (the `http.ServeMux.Handle` pattern) - agents are effectively immutable once running.
- Escape hatch for dynamic tools (MCP, config-driven), a plain method since no generics are involved:

```go
agent.AddRawTool(ai.ToolDefinition{Name: ..., Description: ..., Schema: raw},
    func(ctx context.Context, rawArgs json.RawMessage) (any, error) { ... })
```

### Structured output

- `Output = string` means plain text.
- Any other `Output` defaults to tool-based structured output (a final "output tool" whose schema is reflected from `Output`); providers may opt into native JSON mode via `WithOutputMode`.
- Unmarshal/validation failure goes back to the model as a retry, bounded by the retry cap.
- Optional `ai.WithOutputValidator(fn)` for semantic checks; returning `ai.Retryf(...)` triggers a model retry.

### Errors, usage, limits

```go
type Usage struct { InputTokens, OutputTokens, Requests int; ... }
type UsageLimits struct { RequestLimit, TotalTokenLimit int; ... }  // zero value = unlimited
```

Sentinel/typed errors: `ai.ErrUsageLimitExceeded`, `ai.ErrMaxRetriesExceeded`, `ai.UnexpectedModelBehaviorError`. All inspectable with `errors.Is`/`errors.As`.

### The loop

Private, but shaped as middleware from day one so capabilities can be retrofitted without restructuring:

```
wrapRun(
  for {
    resp := wrapModelRequest(model.Request(...))
    usage.Add(resp.Usage); limits.Check(usage)
    if no tool calls { return finalize(resp) }   // unmarshal Output, retry on validation error
    for each call { msgs += wrapToolCall(execTool(call)) }
  }
)
```

### Capabilities (v0.3)

Reusable, composable units of agent behavior - PydanticAI's primary extension point, translated to Go as small optional interfaces discovered by type assertion (the `http.Flusher` pattern), instead of one 20-method ABC.

```go
type Capability interface {
    Setup(reg *CapabilityRegistry) error   // register tools, append instructions, adjust settings
}

// Opt-in hook interfaces:
type ModelRequestWrapper interface {
    WrapModelRequest(ctx context.Context, rc *RunContextAny, msgs []Message,
        params ModelRequestParams, next ModelRequestFunc) (*ModelResponse, error)
}
type ToolCallWrapper interface {
    WrapToolCall(ctx context.Context, rc *RunContextAny, call ToolCall, next ToolCallFunc) (any, error)
}
type RunWrapper interface {
    WrapRun(ctx context.Context, rc *RunContextAny, next RunFunc) (*RunResultAny, error)
}
type InstructionsProvider interface {
    Instructions(ctx context.Context, rc *RunContextAny) (string, error)
}
```

- Untyped boundary (`RunContextAny`): capabilities are reusable across agents with different `Deps`/`Output`.
- Ordering: slice order, first is outermost. PydanticAI's tier/edge ordering system is deferred until an ecosystem demands it.
- No graph nodes leak into hooks - the loop has exactly three interception points: run, model request, tool call.
- Out of scope permanently: on-demand capability loading, self-extension. Harness territory.

### Tracing

OTel spans following the GenAI semantic conventions, same approach as pydantic-evals-go: agent run span, per-request span, per-tool span. On by default when a global tracer provider is configured; `WithTracerProvider` to override.

## Testing Strategy

- Public-API tests only; most loop tests use `fakes.NewModel` (a `TestModel` that calls every tool then returns schema-conformant output) and `fakes.NewFunctionModel` (user-supplied response function) - no network.
- Provider tests record real traffic with `dnaeon/go-vcr`, credential-filtered cassettes committed, CI replays only so a missing interaction fails loudly.
- 100% coverage, pragmas only with stated reasons.

## Releases

### v0.1 - The loop (shipped)

The minimal useful agent: typed runs against OpenAI.

- Message types (parts model mirroring PydanticAI for interop)
- `Model` interface, `ModelSettings`, `Usage`, `UsageLimits`
- `Agent[Deps, Output]`, `NewAgent`, `Run`
- `RunContext[Deps]`
- `AddTool` / `AddSimpleTool` / `AddRawTool`, schema reflection, `Retryf`, retry caps
- Structured output via output tool, validation-error retries
- Loop implemented middleware-shaped internally (hooks not yet public)
- `models/openai` (Chat Completions), `models/fakes` (`TestModel`, `FunctionModel`)
- OTel tracing
- Error taxonomy, cassette-based provider tests, 100% coverage

### v0.2 - Streaming and providers (shipped)

- `StreamingModel`, `Agent.RunStream`: text deltas, partial tool calls, event stream
- `models/anthropic`, `models/google`
- `WithMessageHistory` (multi-turn), `NewMessages()` on results
- Multimodal user input: `RunParts` / `UserPart` (images, files)
- Native JSON-mode structured output where providers support it

### v0.3 - Capabilities (shipped)

- Public `Capability` + hook interfaces (`RunWrapper`, `ModelRequestWrapper`, `ToolCallWrapper`, `InstructionsProvider`)
- `WithCapabilities`, slice-order composition (first is outermost)
- Dogfood: usage limits reimplemented as an internal `ModelRequestWrapper` capability
- Output validation stays `AddOutputValidator` (a typed method): construction options are untyped, so a `WithOutputValidator` option cannot carry `Deps`/`Output` - Open Decision 1 resolved in favor of methods for typed extension points
- History processors expressed as `ModelRequestWrapper` capabilities (no separate API needed)
- `models/openai` Responses API constructor (`NewResponsesModel`)

### v0.4 - Ecosystem

- `capabilities/mcp`: MCP client tools via `AddRawTool` machinery
- Provider-native tools (web search, web fetch) as capability packages
- Thinking/reasoning part support surfaced per provider
- Evals integration: first-class task adapter for pydantic-evals-go

### Post-parity - Agentic upstream sync

Once the package is up to date with PydanticAI itself, add a [gh-aw](https://github.com/githubnext/gh-aw) agentic workflow so the repo keeps itself current - the same setup as [pydantic-evals-go](https://github.com/Kludex/pydantic-evals-go)'s `agentic-evals-sync`, but running **daily** instead of weekly.

- `.github/workflows/agentic-ai-sync.md` (compiled `.lock.yml` committed alongside), `schedule: daily` + `workflow_dispatch`.
- Upstream commit pinned in `.upstream-sync.json`; the agent diffs `pydantic/pydantic-ai` since the pin, ports relevant changes into Go, bumps the pin, and opens a draft PR (`[ai-sync]` prefix, `agentic-workflows` + `upstream-sync` labels, max 1 PR per run).
- Python-only or non-applicable changes (graph internals, durable execution, harness features) are skipped and noted in the PR body.
- Same guardrails as evals-go: read-only permissions plus `create-pull-request` safe output, allowlisted bash tools (`git`, `rg`, `gofmt`, `go build/vet/test`), `AGENTIC_WORKFLOWS_ENABLED` repo variable as kill switch, concurrency group, turn/timeout caps.

### Explicitly out of scope

- Graph layer, durable execution (Temporal etc.), multi-agent handoff protocols
- HTTP-level retry policy (delegate to the injected `http.Client`/transport)
- On-demand capability loading, self-extension, harness features

## Open Decisions

1. **`Option` typing** - construction options are currently untyped (`Option`); anything needing `Deps` (dynamic instructions) may force `Option[Deps]` or a method-based API. Resolve while stubbing v0.1.
2. **Schema library** - `invopop/jsonschema` vs in-house `internal/schema`. Start with `invopop`, wrap it so it can be swapped.
3. **Provider transport** - direct HTTP (small deps, clean cassettes) vs official SDKs (features faster). Leaning direct HTTP for openai; revisit per provider.
4. **`RunContextAny` shape** - concrete struct with `Deps any`, or interface. Decide when capabilities land (v0.3), but keep `RunContext[Deps]` convertible to it from v0.1.
