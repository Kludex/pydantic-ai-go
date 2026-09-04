# pydantic-ai-go

An idiomatic Go library for the LLM agent loop. Read `PLAN.md` before making design decisions - it records the API sketch, release scope, and the reasoning behind them (no graph layer, generics stop at the `Model`/`Capability` boundaries).

## Code

Write and review all Go code as if you were **Dave Cheney**: simplicity first, small interfaces discovered at the point of use, usable zero values, explicit errors handled once, no speculative abstraction, clarity over cleverness. Before finishing any change, re-read it with that eye and remove what a careful reviewer would question.

- All library packages live under `ai/`: providers under `ai/models/`, implementation details under `ai/internal/`.
- `context.Context` is always the first parameter and the only cancellation carrier - never stored in structs.
- Constructors: bare `New` only when the package name says what is created; otherwise `NewX` (e.g. `openai.NewModel`).
- Generics never cross the `Model` or `Capability` boundaries.
- Avoid comments; a clear name and a precise type are the documentation.

## Docs

Write all documentation (README, guides, doc comments that surface in godoc) as if you were **Sebastián Ramírez** writing the FastAPI docs, minus the emoji: short plain sentences, second person, lead with a complete runnable example, explain the why behind every default, structure as reference not narrative.

## Testing

- Test through the public API only - never import internals or test private functions.
- Loop tests use `ai/models/fakes`; provider tests record real traffic with `go-vcr`, cassettes committed, CI replays only.
- 100% coverage; exclusions need a pragma with a stated reason.

## Workflow

- `gofmt`, `go vet`, `golangci-lint`, and `go test ./...` must pass before any commit.
- Never rebase, never force push, never amend pushed commits.
