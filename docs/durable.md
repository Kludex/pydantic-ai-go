# Durable operation backends

## Bind an operation

```go
package main

import (
    "context"
    "fmt"

    "github.com/Kludex/pydantic-ai-go/durable"
)

type backend struct{}
type bound struct{ registration durable.Registration }

func (backend) Bind(registration durable.Registration) (durable.BackendOperation, error) {
    return bound{registration: registration}, nil
}

func (operation bound) Invoke(ctx context.Context, invocation durable.Invocation) ([]byte, error) {
    fmt.Printf("operation=%s cache=%s\n", invocation.Name, invocation.CacheKey)
    return operation.registration.Handler(ctx, invocation.Params)
}

func main() {
    operation, err := durable.Bind(backend{}, "weather", "default", durable.Operation[string, string]{
        ID: durable.OperationID{
            Kind:      durable.KindToolsetCall,
            ToolsetKind: durable.ToolsetFunction,
            ToolsetID: "forecast",
        },
        Role: durable.RoleTool,
        Handler: func(_ context.Context, city string) (string, error) {
            return "Sunny in " + city, nil
        },
        CacheIdentity: durable.CacheIdentityFunc[string](func(city string) (any, error) {
            return map[string]string{"city": city}, nil
        }),
        InvocationLabel: func(city string) string { return city },
    })
    if err != nil {
        panic(err)
    }
    result, err := operation.Invoke(context.Background(), "London")
    if err != nil {
        panic(err)
    }
    fmt.Println(result)
}
```

`Backend.Bind` receives one immutable worker registration. A registration contains the stable operation name and a handler that accepts serialized parameters. `BackendOperation.Invoke` receives detached parameter and cache-key bytes for each call.

The backend decides whether to execute immediately, schedule an activity, register a worker handler, replay a journal result, or return a cache hit. It must preserve `Invocation.Name` as persisted compatibility data.

## Serialization

`Operation` uses `JSONCodec` for parameters and results by default. A backend never receives live model, dependency, or tool values unless your codec deliberately serializes them.

Implement `Codec` when an engine has its own payload format. Implement `CacheIdentity` when transport parameters include values that should not change cache identity. The projected value is JSON encoded separately into `Invocation.CacheKey`.

Operation names cover model requests and streams, suspended-response cancellation, compaction, event handlers, capability operations, and function, dynamic, or MCP toolset discovery, validation, instruction, and call operations.

## Model ownership

```go
package main

import (
    "context"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/durable"
    "github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
    registry := &durable.ModelRegistry{}
    if err := registry.Register("shared", openai.NewModel("gpt-5-mini")); err != nil {
        panic(err)
    }
    if err := registry.RegisterFactory("rebuilt", func(context.Context) (ai.Model, error) {
        return openai.NewModel("gpt-5-mini"), nil
    }); err != nil {
        panic(err)
    }

    lease, err := registry.Acquire(context.Background(), "rebuilt")
    if err != nil {
        panic(err)
    }
    defer func() { _ = lease.Close(context.Background()) }()
    fmt.Println(lease.Model.Name())
}
```

A caller-owned model registered with `Register` is never rebuilt, opened, or closed by the registry. A factory model is rebuilt inside each operation. If it implements `ModelOpener`, the registry opens it there and returns the matching close function in `ModelLease`.

Use stable registry IDs. Changing one can strand persisted workflows or send replayed work to different credentials and endpoints.

## Current scope

The package provides the public third-party backend contract, stable operation naming, parameter and result serialization, semantic cache identity, callable backend adaptation, and explicit model ownership.

Agent-loop operation binding and concrete Temporal, DBOS, and Prefect packages remain. Startup registration checks, Prefect observer-only runtime capabilities, and separately journaled Prefect dynamic-tool discovery, validation, and calls remain.
