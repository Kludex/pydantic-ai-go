# Agent2Agent

## Serve an agent

```go
package main

import (
    "net/http"

    protocol "github.com/a2aproject/a2a-go/a2a"
    "github.com/a2aproject/a2a-go/a2asrv"

    a2aintegration "github.com/Kludex/pydantic-ai-go/a2a"
    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
    agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
    executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{
        ArtifactName: "answer",
        ArtifactDescription: "The generated answer.",
        ArtifactExtensions: []string{"urn:example:answer"},
        ArtifactMetadata: map[string]any{"format": "assistant-answer"},
    })
    handler := a2asrv.NewHandler(executor)

    card := &protocol.AgentCard{
        Name: "Assistant", Description: "Answers general questions.", Version: "1.0.0",
        ProtocolVersion: string(protocol.Version), URL: "http://localhost:8080/invoke",
        PreferredTransport: protocol.TransportProtocolJSONRPC,
        Capabilities: protocol.AgentCapabilities{Streaming: true},
        DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
        Skills: []protocol.AgentSkill{{
            ID: "answer", Name: "Answer questions", Description: "Answers a question.", Tags: []string{"assistant"},
        }},
    }
    mux := http.NewServeMux()
    mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(handler))
    mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
    server := &http.Server{Addr: ":8080", Handler: mux}
    if err := server.ListenAndServe(); err != nil {
        panic(err)
    }
}
```

Set `OPENAI_API_KEY`, then run the server:

```console
$ go run ./examples/a2a
```

The integration implements the official `a2asrv.AgentExecutor`. You choose JSON-RPC, gRPC, or HTTP+JSON through the official SDK. You also own the agent card and transport server.

## Request mapping

The executor accepts user text, structured data, inline files, and HTTP or HTTPS file URIs. Structured data becomes compact JSON prompt text. File URIs become provider-neutral image, audio, video, or document content based on MIME type.

Stored task history is client-controlled. The executor calls `SanitizeMessages` before passing it to the agent. The zero configuration removes unsafe system, file, upload, and unresolved tool-call state.

`RequestContext.ContextID` becomes the agent conversation ID. Use `Config.ResolveDeps` for authenticated request-specific dependencies. It may run concurrently.

Loaded `RequestContext.RelatedTasks` are added before the current task history. Each task gets a visible boundary, followed by its messages and artifacts. The same sanitization policy covers related and current task content. Configure the official SDK's `ReferencedTasksLoader` with your task store when you want `Message.ReferenceTasks` resolved automatically.

## Task lifecycle

A new task emits submitted and working states. An existing task starts at working. Text and generated files stream as one append-only A2A artifact. `Config.ArtifactName`, `ArtifactDescription`, `ArtifactExtensions`, and `ArtifactMetadata` describe its first event. Metadata must contain JSON-compatible values. A successful run ends completed.

A deferred approval or external tool call ends the response in input-required state. A model failure ends failed with an agent message. First-party cancellation ends canceled. Every terminal response status sets `final`.

The official A2A server owns task storage, event queues, push notifications, retries, and transport cancellation. The executor does not close its queue.

## Use a remote agent as a model

```go
package main

import (
    "context"
    "fmt"
    "log"

    protocol "github.com/a2aproject/a2a-go/a2a"
    "github.com/a2aproject/a2a-go/a2aclient"

    ai "github.com/Kludex/pydantic-ai-go"
    a2aintegration "github.com/Kludex/pydantic-ai-go/a2a"
)

func main() {
    ctx := context.Background()
    endpoint := "http://localhost:8080/invoke"
    client, err := a2aclient.NewFromEndpoints(ctx, []protocol.AgentInterface{{
        URL: endpoint, Transport: protocol.TransportProtocolJSONRPC,
    }})
    if err != nil {
        log.Fatal(err)
    }
    defer func() {
        if err := client.Destroy(); err != nil {
            log.Print(err)
        }
    }()

    model := a2aintegration.NewModel("Assistant", client, a2aintegration.ModelConfig{
        ProviderURL: endpoint,
        SendConfig: &protocol.MessageSendConfig{
            AcceptedOutputModes: []string{"text/plain", "application/json"},
        },
    })
    agent := ai.NewAgent[struct{}, string](model)
    result, err := agent.Run(ctx, "What is the capital of Brazil?", struct{}{})
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result.Output)
}
```

`Model` implements `ai.Model` and `ai.StreamingModel`. It forwards text, inline files, and image, audio, video, or document URIs. Remote task and context IDs persist in provider details. A new completed turn starts a new task in the same context. Use `TaskError` IDs with the official client when an input-required or authentication-required task needs protocol-specific continuation.

A remote agent owns its tools and generation settings. The model rejects caller-provided tools, provider-native tools, uploaded provider files, and generation settings instead of silently dropping them. Structured Go outputs use prompted JSON mode because A2A does not define a portable output-schema field.

Remote responses may contain text, structured data, or inline file bytes. Structured data becomes compact JSON text. URI response files fail because `ai.FilePart` represents detached bytes, not a remote reference.

`ModelConfig.Extensions`, `MessageMetadata`, and `RequestMetadata` carry extensions you negotiated from the agent card. `SendConfig` forwards accepted output modes, history length, and push configuration. The official SDK still owns agent-card discovery, extension interceptors, push callback policy, authentication, and transport selection. Do not enable polling on the official client. An `ai.Model` request requires a terminal result.

Remote failed, rejected, canceled, authentication-required, and input-required tasks return an inspectable `TaskError` with task and context IDs. Transport failures return `ai.ModelTransportError`. Context cancellation keeps its original cause.

## Resume deferred tools

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"

    a2aintegration "github.com/Kludex/pydantic-ai-go/a2a"
)

func main() {
    part, err := a2aintegration.NewDeferredResultsPart(a2aintegration.DeferredResults{
        Calls: map[string]a2aintegration.DeferredCallResult{
            "call-id": {
                Outcome: a2aintegration.DeferredCallSucceeded,
                Value: map[string]any{"result": "done"},
            },
        },
        Approvals: map[string]a2aintegration.DeferredApproval{
            "approval-id": {Approved: true},
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    encoded, err := json.Marshal(part)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(encoded))
}
```

An input-required status contains one `pydantic-ai-go/deferred-tool-requests` data part. It includes pending IDs, arguments, metadata, and opaque model history. Send one `pydantic-ai-go/deferred-tool-results` part in the next user message. `NewDeferredResultsPart` creates that part without requiring you to depend on its wire representation.

You must resolve every pending ID exactly once. External calls accept successful values, terminal failures, or retry messages. Approvals accept a decision and optional replacement arguments. The executor restores the stored model history and resumes without repeating completed tool side effects.

Keep the SDK task store authoritative. The executor reads continuation state from `StoredTask`, not from ordinary user history.

## Current scope

The integration provides server-side execution and client-side model calls through the official Go SDK. It supports text, data, and file content, sanitized task history, named and annotated streamed artifacts, prompted structured output, dependency resolution, deferred input-required state, terminal failure mapping, task continuity, and cancellation.

You compose agent-card discovery, extension negotiation, push delivery, authentication, task storage, and JSON-RPC, gRPC, or HTTP+JSON policy with the official SDK. The integration forwards the resulting message and request configuration without duplicating those transport responsibilities.
