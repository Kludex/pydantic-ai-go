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

## Task lifecycle

A new task emits submitted and working states. An existing task starts at working. Text and generated files stream as one append-only A2A artifact. `Config.ArtifactName`, `ArtifactDescription`, `ArtifactExtensions`, and `ArtifactMetadata` describe its first event. Metadata must contain JSON-compatible values. A successful run ends completed.

A deferred approval or external tool call ends the response in input-required state. A model failure ends failed with an agent message. First-party cancellation ends canceled. Every terminal response status sets `final`.

The official A2A server owns task storage, event queues, push notifications, retries, and transport cancellation. The executor does not close its queue.

## Current scope

The integration provides server-side execution through the official Go SDK, text, data, and file input, sanitized task history, named and annotated streamed artifacts, structured-output fallback, dependency resolution, deferred input-required state, failure state, and cancellation.

Client-side A2A model calls, related-task context, extension negotiation, push notification policy, and mapping deferred A2A follow-up messages back to `DeferredToolResults` remain.
