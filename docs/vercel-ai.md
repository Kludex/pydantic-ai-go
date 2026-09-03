# Vercel AI

## Serve an agent

```go
package main

import (
    "net/http"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/openai"
    "github.com/Kludex/pydantic-ai-go/ui/vercel"
)

func main() {
    agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
    adapter := vercel.NewAdapter(agent, vercel.Config{})

    server := &http.Server{Addr: ":8080", Handler: adapter.Handler(struct{}{})}
    if err := server.ListenAndServe(); err != nil {
        panic(err)
    }
}
```

Set `OPENAI_API_KEY`, then run the server:

```console
$ go run ./examples/vercel-ai
```

Send Vercel AI `submit-message` or `regenerate-message` JSON to `POST /`. The handler returns SSE with `x-vercel-ai-ui-message-stream: v1`. The final record is `data: [DONE]`.

## Trust client history

The adapter treats every `messages` value as untrusted. It converts UI message parts to model history and calls `SanitizeMessages` before every run. The secure default removes client system prompts, provider-hosted file references, unsafe file settings, and unresolved trailing tool calls.

Pass `Config.Sanitization` only after your authenticated application decides which client-held values to trust. The handler does not authenticate requests.

The request `id` becomes the agent conversation ID. `ServerMessageID` controls the assistant message identity returned to the frontend. The adapter generates one when it is empty.

## Transform an existing stream

```go
package main

import (
    "encoding/json"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/ui/vercel"
)

func main() {
    stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
        yield(ai.PartStartEvent{
            PartID: "text",
            Part:   ai.TextPart{Content: "hello"},
        }, nil)
        yield(ai.PartEndEvent{
            PartID: "text",
            Part:   ai.TextPart{Content: "hello"},
        }, nil)
        yield(ai.FinishEvent{FinishReason: ai.FinishReasonStop}, nil)
    })

    for chunk, err := range vercel.TransformStream(stream, "message-1") {
        if err != nil {
            panic(err)
        }
        encoded, err := json.Marshal(chunk)
        if err != nil {
            panic(err)
        }
        fmt.Println(string(encoded))
    }
}
```

`TransformStream` supports event delivery through queues and durable workflows without an HTTP request.

## Current scope

The adapter supports AI SDK UI versions 5 through 7 for text, reasoning, function and provider-native tool inputs and outputs, step boundaries, finish reasons, secure client-held history, standalone transformation, and bounded SSE HTTP serving.

Vercel AI file and source parts, data parts, provider metadata, message metadata, tool approval states, deferred resumes, compaction activities, tool-availability data, cancellation chunks, and version-specific fields remain.
