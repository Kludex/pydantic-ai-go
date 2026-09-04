# AG-UI

## Serve an agent

```go
package main

import (
    "net/http"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/openai"
    "github.com/Kludex/pydantic-ai-go/ui/agui"
)

func main() {
    agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
    adapter := agui.NewAdapter(agent, agui.Config{})

    server := &http.Server{Addr: ":8080", Handler: adapter.Handler(struct{}{})}
    if err := server.ListenAndServe(); err != nil {
        panic(err)
    }
}
```

Set `OPENAI_API_KEY`, then run the server:

```console
$ go run ./examples/agui
```

Send AG-UI `RunAgentInput` JSON to `POST /`. The response uses `text/event-stream` and emits AG-UI JSON in SSE `data` records.

## Trust client history

`RunAgentInput.messages` is untrusted. The adapter converts it to model messages and always calls `SanitizeMessages` before the run. The secure default removes client-authored system prompts, uploaded-file references, unsafe file settings, and unresolved trailing tool calls.

Pass `Config.Sanitization` only when your authenticated application deliberately grants more authority. The adapter is not an authentication boundary.

The input's `threadId` becomes the agent conversation ID. Its `runId` remains the AG-UI protocol identity and is not forced onto the agent's run ID.

## Send multimodal input

```json
{
  "threadId": "thread-1",
  "runId": "run-1",
  "messages": [
    {
      "id": "user-1",
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {
          "type": "image",
          "source": {
            "type": "url",
            "value": "https://example.com/image.png",
            "mimeType": "image/png"
          }
        }
      ]
    }
  ]
}
```

User content accepts AG-UI text, legacy binary, image, audio, video, and document parts. A source can contain a URL or base64 data. Inline binary data is decoded before the model request.

Set `metadata.vendor_metadata` to preserve provider-specific file data. Set `metadata.force_download` to `safe` or `allow-local` only when the matching mode is allowed by `Config.Sanitization`. The secure default removes download authority from client-held history.

## Transform an existing event stream

```go
package main

import (
    "encoding/json"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/ui/agui"
)

func main() {
    stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
        yield(ai.PartStartEvent{
            Index:  0,
            PartID: "text",
            Part:   ai.TextPart{Content: "hello"},
        }, nil)
        yield(ai.FinishEvent{}, nil)
    })

    for event, err := range agui.TransformStream(stream, "thread-1", "run-1") {
        if err != nil {
            panic(err)
        }
        encoded, err := json.Marshal(event)
        if err != nil {
            panic(err)
        }
        fmt.Println(string(encoded))
    }
}
```

`TransformStream` is useful when agent events arrive through a queue or durable workflow instead of an HTTP request.

## Approve deferred tools

Tools registered with `WithApprovalRequired` finish the AG-UI run with an interrupt outcome. Each interrupt ID uses `int-<toolCallId>` and advertises the resume payload schema.

Resume the same client-held history with a `resume` entry:

```json
{
  "interruptId": "int-call_approve",
  "status": "completed",
  "payload": {
    "approved": true,
    "editedArgs": {"city": "London"}
  }
}
```

`approved` must be a JSON boolean. Missing or malformed decisions deny by default. `editedArgs` must be an object and fully replaces the original tool arguments. A cancelled entry denies with a cancellation message.

## Event ordering

Each model response owns one assistant message. The adapter emits `TEXT_MESSAGE_START` before text or tool-call events from that response. A tool call uses that message ID as `parentMessageId`, including responses that start with a tool and contain no text.

Function and output tools emit start, argument, end, and result events. Provider-native tool calls use the same AG-UI lifecycle. Stopping the AG-UI consumer stops the underlying agent event iteration.

## Current scope

The adapter supports text and multimodal user messages, assistant function calls, structured tool results, secure client-held history, normalized text and tool streaming, standalone event transformation, generated protocol IDs, and SSE HTTP responses.

Frontend tools, state snapshots and deltas, activities, reasoning events, file preservation, external-execution interrupts, custom events, forwarded context, and protocol-version negotiation remain.
