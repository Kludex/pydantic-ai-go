# AG-UI

## Serve an agent

```go
package main

import (
    "net/http"

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/models/openai"
    "github.com/Kludex/pydantic-ai-go/ai/ui/agui"
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

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/ui/agui"
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

Cancel the context passed to `RunStream` to stop the agent run. The HTTP handler uses the request context, so a disconnected client cancels generation without a separate adapter token.

## Receive state and context

Implement `RunInputReceiver` on your dependency value to receive frontend state, context, and forwarded properties before each run. The adapter supplies generated or client-provided thread and run IDs with the values.

The receiver gets a detached JSON snapshot. Changes cannot mutate the request. Return an error to reject the run before the model is called. Dependencies that do not implement the interface ignore these optional application values.

`ForwardedInput.Events` is a run-scoped, concurrency-safe queue. Tools and application code can call `EmitStateSnapshot`, `EmitStateDelta`, or `EmitCustom`. The adapter detaches each JSON value and emits queued events before the terminal run event. State deltas use RFC 6902 JSON Patch arrays.

## Run frontend tools

Include AG-UI `tools` in the run input to expose functions implemented by the frontend. The adapter registers them as run-scoped external tools. It never executes them on the server.

A frontend tool emits the normal `TOOL_CALL_START`, `TOOL_CALL_ARGS`, and `TOOL_CALL_END` events without a server result. Keep the assistant tool call and the frontend's matching tool message in the next request history. This lets the model continue without registering the client tool on the reusable agent.

## Resume externally executed tools

Tools registered for external execution finish with an interrupt outcome. Each interrupt keeps the original tool-call ID and uses an `ext-` interrupt ID. Its response schema requires a `result` value.

Send the original assistant tool call back with a matching resume entry:

```json
{
  "interruptId": "ext-call_remote",
  "status": "resolved",
  "payload": {"result": {"temperature": 21}}
}
```

Use `status: "cancelled"` to return a failed tool result to the model. The adapter rejects duplicate, malformed, and unmatched resumes. It restores the external pending kind on sanitized client history before continuing the agent loop.

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

## Select a protocol version

Set `Config.Version` to the AG-UI version used by your frontend. The zero value targets `0.1.19`.

Versions before `0.1.11` receive the legacy `THINKING_*` lifecycle. Version `0.1.11` and later receive `REASONING_*` events. The adapter carries reasoning signatures and provider metadata through `REASONING_ENCRYPTED_VALUE`. Version `0.1.14` and later use the `reasoning` message role.

Every event includes a Unix-millisecond `timestamp`. A canceled agent run closes any open text message and emits `RUN_FINISHED` without a success or interrupt outcome because AG-UI has no cancellation outcome.

## Preserve generated and uploaded files

Set `Config.PreserveFileData` to emit generated `FilePart` values as `pydantic_ai_file` activity snapshots. File replacement deltas reuse the same activity message ID. The activity carries a data URL plus detached provider metadata.

The same setting reconstructs echoed `pydantic_ai_file` and `pydantic_ai_uploaded_file` activities. Uploaded provider references still require `Config.Sanitization.AllowUploadedFiles`. This second trust gate prevents file preservation from granting access to client-supplied provider files by itself.

File activities require AG-UI `0.1.19` or later. The default remains disabled because data URLs duplicate binary output in client-held history.

## Preserve durable activities

AG-UI `0.1.19` and later receives `ACTIVITY_SNAPSHOT` events for provider-neutral compaction boundaries and tools revealed during a run. The reserved activity types are `pydantic_ai_compaction` and `pydantic_ai_tool_availability_delta`.

Keep those activity messages in client-held history. The adapter reconstructs `CompactionPart` and `ToolAvailabilityDeltaPart` values on the next request. Unknown application activity types remain UI-only and are not sent to the model. Older protocol versions omit activity events.

## Preserve typed tools

AG-UI `0.1.11` and later receives namespaced `REASONING_ENCRYPTED_VALUE` events for typed tool calls and non-successful results. Echo those encrypted values on `ToolCall.encryptedValue` and tool-message `encryptedValue` fields. The adapter accepts only known tool kinds and failed, denied, or interrupted outcomes. Invalid client claims degrade to ordinary tool history.

Provider-native calls use `pyd_ai_builtin|<provider>|<call-id>` protocol IDs. The adapter restores the original provider and call ID when history returns.

System and developer messages both become system prompt parts. Secure sanitization still removes them unless you explicitly set `AllowSystemPrompts`.

## Event ordering

Each model response owns one assistant message. The adapter emits `TEXT_MESSAGE_START` before text or tool-call events from that response. A tool call uses that message ID as `parentMessageId`, including responses that start with a tool and contain no text.

Function and output tools emit start, argument, end, and result events. Provider-native tool calls use the same AG-UI lifecycle. Stopping the AG-UI consumer stops the underlying agent event iteration.

## Current scope

The adapter supports text and multimodal user messages, assistant reasoning, function calls, structured tool results, secure client-held history, versioned thinking and reasoning streams, normalized text and tool streaming, event timestamps, cancellation, standalone event transformation, generated protocol IDs, and SSE HTTP responses.

The adapter covers the audited AG-UI message, interrupt, and event surface.
