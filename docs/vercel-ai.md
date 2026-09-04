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

## Send files

Use a `file` part with a hosted URL or a base64 data URL:

```console
$ curl http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -d '{
    "trigger": "submit-message",
    "id": "chat-1",
    "messages": [{
      "id": "user-1",
      "role": "user",
      "parts": [
        {"type": "text", "text": "Describe this image."},
        {"type": "file", "mediaType": "image/png", "url": "https://example.com/image.png"}
      ]
    }]
  }'
```

The media type selects image, audio, video, or document input. The default sanitizer permits only `http` and `https` URLs. Inline data URLs are decoded before the model request. Generated files stream back as `file` chunks with base64 data URLs.

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

## Approve deferred tools

Set `SDKVersion` to 6 or 7 to emit `tool-approval-request` chunks for tools registered with `WithApprovalRequired`. Return the original assistant tool part with an approval response:

```json
{
  "type": "tool-delete_record",
  "toolCallId": "call_delete",
  "state": "approval-responded",
  "input": {"id": "record-1"},
  "approval": {
    "id": "call_delete",
    "approved": false,
    "reason": "Keep this record"
  }
}
```

`approved` is a strict JSON boolean. Missing decisions deny by default. The adapter resumes the original tool-call ID and does not create a second call.

To change the arguments before execution, replace the tool part's `input` value and approve it. The adapter validates and executes that replacement instead of the model-generated arguments.

## Resume externally executed tools

Register external tools with `AddExternalTool`. The adapter stores their pending call IDs in the final assistant message metadata. Preserve that metadata while your application executes each call. Return every result in the trailing assistant message:

```json
{
  "trigger": "submit-message",
  "id": "chat-1",
  "messages": [
    {
      "id": "user-1",
      "role": "user",
      "parts": [{"type": "text", "text": "Run the report."}]
    },
    {
      "id": "assistant-1",
      "role": "assistant",
      "metadata": {
        "pydantic_ai": {
          "external_tool_call_ids": ["call_report"]
        }
      },
      "parts": [
        {
          "type": "tool-report",
          "toolCallId": "call_report",
          "state": "output-available",
          "input": {"month": "August"},
          "output": {"total": 42}
        }
      ]
    }
  ]
}
```

Use `output-error` and `errorText` when execution fails. The adapter resumes the original call IDs as one deferred result batch. It rejects missing, duplicate, incomplete, and malformed result metadata. This prevents a partial batch from rerunning completed work.

## Stream custom data and sources

```go
package main

import (
    "context"
    "net/http"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/openai"
    "github.com/Kludex/pydantic-ai-go/ui/vercel"
)

type ReportArgs struct {
    Topic string `json:"topic"`
}

func main() {
    agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
    ai.AddSimpleTool(agent, "report", func(
        _ context.Context, args ReportArgs,
    ) (ai.ToolReturn, error) {
        metadata, err := vercel.ToolResultMetadata(
            vercel.Chunk{Type: "data-progress", Data: map[string]any{"percent": 100}},
            vercel.Chunk{
                Type:     vercel.ChunkSourceURL,
                SourceID: "source-1",
                URL:      "https://example.com/report",
                Title:    "Report source",
            },
        )
        if err != nil {
            return ai.ToolReturn{}, err
        }
        return ai.ToolReturn{ReturnValue: "Report complete: " + args.Topic, Metadata: metadata}, nil
    })

    adapter := vercel.NewAdapter(agent, vercel.Config{})
    if err := http.ListenAndServe(":8080", adapter.Handler(struct{}{})); err != nil {
        panic(err)
    }
}
```

`ToolResultMetadata` accepts custom `data-*`, `source-url`, `source-document`, and `file` chunks. The adapter emits them after the tool output. It rejects control chunks so tool metadata cannot corrupt stream lifecycle state. The helper copies every chunk before the run stores it.

Set `Transient` to `true` on a custom data chunk when the frontend should not retain it in message history. Retained custom data and source parts remain UI-only when the client sends the history back. They are not added to the next model request.

## Preserve compaction and discovered tools

The adapter uses `data-compaction` parts for provider compaction boundaries. It uses `data-tool-availability-delta` parts for tools revealed during a run. Keep these data parts in client-held history so later requests preserve the compacted context and deferred-tool visibility.

Unrecognized `data-*` parts remain UI-only and are not sent to the model.

## Preserve metadata

The adapter stores part metadata under `providerMetadata.pydantic_ai`. It round-trips provider part IDs, names, details, reasoning signatures, tool kinds, and file metadata.

Message metadata remains application-owned. The adapter reserves `metadata.pydantic_ai.timestamp` for the original message timestamp and `metadata.pydantic_ai.external_tool_call_ids` for deferred external work. Preserve reserved values exactly. It does not trust client-held provider response IDs or provider URLs.

A canceled run emits an `abort` chunk followed by `[DONE]`. This keeps cancellation distinct from model and adapter errors.

## Current scope

The adapter supports AI SDK UI versions 5 through 7 for text, files, reasoning, function and provider-native tool inputs and outputs, approval and external-tool resumes, compaction boundaries, tool-availability changes, step boundaries, finish reasons, secure client-held history, standalone transformation, and bounded SSE HTTP serving.

Remaining version-specific fields remain.
