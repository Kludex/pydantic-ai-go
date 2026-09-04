# Realtime sessions

## Open an OpenAI Realtime session

```go
package main

import (
    "context"
    "fmt"
    "os/signal"

    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
)

func main() {
    ctx, stop := signal.NotifyContext(context.Background())
    defer stop()

    model := openairt.NewModel("gpt-realtime")
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{})
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(context.Background()) }()

    if err := session.Send(ctx, "Explain why the sky is blue in one sentence."); err != nil {
        panic(err)
    }
    for event, err := range session.Events(ctx) {
        if err != nil {
            panic(err)
        }
        if turn, ok := event.(realtime.TurnCompleteEvent); ok {
            fmt.Println(turn.Response.Text())
            break
        }
    }
}
```

`realtime.Open` owns the connection pump and background tool calls. Close the session when you stop consuming events. The session retains portable `ai.ModelMessage` history through `Messages` and `NewMessages`.

OpenAI reads `OPENAI_API_KEY`. Use `openai.WithAPIKey`, `openai.WithBaseURL`, `openai.WithHTTPClient`, and `openai.WithHeaders` when you need explicit transport configuration.

## Use Azure OpenAI or Voice Live

```go
package main

import (
    "context"
    "os"

    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    azurert "github.com/Kludex/pydantic-ai-go/ai/realtime/azure"
)

func main() {
    ctx := context.Background()
    model, err := azurert.NewModel("gpt-realtime", azurert.Config{
        Endpoint: os.Getenv("AZURE_OPENAI_ENDPOINT"),
        APIKey: os.Getenv("AZURE_OPENAI_API_KEY"),
    })
    if err != nil {
        panic(err)
    }
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{})
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(ctx) }()

    if err := session.Send(ctx, "Say hello."); err != nil {
        panic(err)
    }
}
```

Azure OpenAI uses `/openai/v1/realtime`. Pass `Config.TokenProvider` to use a fresh Microsoft Entra token instead of an API key.

Set `azure_voice_live` in `realtime.Settings.Provider` or use `azure.WithSettings` to select Azure AI Voice Live for models served by both APIs. Voice Live-only model families route there automatically. Configure its separate endpoint, key, and API version through `Config.VoiceLiveEndpoint`, `Config.VoiceLiveAPIKey`, and `Config.VoiceLiveAPIVersion`.

Azure OpenAI supports ephemeral client secrets and WebRTC through `CreateClientSecret`, `AnswerWebRTCOffer`, and `ConnectWebRTC`. Voice Live uses a different signaling protocol and rejects those methods.

## Stream audio

```go
package main

import (
    "context"
    "fmt"

    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
)

func main() {
    ctx := context.Background()
    model := openairt.NewModel("gpt-realtime")
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{},
        realtime.WithAudioRetention(realtime.AudioRetentionAll),
    )
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(ctx) }()

    fmt.Printf("input=%d Hz output=%d Hz\n",
        session.AudioInputSampleRate(), session.AudioOutputSampleRate())

    pcm16 := []byte{0, 0, 0, 0}
    if err := session.SendAudio(ctx, pcm16, "audio/pcm"); err != nil {
        panic(err)
    }
    if err := session.CommitAudio(ctx); err != nil {
        panic(err)
    }
    if err := session.CreateResponse(ctx); err != nil {
        panic(err)
    }

    for chunk, err := range session.StreamAudio(ctx) {
        if err != nil {
            panic(err)
        }
        fmt.Printf("received %d PCM bytes\n", len(chunk))
        break
    }
}
```

Raw audio is mono PCM16. Read the input and output rates from the model or session. OpenAI and xAI use 24 kHz input. Gemini Live uses 16 kHz input and 24 kHz output.

Set `WithAudioRetention` when you need raw audio in portable history. Retained audio is stored as WAV. Live audio events remain raw PCM.

## Execute tools

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
)

type input struct {
    City string `json:"city"`
}

func main() {
    ctx := context.Background()
    model := openairt.NewModel("gpt-realtime")
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{
        Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
            Name: "weather",
            Description: "Return the current weather.",
            Schema: map[string]any{
                "type": "object",
                "properties": map[string]any{"city": map[string]any{"type": "string"}},
                "required": []string{"city"},
            },
        }}},
    }, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
        _ context.Context, call ai.ToolCallPart,
    ) (any, error) {
        var arguments input
        if err := json.Unmarshal(call.Args, &arguments); err != nil {
            return nil, ai.Retryf("city is required")
        }
        return fmt.Sprintf("Sunny in %s", arguments.City), nil
    })))
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(ctx) }()

    if err := session.Send(ctx, "What is the weather in London?"); err != nil {
        panic(err)
    }
    for event, err := range session.Events(ctx) {
        if err != nil {
            panic(err)
        }
        if result, ok := event.(ai.FunctionToolResultEvent); ok {
            fmt.Printf("tool result: %#v\n", result.Part)
            break
        }
    }
}
```

Tool calls execute concurrently with media streaming. A `RetryError` becomes model-visible corrective feedback. A `ToolFailedError` becomes a failed result without consuming a retry budget. Rich `ToolReturn` values preserve additional user content and metadata in history.

## Use Gemini Live

```go
package main

import (
    "context"
    "encoding/base64"

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    googlert "github.com/Kludex/pydantic-ai-go/ai/realtime/google"
)

func main() {
    ctx := context.Background()
    model := googlert.NewModel("gemini-2.5-flash-native-audio-latest")
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{})
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(ctx) }()

    frameData, err := base64.StdEncoding.DecodeString(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
    )
    if err != nil {
        panic(err)
    }
    frame := ai.BinaryContent{Data: frameData, MediaType: "image/png"}
    if err := session.Send(ctx, frame); err != nil {
        panic(err)
    }
}
```

Gemini Live uses the official `google.golang.org/genai` SDK. Pass `google.WithClient` to reuse a configured SDK client. Use `google.WithVertex` for Vertex AI. Gemini supports image frames and Google Search, but it does not support manual turn control or text-only output on current speech models.

## Use xAI Grok Voice

```go
package main

import (
    "context"

    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    xairt "github.com/Kludex/pydantic-ai-go/ai/realtime/xai"
)

func main() {
    ctx := context.Background()
    model := xairt.NewModel("grok-voice-latest")
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{})
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(ctx) }()

    if err := session.Send(ctx, "Say hello."); err != nil {
        panic(err)
    }
}
```

xAI reads `XAI_API_KEY`. Grok Voice always produces speech. It supports interruption but not output truncation. Reconnects use xAI conversation IDs so the provider restores the session without replaying local history.

## Reconnect

Set `Settings.Reconnect` to recover from dropped WebSocket sessions. OpenAI starts a configured replacement connection and replays finalized portable history. xAI resumes its provider conversation ID. Gemini uses the latest official SDK session-resumption handle.

A reconnect emits `SessionReconnectEvent`. `StateRestored` reports whether the provider restored in-flight state. Bounded attempts and total reconnect limits prevent a failing provider from reconnecting forever.

## Browser WebRTC

OpenAI supports ephemeral client secrets, SDP negotiation, and server-side control channels through `CreateClientSecret`, `AnswerWebRTCOffer`, and `ConnectWebRTC`. Keep your long-lived API key on the server. Pass the returned `WebRTCSession` to `ConnectWebRTC` so browser media and server tool execution share one provider call.
