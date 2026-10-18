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

`model.Profile().ContextWindow` uses the bundled `genai-prices` metadata when the model has a recorded limit. Zero means unknown. Set `realtime.ProfileOverride.ContextWindow` to replace the detected value, including with an explicit zero.

A string passed to `Send` asks the model to respond. Do not call `CreateResponse` after sending text. That asks for two responses. Pass `realtime.WithResponse(false)` to add passive text context. Images are passive by default. Pass `realtime.WithResponse(true)` with an image to ask for a response in the same operation.

OpenAI reads `OPENAI_API_KEY`. Use `openai.WithAPIKey`, `openai.WithBaseURL`, `openai.WithHTTPClient`, and `openai.WithHeaders` when you need explicit transport configuration.

## Use OpenAI GPT-Live

Use `openai.NewLiveModel("gpt-live-1+gpt-5")` for GPT-Live's separate audio-driven protocol. Your instructions and tools run on the delegated Responses backend. See [GPT-Live](gpt-live.md) for a complete example, settings, usage, reconnect, and WebRTC limits.

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

`azure_voice_live_turn_detection` accepts server VAD and Azure semantic VAD maps. End-of-utterance detection is accepted only by cascade models and fails before the WebSocket connection otherwise. `azure_voice_live_noise_reduction` configures Azure deep suppression or near/far-field processing. `azure_voice_live_echo_cancellation` enables server echo cancellation. On cascade models, the portable `openai_input_noise_reduction` setting maps to Azure deep suppression.

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

Call `StreamAudio` or `StreamTranscripts` before starting the goroutine that consumes it. The subscription starts at the method call, so scheduler delay does not lose the first chunk. Each audio view buffers five minutes of PCM, up to 30,000 chunks. Each transcript view buffers 512 updates. A view drops its oldest item only when its consumer falls behind that bound. A terminal receive error is returned by the first event, audio, transcript, outbound, or close operation that observes it. `Err` keeps the terminal error available for inspection.

Call `WaitForReply` after a responding send when you only need the model turn boundary. It does not consume `Events`, so another goroutine can continue handling events. `WaitForReply` returns immediately when no response is outstanding or when the session closes. Use `WaitForPlayback` separately when you also need buffered audio to finish playing.

A part's `Index` identifies it for the whole session. A provider can finish an output item after a later item started, so text of the item's transcript can arrive after its part ended. The session then streams that part again under the same `Index`, and the new `ai.PartStartEvent` replaces the part as it ended. `StreamTranscripts` yields the grown transcript under the same `Index` too. Keep parts by `Index` rather than adding one for every `ai.PartEndEvent`, and history still records one part per item.

Set `WithAudioRetention` when you need raw audio in portable history. Retained audio is stored as WAV. Live audio events remain raw PCM.

## Handle barge-in

Pass `realtime.WithBargeIn(true)` to `Open` when one device-paced `StreamAudio` consumer plays the output. The session tracks completed chunks, flushes unheard audio when user speech starts, and truncates provider history when supported. Automatic barge-in does nothing when the model profile does not support interruption.

For an application-owned trigger, read `PlayedAudioBytes` and pass the result to `InterruptAtAudio`. Keep your own playback counter when your audio layer buffers ahead of the device. You can pass an exact millisecond position to `Interrupt` instead.

A reply cut off this way is recorded as interrupted, with `ai.SpeechPart.InterruptedAtMS` on the speech part that was cut, even when it had already finished generating. A message you already read from `Messages` is never edited: a later snapshot holds an interrupted copy at the same position. The transcript stays what the model generated, because generation outruns playback.

When you hand that history to a text model or seed it into a new realtime session with `ConnectParams.Messages`, the transcript gets a marker: `[Interrupted after 640 ms]`, or `[Interrupted]` on the last speech part of an interrupted response with no recorded offset. Without it, the model reads the transcript as heard in full and may repeat itself. The marker is never stored in history.

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

A tool can call `session.Close(ctx)` to end the conversation. Closing from the running tool does not wait on itself. Its late result is not sent to the closed provider.

Use `session.Enqueue(ctx, ...)` to add an out-of-band prompt after the current response. Use `EnqueueWhenIdle` to place work after queued ASAP prompts. Both methods are safe to call from another goroutine and return an error after the session closes. Realtime enqueue accepts strings, `ai.TextContent`, `ai.UserPromptPart`, `ai.SystemPromptPart`, and matching `ai.ModelRequest` values. System prompt parts are wrapped in `<system>` tags because the live channel accepts only a user turn.

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

Gemini Live uses the official `google.golang.org/genai` SDK. Pass `google.WithClient` to reuse a configured SDK client. Use `google.WithVertex` for Vertex AI. `gemini-3.8-live` defaults tool declarations to asynchronous, so the adapter declares blocking tools explicitly. `gemini-3.8-live-extended-thinking` always uses asynchronous tools and defaults to low reasoning when you do not set an effort. Gemini supports image frames and Google Search, but it does not support manual turn control or text-only output on current speech models.

!!! note "Native Gemini function history needs SDK support"
    Seeded tool calls and results use readable text, including on `gemini-3.8-live`. The pinned `google.golang.org/genai` SDK does not expose `HistoryConfig` on `LiveConnectConfig`. Native function history needs that field to avoid triggering a reply during seeding. The adapter keeps the text fallback until the SDK supports it.

When you disable input transcription, Gemini 3.x can still send transcript frames. The adapter discards them so user speech stays out of portable history.

Gemini Developer API proactive audio requires `google.WithAPIVersion("v1alpha")`. Pass the same API version when you also provide a preconfigured SDK client. The adapter rejects proactive audio on the default `v1beta` connection before opening the session. Vertex AI uses a different version line and does not apply this check.

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

xAI reads `XAI_API_KEY`. Grok Voice always produces speech. It supports interruption but not output truncation. A reconnect starts a new provider conversation and replays finalized local history. xAI's native conversation resumption is not used because it drops typed user turns and tool calls.

xAI ends a conversation that runs past its maximum duration with a `max_duration` error. With `Settings.Reconnect`, the session reports it as a recoverable `SessionErrorEvent` and reconnects right away, so the call carries on past the limit in a new conversation that gets the replayed history. Without a policy, or once its reconnects are used up, the error ends the session.

## Resolve application providers

Pass `infer.WithProvider` to `infer.Model` when credentials or endpoints come from application state instead of environment variables. The resolver receives the provider-local model name and returns a configured `realtime.Model`. This matches the resolver pattern used by standard and embedding model inference.

## Reconnect

Set `Settings.Reconnect` to recover from dropped WebSocket sessions. OpenAI, Azure OpenAI, and xAI start configured replacement connections and replay finalized portable history. Gemini uses the latest official SDK session-resumption handle.

A reconnect emits `SessionReconnectEvent`. `StateRestored` reports whether the provider restored in-flight state. Bounded attempts and total reconnect limits prevent a failing provider from reconnecting forever.

## Browser WebRTC

OpenAI supports ephemeral client secrets, SDP negotiation, and server-side control channels through `CreateClientSecret`, `AnswerWebRTCOffer`, and `ConnectWebRTC`. Keep your long-lived API key on the server. Pass the returned `WebRTCSession` to `ConnectWebRTC` so browser media and server tool execution share one provider call.
