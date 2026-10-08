# OpenAI GPT-Live

## Open a voice session

```go
package main

import (
    "context"
    "fmt"
    "os"
    "os/signal"
    "time"

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
)

func main() {
    pcm, err := os.ReadFile("question.pcm")
    if err != nil {
        panic(err)
    }
    ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
    defer stop()

    model := openairt.NewLiveModel("gpt-live-1+gpt-5",
        openairt.WithLiveSettings(openairt.LiveSettings{Voice: "marin"}),
    )
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{
        Request: ai.ModelRequestParams{
            Instructions: "Answer accurately. Search when you need current information.",
            NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
        },
        Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{}},
    })
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(context.Background()) }()

    audioCtx, cancelAudio := context.WithCancel(ctx)
    defer cancelAudio()
    go func() {
        ticker := time.NewTicker(20 * time.Millisecond)
        defer ticker.Stop()
        for offset := 0; ; {
            select {
            case <-audioCtx.Done():
                return
            case <-ticker.C:
                chunk := make([]byte, 960)
                offset += copy(chunk, pcm[offset:])
                if err := session.SendAudio(audioCtx, chunk, "audio/pcm"); err != nil {
                    return
                }
            }
        }
    }()

    for event, err := range session.Events(ctx) {
        if err != nil {
            panic(err)
        }
        if turn, ok := event.(realtime.TurnCompleteEvent); ok {
            for _, part := range turn.Response.Parts {
                if speech, ok := part.(ai.SpeechPart); ok && speech.Transcript != nil {
                    fmt.Println(*speech.Transcript)
                }
            }
            break
        }
    }
}
```

Set `OPENAI_API_KEY`. Supply `question.pcm` as little-endian, mono PCM16 at 24 kHz. The example keeps sending silence after the recording ends. GPT-Live advances its timeline only while input audio flows, including while the backend works. Use `session.StreamAudio` when you also want to play the generated speech.

`NewLiveModel` uses `/live/sessions`, not the Realtime API. `infer.Model("openai:gpt-live-1+gpt-5")` selects the same adapter. You can reuse `WithAPIKey`, `WithBaseURL`, `WithHTTPClient`, `WithHeaders`, and `WithProfile`.

## Delegation and settings

A delegation is work that the voice model sends to a Responses backend. Your `ConnectParams.Request.Instructions`, function tools, and native web search go to that backend. `LiveSettings.Instructions` controls the voice model's speaking style instead. Register a `realtime.WithToolExecutor` to execute function calls through the existing [session tool API](realtime.md#execute-tools).

| Setting | Behavior |
| --- | --- |
| Model suffix, such as `+gpt-5` | Selects the delegated Responses model. |
| `LiveSettings.Delegation["model"]` | Overrides the suffix. An empty model or `auto` selects `gpt-6-sol`. |
| `LiveSettings.Delegation` | Accepts extra `instructions`, `max_output_tokens`, `parallel_tool_calls`, `reasoning_effort`, `verbosity`, and `service_tier`. |
| `realtime.Settings.Thinking` | Configures the backend's reasoning. An explicit delegated `reasoning_effort` takes precedence. |
| `realtime.Settings.ParallelToolCalls` | Configures the backend. An explicit delegated value takes precedence. |
| `LiveSettings.TurnSilence` | Defaults to two seconds. Delegated work suspends the silence clock so a tool's wait does not end the turn. |
| `LiveSettings.Store` | Defaults to false. With reconnect enabled, stored sessions are forked; other sessions replay recent text history. |
| `LiveSettings.DataChannel` | Controls browser event permissions. Both event lists are empty by default to keep instructions and tools server-owned. |

Session-level `Settings.Provider` values override the matching model defaults: `openai_voice`, `openai_live_instructions`, `openai_live_delegation`, `openai_live_turn_silence_ms`, `openai_live_store`, and `openai_live_data_channel`. The silence value uses milliseconds.

Tool results can include text, images, documents, and OpenAI uploaded files. They reach the backend as Responses input items. Audio and video attachments are rejected before any result is sent. Parallel calls continue only after every requested result arrives and the response that requested them ends.

## Protocol limits

- `Send(ctx, "text")` appends speakable context. `WithResponse(false)` appends silent context. Neither is an ordinary user-text turn. Each append accepts at most 500 `o200k_base` tokens.
- Images require `WithResponse(true)`. The backend sees the image; the voice model does not.
- Initial history accepts text and speech transcripts. Tool rounds become readable user and assistant text. Raw media cannot be seeded. Reconnect drops media and retains up to 128 recent items within an 8192-token budget.
- Input and output PCM rates must match at 16 or 24 kHz. WebRTC negotiates its own format.
- GPT-Live rejects text-only output, VAD configuration, spoken token limits, transcription-model selection, required tool choice, manual turns, cancellation, and truncation.

!!! warning "Turn completion is inferred"
    GPT-Live has no response terminal event. `Profile.SynthesizesTurnBoundary` is true. A long pause can end a turn before the speaker has finished. Do not treat the turn boundary as confirmation that a task succeeded.

## Usage and errors

`session.Usage().AudioSeconds` accumulates increments from GPT-Live's cumulative duration reports. Backend token usage includes cache and reasoning counters. Backend metadata appears as `delegated_model` and `delegated_response_id` on recorded responses. Spoken turn boundaries do not add backend requests.

`session.ContextWindowUsed()` uses the provider's latest occupancy ratio, including reports without new billed seconds. Backend tokens do not estimate the voice model's context occupancy.

Closing an owned session sends `session.close` and waits up to two seconds for final usage. Cancellation skips that wait. A sideband close leaves browser media running and cannot request final duration. If the provider does not respond, the final unreported seconds remain unknown.

Malformed known events produce recoverable `SessionErrorEvent` values. Unknown events are ignored. Inspect provider failures with `errors.As` and `*openai.LiveError`. An unavailable backend or an abnormal session close ends the session. Failed delegations are recoverable, and late results for abandoned calls are not sent.

## Browser WebRTC

Relay the browser's offer with `model.AnswerWebRTCOffer`. GPT-Live uses a JSON `POST /live/sessions` request. Attach server control with `model.ConnectWebRTC(ctx, answer.Session, params)`.

The offer fixes the session configuration. A sideband cannot reseed history, reconfigure the call, or reconnect automatically. It tracks audio for turn completion but does not forward playback audio. GPT-Live has no ephemeral client secrets.

Call `model.HangUp(ctx, answer.Session)` to end browser media. Closing the sideband alone does not hang up. The Realtime API's `openai.Model` also supports `HangUp` through its separate call endpoint.
