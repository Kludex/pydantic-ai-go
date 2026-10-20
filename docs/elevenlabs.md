# ElevenLabs Agents

```go
package main

import (
    "context"

    "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/realtime"
    "github.com/Kludex/pydantic-ai-go/ai/realtime/elevenlabs"
)

func main() {
    ctx := context.Background()
    model := elevenlabs.NewModel("agent_0101", elevenlabs.WithSettings(elevenlabs.Settings{
        VoiceID:  "21m00Tcm4TlvDq8ikWAM",
        Language: "en",
    }))
    session, err := realtime.Open(ctx, model, realtime.ConnectParams{
        Request: ai.ModelRequestParams{Instructions: "You are a concise support agent."},
    })
    if err != nil {
        panic(err)
    }
    defer func() { _ = session.Close(ctx) }()

    if err := session.Send(ctx, "Where is my order?"); err != nil {
        panic(err)
    }
}
```

An ElevenLabs agent is a hosted pipeline: speech recognition, an LLM, ElevenLabs text-to-speech, and a server-side turn model, all behind one agent ID. `elevenlabs.NewModel` wraps that agent. The model reads `ELEVENLABS_API_KEY`, or you pass `elevenlabs.WithAPIKey`. You can also resolve it with `infer.Model("elevenlabs-agents:agent_0101")`.

Audio is 16 kHz mono PCM16 in both directions. If you configure the agent for another PCM rate, declare it with `elevenlabs.WithProfile`. The connect fails when the agent reports a format the profile does not match, so you never stream audio at the wrong rate.

## Enable overrides on the agent

Your instructions, voice, language, and other per-conversation settings replace the agent's own only when the agent allows it. Each one has an override toggle under **Security > Overrides** in the agent's settings.

| Input | Toggle |
| --- | --- |
| `Request.Instructions` | `agent.prompt.prompt` |
| `Settings.LLM` | `agent.prompt.llm` |
| `Settings.FirstMessage` | `agent.first_message` |
| `Settings.Language` | `agent.language` |
| `Settings.VoiceID` | `tts.voice_id` |
| `Settings.TTS` | `tts.model_id`, `tts.stability`, `tts.speed`, `tts.similarity_boost` |
| `OutputModality: realtime.OutputModalityText` | `conversation.text_only` |
| `ToolChoice: realtime.ToolChoiceNone` | `agent.prompt.tool_ids` |
| `Settings.CustomLLMExtraBody` | `custom_llm_extra_body` (platform setting) |

The model reads the agent before it dials. A missing toggle fails the connect with an error naming the toggle. Without this check, ElevenLabs would close the socket mid-handshake with no explanation.

`DynamicVariables` and `UserID` need no toggle. `ConfigOverride` is deep-merged last and is not checked, so a field the agent does not permit closes the socket.

## Sync tools

The agent stores its own client tool definitions. The model compares them with your run's tools before it dials. `Settings.ToolSync` decides what happens on a difference.

| Mode | Behavior |
| --- | --- |
| `elevenlabs.ToolSyncError` (default) | Fail the connect and list every difference. |
| `elevenlabs.ToolSyncSync` | Update the workspace tools over REST, then dial. |
| `elevenlabs.ToolSyncOff` | Trust the agent's tools as they are. |

You can also set the mode per session with the provider key `elevenlabs_agents_tool_sync`.

!!! warning "Sync changes shared workspace state"
    A synced tool changes for every conversation with the agent, including ones already running. A tool used by another agent is copied instead of edited, so the other agent keeps its definition.

ElevenLabs stores a narrower JSON Schema dialect: every parameter needs a description, and unions other than a nullable type cannot be expressed. Sync fails before any write when a tool does not fit. Add descriptions, simplify the types, or keep the tool configured in the dashboard with `ToolSyncOff`.

Run tools with `realtime.WithToolExecutor`, as for any [realtime session](realtime.md#execute-tools). Results go back as text. Media in a tool result fails the send because the agent's tool channel carries only text.

## Hosted MCP tools

Tools from MCP servers attached to the agent run on ElevenLabs. They appear in history as `ai.NativeToolCallPart` and `ai.NativeToolReturnPart` with the tool name `mcp_server:<service_id>`. A call that waits for approval is denied, because the session has no way to ask you, and a warning tells you to change the server's approval policy.

## Audio tags

Eleven v3 voices act on expressive tags such as `[laughs]`. The model removes them from transcripts so history holds what was said, and keeps them in `ProviderDetails["audio_tags"]` on the response. Set `Settings.KeepAudioTags` to keep them in the text. Bracketed text that is not a tag, such as `[1]`, `[Your Name]`, or a Markdown link, stays.

## Limits

| Feature | Behavior |
| --- | --- |
| Reconnect | Rejected. A new conversation would remember nothing. |
| Seeded history | Rejected. The agent cannot replay earlier turns. |
| Turn detection | Always on; disabling or tuning it is rejected. |
| Input transcription | Always on; disabling it is rejected. |
| `ToolChoiceRequired` | Rejected. Agents have no per-response tool choice. |
| Barge-in truncation | Not supported. The interrupted reply ends with what was sent. |
| Concurrency queue | A conversation that times out in the queue ends with a non-recoverable `SessionErrorEvent`. |

Enable the `context_usage` client event on the agent (**Advanced > Client events**) to get session usage with the context window fill. Without it the model warns and reports no usage.
