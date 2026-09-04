# Terminal and web chat

## Run terminal chat

```console
$ OPENAI_API_KEY=... go run ./cmd/pydantic-ai-go -model openai:gpt-5-mini
> Explain structured concurrency.
```

The command streams assistant text as it arrives. Function calls print as `[tool]` and results print as `[tool result]`.

Use `/usage` for session totals. Use `/clear` to discard history and totals. Use `/exit` or `/quit` to stop.

Load common `mcpServers` JSON with `-mcp-config`:

```console
$ go run ./cmd/pydantic-ai-go \
    -model anthropic:claude-sonnet-4-5 \
    -mcp-config ./mcp.json
```

The command supports `openai`, `openai-responses`, `anthropic`, `google`, `bedrock`, `bedrock-mantle`, `cerebras`, `cohere`, `crusoe`, `groq`, `huggingface`, `mistral`, `ollama`, `openrouter`, `snowflake`, `xai`, and `zai` provider prefixes.

## Run terminal chat from your program

```go
package main

import (
    "context"

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/cli"
    "github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func main() {
    agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
    if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
        MCPConfigPath: "mcp.json",
    }); err != nil {
        panic(err)
    }
}
```

`cli.Config` accepts custom input and output streams for embedding or tests. History and run options are detached before each turn.

## Run web chat

```console
$ OPENAI_API_KEY=... go run ./cmd/pydantic-ai-go web \
    -model openai:gpt-5-mini \
    -listen :8080 \
    -allowed-hosts localhost
```

Open <http://localhost:8080>. The command serves PydanticAI's official `@pydantic/ai-chat-ui`. The browser streams text, renders rich content, and handles tool execution and approval states.

IP addresses, `localhost`, and names below `.localhost` are accepted by default. `-allowed-hosts` adds comma-separated host names. `*.example.com` accepts subdomains only. Use `*` only when an authenticated proxy protects the server.

## Serve web chat from your program

```go
package main

import (
    "net/http"

    ai "github.com/Kludex/pydantic-ai-go/ai"
    "github.com/Kludex/pydantic-ai-go/ai/models/openai"
    "github.com/Kludex/pydantic-ai-go/ai/webchat"
)

func main() {
    model := openai.NewModel("gpt-5-mini")
    agent := ai.NewAgent[struct{}, string](model)
    handler, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{
        DefaultModelID: "openai:gpt-5-mini",
        Models: []webchat.ModelOption{
            {ID: "openai:gpt-5.2", Name: "GPT-5.2", Model: openai.NewModel("gpt-5.2")},
        },
        NativeTools:  []ai.NativeTool{ai.WebSearchTool{}, ai.CodeExecutionTool{}},
        AllowedHosts: []string{"ui.example.com"},
        MCPConfigPath: "mcp.json",
    })
    if err != nil {
        panic(err)
    }
    server := &http.Server{Addr: ":8080", Handler: handler}
    if err := server.ListenAndServe(); err != nil {
        panic(err)
    }
}
```

The handler fetches `@pydantic/ai-chat-ui` from `webchat.DefaultHTMLURL` and caches it in your user cache directory. Set `HTMLSource` to `webchat.OfflineHTMLURL`, another HTTP URL, or a local file. `CacheDir` overrides remote HTML caching, and `HTTPClient` controls remote fetching.

The official UI reads model and native-tool choices from `/api/configure`, streams Vercel AI SDK v7 from `/api/chat`, and checks `/api/health`. The agent's model is included automatically. Set `DefaultModelID` or `DefaultModelName` to customize it. Add other choices with `Models`.

The chat endpoint requires `Content-Type: application/json` and refuses cross-origin preflights. Use the AG-UI or Vercel AI adapters directly for a custom frontend.

## Resolve model names

```go
package main

import (
    "fmt"

    "github.com/Kludex/pydantic-ai-go/ai/models/infer"
)

func main() {
    model, err := infer.Model("groq:openai/gpt-oss-20b")
    if err != nil {
        panic(err)
    }
    fmt.Println(model.Name())
}
```

Use `infer.WithProvider` to register an application-specific prefix. Resolvers must be safe for concurrent calls and must not return nil models.
