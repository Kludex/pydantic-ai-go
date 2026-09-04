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

Open <http://localhost:8080>. The browser streams text and shows tool start, completion, and failure states.

`-allowed-hosts` accepts comma-separated exact host names. The listen port is ignored during host matching. Configure this allowlist when the server is reachable through an untrusted proxy or network.

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
    agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
    handler, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{
        AllowedHosts:  []string{"localhost"},
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

The web endpoint uses the Vercel AI adapter and applies its secure untrusted-history sanitization. The built-in page is intentionally small. Use the AG-UI or Vercel AI adapters directly for a custom frontend.

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
