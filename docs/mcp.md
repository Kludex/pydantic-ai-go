# MCP clients

Use an MCP server as a toolset or connect a shared session.

## Create a toolset for each run

Use a run-scoped toolset for most agents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	agent.AddToolset(aimcp.NewStreamableHTTPToolset[struct{}](
		"http://localhost:8000/mcp",
		aimcp.WithID("catalog"),
	))

	result, err := agent.Run(
		context.Background(),
		"Find the installation guide in the catalog.",
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

The toolset opens a fresh MCP session for the run. It closes the session after the run completes, fails, pauses, or is canceled. This default prevents state and credentials from leaking between concurrent runs.

The server's tools and instructions are refreshed before each model step. Tool input and output schemas, annotations, and metadata are preserved.

## Share an explicit session

Use `Connect` when several runs must share one server session or when you need prompts and resources directly:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	ctx := context.Background()
	session, err := aimcp.Connect(
		ctx,
		&mcpsdk.StreamableClientTransport{Endpoint: "http://localhost:8000/mcp"},
		aimcp.WithID("catalog"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := session.Close(); err != nil {
			log.Printf("close MCP session: %v", err)
		}
	}()

	prompts, err := session.ListPrompts(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, prompt := range prompts {
		fmt.Println(prompt.Name)
	}

	resources, err := session.ListResources(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, resource := range resources {
		fmt.Println(resource.URI)
	}

	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	agent.AddToolset(aimcp.NewSessionToolset[struct{}](session))

	result, err := agent.Run(ctx, "Use the catalog tools.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

You own a shared session's lifecycle. Stop every agent run that uses it before calling `Close`. `SessionToolset` never closes the session.

A `Session` is safe for concurrent direct operations and agent runs. `WithReadTimeout` bounds each list, prompt, resource, ping, and tool request. `WithInitTimeout` bounds connection and initialization.

## Use prompts, resources, and tools directly

A shared session exposes these operations:

| Operation | Method |
| --- | --- |
| Check the connection | `Ping` |
| Inspect negotiated server state | `InitializeResult` |
| List and render prompts | `ListPrompts`, `GetPrompt` |
| List and read resources | `ListResources`, `ListResourceTemplates`, `ReadResource` |
| List and call tools | `ListTools`, `CallTool` |

Returned protocol values are detached copies. You can modify them without changing SDK caches or later results.

`CallTool` returns MCP tool failures through `CallToolResult.IsError`. It returns a Go error only when the protocol request fails. `SessionToolset` converts the same tool failure into the configured agent behavior: retry by default, a failed result with `ToolErrorFailed`, or a run error with `ToolErrorAbort`.

## Choose a transport

Use the transport that your server supports:

| Transport | Constructor |
| --- | --- |
| Streamable HTTP | `NewStreamableHTTPToolset` or `mcpsdk.StreamableClientTransport` |
| Local subprocess | `NewCommandToolset` or `mcpsdk.CommandTransport` |
| Legacy HTTP and SSE | `NewSSEToolset` or `mcpsdk.SSEClientTransport` |
| Custom per-run transport | `NewToolset` with `TransportFactory` |

Pass `WithClientOptions` to configure server-initiated MCP handlers and notifications from the official Go SDK. Pass `WithSessionOptions` to configure each protocol session.

## Load `.mcp.json`

Load the `mcpServers` format used by Claude Desktop and Cursor:

```go
package main

import (
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	toolsets, err := aimcp.LoadToolsets[struct{}](".mcp.json")
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	for _, toolset := range toolsets {
		agent.AddToolset(toolset)
	}
}
```

Each server name becomes the toolset ID and a tool-name prefix. HTTP entries support headers. Command entries support arguments, environment overrides, and a working directory. String values support `${NAME}` and `${NAME:-default}` expansion.

> [!WARNING]
> Treat `.mcp.json` as trusted input. Command entries execute local programs. Environment expansion can read process environment variables.
