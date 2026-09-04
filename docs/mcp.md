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

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	agent.AddToolset(aimcp.NewHTTPToolset[struct{}](
		aimcp.HTTPToolsetConfig{URL: "http://localhost:8000/mcp"},
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

The server's tools and instructions are refreshed before each model step. Tool input and output schemas, annotations, and metadata are preserved. `NewHTTPToolset` uses legacy SSE for URLs ending in `/sse` and Streamable HTTP for every other HTTP URL.

## Prefer native MCP with a local fallback

Use one capability when the provider should manage MCP where supported and your application should connect otherwise:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	server := ai.MCPServerTool{
		ID:                 "catalog",
		URL:                "https://mcp.example.com/mcp",
		AuthorizationToken: os.Getenv("MCP_AUTH_TOKEN"),
		AllowedTools:       []string{"search"},
	}
	capability := aimcp.NewHTTPServerCapability[struct{}](
		aimcp.HTTPServerCapabilityConfig{Native: server},
	)
	agent := ai.NewAgent[struct{}, string](
		openai.NewResponsesModel("gpt-5"),
		ai.WithCapabilities(capability),
	)

	result, err := agent.Run(context.Background(), "Find the installation guide.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`NewHTTPServerCapability` applies the URL, authorization value, headers, server ID, and allowlist to both paths. It keeps provider-owned calls out of local function-tool execution. Cross-origin redirects never receive configured headers. Pass a caller-owned `http.Client` in `HTTPServerCapabilityConfig` when you need custom TLS, cookies, proxy behavior, or retries.

Use `ai.NewMCPServerCapability` when you already have a local toolset. Use `ai.NewDynamicMCPServerCapability` when dependencies select the provider-hosted URL or credentials for each request. Its stable `ID` must match every resolved `MCPServerTool.ID`; its `AllowedTools` value is authoritative for both paths.

## Share an explicit session

Use `Connect` when several runs must share one server session or when you need prompts and resources directly:

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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

## Handle sampling and elicitation

Let the server call a model through your MCP client and request approved structured input:

```go
package main

import (
	"context"
	"fmt"
	"log"

	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	ctx := context.Background()
	session, err := aimcp.Connect(
		ctx,
		&mcpsdk.StreamableClientTransport{Endpoint: "http://localhost:8000/mcp"},
		aimcp.WithSamplingModel(openai.NewModel("gpt-5-mini")),
		aimcp.WithElicitationHandler(func(
			_ context.Context,
			request *aimcp.ElicitationRequest,
		) (*aimcp.ElicitationResult, error) {
			fmt.Println(request.Params.Message)
			return &aimcp.ElicitationResult{
				Action:  "accept",
				Content: map[string]any{"project": "pydantic-ai-go"},
			}, nil
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	result, err := session.CallTool(ctx, "prepare_release", map[string]any{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Content)
}
```

`WithSamplingModel` maps the server's system prompt, text, image, audio, token limit, temperature, and stop sequences to a direct model request. It returns text to the server and never exposes model reasoning. Use `WithSamplingHandler` when you need complete control over the sampling result.

The SDK fulfills current-protocol multi-round-trip input requests and retries the original tool call automatically. This supports a server that pauses one operation for sampling or elicitation without storing client state.

Elicitation is a trust boundary. Show the server's request to the user and obtain informed approval before returning personal, secret, or destructive input. Returning `Action: "decline"` or `Action: "cancel"` sends no content.

Sampling models and handlers are mutually exclusive. Handler requests and results are detached, so a server cannot mutate application-owned values after the callback returns.

> [!NOTE]
> MCP deprecated sampling in protocol version 2026-07-28. The official SDK and this package keep it available during the protocol's compatibility window. Prefer a server-owned model integration for new systems.

## Choose a transport

Use the transport that your server supports:

| Transport | Constructor |
| --- | --- |
| Streamable HTTP | `NewStreamableHTTPToolset` or `mcpsdk.StreamableClientTransport` |
| Local subprocess | `NewCommandToolset` or `mcpsdk.CommandTransport` |
| Legacy HTTP and SSE | `NewSSEToolset` or `mcpsdk.SSEClientTransport` |
| Custom per-run transport | `NewToolset` with `TransportFactory` |

Pass `WithClientOptions` to configure server-initiated MCP handlers and notifications from the official Go SDK. Pass `WithSessionOptions` to configure each protocol session.

## Authorize Streamable HTTP with OAuth

Use the official SDK's authorization-code handler:

```go
package main

import (
	"context"
	"fmt"
	"log"

	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	mcpauth "github.com/modelcontextprotocol/go-sdk/auth"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
)

func main() {
	handler, err := mcpauth.NewAuthorizationCodeHandler(&mcpauth.AuthorizationCodeHandlerConfig{
		PreregisteredClient: &oauthex.ClientCredentials{ClientID: "my-client"},
		RedirectURL:         "http://127.0.0.1:8080/oauth/callback",
		AuthorizationCodeFetcher: func(
			_ context.Context,
			arguments *mcpauth.AuthorizationArgs,
		) (*mcpauth.AuthorizationResult, error) {
			fmt.Println("Open this URL:", arguments.URL)
			var code, state string
			fmt.Print("Authorization code and state: ")
			if _, err := fmt.Scanln(&code, &state); err != nil {
				return nil, err
			}
			return &mcpauth.AuthorizationResult{Code: code, State: state}, nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	session, err := aimcp.Connect(
		context.Background(),
		&mcpsdk.StreamableClientTransport{Endpoint: "https://example.com/mcp"},
		aimcp.WithOAuthHandler(handler),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	if err := session.Ping(context.Background()); err != nil {
		log.Fatal(err)
	}
}
```

`WithOAuthHandler` clones the Streamable HTTP transport before attaching the handler. It works with `Connect` and `NewStreamableHTTPToolset`. Other transports fail during setup because they cannot perform the MCP OAuth challenge flow.

The callback must return the code, state, and optional issuer from the redirect without changing them. The SDK verifies the state and advertised issuer. Store refresh tokens with operating-system or cloud secret storage. Use `AuthorizationCodeHandlerConfig.NewTokenSource` to wrap token persistence. The SDK coordinates token refreshes, retries one failed authorization, and avoids repeating a canceled interactive flow.

## Load `.mcp.json`

Load the `mcpServers` format used by Claude Desktop and Cursor:

```go
package main

import (
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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
