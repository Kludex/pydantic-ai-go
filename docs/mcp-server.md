# MCP servers

Expose an agent as a typed MCP tool. The official MCP Go SDK infers the input and output schemas and serves the protocol over standard input and output.

```go
package main

import (
	"context"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

type AskInput struct {
	Question string `json:"question" jsonschema:"A question for the agent"`
}

type AskOutput struct {
	Answer string `json:"answer"`
}

func main() {
	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-5-mini"),
		ai.WithAgentName("mcp-assistant"),
		ai.WithInstructions("Answer accurately and concisely."),
	)
	server := mcpsdk.NewServer(&mcpsdk.Implementation{
		Name: "pydantic-ai-go-agent", Version: "1.0.0",
	}, nil)
	mcpsdk.AddTool(server, &mcpsdk.Tool{
		Name: "ask", Description: "Ask the agent a question",
	}, func(
		ctx context.Context, _ *mcpsdk.CallToolRequest, input AskInput,
	) (*mcpsdk.CallToolResult, AskOutput, error) {
		result, err := agent.Run(ctx, input.Question, struct{}{})
		if err != nil {
			return nil, AskOutput{}, err
		}
		return nil, AskOutput{Answer: result.Output}, nil
	})
	if err := server.Run(context.Background(), &mcpsdk.StdioTransport{}); err != nil {
		panic(err)
	}
}
```

The complete source is available at [`examples/mcp-server/main.go`](../examples/mcp-server/main.go).

The request context carries client cancellation and deadlines into the agent. Keep the agent outside the handler so model configuration, connections, and concurrency controls are reused. Pass tenant-scoped clients and authorization data through typed dependencies instead of global state.

The typed SDK handler turns an agent error into an MCP tool error. The client model can inspect that error and choose another action. Return a protocol error only for server failures that the client cannot correct.

## Client-provided sampling

`SamplingModel` adapts MCP sampling to the provider-neutral `Model` interface. It translates text, image, audio, function tools, structured output tools, retry results, stop reasons, and client-selected model identity. `SamplingSettings` adds detached `ModelPreferences`. `WithSamplingDefaultMaxTokens` changes the required MCP fallback from 16,384 tokens.

MCP sampling requires informed user approval in the client. Do not use it to bypass the client's model or data-access policy.

!!! warning "Sampling changed in the 2026-07-28 protocol"

    MCP deprecated sampling and direct server-initiated requests in protocol version `2026-07-28`. The official SDK's `ServerSession` implements `SamplingSession` for older negotiated protocol versions. New protocol handlers must return sampling requests through `CallToolResult.InputRequests` and resume from `InputResponses`.

    Use a directly configured provider model inside the MCP handler, as shown above, for new servers. A future resumable agent adapter is required before a current-protocol multi-round-trip sampling request can transparently suspend and continue an agent loop without repeating tool side effects.

`SamplingSession` is a small interface. You can use an alternate transport implementation or a deterministic fake without depending on an SDK session concrete type. A tool-enabled session must preserve tool-use IDs so the agent can pair concurrent calls and results.

## Transport security

A stdio server inherits the launching process's credentials. Restrict which clients can launch it and keep protocol output off standard output.

For Streamable HTTP, authenticate before the MCP handler. Derive dependencies from the authenticated request. Do not accept a tenant or user ID from tool arguments as proof of identity.

Keep model credentials on the server when the server pays for generation. Use sampling only when the client deliberately owns model selection and billing.
