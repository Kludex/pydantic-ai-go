// Command mcp-server exposes an agent through an MCP tool over stdio.
package main

import (
	"context"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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
