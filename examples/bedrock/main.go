// Command bedrock runs a minimal agent through Amazon Bedrock Converse.
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/bedrock"
)

func main() {
	model := bedrock.NewModel("us.amazon.nova-lite-v1:0")
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithInstructions("Answer in one short sentence."),
	)

	result, err := agent.Run(context.Background(), "Why is the sky blue?", struct{}{})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
