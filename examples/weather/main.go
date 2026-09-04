// Command weather runs a minimal weather agent against OpenAI.
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

type Deps struct {
	DefaultUnit string
}

type WeatherArgs struct {
	City string `json:"city" jsonschema:"description=City name"`
}

func main() {
	model := openai.NewModel("gpt-4o-mini")
	agent := ai.NewAgent[Deps, string](model,
		ai.WithInstructions("You are a weather assistant."),
	)
	ai.AddTool(agent, "get_weather",
		func(ctx context.Context, rc *ai.RunContext[Deps], args WeatherArgs) (string, error) {
			return fmt.Sprintf("sunny, 21 %s in %s", rc.Deps.DefaultUnit, args.City), nil
		},
		ai.WithDescription("Get current weather for a city"),
	)

	result, err := agent.Run(context.Background(), "What's the weather in Berlin?", Deps{DefaultUnit: "C"})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
