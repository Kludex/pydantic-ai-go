package ai_test

import (
	"context"
	"fmt"
	"os"

	"github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func ExampleWorkspace() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "agent-workspace-")
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := os.RemoveAll(directory); err != nil {
			panic(err)
		}
	}()
	local, err := ai.NewLocalWorkspace(directory, nil)
	if err != nil {
		panic(err)
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(local))
	ai.AddTool(agent, "save_note", func(ctx context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		return "saved", rc.Workspace.WriteText(ctx, "notes/today.txt", "ready")
	})
	result, err := agent.Run(ctx, "Save a note.", struct{}{})
	if err != nil {
		panic(err)
	}
	text, err := result.Workspace().ReadText(ctx, "notes/today.txt")
	if err != nil {
		panic(err)
	}
	fmt.Println(text)
	reviewer := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	review, err := reviewer.Run(ctx, "Review the note.", struct{}{},
		ai.WithRunWorkspace(ai.ReadOnlyWorkspace(result.Workspace())),
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(review.Workspace().ReadOnly())
	// Output:
	// ready
	// true
}
