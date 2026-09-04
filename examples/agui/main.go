// Command agui serves an OpenAI agent through an AG-UI SSE endpoint.
package main

import (
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/ui/agui"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	adapter := agui.NewAdapter(agent, agui.Config{})

	server := &http.Server{Addr: ":8080", Handler: adapter.Handler(struct{}{})}
	if err := server.ListenAndServe(); err != nil {
		panic(err)
	}
}
