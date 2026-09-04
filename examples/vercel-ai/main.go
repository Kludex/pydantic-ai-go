// Command vercel-ai serves an OpenAI agent through the Vercel AI UI protocol.
package main

import (
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/ui/vercel"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	adapter := vercel.NewAdapter(agent, vercel.Config{})

	server := &http.Server{Addr: ":8080", Handler: adapter.Handler(struct{}{})}
	if err := server.ListenAndServe(); err != nil {
		panic(err)
	}
}
