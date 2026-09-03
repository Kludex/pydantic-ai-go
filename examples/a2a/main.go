// Command a2a serves an OpenAI agent through the official A2A JSON-RPC transport.
package main

import (
	"net/http"

	protocol "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	ai "github.com/Kludex/pydantic-ai-go"
	a2aintegration "github.com/Kludex/pydantic-ai-go/a2a"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5-mini"))
	executor := a2aintegration.NewExecutor(agent, a2aintegration.Config[struct{}]{})
	handler := a2asrv.NewHandler(executor)

	card := &protocol.AgentCard{
		Name: "Assistant", Description: "Answers general questions.", Version: "1.0.0",
		ProtocolVersion: string(protocol.Version), URL: "http://localhost:8080/invoke",
		PreferredTransport: protocol.TransportProtocolJSONRPC,
		Capabilities:       protocol.AgentCapabilities{Streaming: true},
		DefaultInputModes:  []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		Skills: []protocol.AgentSkill{{
			ID: "answer", Name: "Answer questions", Description: "Answers a question.", Tags: []string{"assistant"},
		}},
	}
	mux := http.NewServeMux()
	mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(handler))
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	server := &http.Server{Addr: ":8080", Handler: mux}
	if err := server.ListenAndServe(); err != nil {
		panic(err)
	}
}
