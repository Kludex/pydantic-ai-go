package openaicodex_test

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaicodex"
)

type echoArgs struct {
	Value string `json:"value"`
}

func TestAgentToolRoundTrip(t *testing.T) {
	var mu sync.Mutex
	requests := 0
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		mu.Lock()
		defer mu.Unlock()
		requests++
		if request.Header.Get("Authorization") != "Bearer access" {
			t.Fatalf("unexpected authorization: %q", request.Header.Get("Authorization"))
		}
		if requests == 1 {
			return codexEventStream(
				`{"type":"response.output_item.added","item":{"id":"call-item","type":"function_call",`+
					`"call_id":"call","name":"echo","arguments":""}}`,
				`{"type":"response.function_call_arguments.delta","item_id":"call-item",`+
					`"delta":"{\"value\":\"Moo!\"}"}`,
				`{"type":"response.completed","response":{"id":"first","model":"gpt-5.6-luna",`+
					`"status":"completed","output":[],"usage":{}}}`,
			), nil
		}
		return codexEventStream(
			`{"type":"response.output_item.added","item":{"id":"message","type":"message"}}`,
			`{"type":"response.output_text.delta","item_id":"message","delta":"Moo!"}`,
			`{"type":"response.completed","response":{"id":"second","model":"gpt-5.6-luna",`+
				`"status":"completed","output":[],"usage":{}}}`,
		), nil
	})
	model, err := openaicodex.NewModel("gpt-5.6-luna",
		openaicodex.WithCredentials(openaicodex.Credentials{
			AccessToken: "access", RefreshToken: "refresh", AccountID: "account",
		}),
		openaicodex.WithHTTPClient(&http.Client{Transport: transport}),
	)
	if err != nil {
		t.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](model, ai.WithInstructions("Call echo once."))
	calls := 0
	ai.AddSimpleTool(agent, "echo", func(_ context.Context, args echoArgs) (string, error) {
		calls++
		return args.Value, nil
	})
	result, err := agent.Run(t.Context(), "Use the tool.", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Moo!" || calls != 1 || requests != 2 {
		t.Fatalf("unexpected tool round trip: output=%q calls=%d requests=%d", result.Output, calls, requests)
	}
}

func codexEventStream(events ...string) *http.Response {
	lines := make([]string, len(events))
	for index, event := range events {
		lines[index] = fmt.Sprintf("data: %s\n", event)
	}
	result := response(http.StatusOK, strings.Join(lines, "\n"))
	result.Header.Set("Content-Type", "text/event-stream")
	return result
}
