package openai_test

import (
	"context"
	"net/http"
	"os"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// vcrModel replays committed cassettes; with a real OPENAI_API_KEY and a
// missing cassette it records new interactions, filtering credentials.
func vcrModel(t *testing.T, name string) *openai.Model {
	t.Helper()
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("OPENAI_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	r, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(i *cassette.Interaction) error {
			delete(i.Request.Headers, "Authorization")
			return nil
		}, recorder.AfterCaptureHook),
		recorder.WithMatcher(cassette.MatcherFunc(func(r *http.Request, i cassette.Request) bool {
			return r.Method == i.Method && r.URL.String() == i.URL
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})
	return openai.NewModel("gpt-4o-mini", openai.WithHTTPClient(r.GetDefaultClient()))
}

func TestRecordedSimpleRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrModel(t, "simple_run"),
		ai.WithInstructions("Answer with a single word."),
	)
	result, err := agent.Run(t.Context(), "What is the capital of France?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output == "" {
		t.Fatal("expected non-empty output")
	}
	if result.Usage().TotalTokens() == 0 {
		t.Fatal("expected token usage from the API")
	}
}

func TestRecordedToolRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrModel(t, "tool_run"),
		ai.WithInstructions("Use the get_weather tool, then answer briefly."),
	)
	called := false
	ai.AddSimpleTool(agent, "get_weather", func(_ context.Context, args struct {
		City string `json:"city" jsonschema:"description=City name"`
	}) (string, error) {
		called = true
		return "sunny, 21C in " + args.City, nil
	}, ai.WithDescription("Get current weather for a city"))

	result, err := agent.Run(t.Context(), "What is the weather in Berlin?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("expected the tool to be called")
	}
	if result.Output == "" {
		t.Fatal("expected non-empty output")
	}
}

func TestRecordedStructuredOutput(t *testing.T) {
	type cityInfo struct {
		City    string `json:"city" jsonschema:"description=City name"`
		Country string `json:"country" jsonschema:"description=Country name"`
	}
	agent := ai.NewAgent[struct{}, cityInfo](vcrModel(t, "structured_output"))
	result, err := agent.Run(t.Context(), "Where is the Eiffel Tower?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.City == "" || result.Output.Country == "" {
		t.Fatalf("expected populated output, got %+v", result.Output)
	}
}

func TestRecordedStreamingRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrModel(t, "streaming_run"),
		ai.WithInstructions("Answer with a single word."),
	)
	stream := agent.RunStream(t.Context(), "What is the capital of Italy?", struct{}{})
	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if delta, ok := event.(ai.TextDeltaEvent); ok {
			text += delta.Delta
		}
	}
	if text == "" || stream.Result() == nil {
		t.Fatalf("expected streamed text and result, got %q", text)
	}
	if stream.Result().Usage().TotalTokens() == 0 {
		t.Fatal("expected usage from the API")
	}
}
