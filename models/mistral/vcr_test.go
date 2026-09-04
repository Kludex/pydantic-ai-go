package mistral_test

import (
	"net/http"
	"os"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/mistral"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestRecordedSimpleRun(t *testing.T) {
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/simple_run.yaml"); os.IsNotExist(err) && os.Getenv("MISTRAL_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	recording, err := recorder.New("testdata/simple_run",
		recorder.WithMode(mode),
		recorder.WithHook(func(interaction *cassette.Interaction) error {
			delete(interaction.Request.Headers, "Authorization")
			delete(interaction.Response.Headers, "Set-Cookie")
			return nil
		}, recorder.AfterCaptureHook),
		recorder.WithMatcher(cassette.MatcherFunc(func(request *http.Request, interaction cassette.Request) bool {
			return request.Method == interaction.Method && request.URL.String() == interaction.URL
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := recording.Stop(); err != nil {
			t.Error(err)
		}
	})
	model := mistral.NewModel("mistral-small-latest", mistral.WithHTTPClient(recording.GetDefaultClient()))
	agent := ai.NewAgent[struct{}, string](model, ai.WithInstructions("Answer with a single word."))
	result, err := agent.Run(t.Context(), "What is the capital of France?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output == "" || result.Usage().TotalTokens() == 0 {
		t.Fatalf("expected output and usage: %#v", result)
	}
}
