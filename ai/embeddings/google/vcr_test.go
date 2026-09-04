package google_test

import (
	"net/http"
	"os"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	embeddinggoogle "github.com/Kludex/pydantic-ai-go/ai/embeddings/google"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func googleVCR(t *testing.T, name string) *http.Client {
	t.Helper()
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("GOOGLE_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	recorder, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(interaction *cassette.Interaction) error {
			delete(interaction.Request.Headers, "X-Goog-Api-Key")
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
		if err := recorder.Stop(); err != nil {
			t.Error(err)
		}
	})
	return recorder.GetDefaultClient()
}

func TestRecordedGeminiEmbedding(t *testing.T) {
	model := embeddinggoogle.NewModel(
		"gemini-embedding-001", embeddinggoogle.WithHTTPClient(googleVCR(t, "gemini_embedding")),
	)
	dimensions := 8
	result, err := embeddings.New(model).EmbedQuery(
		t.Context(), "What is durable execution?", embeddings.Settings{Dimensions: &dimensions},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != dimensions || result.Usage.Requests != 1 {
		t.Fatalf("unexpected embedding result: dimensions=%d usage=%#v", len(result.Embeddings[0]), result.Usage)
	}
}
