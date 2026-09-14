package google_test

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	imagegoogle "github.com/Kludex/pydantic-ai-go/ai/images/google"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func imageVCR(t *testing.T, name string) *http.Client {
	t.Helper()
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("GOOGLE_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	recording, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(interaction *cassette.Interaction) error {
			delete(interaction.Request.Headers, "X-Goog-Api-Key")
			var response map[string]any
			if err := json.Unmarshal([]byte(interaction.Response.Body), &response); err != nil {
				return err
			}
			for _, candidate := range response["candidates"].([]any) {
				content := candidate.(map[string]any)["content"].(map[string]any)
				for _, part := range content["parts"].([]any) {
					inline, ok := part.(map[string]any)["inlineData"].(map[string]any)
					if ok {
						inline["data"] = base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\nrecorded"))
					}
				}
			}
			body, err := json.Marshal(response)
			if err != nil {
				return err
			}
			interaction.Response.Body = string(body)
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
	return recording.GetDefaultClient()
}

func TestRecordedGeminiImageGeneration(t *testing.T) {
	model := imagegoogle.NewModel(
		"gemini-2.5-flash-image", imagegoogle.WithHTTPClient(imageVCR(t, "gemini_image_generation")),
	)
	result, err := images.New(model).Generate(t.Context(), "A tiny blue square on a white background.", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 || result.Images[0].Content.MediaType == "" ||
		result.Usage.InputTokens == 0 || result.Usage.OutputTokens == 0 || result.ProviderResponseID == "" {
		t.Fatalf("unexpected recorded result: %#v", result)
	}
}
