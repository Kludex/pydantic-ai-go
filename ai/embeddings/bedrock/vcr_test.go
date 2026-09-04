package bedrock_test

import (
	"net/http"
	"os"
	"testing"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	embeddingbedrock "github.com/Kludex/pydantic-ai-go/ai/embeddings/bedrock"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func bedrockVCR(t *testing.T, name string) (*http.Client, bool) {
	t.Helper()
	mode := recorder.ModeReplayOnly
	replaying := true
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("AWS_ACCESS_KEY_ID") != "" {
		mode = recorder.ModeRecordOnce
		replaying = false
	}
	recorder, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(interaction *cassette.Interaction) error {
			for _, header := range []string{
				"Authorization", "X-Amz-Security-Token", "X-Amz-Date", "X-Amz-Content-Sha256",
			} {
				delete(interaction.Request.Headers, header)
			}
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
	return recorder.GetDefaultClient(), replaying
}

func TestRecordedBedrockEmbedding(t *testing.T) {
	httpClient, replaying := bedrockVCR(t, "titan_embedding")
	options := []func(*awsconfig.LoadOptions) error{
		awsconfig.WithHTTPClient(httpClient), awsconfig.WithRegion("us-east-1"),
	}
	if replaying {
		options = append(options, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider("vcr-access-key", "vcr-secret-key", ""),
		))
	}
	config, err := awsconfig.LoadDefaultConfig(t.Context(), options...)
	if err != nil {
		t.Fatal(err)
	}
	model, err := embeddingbedrock.NewModel(
		"amazon.titan-embed-text-v2:0", embeddingbedrock.WithAWSConfig(config),
	)
	if err != nil {
		t.Fatal(err)
	}
	dimensions := 256
	result, err := embeddings.New(model).EmbedQuery(
		t.Context(), "What is durable execution?", embeddings.Settings{Dimensions: &dimensions},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 || len(result.Embeddings[0]) != dimensions || result.Usage.InputTokens == 0 {
		t.Fatalf("unexpected embedding result: dimensions=%d usage=%#v", len(result.Embeddings[0]), result.Usage)
	}
}
