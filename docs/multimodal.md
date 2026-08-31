# Multimodal input

## Send an image URL

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5"))
	result, err := agent.RunParts(
		context.Background(),
		[]ai.UserContent{
			ai.TextContent{Text: "Describe this image in one sentence."},
			ai.ImageURL{URL: "https://example.com/diagram.png"},
		},
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`RunParts` preserves content order. Put the instruction before or after the image according to the prompt you want the provider to receive.

OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google Gemini accept `ImageURL`. The remote server must be able to fetch the URL.

OpenAI Responses renders ordered `input_text` and `input_image` items. Its `/responses/input_tokens` count includes the same multimodal request.

## Send inline image bytes

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func main() {
	image, err := os.ReadFile("diagram.png")
	if err != nil {
		log.Fatal(err)
	}

	agent := ai.NewAgent[struct{}, string](google.NewModel("gemini-2.5-flash"))
	result, err := agent.RunParts(
		context.Background(),
		[]ai.UserContent{
			ai.TextContent{Text: "List the components shown in this diagram."},
			ai.BinaryContent{Data: image, MediaType: "image/png"},
		},
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`BinaryContent` copies the byte slice before it reaches reusable agent state or provider callbacks. Set `MediaType` to the real IANA media type, such as `image/png` or `image/jpeg`.

Inline content increases request size. Prefer `ImageURL` when the provider can fetch a stable private or public URL securely.

> [!WARNING]
> A URL can expose its host, path, query values, and access token to the model provider. Use short-lived signed URLs. Do not place long-lived credentials in the URL.

## Reference an uploaded file

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewResponsesModel("gpt-5.4"))
	result, err := agent.RunParts(
		context.Background(),
		[]ai.UserContent{
			ai.TextContent{Text: "Summarize the uploaded report."},
			ai.UploadedFile{
				FileID:       "file_abc123",
				ProviderName: "openai",
				MediaType:    "application/pdf",
			},
		},
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`UploadedFile` references bytes already stored by a provider. File IDs are not portable. `ProviderName` must match the selected provider or the request fails before transport.

OpenAI Responses sends images as `input_image` and other files as `input_file`. Anthropic sends uploaded images and documents through file sources. Gemini accepts HTTPS references from the Google Files API. Vertex AI accepts `gs://` Google Cloud Storage references. OpenAI Chat Completions, Bedrock, xAI, and provider-specific file metadata remain parity work.

## Read generated files

`FilePart` represents binary output from a model or provider-native tool. OpenAI code interpreter and image generation use this part. Google image models return inline generated images as `FilePart` values and replay them as inline data in later Google requests. `FilePart.Content.Data`, provider details, histories, and stream events are detached before they reach consumers.

## Reuse multimodal history

`Result.Messages()` contains the original `UserPromptPart.Contents`. You can pass those messages to `WithMessageHistory` on another run.

The returned history is detached. Mutating `BinaryContent.Data`, `UploadedFile.VendorMetadata`, `FilePart.Content.Data`, or replacing a content item does not change the completed run.
