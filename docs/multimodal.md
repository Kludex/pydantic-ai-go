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

## Send a video

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openrouter"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		openrouter.NewModel("google/gemini-3-pro-preview"),
	)
	result, err := agent.RunParts(
		context.Background(),
		[]ai.UserContent{
			ai.TextContent{Text: "Describe this video in one sentence."},
			ai.VideoURL{URL: "https://example.com/demo.mp4"},
		},
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

OpenRouter sends `VideoURL` as its `video_url` extension. It also sends `BinaryContent` with a `video/*` media type as an inline video data URL.

`VideoURL` infers common video media types from the URL extension. Set `MediaType` when the URL has no recognizable extension. `ResolvedIdentifier()` returns a stable six-character identifier unless you provide `Identifier`.

Google sends YouTube URLs, Gemini Files API URLs, and Vertex `gs://` URLs directly. It downloads other video URLs and sends the bytes inline. The downloader limits responses to 50 MiB, pins validated DNS addresses to prevent rebinding, and blocks private networks and cloud metadata endpoints.

Set `ForceDownload: ai.FileDownloadSafe` to download an OpenRouter URL with the same protections. Set `ForceDownload: ai.FileDownloadAllowLocal` only when you intentionally need a private host. Cloud metadata endpoints remain blocked in both modes.

> [!WARNING]
> `FileDownloadAllowLocal` permits requests to loopback and private network addresses. Use it only with trusted URLs. The URL and every redirect remain restricted to HTTP or HTTPS.

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

## Mark a prompt-cache boundary

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func main() {
	agent := ai.NewAgent[struct{}, string](anthropic.NewModel("claude-sonnet-4-6"))
	result, err := agent.RunParts(
		context.Background(),
		[]ai.UserContent{
			ai.TextContent{Text: "A long, stable reference document..."},
			ai.CachePoint{TTL: ai.CachePointTTL1Hour},
			ai.TextContent{Text: "Summarize the reference document."},
		},
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`CachePoint` marks the preceding content item. It cannot be the first item. The zero value uses five minutes. Anthropic accepts five-minute and one-hour boundaries and keeps the newest four. OpenAI GPT-5.6 Chat Completions and Responses use explicit cache breakpoints and ignore the per-marker TTL.

OpenRouter maps cache points for Anthropic and Gemini routes. It includes TTL only for Anthropic, enforces Anthropic's four-point limit, and maps OpenAI GPT-5.6 routes to OpenAI breakpoints. Unsupported providers validate and omit the marker.

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
