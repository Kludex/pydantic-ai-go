# Multimodal input

## Send an image URL

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewModel("gpt-5"))
	result, err := agent.RunParts(
		context.Background(),
		[]ai.UserContent{
			ai.TextContent{
				Text:     "Describe this image in one sentence.",
				Metadata: map[string]any{"source": "diagram-review"},
			},
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

`RunParts` preserves content order.
Put the instruction before or after the image according to the prompt you want the provider to receive.
`TextContent.Metadata` remains in message history for your application but is not sent to the model.

OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google models accept `ImageURL`. OpenAI and Anthropic can fetch the URL directly. Google downloads ordinary Gemini API URLs with SSRF protection, but sends Gemini Files API and Vertex AI URLs directly.

Set `VendorMetadata: map[string]any{"detail": "high"}` for OpenAI image detail. Set `ForceDownload` when you do not want a provider to fetch the URL directly.

OpenAI Responses renders ordered `input_text` and `input_image` items. Its `/responses/input_tokens` count includes the same multimodal request.

## Send a document and audio

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	agent := ai.NewAgent[struct{}, string](openai.NewResponsesModel("gpt-5.4"))
	result, err := agent.RunParts(
		context.Background(),
		[]ai.UserContent{
			ai.TextContent{Text: "Compare the report with the meeting recording."},
			ai.DocumentURL{URL: "https://example.com/report.pdf"},
			ai.AudioURL{URL: "https://example.com/meeting.mp3"},
		},
		struct{}{},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`DocumentURL` and `AudioURL` infer their media types from common file extensions. Set `MediaType` when a URL has no extension. Each type provides `ResolvedIdentifier()` for a stable six-character file identifier.

OpenAI Responses sends document and audio URLs as `input_file`. Forced downloads become base64 file data. OpenAI Chat Completions downloads audio and accepts MP3 or WAV data. It downloads documents, sends supported binary documents as files, and wraps text-like documents in stable file delimiters.

Anthropic accepts PDF URLs directly. It downloads forced PDFs and all plain-text documents. Anthropic does not accept audio input. Google models accept image, audio, document, and video URLs. Gemini API downloads ordinary URLs, while Vertex AI and Gemini Files API references remain remote file data.

OpenRouter sends document URLs directly through its Chat Completions file extension. Audio URLs are downloaded before they are sent.

## Send a video

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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

Google sends YouTube URLs, Gemini Files API URLs, and Vertex `gs://` URLs directly. It downloads other Gemini API video URLs and sends the bytes inline.

## Control URL downloads

The zero-value `FileDownloadNever` mode lets a provider fetch a URL directly when its API supports that behavior. An adapter still downloads the file when its API requires inline data.

Set `ForceDownload: ai.FileDownloadSafe` to always download an image, document, audio file, or video. The downloader limits responses to 50 MiB, pins validated DNS addresses to prevent rebinding, and blocks private networks and cloud metadata endpoints. YouTube and Vertex `gs://` video references remain provider-hosted because they cannot be downloaded safely as ordinary HTTP files.

Set `ForceDownload: ai.FileDownloadAllowLocal` only when you intentionally need a private host. Cloud metadata endpoints remain blocked.

> [!WARNING]
> `FileDownloadAllowLocal` permits requests to loopback and private network addresses. Use it only with trusted URLs. The URL and every redirect remain restricted to HTTP or HTTPS.

## Send inline file bytes

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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

`BinaryContent` copies the byte slice and vendor metadata before they reach reusable agent state or provider callbacks. Set `MediaType` to the real IANA media type, such as `image/png`, `audio/wav`, or `application/pdf`. `ResolvedIdentifier()` uses a stable digest unless you set `Identifier`.

Provider support depends on the media type. OpenAI Chat Completions accepts images, MP3 or WAV audio, and supported documents. OpenAI Responses accepts images and supported documents. Anthropic accepts images, PDFs, and plain text. Google accepts each supported media category.

Inline content increases request size. Prefer a URL when the provider can fetch a stable private or public URL securely.

> [!WARNING]
> A URL can expose its host, path, query values, and access token to the model provider. Use short-lived signed URLs. Do not place long-lived credentials in the URL.

## Mark a prompt-cache boundary

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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

	ai "github.com/Kludex/pydantic-ai-go/ai"
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

The returned history is detached. Mutating file bytes, URL metadata, uploaded-file metadata, generated-file metadata, or replacing a content item does not change the completed run.
