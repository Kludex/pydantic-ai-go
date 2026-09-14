# Direct image generation

Use `images.Generator` when your application decides when to create an image. Use the image generation capability when the model should decide.

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/openai"
)

func main() {
	generator := images.New(openai.NewModel("gpt-image-2"))
	result, err := generator.Generate(context.Background(), "A watercolor map of a floating city.", nil)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("floating-city.png", result.Image().Data, 0o600); err != nil {
		log.Fatal(err)
	}
}
```

Set `OPENAI_API_KEY` before you run the example.

## Providers

| Provider | Constructor | Environment variable | Editing | Batches |
| --- | --- | --- | --- | --- |
| OpenAI | `openai.NewModel("gpt-image-2")` | `OPENAI_API_KEY` | Yes | `openai.Settings.N` |
| Google Gemini | `google.NewModel("gemini-3.1-flash-image")` | `GOOGLE_API_KEY` | Yes | No |
| Google Cloud | `google.NewVertexModel(...)` | Application Default Credentials | Yes | No |
| xAI | `xai.NewModel("grok-imagine-image")` | `XAI_API_KEY` | Yes | `xai.Settings.N` |

OpenAI and Google accept `WithHTTPClient`. The model never closes or replaces your client. Use `WithBaseURL` for a compatible HTTP endpoint. OpenAI accepts `ai/models/openai.ProviderConfig`. Google accepts `ai/models/google.ProviderConfig` and `VertexConfig`.

xAI uses the `GenerateImage` gRPC method used by the official xAI Python SDK. Use `xai.WithClient` to pass a caller-owned `grpc.ClientConnInterface`. The model never closes that connection. The default connection targets `api.x.ai:443`; call `model.Close()` when you let `xai.NewModel` create that connection. Use `xai.WithTarget` for a compatible gRPC endpoint.

Use `infer.Model("openai:gpt-image-2")` when a provider-prefixed name comes from configuration. The built-in prefixes are `openai`, `google`, `google-cloud`, and `xai`.

## Edit images

Pass `ai.BinaryContent`, `ai.ImageURL`, or `ai.UploadedFile` values as references.

```go
package main

import (
	"context"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/google"
)

func edit(source []byte) ([]byte, error) {
	generator := images.New(google.NewModel("gemini-3.1-flash-image"))
	result, err := generator.Generate(
		context.Background(),
		"Replace the cat with a dog while preserving the composition.",
		[]images.Input{ai.BinaryContent{Data: source, MediaType: "image/png"}},
	)
	if err != nil {
		return nil, err
	}
	return result.Image().Data, nil
}

func main() {
	if _, err := edit([]byte("image bytes")); err != nil {
		log.Fatal(err)
	}
}
```

OpenAI downloads URL references and sends PNG, JPEG, or WebP bytes to its edit endpoint. It does not accept `UploadedFile` IDs. The Gemini API accepts its HTTPS Files API URIs. Vertex AI does not accept hosted-file references for direct generation. xAI accepts hosted file IDs, URLs, and inline data. Put xAI uploaded files before URL or binary references so their order remains stable.

Forced downloads use the same SSRF protection and 50 MiB limit as multimodal model input. `FileDownloadAllowLocal` is intended for trusted local development and tests.

## Settings

Portable settings describe exact dimensions or an aspect ratio.

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/openai"
)

func main() {
	providerSettings, err := (openai.Settings{
		Common: images.Settings{AspectRatio: images.AspectRatio16To9},
		Quality: openai.QualityHigh,
		OutputFormat: openai.OutputFormatPNG,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}
	generator := images.New(openai.NewModel("gpt-image-2"), images.WithSettings(providerSettings))
	if _, err := generator.Generate(context.Background(), "A desert observatory at dusk.", nil); err != nil {
		log.Fatal(err)
	}
}
```

Model defaults apply first. Generator defaults apply next. Per-call settings apply last. Every layer is detached and safe for concurrent calls.

`Dimensions` and `AspectRatio` are mutually exclusive. Exact dimensions never round to a nearby provider shape:

- GPT Image 1.x accepts `1024x1024`, `1024x1536`, and `1536x1024`.
- GPT Image 2 requires multiples of 16, a maximum edge of 3840, at most a 3:1 ratio, and 655,360 through 8,294,400 pixels.
- Gemini dimensions map to the documented 512, 1K, 2K, or 4K model tiers.
- Grok Imagine dimensions map to the 1K and 2K ratio table published through the official SDK.

Provider geometry takes precedence over portable geometry. The result reports the conflict in `Warnings` instead of silently hiding it. An unsupported portable shape fails before the request when the provider wire format cannot represent it.

Use provider settings for controls that are not portable:

- OpenAI: image count, output format, size, quality, background, input fidelity, moderation, compression, and user ID.
- Google: native aspect ratio, image size, output MIME type, and compression quality.
- xAI: image count, user ID, native aspect ratio, and resolution tier.

`ExtraHeaders` applies after provider and dynamic authentication headers on the HTTP providers. `ExtraBody` cannot replace typed request fields. xAI uses gRPC, so it reports both HTTP-specific escape hatches as ignored warnings.

## Results and usage

`Result.Images` contains normalized `GeneratedImage` values. `Result.Image()` returns a detached copy of the first image. Successful generators always return at least one image.

Each result includes the prompt, model name, provider name and URL, timestamp, response ID, usage, warnings, and detached provider details. OpenAI usage includes text and image token details. Gemini includes prompt, tool-use, candidate, thinking, cache, and per-modality token details. xAI preserves batch-wide token and cost details. `Result.Price()` uses the bundled `genai-prices` data when the model has a price entry.

A provider moderation refusal returns `*ai.ContentFilterError`. A malformed successful response returns `*ai.UnexpectedModelBehaviorError`. Provider status errors remain inspectable through the provider's API error type. Connection and response-read failures implement `ai.ModelAPIError` and preserve their underlying error.

## OpenTelemetry

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/openai"
)

func main() {
	generator := images.New(openai.NewModel("gpt-image-2"), images.WithInstrumentation())
	if _, err := generator.Generate(context.Background(), "A small observatory.", nil); err != nil {
		log.Fatal(err)
	}
}
```

Instrumentation emits one `image_generation <model>` client span. It records model and provider identity, request geometry, image formats, usage, cost, and token and cost histograms. Prompt content is included by default. Use `images.WithInstrumentationContent(false)` before exporting telemetry outside your trust boundary. Headers, extra body fields, and image bytes are never recorded.

`InstrumentAll` enables instrumentation for generators without an explicit option. `WithoutInstrumentation` keeps one generator disabled. `InstrumentModel` wraps a lower-level model directly.

## Agent fallback

Pass a direct generator to `images.NewImageGenerationCapability`.

```go
package main

import (
	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	imageopenai "github.com/Kludex/pydantic-ai-go/ai/images/openai"
	modelopenai "github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func main() {
	generator := images.New(imageopenai.NewModel("gpt-image-2"))
	capability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		Native:    ai.ImageGenerationTool{},
		Generator: generator,
		Settings: images.Settings{
			AspectRatio: images.AspectRatio16To9,
		},
	})
	_ = ai.NewAgent[struct{}, string](
		modelopenai.NewResponsesModel("gpt-5-mini"),
		ai.WithCapabilities(capability),
	)
}
```

The outer model uses native image generation when it supports it. Otherwise, it receives a local `generate_image` function backed by the direct generator. Pass `FallbackModel` instead of `Generator` when you do not need generator-level defaults. Pass `ResolveNative` when dependencies choose native settings for each request. A resolved native aspect ratio also reaches the direct fallback unless `Settings` defines direct geometry.

Repeated declarations merge native fields and direct `Settings` in registration order. Later non-zero fields take precedence. `ProviderSettings` merges by provider key. A generator or local fallback is inherited from another declaration in the same layer. Conflicting `Dimensions` and `AspectRatio` values fail during capability registration.

The fallback expects exactly one generated image because one tool result represents one artifact. A content-filter refusal becomes a model retry, so the outer model can rephrase its request. An edit-only native request fails if it reaches the direct fallback because the `generate_image` function receives no reference images. Call `Generator.Generate` directly for editing and batches.

## Tests

Use `fakes.Model` to generate a deterministic PNG without network access.

```go
package main

import (
	"context"
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/fakes"
)

func main() {
	model := fakes.NewModel()
	result, _ := images.New(model).Generate(context.Background(), "test image", nil)
	fmt.Println(result.Image().MediaType)
}
```

`LastCall` returns the latest reference inputs and merged settings for assertions.

The Google request path has a credential-filtered cassette recorded against the Gemini API. No xAI credential was available for this audit, so the xAI path is verified against the official `xai-sdk-python` v1.18.0 protobuf schema and public in-process gRPC request tests, not a live Go traffic recording.
