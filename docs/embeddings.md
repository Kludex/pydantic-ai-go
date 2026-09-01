# Embeddings

Use `embeddings.Embedder` to generate vectors for search queries and documents.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/openai"
)

func main() {
	model := openai.NewModel("text-embedding-3-small")
	embedder := embeddings.New(model)

	result, err := embedder.EmbedQuery(context.Background(), "What is structured concurrency?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(result.Embeddings[0]))
}
```

Set `OPENAI_API_KEY` before you run the example. The result contains one vector for each input, the original inputs, usage, provider identity, and a timestamp.

## Queries and documents

Use query embeddings for text that searches a collection. Use document embeddings for the collection itself. Some providers optimize these input types differently.

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/openai"
)

func main() {
	embedder := embeddings.New(openai.NewModel("text-embedding-3-small"))
	ctx := context.Background()

	documents, err := embedder.EmbedDocuments(ctx, []string{
		"Structured concurrency ties child tasks to a parent scope.",
		"A goroutine is a lightweight concurrent function.",
	})
	if err != nil {
		log.Fatal(err)
	}

	query, err := embedder.EmbedQuery(ctx, "How are child tasks scoped?")
	if err != nil {
		log.Fatal(err)
	}

	_ = documents.Embeddings
	_ = query.Embeddings[0]
}
```

`EmbedQuery` and `EmbedDocument` accept one string. `EmbedQueries` and `EmbedDocuments` preserve batch order. `Result.At` and `Result.ForInput` return detached vectors.

## Model names and temporary overrides

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/fakes"
	"github.com/Kludex/pydantic-ai-go/embeddings/infer"
)

func main() {
	model, err := infer.Model("openai:text-embedding-3-small")
	if err != nil {
		log.Fatal(err)
	}
	embedder := embeddings.New(model)

	ctx := embeddings.WithModel(context.Background(), fakes.NewModel())
	_, err = embedder.EmbedQuery(ctx, "test without an API request")
	if err != nil {
		log.Fatal(err)
	}
}
```

`infer.Model` requires a provider prefix. It supports `openai`, `azure`, `bedrock`, `cohere`, `google`, `google-cloud`, and `voyageai`. Use `infer.WithProvider` to register a custom resolver. Use `infer.WithAzureConfig` or `infer.WithVertexConfig` to configure the matching cloud provider.

`WithModel` scopes an override to one context tree. It does not mutate the reusable `Embedder`, so concurrent requests can select different models safely.

## Settings

Pass defaults to `embeddings.New`. Pass one `embeddings.Settings` value to override them for one call.

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/openai"
)

func main() {
	defaultDimensions := 1_536
	embedder := embeddings.New(
		openai.NewModel("text-embedding-3-small"),
		embeddings.WithSettings(embeddings.Settings{Dimensions: &defaultDimensions}),
	)

	dimensions := 512
	_, err := embedder.EmbedQuery(
		context.Background(),
		"Use a smaller vector for this index.",
		embeddings.Settings{Dimensions: &dimensions},
	)
	if err != nil {
		log.Fatal(err)
	}
}
```

Settings are copied at configuration and request boundaries. `ExtraHeaders` applies last. `ExtraBody` cannot replace `model`, `input`, or `dimensions`.

## OpenAI-compatible endpoints

Use `models/openai.ProviderConfig` to set endpoint identity, authentication, query parameters, and dynamic request preparation.

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	embeddingopenai "github.com/Kludex/pydantic-ai-go/embeddings/openai"
	modelopenai "github.com/Kludex/pydantic-ai-go/models/openai"
)

func main() {
	model := embeddingopenai.NewModel(
		"text-embedding-3-small",
		embeddingopenai.WithProvider(modelopenai.ProviderConfig{
			Name:    "my-provider",
			BaseURL: "https://api.example.com/v1",
			APIKey:  "secret",
		}),
	)

	_, err := embeddings.New(model).EmbedQuery(context.Background(), "hello")
	if err != nil {
		log.Fatal(err)
	}
}
```

The provider name and URL remain attached to each result. `Result.Price` uses them with the bundled `genai-prices` snapshot.

## Google Gemini

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	embeddinggoogle "github.com/Kludex/pydantic-ai-go/embeddings/google"
)

func main() {
	model := embeddinggoogle.NewModel("gemini-embedding-2")
	embedder := embeddings.New(model)

	settings, err := (embeddinggoogle.Settings{
		Task:  embeddinggoogle.TaskQuestionAnswering,
		Title: "Structured concurrency",
	}).Build()
	if err != nil {
		log.Fatal(err)
	}
	_, err = embedder.EmbedDocuments(
		context.Background(),
		[]string{"Child tasks remain inside their parent scope."},
		settings,
	)
	if err != nil {
		log.Fatal(err)
	}
}
```

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` before you run the example. `gemini-embedding-2` uses a task prefix. Earlier models use Google's `taskType` request field. If you pass a setting that the selected family ignores, the result reports it in `Warnings`.

## Google Cloud Vertex AI

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	embeddinggoogle "github.com/Kludex/pydantic-ai-go/embeddings/google"
	modelgoogle "github.com/Kludex/pydantic-ai-go/models/google"
)

func main() {
	model, err := embeddinggoogle.NewVertexModel(
		"gemini-embedding-001",
		modelgoogle.VertexConfig{
			Project:  "my-project",
			Location: "global",
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	_, err = embeddings.New(model).EmbedQuery(context.Background(), "How are child tasks scoped?")
	if err != nil {
		log.Fatal(err)
	}
}
```

Vertex AI uses Application Default Credentials unless you pass a `TokenProvider` or an Express Mode API key. The same transport supports regional, global, multi-region, and custom endpoints.

## Cohere

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/cohere"
)

func main() {
	model := cohere.NewModel("embed-v4.0")
	embedder := embeddings.New(model)

	maxTokens := 256
	settings, err := (cohere.Settings{
		InputType: cohere.InputTypeClassification,
		MaxTokens: &maxTokens,
		Truncate:  cohere.TruncationEnd,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}
	_, err = embedder.EmbedDocuments(
		context.Background(),
		[]string{"Child tasks stay inside their parent scope."},
		settings,
	)
	if err != nil {
		log.Fatal(err)
	}
}
```

Set `CO_API_KEY` before you run the example. Cohere-specific input and truncation settings take precedence over the portable defaults. `CountTokens` uses Cohere's v1 tokenizer while embedding requests use the v2 API.

## VoyageAI

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/voyageai"
)

func main() {
	model := voyageai.NewModel("voyage-4")
	embedder := embeddings.New(model)

	settings, err := (voyageai.Settings{
		InputType: voyageai.InputTypeNone,
	}).Build()
	if err != nil {
		log.Fatal(err)
	}
	_, err = embedder.EmbedQuery(context.Background(), "What is structured concurrency?", settings)
	if err != nil {
		log.Fatal(err)
	}
}
```

Set `VOYAGE_API_KEY` before you run the example. Use `InputTypeNone` when you do not want VoyageAI to add a retrieval prefix. The adapter requests compact base64 vectors and returns detached `float64` values.

## Amazon Bedrock

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/bedrock"
)

func main() {
	model, err := bedrock.NewModel("amazon.titan-embed-text-v2:0")
	if err != nil {
		log.Fatal(err)
	}
	embedder := embeddings.New(model)

	_, err = embedder.EmbedDocuments(context.Background(), []string{"first document", "second document"})
	if err != nil {
		log.Fatal(err)
	}
}
```

The default client uses the AWS SDK credential and region chain. Use `bedrock.WithAWSConfig` with a loaded `aws.Config`. Use `bedrock.WithClient` to inject a transport client. The client supports Titan, Cohere, and Nova request formats.

Titan and Nova require one request per input. The model preserves input order and allows five parallel requests by default. Set `bedrock.Settings.MaxConcurrency` when your Bedrock quota requires a different limit. Cohere batches all inputs in one request.

Use `bedrock.Settings` for dimensions, normalization, input types, truncation, Nova purposes, and inference profiles. Provider-specific defaults merge field by field with per-call settings.

## Token limits

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/openai"
)

func main() {
	embedder := embeddings.New(openai.NewModel("text-embedding-3-small"))
	ctx := context.Background()

	count, err := embedder.CountTokens(ctx, "What is structured concurrency?")
	if err != nil {
		log.Fatal(err)
	}
	maximum, known, err := embedder.MaxInputTokens(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(count, maximum, known)
}
```

OpenAI token counting runs locally with the model's tokenizer. Models without token counting return `embeddings.ErrTokenCountingUnsupported`. An unknown maximum returns `known == false`.

## OpenTelemetry

```go
package main

import (
	"context"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/openai"
)

func main() {
	model := embeddings.InstrumentModel(openai.NewModel("text-embedding-3-small"))
	result, err := embeddings.New(model).EmbedQuery(context.Background(), "What is structured concurrency?")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("dimensions: %d", len(result.Embeddings[0]))
}
```

`InstrumentModel` emits one `embeddings <model>` client span. It records provider and model identity, usage, dimensions, cost, and OpenTelemetry token and cost histograms. Input and vector content are enabled by default. Pass `embeddings.WithInstrumentationContent(false)` before you export telemetry outside your trust boundary.

Instrumentation preserves optional token-counting and input-limit capabilities through model wrappers.

## Tests

Use `fakes.Model` for deterministic vectors without network access.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/fakes"
)

func main() {
	embedder := embeddings.New(fakes.NewModel(fakes.WithDimensions(3)))
	result, err := embedder.EmbedQuery(context.Background(), "test")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Embeddings[0])
}
```

The fake model returns vectors containing `1`. It also provides deterministic token counting and a detached `LastSettings` snapshot.
