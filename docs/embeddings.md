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
