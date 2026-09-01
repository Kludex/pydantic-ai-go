# HTTP retries

## Retry transient provider failures

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/retries"
)

func main() {
	transport, err := retries.NewTransport(retries.Config{
		MaxAttempts:      5,
		ValidateResponse: retries.ValidateStatus,
		ShouldRetry:      retries.RetryTransient,
		Wait: retries.WaitRetryAfter(
			retries.ExponentialBackoff(500*time.Millisecond, 10*time.Second),
			2*time.Minute,
		),
	}, http.DefaultTransport)
	if err != nil {
		log.Fatal(err)
	}

	model := openai.NewModel(
		"gpt-5",
		openai.WithHTTPClient(&http.Client{Transport: transport}),
	)
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "What is 2 + 2?", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`MaxAttempts` includes the initial request. This example retries connection errors, HTTP 429, and HTTP 5xx responses. HTTP 4xx responses other than 429 fail immediately.

`WaitRetryAfter` accepts both integer seconds and HTTP dates. It caps the server delay before using the exponential fallback. Cancellation of the request context interrupts the wait.

## Choose which responses are errors

An HTTP response is not a transport error by itself. `ValidateResponse` decides whether a response enters the retry policy.

`ValidateStatus` rejects every response outside 200-299. `RetryTransient` then retries only 429 and 5xx errors. The final `ResponseError` retains the response and implements `ModelAPIError`, so a `FallbackModel` can move to its next model after retries are exhausted.

Leave `ValidateResponse` unset when you only want to retry network errors. You can also provide your own `ResponseValidator` and `RetryFunc` for provider-specific status or body rules.

## Replay request bodies

A retry must send the same request. JSON requests created by the bundled providers are replayable because `http.NewRequest` sets `GetBody` for byte readers.

A custom streaming body needs its own `Request.GetBody` function. If a retry is requested after a non-replayable body is consumed, the transport returns `ErrBodyNotReplayable` instead of sending an empty or partial request.

## Observe attempts

Set `Config.BeforeRetry` to inspect the completed `Attempt`. It contains the one-based attempt number, response when available, and error. The callback can run concurrently when one client serves concurrent agent runs.

Transport retries are below the agent loop. They do not add model messages, consume tool or output retry budgets, or increment model request usage. The provider sees each HTTP attempt, while the model sees one logical request.

## Report failures from a custom model

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
)

type primaryModel struct{}

func (primaryModel) Name() string { return "primary" }

func (model primaryModel) Request(
	ctx context.Context,
	_ []ai.ModelMessage,
	_ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return nil, ai.NewModelTransportError(ctx, model, "request", errors.New("connection refused"))
}

type backupModel struct{}

func (backupModel) Name() string { return "backup" }

func (backupModel) Request(
	_ context.Context,
	_ []ai.ModelMessage,
	_ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "available"}}}, nil
}

func main() {
	model := ai.NewFallbackModel(primaryModel{}, ai.WithFallbackModels(backupModel{}))
	response, err := ai.RequestModel(context.Background(), model, nil, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response.Text())
}
```

Return `NewModelTransportError` for connection, client-timeout, and response-read failures.

The error implements `ModelAPIError`. The default fallback policy can try another model when a request fails before streaming starts.

If the passed context is already canceled or expired, the function preserves the cause without classifying it as a provider failure.
