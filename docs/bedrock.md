# Amazon Bedrock

## Run an agent

```go
package main

import (
    "context"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/bedrock"
)

func main() {
    model := bedrock.NewModel("us.amazon.nova-lite-v1:0")
    agent := ai.NewAgent[struct{}, string](model,
        ai.WithInstructions("Answer in one short sentence."),
    )

    result, err := agent.Run(context.Background(), "Why is the sky blue?", struct{}{})
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Output)
}
```

Run the example with your normal AWS credentials:

```console
$ AWS_REGION=us-east-1 go run ./examples/bedrock
```

`NewModel` loads the default AWS SDK configuration on the first operation. This keeps construction free of network and credential-provider work. Concurrent calls share the loaded client.

You can pass a foundation model ID, an inference profile ID, or an ARN. Bedrock validates whether that resource supports the Converse API.

## Count input tokens

```go
package main

import (
    "context"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/bedrock"
)

func main() {
    model := bedrock.NewModel("us.amazon.nova-lite-v1:0")
    messages := []ai.ModelMessage{
        ai.ModelRequest{Parts: []ai.RequestPart{
            ai.UserPromptPart{Content: "Why is the sky blue?"},
        }},
    }

    usage, err := ai.CountModelTokens(context.Background(), model, messages, ai.ModelRequestParams{})
    if err != nil {
        panic(err)
    }
    fmt.Println(usage.InputTokens)
}
```

Token counting sends the same Converse messages, system blocks, tool definitions, and additional model fields to Bedrock's `CountTokens` operation. Generation-only settings are not part of the token-counting input.

A custom `bedrock.Client` may omit `bedrock.TokenCountingClient`. `CountModelTokens` then returns `ai.ErrTokenCountingUnsupported`.

## Configure the AWS SDK

```go
package main

import (
    "context"
    "fmt"

    "github.com/aws/aws-sdk-go-v2/aws"
    "github.com/aws/aws-sdk-go-v2/credentials"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/bedrock"
)

func main() {
    config := aws.Config{
        Region:      "us-east-1",
        Credentials: credentials.NewStaticCredentialsProvider("access-key", "secret-key", ""),
    }
    model := bedrock.NewModel("us.amazon.nova-lite-v1:0", bedrock.WithAWSConfig(config))
    agent := ai.NewAgent[struct{}, string](model)

    result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Output)
}
```

Use `WithAWSConfig` when your application already owns a loaded `aws.Config`. The option copies the configuration before creating the client.

Use `WithAWSLoadOptions` to customize lazy default loading. Use `WithClient` for a caller-owned Bedrock Runtime implementation. Caller-owned clients must be safe for concurrent calls and remain owned by the caller.

`WithProviderURL` records endpoint identity for telemetry when you supply a custom client. It does not change where that client sends requests.

## Messages and files

The Converse adapter maps these provider-neutral values:

- Text, function tool calls, tool results, validation retries, and signed reasoning.
- Inline image, audio, video, and document content.
- HTTP and HTTPS file URLs through the shared SSRF-safe downloader.
- `UploadedFile` values that belong to `bedrock` and use an `s3://` URI.
- Explicit five-minute and one-hour `CachePoint` values.

Bedrock requires text beside a document. The adapter inserts a neutral text block when a user prompt contains documents without text. It preserves the newest four explicit cache points because Bedrock rejects requests above that limit.

Bedrock receives detached byte slices. A request does not retain caller-owned file buffers.

## Settings

Portable maximum-token, temperature, top-p, stop-sequence, service-tier, and extra-header settings map to Converse fields. `ModelSettings.ExtraBody` maps to `additionalModelRequestFields` for model-specific Bedrock parameters.

`WithDefaultSettings` stores a detached copy. Agent and run settings override those defaults through the standard fieldwise merge.

## Current scope

The adapter currently uses non-streaming Converse generation. Converse streaming, native Bedrock tools, native structured output, automatic cache placement, guardrails, performance settings, request metadata, and legacy Anthropic `InvokeModel` transport remain outside this implementation.
