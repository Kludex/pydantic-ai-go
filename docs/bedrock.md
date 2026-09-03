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

`NewModel` loads the default AWS SDK configuration on the first operation. This keeps construction free of network and credential-provider work. Concurrent calls share the loaded client. `RunStream` uses `ConverseStream` when the client supports it and emits normalized text, reasoning, and function-tool events.

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

A custom client implements `bedrock.Client` for static generation. It can also implement `bedrock.StreamingClient` and `bedrock.TokenCountingClient`. `RunStream` falls back to one final event when the client does not implement streaming. Each returned `bedrock.EventStream` is closed when iteration ends or the consumer stops early.

`WithProviderURL` records endpoint identity for telemetry when you supply a custom client. It does not change where that client sends requests.

## Use the legacy Anthropic InvokeModel transport

```go
package main

import (
    "context"
    "fmt"

    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func main() {
    awsConfig, err := config.LoadDefaultConfig(context.Background())
    if err != nil {
        panic(err)
    }
    client := bedrockruntime.NewFromConfig(awsConfig)
    model := anthropic.NewLegacyBedrockModel(
        "anthropic.claude-sonnet-4-20250514-v1:0",
        anthropic.LegacyBedrockConfig{
            Client:      client,
            ProviderURL: "https://bedrock-runtime.us-east-1.amazonaws.com",
        },
    )
    agent := ai.NewAgent[struct{}, string](model)

    result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Output)
}
```

`NewLegacyBedrockModel` reuses Anthropic message normalization with Bedrock's `InvokeModel` request body. It uses Bedrock's `CountTokens` operation because Anthropic's Messages token-count endpoint is unavailable on this transport.

The legacy transport defaults provider-native deferred tool search to regex. Bedrock rejects BM25 on this API, so an explicit BM25 strategy fails before transport. `RunStream` uses `InvokeModelWithResponseStream` with the AWS SDK or a custom `LegacyBedrockStreamingClient`. A custom client without streaming support falls back to one complete `InvokeModel` response.

You own the AWS client and its lifecycle. `ProviderURL` records telemetry identity and does not reconfigure the client.

## Messages and files

The Converse adapter maps these provider-neutral values:

- Text, function tool calls, tool results, validation retries, and signed reasoning.
- Inline image, audio, video, and document content.
- HTTP and HTTPS file URLs through the shared SSRF-safe downloader.
- `UploadedFile` values that belong to `bedrock` and use an `s3://` URI.
- Explicit five-minute and one-hour `CachePoint` values.

Bedrock requires text beside a document. The adapter inserts a neutral text block when a user prompt contains documents without text. It preserves the newest four explicit cache points because Bedrock rejects requests above that limit.

Bedrock receives detached byte slices. A request does not retain caller-owned file buffers.

## Prompt caching and native output

```go
package main

import (
    "context"
    "fmt"

    ai "github.com/Kludex/pydantic-ai-go"
    "github.com/Kludex/pydantic-ai-go/models/bedrock"
)

type Answer struct {
    Summary string `json:"summary"`
}

func main() {
    settings, err := (bedrock.Settings{
        CacheInstructions: bedrock.CacheTTL5Minutes,
        CacheMessages:     bedrock.CacheTTL5Minutes,
    }).Build()
    if err != nil {
        panic(err)
    }

    model := bedrock.NewModel("us.amazon.nova-lite-v1:0")
    agent := ai.NewAgent[struct{}, Answer](model,
        ai.WithInstructions("Return a concise summary."),
        ai.WithModelSettings(settings),
        ai.WithOutputMode(ai.OutputModeNative),
    )
    result, err := agent.Run(context.Background(), "Explain prompt caching.", struct{}{})
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Output.Summary)
}
```

`bedrock.Settings` places typed cache points after instructions, on the newest user message, and after tool definitions. Fixed instruction and tool points reserve their share of Bedrock's four-point limit. The adapter keeps the newest remaining message points. `ResolvePromptCacheRetention` reports the longest typed lifetime.

Native output sends your reflected JSON Schema through Converse `outputConfig.textFormat`. Use tool output for models that do not support Bedrock structured output.

## Portable settings

Portable maximum-token, temperature, top-p, stop-sequence, service-tier, and extra-header settings map to Converse fields. `ModelSettings.ExtraBody` maps to `additionalModelRequestFields` for model-specific Bedrock parameters.

`WithDefaultSettings` stores a detached copy. Agent and run settings override those defaults through the standard fieldwise merge.

## Current scope

Native Bedrock tools, guardrails, performance settings, request metadata, and streamed image and provider-tool blocks remain outside this implementation.
