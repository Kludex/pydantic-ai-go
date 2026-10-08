# Prompt caching

```go
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/anthropic"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-5"),
		ai.WithInstructions("Answer briefly."),
		ai.WithCapabilities(ai.Caching{}),
	)
	result, err := agent.Run(context.Background(), "What is prompt caching?", struct{}{})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
```

Set `ANTHROPIC_API_KEY` before running this example. Prompt caching lets a provider reuse a recently processed prompt prefix. Long instructions, tool definitions, and repeated conversation turns can cost less and run faster.

`Caching{}` enables the provider's default retention. It caches the conversation as well as the stable prefix. Anthropic's HTTP API manages the moving conversation boundary itself. Other supported transports receive explicit boundaries.

Cache writes cost more than uncached input. Anthropic's five-minute cache and OpenAI's GPT-5.6 cache charge 1.25 times the input price to write and about 0.1 times to read. Anthropic's one-hour cache charges twice the input price to write. A one-off request pays the write premium without a later read to recover it.

## Cache only the stable prefix

```go
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/anthropic"
)

func main() {
	messages := false
	agent := ai.NewAgent[struct{}, string](
		anthropic.NewModel("claude-sonnet-4-5"),
		ai.WithInstructions("You answer questions about a shared product reference."),
		ai.WithCapabilities(ai.Caching{
			Retention: ai.CacheRetention1Hour,
			Messages:  &messages,
		}),
	)
	result, err := agent.Run(context.Background(), "Explain your role.", struct{}{})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
```

`Messages: &messages` with `false` caches only static instructions and tool definitions. Use it when many short conversations share a long prefix but their individual messages will not be reused. A nil `Messages` defaults to `true`.

## Configure settings directly

```go
package main

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func main() {
	agent := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(),
		ai.WithModelSettings(ai.ModelSettings{
			Cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour},
		}),
	)
	result, err := agent.Run(context.Background(), "Check configuration without a network request.", struct{}{},
		ai.WithRunModelSettings(ai.ModelSettings{
			Cache: &ai.CacheConfig{Retention: ai.CacheRetentionDisabled},
		}),
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Output)
}
```

Use `ModelSettings.Cache` without a capability when you already manage settings. A nil cache leaves configuration unset. An empty `CacheConfig` enables defaults. `CacheRetentionDisabled` overrides a lower-precedence portable cache setting.

| Retention | Behavior |
| --- | --- |
| Empty | Enable the provider's default retention |
| `CacheRetentionDisabled` | Disable library-managed caching |
| `CacheRetention5Minutes` | Request five minutes |
| `CacheRetention30Minutes` | Request thirty minutes |
| `CacheRetention1Hour` | Request one hour |

Unsupported retention tiers snap down to the nearest shorter supported tier. If none is shorter, they snap up to the shortest tier. Providers without configurable tiers use their own default.

Provider-local cache controls take precedence over the whole portable configuration, including when a local option in `ExtraBody` is explicitly `false`. OpenAI's legacy maximum-retention policy and cache key are independent settings. They do not suppress the portable configuration.

Disabling portable caching does not disable explicit `CachePoint` markers or provider-local controls. Providers that cache implicitly still do so. No cached data is stored by this Go library.

Repeated `Caching` capabilities share the ID `caching`. The last configuration wins within a registration layer. A matching run capability replaces the agent configuration. Set `ID` when you need a separate capability identity.

## Provider translation

| Provider | Default portable configuration | Stable-prefix-only behavior |
| --- | --- | --- |
| Anthropic HTTP API | Automatic cache, five minutes; one hour supported | Static instruction and tool-definition boundaries |
| Anthropic legacy Bedrock client | Instruction, tool-definition, and conversation boundaries | Omit the conversation boundary |
| Bedrock Converse Claude | Instruction, tool-definition, and conversation boundaries | Omit the conversation boundary |
| Bedrock Converse Nova | Instruction and conversation boundaries, five minutes | Cache instructions only |
| OpenRouter Anthropic routes | Instruction, tool-definition, and conversation boundaries | Omit the conversation boundary |
| OpenRouter Gemini 2.5 and later routes | Instruction and conversation boundaries; no TTL on the wire | Cache instructions only |
| OpenAI GPT-5.6 and GPT-6 families | Implicit mode, thirty-minute TTL, static instruction boundary | Explicit mode with only the instruction boundary |
| Other models and implicitly caching providers | No portable translation | No effect |

Bedrock grants the one-hour tier only to selected Claude models. Older Claude models snap to five minutes. Nova does not cache tool definitions.

OpenAI Responses requests that continue a `previous_response_id` or a `conversation` cannot place an instruction boundary. In stable-prefix-only mode, those requests therefore add no library-managed cache boundary.

On Bedrock and OpenRouter, a wide tool turn can move the newest boundary beyond the provider's cache lookback. The previous request's tail gets a boundary too. Instruction and tool-definition boundaries take priority. Excess message boundaries are dropped oldest-first to stay within the four-boundary limit. An explicit marker at Anthropic's final cacheable block takes precedence over its automatic boundary.

## Retention and health

Use `ai.ResolveCacheRetention` with the model and settings to resolve the active requested lifetime. It respects provider-local precedence. `ai.PromptCacheOutlook` uses that lifetime or `ModelProfile.DefaultCacheRetention` to classify served history as warm, cold, or unknown. A marker's TTL extends the estimate only when the profile says the provider honors that tier. OpenAI ignores a marker's one-hour TTL.

[Instrumentation](observability.md) records cache reads, writes, and cache health. When a model needs request-side configuration, an unconfigured request with at least 4,096 input tokens and no reported cache use gets `pydantic_ai.cache.not_enabled`. The request span emits an event with the same name once per conversation and provider/endpoint/model key. Explicit markers, deliberately disabled caching, provider-local settings, and implicitly caching models suppress this diagnostic.
