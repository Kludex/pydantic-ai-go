# OpenAI Codex subscription

Use your ChatGPT/Codex subscription with the OpenAI Responses API. Use the regular [`openai`](providers.md#openai) package when you have an OpenAI API key.

## Use Codex CLI credentials

```go
package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaicodex"
)

func main() {
	model, err := openaicodex.NewModel("gpt-5.6-luna")
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Explain structured concurrency.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

Run `codex login` before you start the program. `NewModel` reads `$CODEX_HOME/auth.json` or `~/.codex/auth.json` when you do not provide credentials.

The model never writes the Codex CLI file. A rotated token remains in memory until the process exits. This prevents the library from corrupting credentials owned by another program.

## Persist rotated credentials

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaicodex"
)

type FileCredentialSource struct {
	Path string
}

func (source FileCredentialSource) Load(_ context.Context) (openaicodex.Credentials, error) {
	data, err := os.ReadFile(source.Path)
	if err != nil {
		return openaicodex.Credentials{}, err
	}
	var credentials openaicodex.Credentials
	if err := json.Unmarshal(data, &credentials); err != nil {
		return openaicodex.Credentials{}, err
	}
	return credentials, nil
}

func (source FileCredentialSource) Save(_ context.Context, credentials openaicodex.Credentials) error {
	data, err := json.Marshal(credentials)
	if err != nil {
		return err
	}
	directory := filepath.Dir(source.Path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, ".codex-credentials-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryPath, source.Path)
}

func main() {
	source := FileCredentialSource{Path: "codex-credentials.json"}
	model, err := openaicodex.NewModel(
		"gpt-5.6-luna",
		openaicodex.WithCredentialSource(source),
		openaicodex.WithHTTPClient(http.DefaultClient),
	)
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(context.Background(), "Say hello.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

`CredentialSource.Load` runs on first use. It runs again before each refresh so the model can adopt credentials rotated by another process. `CredentialSource.Save` receives the complete new set after refresh.

Refresh tokens are single-use. Coordinate writers when multiple processes share one source. The built-in refresh is single-flight inside one model instance, but it cannot lock your external store across processes.

If `Save` fails, the new credentials remain active in memory and the request returns `CredentialsPersistenceError`. This makes failed durability visible without reverting to a token that OpenAI already rotated.

Store application credentials outside `~/.codex`. That directory belongs to the Codex CLI.

## Log in with OAuth PKCE

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/Kludex/pydantic-ai-go/ai/models/openaicodex"
)

func main() {
	ctx := context.Background()
	flow, err := openaicodex.NewOAuthFlow(http.DefaultClient)
	if err != nil {
		log.Fatal(err)
	}
	authorizationURL, err := flow.AuthorizationURL("", nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Open this URL:\n%s\n", authorizationURL)

	credentials, err := flow.ExchangeCodeFromCallback(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Connected account %s\n", credentials.AccountID)
}
```

The public Codex OAuth client pins `http://localhost:1455/auth/callback`. `ExchangeCodeFromCallback` listens on that address until the matching state arrives. You can call `ExchangeCode` from your own callback handler instead.

`OAuthFlow` uses authorization code flow with an S256 PKCE challenge. The HTTP client remains caller-owned and is used for the token exchange.

## Credential safety

The model adds `Authorization`, `chatgpt-account-id`, and `originator` only for HTTPS requests to `chatgpt.com`. A redirect to another host or to plaintext HTTP does not receive those values.

The model copies your `http.Client` before wrapping its transport. It never changes or closes the client you pass to `WithHTTPClient`.

Avoid full HTTP body capture around OAuth calls. Authorization codes, access tokens, refresh tokens, and ID tokens are secrets.

## Wire behavior

Codex uses a narrower Responses API dialect:

- Every request uses streaming, including `Model.Request` calls. The wrapper drains the stream into a normal `ModelResponse`.
- Every request sends `store: false`.
- `MaxTokens`, `Temperature`, and `TopP` are omitted.
- `CountTokens` returns `ai.ErrTokenCountingUnsupported` because the subscription endpoint does not expose `/responses/input_tokens`.
- Suspended response continuation is rejected because `store: false` leaves no server-side response to retrieve.

The model derives `session-id`, `thread-id`, `x-client-request-id`, and `prompt_cache_key` from the latest message `ConversationID`. Explicit request headers and an explicit OpenAI prompt cache key win. These values improve prompt-cache affinity but do not guarantee a cache hit.
