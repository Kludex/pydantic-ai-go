# OpenAI ChatGPT plan usage

[Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source) lets the users of your open-source or locally run app pay for its requests with their ChatGPT plan instead of an API key. The `openaichatgpt` package runs the sign-in, sends Responses API requests with the signed-in user's token, and refreshes the token as it expires.

Offering plan usage in a paid or remotely hosted app needs OpenAI's approval first, through its [interest form](https://openai.com/form/sign-in-with-chatgpt-interest/). OpenAI's [terms and policies](https://openai.com/policies/) govern your use.

This is different from the [OpenAI Codex subscription](openai-codex.md) package, which signs in as the Codex CLI to use your own subscription. With Sign in with ChatGPT, each user approves your app by name, requests go to the public OpenAI API, and users can see and limit your app's usage in [ChatGPT settings](https://chatgpt.com/settings/usage).

## Sign in

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaichatgpt"
)

// Create once per installation and persist it, for example with uuid.New().URN().
const hostID = "urn:uuid:6f1c3a52-2b0e-4b0f-9a35-6d2e4f1a7b9c"

func main() {
	ctx := context.Background()
	flow, err := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App", openaichatgpt.WithHostID(hostID))
	if err != nil {
		log.Fatal(err)
	}
	authorization, err := flow.AuthorizationURL("", nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Open this URL to sign in:", authorization)
	credentials, err := flow.ExchangeCodeFromCallback(ctx)
	if err != nil {
		log.Fatal(err)
	}

	model, err := openaichatgpt.NewModel("gpt-6.1-sol", openaichatgpt.WithCredentials(credentials))
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(ctx, `Where does "hello world" come from?`, struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Output)
}
```

OpenAI requires the callback on `http://127.0.0.1:PORT/auth/callback`. `ExchangeCodeFromCallback` listens on that port, 1455 unless you pass `WithRedirectPort`, until the browser redirects there. It then exchanges the code for credentials. Opening the browser stays with your app.

The app name is what the user sees when approving your app. Use the same name on every installation. `WithHostID` sets an opaque identifier for one installation, such as a laptop or a VM. Create it once and reuse it for every sign-in on that installation. OpenAI asks apps to send it, but it is optional, because sign-ins that send it have been reported to time out.

The first sign-in to a ChatGPT account registers your app with it, and OpenAI issues the credentials' `ClientID` for that registration. The user can decline to let your app use their plan. The sign-in then still returns credentials, so you can save the registration, but `PlanUsageGranted` reports `false` and `NewModel` refuses them with `ErrPlanUsageNotGranted`.

`NewModel` never falls back to `OPENAI_API_KEY`. The `openai-chatgpt:` inference prefix exists, but it has no credentials to use and returns an error that says to sign in first. Build the model with `NewModel` as shown.

## Persist credentials

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaichatgpt"
)

type FileCredentialSource struct{ Path string }

func (source FileCredentialSource) Load(context.Context) (openaichatgpt.Credentials, error) {
	var credentials openaichatgpt.Credentials
	data, err := os.ReadFile(source.Path)
	if err != nil {
		return credentials, err
	}
	return credentials, json.Unmarshal(data, &credentials)
}

func (source FileCredentialSource) Save(_ context.Context, credentials openaichatgpt.Credentials) error {
	data, err := json.Marshal(credentials)
	if err != nil {
		return err
	}
	return os.WriteFile(source.Path, data, 0o600)
}

func main() {
	source := FileCredentialSource{Path: "chatgpt-credentials.json"}
	model, err := openaichatgpt.NewModel("gpt-6.1-sol", openaichatgpt.WithCredentialSource(source))
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

`WithCredentials` keeps credentials in memory only, so the next process has to sign in again. The model refreshes expired tokens automatically, and refresh tokens are single-use, so the stored copy has to keep up. With `WithCredentialSource`, the model calls `Load` on first use and again before every refresh, to adopt a set another process already rotated. It calls `Save` with the complete set after every refresh.

!!! warning "Protect the credentials file"
    The file holds a refresh token that grants access to the user's ChatGPT plan. Keep it out of source control and logs, and restrict it to the current user, as the `0o600` mode does.

If `Save` fails, the refreshed credentials stay active in memory and the request returns `CredentialsPersistenceError`. A later refresh does not adopt the spent set the source still holds. A rejected refresh returns `CredentialsRefreshError`. When its message says the grant was rejected, the user has to sign in again. Both implement `ai.ModelAPIError`, so a `FallbackModel` treats an unusable sign-in like any other provider failure.

## Sign in again

```go
package main

import (
	"context"
	"fmt"
	"net/http"
	"net/url"

	"github.com/Kludex/pydantic-ai-go/ai/models/openaichatgpt"
)

func signInAgain(ctx context.Context, saved openaichatgpt.Credentials) (openaichatgpt.Credentials, error) {
	flow, err := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App",
		openaichatgpt.WithClientID(saved.ClientID), openaichatgpt.WithIDTokenHint(saved.IDToken))
	if err != nil {
		return openaichatgpt.Credentials{}, err
	}
	var extra url.Values
	if !saved.PlanUsageGranted() {
		extra = url.Values{"prompt": {"consent"}}
	}
	authorization, err := flow.AuthorizationURL("", extra)
	if err != nil {
		return openaichatgpt.Credentials{}, err
	}
	fmt.Println("Open this URL to sign in again:", authorization)
	return flow.ExchangeCodeFromCallback(ctx)
}

func main() {}
```

To sign the same account in again, for example after its refresh token expired, pass the saved `ClientID` so the sign-in reuses the registration instead of creating another. Pass the saved `IDToken` so the user does not have to pick the account. `prompt=consent` asks a user who declined plan usage again.

With an ID token hint, the sign-in fails if the user completes it with a different ChatGPT account. The authorization URL carries the ID token, so keep it out of logs. A model checks plan usage before every request, so give new credentials to a new model.

## Errors

A sign-in the token endpoint rejects, or whose ID token fails validation, returns `CredentialsRefreshError`. Start again with a new flow that passes `WithClientID(flow.ClientID())`: the failed flow keeps the client ID issued for it, so the retry reuses that registration. The ID token's signature is not verified, because it comes straight from the token endpoint over TLS, which OpenID Connect Core 1.0 section 3.1.3.7 accepts in its place. Its issuer, audience, expiry, nonce, and subject are checked.

When the user's plan cannot serve a request, the request fails with an `openai.APIError` whose body carries an error code such as `subscription_sharing_usage_limit_exceeded`. OpenAI never moves the request to another billing method. [Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery) lists the codes and what to tell the user.

## Limitations

ChatGPT plan usage is a preview with the [limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations) OpenAI documents:

- Requests stream. `Request` drains the stream, so `agent.Run` works as usual.
- Every request is sent with `store=false`, so resuming a suspended run returns an error.
- System prompts are sent as developer messages, because system messages are rejected.
- `MaxTokens`, `ServiceTier`, `Temperature`, and `TopP` are dropped before sending. Raw `ExtraBody` fields are sent unchanged, so ones the route rejects surface as errors.
- `WebSearchTool` is the only supported native tool. Deferred tools are discovered with local tool search.
- Prompt caching has no effect, and `CountTokens` returns `ai.ErrTokenCountingUnsupported`.
- The callback must reach the computer running the browser. For an app on a remote VM, sign in locally and copy the credentials to it, as described in [Self-hosted VMs](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms).
