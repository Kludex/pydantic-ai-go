package openaicodex

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/url"
)

var flowBoundParameters = map[string]struct{}{
	"client_id": {}, "redirect_uri": {}, "state": {}, "code_challenge": {},
}

// OAuthFlow is an authorization-code and PKCE login for the public Codex client.
// It performs no I/O until ExchangeCode or ExchangeCodeFromCallback is called.
type OAuthFlow struct {
	client       *http.Client
	redirectURI  string
	state        string
	codeVerifier string
}

// OAuthOption configures an OAuthFlow.
type OAuthOption func(*OAuthFlow)

// WithRedirectURI changes the callback URI. The public Codex client accepts
// only http://localhost:1455/auth/callback. Use another URI only with your own client registration.
func WithRedirectURI(redirectURI string) OAuthOption {
	return func(flow *OAuthFlow) { flow.redirectURI = redirectURI }
}

// WithOAuthState sets the CSRF state value. The default is cryptographically random.
func WithOAuthState(state string) OAuthOption { return func(flow *OAuthFlow) { flow.state = state } }

// NewOAuthFlow creates a login flow that uses the caller-owned HTTP client for token exchange.
func NewOAuthFlow(client *http.Client, options ...OAuthOption) (*OAuthFlow, error) {
	if client == nil {
		return nil, errors.New("openai-codex: OAuth HTTP client must not be nil")
	}
	flow := &OAuthFlow{
		client: client, redirectURI: defaultRedirect,
		state: rand.Text(), codeVerifier: rand.Text() + rand.Text(),
	}
	for _, option := range options {
		option(flow)
	}
	if flow.state == "" {
		return nil, errors.New("openai-codex: OAuth state must not be empty")
	}
	if _, err := url.ParseRequestURI(flow.redirectURI); err != nil {
		return nil, fmt.Errorf("openai-codex: invalid OAuth redirect URI: %w", err)
	}
	return flow, nil
}

// RedirectURI returns the callback URI bound to authorization codes.
func (flow *OAuthFlow) RedirectURI() string { return flow.redirectURI }

// State returns the CSRF state bound to the callback.
func (flow *OAuthFlow) State() string { return flow.state }

// CodeVerifier returns the PKCE verifier used during code exchange.
func (flow *OAuthFlow) CodeVerifier() string { return flow.codeVerifier }

// CodeChallenge returns the unpadded S256 challenge for CodeVerifier.
func (flow *OAuthFlow) CodeChallenge() string {
	digest := sha256.Sum256([]byte(flow.codeVerifier))
	return base64.RawURLEncoding.EncodeToString(digest[:])
}

// AuthorizationURL returns the URL to open in the user's browser. An empty
// scope selects the standard Codex scopes. Extra parameters cannot replace
// values bound to this flow.
func (flow *OAuthFlow) AuthorizationURL(scope string, extra url.Values) (string, error) {
	for name := range extra {
		if _, bound := flowBoundParameters[name]; bound {
			return "", fmt.Errorf("openai-codex: OAuth parameter %q is bound to the flow", name)
		}
	}
	if scope == "" {
		scope = defaultScope
	}
	query := url.Values{
		"response_type":              {"code"},
		"client_id":                  {publicClientID},
		"redirect_uri":               {flow.redirectURI},
		"scope":                      {scope},
		"state":                      {flow.state},
		"code_challenge":             {flow.CodeChallenge()},
		"code_challenge_method":      {"S256"},
		"id_token_add_organizations": {"true"},
		"codex_cli_simplified_flow":  {"true"},
	}
	for name, values := range extra {
		query[name] = append([]string(nil), values...)
	}
	return authorizeURL + "?" + query.Encode(), nil
}

// ExchangeCode exchanges one authorization code for Codex credentials.
func (flow *OAuthFlow) ExchangeCode(ctx context.Context, code string) (Credentials, error) {
	if code == "" {
		return Credentials{}, errors.New("openai-codex: authorization code must not be empty")
	}
	response, err := postToken(ctx, flow.client, map[string]string{
		"grant_type": "authorization_code", "code": code, "code_verifier": flow.codeVerifier,
		"redirect_uri": flow.redirectURI, "client_id": publicClientID,
	})
	if err != nil {
		return Credentials{}, err
	}
	return credentialsFromTokenResponse(response, "")
}
