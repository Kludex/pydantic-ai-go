package openaichatgpt

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

const defaultRedirectPort = 1455

// Authorization parameters the callback validation and code exchange rely on.
var flowBoundParameters = []string{"client_id", "redirect_uri", "state", "code_challenge", "nonce", "resource"}

// Opaque installation identifier forms OpenAI accepts.
var hostIDPrefixes = []string{"urn:uuid:", "urn:ietf:params:oauth:jwk-thumbprint:", "did:key:"}

// OAuthFlow is one authorization-code and PKCE sign-in with ChatGPT plan usage.
// Without a client ID, the sign-in registers a new OAuth client for the user's
// ChatGPT account, and the credentials carry the issued ClientID that later
// sign-ins pass back to reuse it. Construction does no I/O; the browser and
// credential storage stay application-owned.
type OAuthFlow struct {
	client       *http.Client
	appName      string
	hostID       string
	clientID     string
	idTokenHint  string
	hintSubject  string
	redirectPort int
	state        string
	nonce        string
	codeVerifier string
}

// OAuthOption configures an OAuthFlow.
type OAuthOption func(*OAuthFlow)

// WithHostID sets a stable, opaque identifier for this installation of your
// app, created once and persisted: a urn:uuid, an RFC 9278 JWK thumbprint URI,
// or a did:key identifier.
func WithHostID(hostID string) OAuthOption { return func(flow *OAuthFlow) { flow.hostID = hostID } }

// WithClientID reuses the registration from an earlier sign-in to the same
// ChatGPT account. Pass the ClientID of its saved credentials.
func WithClientID(clientID string) OAuthOption {
	return func(flow *OAuthFlow) { flow.clientID = clientID }
}

// WithIDTokenHint passes the IDToken of earlier credentials, so the user skips
// choosing the account. The sign-in must then complete for that account.
func WithIDTokenHint(token string) OAuthOption {
	return func(flow *OAuthFlow) { flow.idTokenHint = token }
}

// WithRedirectPort changes the loopback callback port. OpenAI only lets the
// port vary in http://127.0.0.1:PORT/auth/callback.
func WithRedirectPort(port int) OAuthOption {
	return func(flow *OAuthFlow) { flow.redirectPort = port }
}

// WithOAuthState sets the CSRF state bound to the callback. The default is random.
func WithOAuthState(state string) OAuthOption { return func(flow *OAuthFlow) { flow.state = state } }

// NewOAuthFlow creates a sign-in for appName, the name shown to the user when a
// new registration is approved. Use the same name across installations. The
// caller-owned client sends the token exchange.
func NewOAuthFlow(client *http.Client, appName string, options ...OAuthOption) (*OAuthFlow, error) {
	if client == nil {
		return nil, errors.New("openai-chatgpt: OAuth HTTP client must not be nil")
	}
	flow := &OAuthFlow{
		client: client, appName: appName, redirectPort: defaultRedirectPort,
		state: rand.Text(), nonce: rand.Text(), codeVerifier: rand.Text() + rand.Text(),
	}
	for _, option := range options {
		option(flow)
	}
	switch {
	case flow.state == "":
		return nil, errors.New("openai-chatgpt: OAuth state must not be empty")
	case flow.redirectPort < 1 || flow.redirectPort > 65535:
		return nil, fmt.Errorf(
			"openai-chatgpt: the redirect URI must be http://127.0.0.1:PORT/auth/callback; "+
				"Sign in with ChatGPT only lets the port vary, got port %d", flow.redirectPort,
		)
	case flow.hostID != "" && !hasAnyPrefix(flow.hostID, hostIDPrefixes):
		return nil, fmt.Errorf(
			"openai-chatgpt: host ID must be an opaque urn:uuid, urn:ietf:params:oauth:jwk-thumbprint, "+
				"or did:key identifier, got %q", flow.hostID,
		)
	}
	if flow.idTokenHint != "" {
		decoded, ok := claims(flow.idTokenHint)
		if !ok || decoded.Subject == "" {
			return nil, errors.New("openai-chatgpt: the ID token hint is not an ID token from an earlier sign-in")
		}
		flow.hintSubject = decoded.Subject
	}
	return flow, nil
}

// RedirectURI returns the loopback callback the authorization code is bound to.
func (flow *OAuthFlow) RedirectURI() string {
	return "http://127.0.0.1:" + strconv.Itoa(flow.redirectPort) + "/auth/callback"
}

// State returns the CSRF state bound to the callback.
func (flow *OAuthFlow) State() string { return flow.state }

// Nonce returns the value the ID token must carry.
func (flow *OAuthFlow) Nonce() string { return flow.nonce }

// CodeVerifier returns the PKCE verifier used during code exchange.
func (flow *OAuthFlow) CodeVerifier() string { return flow.codeVerifier }

// CodeChallenge returns the unpadded S256 challenge for CodeVerifier.
func (flow *OAuthFlow) CodeChallenge() string {
	digest := sha256.Sum256([]byte(flow.codeVerifier))
	return base64.RawURLEncoding.EncodeToString(digest[:])
}

// ClientID returns the registration this flow uses. After a new registration's
// callback is accepted it is the issued client ID, so a retry after a failed
// exchange can reuse the registration.
func (flow *OAuthFlow) ClientID() string { return flow.clientID }

// AuthorizationURL returns the URL to send the user to. It carries the ID token
// hint when one was given, so keep it out of logs. An empty scope requests the
// identity scopes plus plan usage; keep offline_access and
// chatgpt.tokens.use.direct. Extra parameters cannot replace values the flow
// validates. Pass prompt=consent to ask a user who declined plan usage again.
func (flow *OAuthFlow) AuthorizationURL(scope string, extra url.Values) (string, error) {
	for _, name := range flowBoundParameters {
		if _, bound := extra[name]; bound {
			return "", fmt.Errorf(
				"openai-chatgpt: extra parameters cannot override %s: the flow validates the callback and "+
					"exchanges the code with its own values", name,
			)
		}
	}
	if scope == "" {
		scope = defaultScope
	}
	clientID := flow.clientID
	if clientID == "" {
		clientID = registrationClientID
	}
	query := url.Values{
		"response_type": {"code"}, "client_id": {clientID}, "redirect_uri": {flow.RedirectURI()},
		"scope": {scope}, "resource": {resource}, "state": {flow.state}, "nonce": {flow.nonce},
		"code_challenge": {flow.CodeChallenge()}, "code_challenge_method": {"S256"},
	}
	if flow.clientID == "" {
		query.Set("agent_name_hint", flow.appName)
	}
	if flow.hostID != "" {
		query.Set("ext_agent_host_id", flow.hostID)
	}
	if flow.idTokenHint != "" {
		query.Set("id_token_hint", flow.idTokenHint)
	}
	for name, values := range extra {
		query[name] = append([]string(nil), values...)
	}
	return authorizeURL + "?" + query.Encode(), nil
}

// ExchangeCode exchanges the callback's code for credentials. callbackClientID
// is the callback's client_id parameter, which carries the issued client ID
// after a new registration; a sign-in that reused a client ID may omit it.
// Credentials without plan usage are returned too, so the registration can be
// saved; check PlanUsageGranted.
func (flow *OAuthFlow) ExchangeCode(ctx context.Context, code, callbackClientID string) (Credentials, error) {
	switch {
	case code == "":
		return Credentials{}, errors.New("openai-chatgpt: authorization code must not be empty")
	case flow.clientID != "":
		if callbackClientID != "" && callbackClientID != flow.clientID {
			return Credentials{}, fmt.Errorf(
				"openai-chatgpt: the authorization callback carried client %q instead of %q; its code was not exchanged",
				callbackClientID, flow.clientID,
			)
		}
	case callbackClientID == "" || callbackClientID == registrationClientID:
		return Credentials{}, errors.New(
			"openai-chatgpt: the registration is incomplete: the authorization callback did not carry the issued client ID",
		)
	default:
		flow.clientID = callbackClientID
	}
	response, err := postToken(ctx, flow.client, url.Values{
		"grant_type": {"authorization_code"}, "code": {code}, "code_verifier": {flow.codeVerifier},
		"redirect_uri": {flow.RedirectURI()}, "client_id": {flow.clientID}, "resource": {resource},
	})
	if err != nil {
		return Credentials{}, err
	}
	if response.IDToken == "" {
		return Credentials{}, &CredentialsRefreshError{err: errors.New("the token response carried no ID token")}
	}
	if err := validateIDToken(response.IDToken, flow.clientID, flow.nonce, flow.hintSubject); err != nil {
		return Credentials{}, err
	}
	credentials := Credentials{
		ClientID: flow.clientID, AccessToken: response.AccessToken, RefreshToken: response.RefreshToken,
		IDToken: response.IDToken,
	}
	if response.Scope != nil {
		credentials.Scope = *response.Scope
	}
	return credentials, nil
}

func hasAnyPrefix(value string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(value, prefix) {
			return true
		}
	}
	return false
}
