package githubcopilot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

const githubOAuthBaseURL = "https://github.com"

var oauthErrorPattern = regexp.MustCompile(`^[a-z_]+$`)

// DeviceAuthorization is a GitHub device-login challenge. Display UserCode at
// VerificationURI. DeviceCode is used only by WaitForAuthorization.
type DeviceAuthorization struct {
	DeviceCode      string `json:"device_code"`
	UserCode        string `json:"user_code"`
	VerificationURI string `json:"verification_uri"`
	ExpiresIn       int    `json:"expires_in"`
	Interval        int    `json:"interval"`
}

// String omits the secret device code.
func (authorization DeviceAuthorization) String() string {
	return fmt.Sprintf(
		"DeviceAuthorization{UserCode:%q VerificationURI:%q ExpiresIn:%d Interval:%d}",
		authorization.UserCode,
		authorization.VerificationURI,
		authorization.ExpiresIn,
		authorization.Interval,
	)
}

// Credentials contains GitHub OAuth credentials returned by device login.
// Applications own secure persistence and token renewal.
type Credentials struct {
	AccessToken           string `json:"access_token"`
	TokenType             string `json:"token_type"`
	Scope                 string `json:"scope"`
	RefreshToken          string `json:"refresh_token,omitempty"`
	ExpiresIn             int    `json:"expires_in,omitempty"`
	RefreshTokenExpiresIn int    `json:"refresh_token_expires_in,omitempty"`
}

// String omits access and refresh tokens.
func (credentials Credentials) String() string {
	return fmt.Sprintf(
		"Credentials{TokenType:%q Scope:%q ExpiresIn:%d RefreshTokenExpiresIn:%d}",
		credentials.TokenType,
		credentials.Scope,
		credentials.ExpiresIn,
		credentials.RefreshTokenExpiresIn,
	)
}

// OAuthOption configures a GitHub device authorization flow.
type OAuthOption func(*OAuthFlow)

// WithOAuthScope sets the space-separated GitHub OAuth scopes. The default is no scopes.
func WithOAuthScope(scope string) OAuthOption { return func(flow *OAuthFlow) { flow.scope = scope } }

// WithOAuthHTTPClient sets the caller-owned client used for GitHub OAuth requests.
func WithOAuthHTTPClient(client *http.Client) OAuthOption {
	if client == nil {
		panic("githubcopilot: OAuth HTTP client must not be nil")
	}
	return func(flow *OAuthFlow) { flow.httpClient = client }
}

// OAuthFlow performs one GitHub.com device authorization flow. It does not
// open a browser, persist credentials, refresh tokens, or verify Copilot access.
type OAuthFlow struct {
	clientID   string
	scope      string
	httpClient *http.Client

	mu        sync.Mutex
	challenge *DeviceAuthorization
	deadline  time.Time
}

// NewOAuthFlow creates a device flow for a caller-owned GitHub OAuth application.
func NewOAuthFlow(clientID string, options ...OAuthOption) (*OAuthFlow, error) {
	if strings.TrimSpace(clientID) == "" {
		return nil, errors.New("githubcopilot: OAuth client ID must not be empty")
	}
	flow := &OAuthFlow{clientID: clientID, httpClient: http.DefaultClient}
	for _, option := range options {
		option(flow)
	}
	return flow, nil
}

// Start requests a new device code and replaces any previous challenge.
func (flow *OAuthFlow) Start(ctx context.Context) (DeviceAuthorization, error) {
	flow.mu.Lock()
	flow.challenge = nil
	flow.deadline = time.Time{}
	flow.mu.Unlock()

	body, err := flow.post(ctx, "/login/device/code", url.Values{
		"client_id": {flow.clientID},
		"scope":     {flow.scope},
	})
	if err != nil {
		return DeviceAuthorization{}, err
	}
	receivedAt := time.Now()
	if oauthError, ok := decodeOAuthError(body); ok {
		return DeviceAuthorization{}, fmt.Errorf("githubcopilot: GitHub device authorization failed: %s", oauthError.Error)
	}
	var wire struct {
		DeviceCode      string `json:"device_code"`
		UserCode        string `json:"user_code"`
		VerificationURI string `json:"verification_uri"`
		ExpiresIn       *int   `json:"expires_in"`
		Interval        *int   `json:"interval"`
	}
	if err := decodeJSON(body, &wire); err != nil || wire.DeviceCode == "" || wire.UserCode == "" ||
		wire.VerificationURI != "https://github.com/login/device" || wire.ExpiresIn == nil || *wire.ExpiresIn <= 0 ||
		wire.Interval != nil && *wire.Interval <= 0 {
		return DeviceAuthorization{}, errors.New("githubcopilot: GitHub returned an invalid device authorization response")
	}
	challenge := DeviceAuthorization{
		DeviceCode: wire.DeviceCode, UserCode: wire.UserCode, VerificationURI: wire.VerificationURI,
		ExpiresIn: *wire.ExpiresIn, Interval: 5,
	}
	if wire.Interval != nil {
		challenge.Interval = *wire.Interval
	}
	flow.mu.Lock()
	flow.challenge = &challenge
	flow.deadline = receivedAt.Add(time.Duration(challenge.ExpiresIn) * time.Second)
	flow.mu.Unlock()
	return challenge, nil
}

// WaitForAuthorization polls until approval, rejection, expiry, or cancellation.
// A challenge is consumed once. Call Start again after any outcome.
func (flow *OAuthFlow) WaitForAuthorization(ctx context.Context) (Credentials, error) {
	parent := ctx
	flow.mu.Lock()
	if flow.challenge == nil {
		flow.mu.Unlock()
		return Credentials{}, errors.New("githubcopilot: call Start before WaitForAuthorization")
	}
	challenge := *flow.challenge
	deadline := flow.deadline
	flow.challenge = nil
	flow.deadline = time.Time{}
	flow.mu.Unlock()

	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	interval := time.Duration(challenge.Interval) * time.Second
	for {
		timer := time.NewTimer(interval)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			return Credentials{}, oauthContextError(parent)
		}
		body, err := flow.post(ctx, "/login/oauth/access_token", url.Values{
			"client_id":   {flow.clientID},
			"device_code": {challenge.DeviceCode},
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
		})
		if err != nil {
			if ctx.Err() != nil {
				return Credentials{}, oauthContextError(parent)
			}
			return Credentials{}, err
		}
		if oauthError, ok := decodeOAuthError(body); ok {
			switch oauthError.Error {
			case "authorization_pending":
				continue
			case "slow_down":
				interval += 5 * time.Second
				if requested := time.Duration(oauthError.Interval) * time.Second; requested > interval {
					interval = requested
				}
				continue
			default:
				return Credentials{}, fmt.Errorf(
					"githubcopilot: GitHub device authorization failed: %s", oauthError.Error,
				)
			}
		}
		var wire struct {
			AccessToken           string  `json:"access_token"`
			TokenType             string  `json:"token_type"`
			Scope                 string  `json:"scope"`
			RefreshToken          *string `json:"refresh_token"`
			ExpiresIn             *int    `json:"expires_in"`
			RefreshTokenExpiresIn *int    `json:"refresh_token_expires_in"`
		}
		if err := decodeJSON(body, &wire); err != nil || wire.AccessToken == "" || wire.TokenType != "bearer" ||
			wire.RefreshToken != nil && *wire.RefreshToken == "" || wire.ExpiresIn != nil && *wire.ExpiresIn <= 0 ||
			wire.RefreshTokenExpiresIn != nil && *wire.RefreshTokenExpiresIn <= 0 {
			return Credentials{}, errors.New("githubcopilot: GitHub returned an invalid device token response")
		}
		credentials := Credentials{AccessToken: wire.AccessToken, TokenType: wire.TokenType, Scope: wire.Scope}
		if wire.RefreshToken != nil {
			credentials.RefreshToken = *wire.RefreshToken
		}
		if wire.ExpiresIn != nil {
			credentials.ExpiresIn = *wire.ExpiresIn
		}
		if wire.RefreshTokenExpiresIn != nil {
			credentials.RefreshTokenExpiresIn = *wire.RefreshTokenExpiresIn
		}
		return credentials, nil
	}
}

type oauthErrorResponse struct {
	Error    string `json:"error"`
	Interval int    `json:"interval"`
}

func decodeOAuthError(body []byte) (oauthErrorResponse, bool) {
	var response oauthErrorResponse
	if decodeJSON(body, &response) != nil || !oauthErrorPattern.MatchString(response.Error) || response.Interval < 0 {
		return oauthErrorResponse{}, false
	}
	return response, true
}

func decodeJSON(body []byte, target any) error {
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("trailing JSON value")
	}
	return nil
}

func oauthContextError(parent context.Context) error {
	if parent.Err() != nil {
		return context.Cause(parent)
	}
	return errors.New("githubcopilot: GitHub device authorization expired; call Start to request a new code")
}

func (flow *OAuthFlow) post(ctx context.Context, path string, form url.Values) ([]byte, error) {
	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	encoded := form.Encode()
	request := (&http.Request{
		Method: http.MethodPost,
		URL:    &url.URL{Scheme: "https", Host: strings.TrimPrefix(githubOAuthBaseURL, "https://"), Path: path},
		Header: http.Header{"Accept": {"application/json"}, "Content-Type": {"application/x-www-form-urlencoded"}},
		Body:   io.NopCloser(strings.NewReader(encoded)),
	}).WithContext(requestCtx)
	client := *flow.httpClient
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("githubcopilot: GitHub device authorization request failed (HTTP %d)", response.StatusCode)
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	return body, nil
}
