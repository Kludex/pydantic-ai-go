package openaichatgpt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/internal/oauthcreds"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	resource       = "https://api.openai.com/v1"
	issuer         = "https://auth.openai.com"
	authorizeURL   = "https://auth.openai.com/api/accounts/authorize"
	tokenURL       = "https://auth.openai.com/api/accounts/oauth/token"
	// registrationClientID starts a first-time registration. It is never the
	// client ID to save or to exchange codes with.
	registrationClientID = "dynamic_agent_client"
	defaultScope         = "openid profile email offline_access resource.invoke " + planUsageScope
	clockSkew            = 60 * time.Second
)

// Refresh error codes that mean the user has to sign in again.
var unusableRefreshTokenErrors = []string{
	"invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
	"refresh_token_invalidated", "refresh_token_reused",
}

type tokenResponse struct {
	AccessToken  string  `json:"access_token"`
	RefreshToken string  `json:"refresh_token"`
	IDToken      string  `json:"id_token"`
	Scope        *string `json:"scope"`
}

// RefreshCredentials exchanges the single-use refresh token against the
// account's issued client. The caller owns client.
func RefreshCredentials(ctx context.Context, client *http.Client, credentials Credentials) (Credentials, error) {
	response, err := postToken(ctx, client, url.Values{
		"grant_type": {"refresh_token"}, "refresh_token": {credentials.RefreshToken},
		"client_id": {credentials.ClientID}, "resource": {resource},
	})
	if err != nil {
		return Credentials{}, err
	}
	refreshed := Credentials{
		ClientID: credentials.ClientID, AccessToken: response.AccessToken, RefreshToken: response.RefreshToken,
		IDToken: response.IDToken, Scope: credentials.Scope,
	}
	if refreshed.IDToken == "" {
		refreshed.IDToken = credentials.IDToken
	}
	// Omitting scope from a refresh keeps the original grant.
	if response.Scope != nil {
		refreshed.Scope = *response.Scope
	}
	return refreshed, nil
}

func postToken(ctx context.Context, client *http.Client, form url.Values) (tokenResponse, error) {
	request, _ := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := client.Do(request)
	if err != nil {
		return tokenResponse{}, &CredentialsRefreshError{err: err}
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		var body struct {
			Error       string `json:"error"`
			Description string `json:"error_description"`
		}
		_ = json.NewDecoder(response.Body).Decode(&body)
		detail := body.Description
		if detail == "" {
			detail = body.Error
		}
		if detail == "" {
			detail = http.StatusText(response.StatusCode)
		}
		if slices.Contains(unusableRefreshTokenErrors, body.Error) {
			detail += "; the grant was rejected, rerun the authorization flow"
		}
		return tokenResponse{}, &CredentialsRefreshError{err: fmt.Errorf(
			"token request failed with status %d: %s", response.StatusCode, detail,
		)}
	}
	var body tokenResponse
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil || body.AccessToken == "" ||
		body.RefreshToken == "" {
		return tokenResponse{}, &CredentialsRefreshError{err: errors.New("token endpoint returned an unexpected response")}
	}
	return body, nil
}

type idTokenClaims struct {
	Issuer   string          `json:"iss"`
	Audience json.RawMessage `json:"aud"`
	Subject  string          `json:"sub"`
	Nonce    string          `json:"nonce"`
}

// claims decodes the ID token claims checked after a sign-in.
func claims(token string) (idTokenClaims, bool) {
	payload, ok := oauthcreds.Claims(token)
	if !ok {
		return idTokenClaims{}, false
	}
	encoded, _ := json.Marshal(payload)
	var decoded idTokenClaims
	return decoded, json.Unmarshal(encoded, &decoded) == nil
}

// validateIDToken checks the issuer, audience, expiry, nonce, and subject. The
// signature is not verified: the token comes straight from the token endpoint
// over TLS, which OpenID Connect Core 1.0 section 3.1.3.7 accepts in its place.
func validateIDToken(token, clientID, nonce, subject string) error {
	decoded, ok := claims(token)
	if !ok {
		return &CredentialsRefreshError{err: errors.New("the token response carried a malformed ID token")}
	}
	var audiences []string
	var single string
	if json.Unmarshal(decoded.Audience, &single) == nil {
		audiences = []string{single}
	} else {
		_ = json.Unmarshal(decoded.Audience, &audiences)
	}
	expiresAt, hasExpiry := oauthcreds.ExpiresAt(token)
	switch {
	case decoded.Issuer != issuer:
		return &CredentialsRefreshError{err: fmt.Errorf("the ID token was issued by %q, not %q", decoded.Issuer, issuer)}
	case !slices.Contains(audiences, clientID):
		return &CredentialsRefreshError{err: fmt.Errorf("the ID token was not issued for client %q", clientID)}
	case !hasExpiry || expiresAt.Add(clockSkew).Before(time.Now()):
		return &CredentialsRefreshError{err: errors.New("the ID token has expired; start the sign-in again")}
	case decoded.Nonce != nonce:
		return &CredentialsRefreshError{err: errors.New(
			"the ID token does not carry this sign-in attempt's nonce; start the sign-in again",
		)}
	case decoded.Subject == "":
		return &CredentialsRefreshError{err: errors.New("the ID token does not identify an account")}
	case subject != "" && decoded.Subject != subject:
		return errors.New("openai-chatgpt: the sign-in completed for a different ChatGPT account than the ID " +
			"token hint identifies; its credentials were not used")
	}
	return nil
}
