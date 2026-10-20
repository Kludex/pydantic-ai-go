package openaichatgpt

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
)

const planUsageScope = "chatgpt.tokens.use.direct"

// Credentials are one ChatGPT account's plan-usage tokens, tied to the OAuth
// client registered for that account. Persist every field together: ClientID
// identifies the registration a later sign-in reuses, IDToken is that
// sign-in's hint, and Scope records whether the user granted plan usage.
type Credentials struct {
	ClientID     string `json:"client_id"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token,omitempty"`
	Scope        string `json:"scope,omitempty"`
}

// PlanUsageGranted reports whether the user granted the chatgpt.tokens.use.direct
// scope, which inference requires.
func (credentials Credentials) PlanUsageGranted() bool {
	return slices.Contains(strings.Fields(credentials.Scope), planUsageScope)
}

// String formats credentials without exposing any token.
func (credentials Credentials) String() string {
	return fmt.Sprintf("OpenAIChatGPTCredentials{ClientID:%q, Scope:%q}", credentials.ClientID, credentials.Scope)
}

// GoString formats credentials without exposing any token.
func (credentials Credentials) GoString() string { return credentials.String() }

// CredentialSource is application-owned storage for rotating credentials.
// Load is called on first use and immediately before a refresh, to adopt a set
// another process already rotated. Save is called with the complete rotated
// set. Refresh tokens are single-use, so processes sharing one source must
// coordinate atomic replacement.
type CredentialSource interface {
	Load(ctx context.Context) (Credentials, error)
	Save(ctx context.Context, credentials Credentials) error
}

// CredentialsRefreshError reports a rejected or malformed exchange at the
// token endpoint, including an ID token that fails validation.
type CredentialsRefreshError struct{ err error }

// Error describes the token exchange failure.
func (err *CredentialsRefreshError) Error() string { return "openai-chatgpt: " + err.err.Error() }

// Unwrap returns the token exchange failure.
func (err *CredentialsRefreshError) Unwrap() error { return err.err }

// IsModelAPIError marks credential failures as eligible for model fallback.
func (*CredentialsRefreshError) IsModelAPIError() bool { return true }

// CredentialsPersistenceError reports that rotated credentials are active in
// memory but the caller-owned source failed to save them.
type CredentialsPersistenceError struct{ err error }

// Error describes the persistence failure.
func (err *CredentialsPersistenceError) Error() string {
	return "openai-chatgpt: credentials were refreshed in memory but saving them failed: " + err.err.Error()
}

// Unwrap returns the caller-owned persistence failure.
func (err *CredentialsPersistenceError) Unwrap() error { return err.err }

// IsModelAPIError marks credential failures as eligible for model fallback.
func (*CredentialsPersistenceError) IsModelAPIError() bool { return true }

// ErrPlanUsageNotGranted reports credentials without ChatGPT plan usage.
var ErrPlanUsageNotGranted = errors.New("openai-chatgpt: these credentials do not grant ChatGPT plan usage (" +
	planUsageScope + "); sign in again with the saved client ID, add prompt=consent to the authorization URL, " +
	"and approve using your ChatGPT plan")

func (credentials Credentials) validate() error {
	switch {
	case credentials.ClientID == "":
		return errors.New("openai-chatgpt: client_id must not be empty")
	case credentials.AccessToken == "":
		return errors.New("openai-chatgpt: access_token must not be empty")
	case credentials.RefreshToken == "":
		return errors.New("openai-chatgpt: refresh_token must not be empty")
	case !credentials.PlanUsageGranted():
		return fmt.Errorf("%w (client %q)", ErrPlanUsageNotGranted, credentials.ClientID)
	default:
		return nil
	}
}
