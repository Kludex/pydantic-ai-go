package openaicodex

import (
	"context"
	"errors"
	"fmt"
)

// Credentials contains the OAuth tokens and ChatGPT account identity used by Codex.
type Credentials struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	AccountID    string `json:"account_id"`
}

// String formats credentials without exposing either token.
func (credentials Credentials) String() string {
	return fmt.Sprintf("OpenAICodexCredentials{AccountID:%q}", credentials.AccountID)
}

// GoString formats credentials without exposing either token.
func (credentials Credentials) GoString() string { return credentials.String() }

// CredentialSource is application-owned storage for rotating Codex credentials.
// Load is called on first use and immediately before a refresh. Save is called
// after the complete rotated credential set has replaced the in-memory set.
// Implementations shared across processes must coordinate atomic replacement.
type CredentialSource interface {
	Load(ctx context.Context) (Credentials, error)
	Save(ctx context.Context, credentials Credentials) error
}

// CredentialsRefreshError reports a rejected or malformed OAuth token exchange.
type CredentialsRefreshError struct{ err error }

// Error describes the credential refresh failure.
func (err *CredentialsRefreshError) Error() string {
	return "openai-codex: refresh credentials: " + err.err.Error()
}

// Unwrap returns the token exchange failure.
func (err *CredentialsRefreshError) Unwrap() error { return err.err }

// IsModelAPIError marks credential failures as eligible for model fallback.
func (*CredentialsRefreshError) IsModelAPIError() bool { return true }

// CredentialsPersistenceError reports that rotated credentials are active in
// memory but the caller-owned source failed to save them.
type CredentialsPersistenceError struct{ err error }

// Error describes the persistence failure.
func (err *CredentialsPersistenceError) Error() string {
	return "openai-codex: credentials were refreshed in memory but saving them failed: " + err.err.Error()
}

// Unwrap returns the caller-owned persistence failure.
func (err *CredentialsPersistenceError) Unwrap() error { return err.err }

// IsModelAPIError marks credential failures as eligible for model fallback.
func (*CredentialsPersistenceError) IsModelAPIError() bool { return true }

func (credentials Credentials) validate() error {
	switch {
	case credentials.AccessToken == "":
		return errors.New("access_token must not be empty")
	case credentials.RefreshToken == "":
		return errors.New("refresh_token must not be empty")
	case credentials.AccountID == "":
		return errors.New("account_id must not be empty")
	default:
		return nil
	}
}
