package openaicodex

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// RefreshCredentials exchanges a single-use refresh token with OpenAI. The
// caller owns client and remains responsible for its transport and lifetime.
func RefreshCredentials(ctx context.Context, client *http.Client, credentials Credentials) (Credentials, error) {
	if err := credentials.validate(); err != nil {
		return Credentials{}, &CredentialsRefreshError{err: err}
	}
	response, err := postToken(ctx, client, map[string]string{
		"grant_type": "refresh_token", "refresh_token": credentials.RefreshToken, "client_id": publicClientID,
	})
	if err != nil {
		return Credentials{}, err
	}
	return credentialsFromTokenResponse(response, credentials.AccountID)
}

type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
	AccountID    string `json:"account_id"`
}

type tokenErrorResponse struct {
	Error       string `json:"error"`
	Description string `json:"error_description"`
}

func postToken(ctx context.Context, client *http.Client, form map[string]string) (tokenResponse, error) {
	if client == nil {
		return tokenResponse{}, &CredentialsRefreshError{err: errors.New("HTTP client must not be nil")}
	}
	values := make(url.Values, len(form))
	for name, value := range form {
		values.Set(name, value)
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(values.Encode()))
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := client.Do(req)
	if err != nil {
		return tokenResponse{}, &CredentialsRefreshError{err: err}
	}
	defer func() { _ = resp.Body.Close() }()
	var raw json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return tokenResponse{}, &CredentialsRefreshError{err: errors.New("token endpoint returned invalid JSON")}
	}
	if resp.StatusCode != http.StatusOK {
		var body tokenErrorResponse
		_ = json.Unmarshal(raw, &body)
		detail := body.Description
		if detail == "" {
			detail = body.Error
		}
		if detail == "" {
			detail = http.StatusText(resp.StatusCode)
		}
		if body.Error == "invalid_grant" {
			detail += "; the grant was rejected, rerun the authorization flow"
		}
		return tokenResponse{}, &CredentialsRefreshError{err: fmt.Errorf(
			"token endpoint returned status %d: %s", resp.StatusCode, detail,
		)}
	}
	if data := bytes.TrimSpace(raw); len(data) == 0 || data[0] != '{' {
		return tokenResponse{}, &CredentialsRefreshError{err: errors.New("token endpoint returned an unexpected response")}
	}
	var response tokenResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return tokenResponse{}, &CredentialsRefreshError{err: errors.New("token endpoint returned an unexpected response")}
	}
	if response.AccessToken == "" {
		return tokenResponse{}, &CredentialsRefreshError{err: errors.New("token endpoint response omitted access_token")}
	}
	if response.RefreshToken == "" {
		return tokenResponse{}, &CredentialsRefreshError{err: errors.New("token endpoint response omitted refresh_token")}
	}
	return response, nil
}

func credentialsFromTokenResponse(response tokenResponse, fallbackAccountID string) (Credentials, error) {
	accountID := response.AccountID
	if accountID == "" {
		accountID = accountIDFromJWT(response.IDToken)
	}
	if accountID == "" {
		accountID = fallbackAccountID
	}
	credentials := Credentials{
		AccessToken: response.AccessToken, RefreshToken: response.RefreshToken, AccountID: accountID,
	}
	if err := credentials.validate(); err != nil {
		return Credentials{}, &CredentialsRefreshError{err: err}
	}
	return credentials, nil
}
