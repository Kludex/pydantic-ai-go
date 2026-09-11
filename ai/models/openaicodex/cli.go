package openaicodex

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// ParseCodexCLIAuth parses the credential subset of a Codex CLI auth.json file.
func ParseCodexCLIAuth(data []byte) (Credentials, error) {
	var document struct {
		Tokens struct {
			AccessToken  string `json:"access_token"`
			RefreshToken string `json:"refresh_token"`
			AccountID    string `json:"account_id"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return Credentials{}, fmt.Errorf("openai-codex: malformed Codex CLI credentials: %w", err)
	}
	credentials := Credentials{
		AccessToken: document.Tokens.AccessToken, RefreshToken: document.Tokens.RefreshToken,
		AccountID: document.Tokens.AccountID,
	}
	if err := credentials.validate(); err != nil {
		return Credentials{}, fmt.Errorf(
			"openai-codex: malformed Codex CLI credentials; run `codex login` to regenerate them: %w", err,
		)
	}
	return credentials, nil
}

// LoadCodexCLICredentials reads $CODEX_HOME/auth.json or ~/.codex/auth.json.
// It never writes or changes the Codex CLI credential store.
func LoadCodexCLICredentials() (Credentials, error) {
	home := os.Getenv("CODEX_HOME")
	if home == "" {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return Credentials{}, fmt.Errorf("openai-codex: locate Codex CLI credentials: %w", err)
		}
		home = filepath.Join(userHome, ".codex")
	}
	path := filepath.Join(home, "auth.json")
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Credentials{}, fmt.Errorf(
			"openai-codex: no Codex CLI credentials found at %q; run `codex login` or provide credentials", path,
		)
	}
	if err != nil {
		return Credentials{}, fmt.Errorf("openai-codex: read Codex CLI credentials at %q: %w", path, err)
	}
	credentials, err := ParseCodexCLIAuth(data)
	if err != nil {
		return Credentials{}, fmt.Errorf("openai-codex: read Codex CLI credentials at %q: %w", path, err)
	}
	return credentials, nil
}
