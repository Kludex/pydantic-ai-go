package openaicodex_test

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaicodex"
)

func TestCodexCLICredentialsAreReadOnly(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "auth.json")
	original := `{"OPENAI_API_KEY":null,"tokens":{` +
		`"access_token":"secret-access","refresh_token":"secret-refresh","account_id":"account"},` +
		`"future":{"value":1}}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CODEX_HOME", directory)
	credentials, err := openaicodex.LoadCodexCLICredentials()
	if err != nil {
		t.Fatal(err)
	}
	if credentials.AccessToken != "secret-access" || credentials.RefreshToken != "secret-refresh" ||
		credentials.AccountID != "account" {
		t.Fatalf("unexpected credentials: %v", credentials)
	}
	unchanged, err := os.ReadFile(path)
	if err != nil || string(unchanged) != original {
		t.Fatalf("Codex CLI auth was changed: %q %v", unchanged, err)
	}
	for _, formatted := range []string{fmt.Sprint(credentials), fmt.Sprintf("%#v", credentials)} {
		if strings.Contains(formatted, "secret-access") || strings.Contains(formatted, "secret-refresh") {
			t.Fatalf("formatted credentials leaked tokens: %s", formatted)
		}
	}
}

func TestCodexCLIUsesDefaultHome(t *testing.T) {
	home := t.TempDir()
	directory := filepath.Join(home, ".codex")
	if err := os.Mkdir(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "auth.json"), []byte(
		`{"tokens":{"access_token":"access","refresh_token":"refresh","account_id":"account"}}`,
	), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CODEX_HOME", "")
	t.Setenv("HOME", home)
	if _, err := openaicodex.LoadCodexCLICredentials(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", "")
	if _, err := openaicodex.LoadCodexCLICredentials(); err == nil || !strings.Contains(err.Error(), "locate") {
		t.Fatalf("unexpected missing home error: %v", err)
	}
}

func TestCodexCLICredentialErrorsDoNotLeak(t *testing.T) {
	for name, data := range map[string]string{
		"JSON":    `{`,
		"access":  `{"tokens":{"refresh_token":"SENTINEL_REFRESH","account_id":"account"}}`,
		"refresh": `{"tokens":{"access_token":"SENTINEL_ACCESS","account_id":"account"}}`,
		"account": `{"tokens":{"access_token":"SENTINEL_ACCESS","refresh_token":"SENTINEL_REFRESH"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := openaicodex.ParseCodexCLIAuth([]byte(data))
			if err == nil {
				t.Fatal("malformed credentials succeeded")
			}
			if strings.Contains(err.Error(), "SENTINEL_ACCESS") || strings.Contains(err.Error(), "SENTINEL_REFRESH") {
				t.Fatalf("credential error leaked a token: %v", err)
			}
		})
	}
	directory := t.TempDir()
	t.Setenv("CODEX_HOME", directory)
	if _, err := openaicodex.LoadCodexCLICredentials(); err == nil || !strings.Contains(err.Error(), "codex login") {
		t.Fatalf("unexpected missing credential error: %v", err)
	}
	if err := os.Mkdir(filepath.Join(directory, "auth.json"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := openaicodex.LoadCodexCLICredentials(); err == nil || !strings.Contains(err.Error(), "read") {
		t.Fatalf("unexpected unreadable credential error: %v", err)
	}
	if err := os.Remove(filepath.Join(directory, "auth.json")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "auth.json"), []byte(`{`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := openaicodex.LoadCodexCLICredentials(); err == nil || !strings.Contains(err.Error(), "malformed") {
		t.Fatalf("unexpected malformed credential error: %v", err)
	}
}

func TestNewModelLoadsCodexCLIWithoutAPIKeyFallback(t *testing.T) {
	directory := t.TempDir()
	t.Setenv("CODEX_HOME", directory)
	t.Setenv("OPENAI_API_KEY", "must-not-be-used")
	if _, err := openaicodex.NewModel("model"); err == nil || !strings.Contains(err.Error(), "codex login") {
		t.Fatalf("unexpected missing CLI error: %v", err)
	}
	expired := jwt(map[string]any{"exp": 0})
	data := `{"tokens":{"access_token":` + fmt.Sprintf("%q", expired) +
		`,"refresh_token":"refresh","account_id":"account"}}`
	if err := os.WriteFile(filepath.Join(directory, "auth.json"), []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	model, err := openaicodex.NewModel("model", openaicodex.WithHTTPClient(&http.Client{
		Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if request.URL.Hostname() == "auth.openai.com" {
				return response(http.StatusOK,
					`{"access_token":"new","refresh_token":"new-refresh","account_id":"account"}`,
				), nil
			}
			return codexSSE("done"), nil
		}),
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	unchanged, err := os.ReadFile(filepath.Join(directory, "auth.json"))
	if err != nil || string(unchanged) != data {
		t.Fatalf("CLI credentials changed: %q %v", unchanged, err)
	}
}

func TestRefreshCredentials(t *testing.T) {
	for name, body := range map[string]string{
		"direct": `{"access_token":"new","refresh_token":"rotated","account_id":"direct"}`,
		"nested JWT": `{"access_token":"new","refresh_token":"rotated","id_token":` + fmt.Sprintf("%q", jwt(
			map[string]any{"https://api.openai.com/auth": map[string]any{"chatgpt_account_id": "nested"}},
		)) + `}`,
		"top JWT": `{"access_token":"new","refresh_token":"rotated","id_token":` +
			fmt.Sprintf("%q", jwt(map[string]any{"chatgpt_account_id": "top"})) + `}`,
		"legacy JWT": `{"access_token":"new","refresh_token":"rotated","id_token":` +
			fmt.Sprintf("%q", jwt(map[string]any{"account_id": "legacy"})) + `}`,
		"fallback": `{"access_token":"new","refresh_token":"rotated"}`,
	} {
		t.Run(name, func(t *testing.T) {
			client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.URL.String() != "https://auth.openai.com/oauth/token" || request.Method != http.MethodPost ||
					request.Header.Get("Accept") != "application/json" ||
					request.Header.Get("Content-Type") != "application/x-www-form-urlencoded" {
					t.Fatalf("unexpected token request: %s %s %v", request.Method, request.URL, request.Header)
				}
				data, _ := io.ReadAll(request.Body)
				values, parseErr := url.ParseQuery(string(data))
				if parseErr != nil || values.Get("grant_type") != "refresh_token" ||
					values.Get("refresh_token") != "old-refresh" || values.Get("client_id") == "" {
					t.Fatalf("unexpected refresh form: %s %v", data, parseErr)
				}
				return response(http.StatusOK, body), nil
			})}
			rotated, err := openaicodex.RefreshCredentials(t.Context(), client, openaicodex.Credentials{
				AccessToken: "old", RefreshToken: "old-refresh", AccountID: "fallback",
			})
			if err != nil {
				t.Fatal(err)
			}
			want := map[string]string{
				"direct": "direct", "nested JWT": "nested", "top JWT": "top", "legacy JWT": "legacy", "fallback": "fallback",
			}[name]
			if rotated.AccessToken != "new" || rotated.RefreshToken != "rotated" || rotated.AccountID != want {
				t.Fatalf("unexpected rotated credentials: %v", rotated)
			}
		})
	}
}

func TestRefreshCredentialErrors(t *testing.T) {
	if _, err := openaicodex.RefreshCredentials(t.Context(), nil, credentials("access")); err == nil {
		t.Fatal("nil client succeeded")
	}
	for _, invalid := range []openaicodex.Credentials{
		{}, {AccessToken: "access"}, {AccessToken: "access", RefreshToken: "refresh"},
	} {
		if _, err := openaicodex.RefreshCredentials(t.Context(), http.DefaultClient, invalid); err == nil {
			t.Fatal("invalid credentials succeeded")
		}
	}
	transportFailure := errors.New("transport failed")
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, transportFailure
	})}
	_, err := openaicodex.RefreshCredentials(t.Context(), client, credentials("access"))
	var refresh *openaicodex.CredentialsRefreshError
	if !errors.As(err, &refresh) || !errors.Is(refresh, transportFailure) || !refresh.IsModelAPIError() {
		t.Fatalf("unexpected refresh transport error: %v", err)
	}

	cases := []struct {
		status int
		body   string
		match  string
	}{
		{http.StatusOK, `not-json`, "invalid JSON"},
		{http.StatusBadRequest, `{"error":"invalid_grant","error_description":"expired"}`, "rerun"},
		{http.StatusForbidden, `{"error":"access_denied"}`, "access_denied"},
		{http.StatusInternalServerError, `{}`, "Internal Server Error"},
		{http.StatusOK, `null`, "unexpected response"},
		{http.StatusOK, `[]`, "unexpected response"},
		{http.StatusOK, `{"access_token":1}`, "unexpected response"},
		{http.StatusOK, `{}`, "access_token"},
		{http.StatusOK, `{"access_token":"SENTINEL_ACCESS"}`, "refresh_token"},
		{http.StatusOK, `{"access_token":"SENTINEL_ACCESS","refresh_token":"SENTINEL_REFRESH"}`, "account_id"},
	}
	for _, test := range cases {
		t.Run(test.match, func(t *testing.T) {
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return response(test.status, test.body), nil
			})}
			fallback := credentials("access")
			if test.match == "account_id" {
				fallback.AccountID = ""
			}
			_, err := openaicodex.RefreshCredentials(t.Context(), client, fallback)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected refresh error: %v", err)
			}
			if strings.Contains(err.Error(), "SENTINEL_ACCESS") || strings.Contains(err.Error(), "SENTINEL_REFRESH") {
				t.Fatalf("refresh error leaked credentials: %v", err)
			}
		})
	}
}
