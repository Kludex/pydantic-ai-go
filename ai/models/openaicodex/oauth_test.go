package openaicodex_test

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/models/openaicodex"
)

func TestOAuthAuthorizationAndExchange(t *testing.T) {
	var form url.Values
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		data := make([]byte, request.ContentLength)
		_, _ = request.Body.Read(data)
		form, _ = url.ParseQuery(string(data))
		return response(http.StatusOK, `{"access_token":"access","refresh_token":"refresh","account_id":"account"}`), nil
	})}
	flow, err := openaicodex.NewOAuthFlow(client, openaicodex.WithOAuthState("state"))
	if err != nil {
		t.Fatal(err)
	}
	if flow.RedirectURI() != "http://localhost:1455/auth/callback" || flow.State() != "state" ||
		len(flow.CodeVerifier()) < 43 {
		t.Fatalf("unexpected flow: redirect=%q state=%q verifier=%q", flow.RedirectURI(), flow.State(), flow.CodeVerifier())
	}
	digest := sha256.Sum256([]byte(flow.CodeVerifier()))
	if flow.CodeChallenge() != base64.RawURLEncoding.EncodeToString(digest[:]) {
		t.Fatal("unexpected PKCE challenge")
	}
	authorization, err := flow.AuthorizationURL("", url.Values{
		"prompt": {"login"}, "codex_cli_simplified_flow": {"false"},
	})
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := url.Parse(authorization)
	if err != nil {
		t.Fatal(err)
	}
	query := parsed.Query()
	if parsed.Hostname() != "auth.openai.com" || query.Get("response_type") != "code" ||
		query.Get("state") != "state" || query.Get("scope") != "openid profile email offline_access" ||
		query.Get("prompt") != "login" || query.Get("codex_cli_simplified_flow") != "false" ||
		query.Get("id_token_add_organizations") != "true" || query.Get("code_challenge_method") != "S256" {
		t.Fatalf("unexpected authorization URL: %s", authorization)
	}
	credentials, err := flow.ExchangeCode(t.Context(), "code")
	if err != nil {
		t.Fatal(err)
	}
	if credentials.AccountID != "account" || form.Get("grant_type") != "authorization_code" ||
		form.Get("code") != "code" || form.Get("code_verifier") != flow.CodeVerifier() ||
		form.Get("redirect_uri") != flow.RedirectURI() || form.Get("client_id") == "" {
		t.Fatalf("unexpected exchange: credentials=%v form=%v", credentials, form)
	}
	customURL, err := flow.AuthorizationURL("custom scope", nil)
	if err != nil || !strings.Contains(customURL, "scope=custom+scope") {
		t.Fatalf("unexpected custom scope: %s %v", customURL, err)
	}
}

func TestOAuthValidation(t *testing.T) {
	if _, err := openaicodex.NewOAuthFlow(nil); err == nil {
		t.Fatal("nil OAuth client succeeded")
	}
	if _, err := openaicodex.NewOAuthFlow(http.DefaultClient, openaicodex.WithOAuthState("")); err == nil {
		t.Fatal("empty state succeeded")
	}
	if _, err := openaicodex.NewOAuthFlow(http.DefaultClient, openaicodex.WithRedirectURI("%")); err == nil {
		t.Fatal("invalid redirect succeeded")
	}
	flow, err := openaicodex.NewOAuthFlow(http.DefaultClient)
	if err != nil {
		t.Fatal(err)
	}
	for _, parameter := range []string{"client_id", "redirect_uri", "state", "code_challenge"} {
		if _, err := flow.AuthorizationURL("", url.Values{parameter: {"override"}}); err == nil {
			t.Fatalf("bound parameter %q was accepted", parameter)
		}
	}
	if _, err := flow.ExchangeCode(t.Context(), ""); err == nil {
		t.Fatal("empty authorization code succeeded")
	}
	nonLoopback, err := openaicodex.NewOAuthFlow(http.DefaultClient,
		openaicodex.WithRedirectURI("https://example.com/callback"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nonLoopback.ExchangeCodeFromCallback(t.Context()); err == nil {
		t.Fatal("non-loopback callback succeeded")
	}
	missingPort, err := openaicodex.NewOAuthFlow(http.DefaultClient,
		openaicodex.WithRedirectURI("http://localhost/callback"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := missingPort.ExchangeCodeFromCallback(t.Context()); err == nil || !strings.Contains(err.Error(), "port") {
		t.Fatalf("unexpected missing port error: %v", err)
	}
}

func TestOAuthCallbackExchange(t *testing.T) {
	port := freePort(t)
	redirect := fmt.Sprintf("http://127.0.0.1:%d/auth/callback", port)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Hostname() != "auth.openai.com" {
			t.Fatalf("unexpected exchange host: %s", request.URL)
		}
		return response(http.StatusOK, `{"access_token":"access","refresh_token":"refresh","account_id":"account"}`), nil
	})}
	flow, err := openaicodex.NewOAuthFlow(client,
		openaicodex.WithRedirectURI(redirect), openaicodex.WithOAuthState("state"))
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct {
		credentials openaicodex.Credentials
		err         error
	}, 1)
	go func() {
		credentials, callbackErr := flow.ExchangeCodeFromCallback(t.Context())
		done <- struct {
			credentials openaicodex.Credentials
			err         error
		}{credentials, callbackErr}
	}()
	getCallback(t, redirect+"?state=wrong&code=stray")
	select {
	case result := <-done:
		t.Fatalf("callback server stopped after mismatched state: %v", result.err)
	default:
	}
	body := getCallback(t, redirect+"?state=state&code=accepted")
	if !strings.Contains(body, "close this tab") {
		t.Fatalf("unexpected callback body: %q", body)
	}
	result := <-done
	if result.err != nil || result.credentials.AccountID != "account" {
		t.Fatalf("unexpected callback exchange: %v %v", result.credentials, result.err)
	}
}

func TestOAuthCallbackErrorsAndCancellation(t *testing.T) {
	t.Run("denied", func(t *testing.T) {
		redirect := fmt.Sprintf("http://localhost:%d/callback", freePort(t))
		flow, err := openaicodex.NewOAuthFlow(http.DefaultClient,
			openaicodex.WithRedirectURI(redirect), openaicodex.WithOAuthState("state"))
		if err != nil {
			t.Fatal(err)
		}
		done := make(chan error, 1)
		go func() {
			_, callbackErr := flow.ExchangeCodeFromCallback(t.Context())
			done <- callbackErr
		}()
		getCallback(t, redirect+"?state=state&error=access_denied")
		if err := <-done; err == nil || !strings.Contains(err.Error(), "access_denied") {
			t.Fatalf("unexpected denial error: %v", err)
		}
	})
	t.Run("unknown denial", func(t *testing.T) {
		redirect := fmt.Sprintf("http://[::1]:%d/callback", freeIPv6Port(t))
		flow, err := openaicodex.NewOAuthFlow(http.DefaultClient,
			openaicodex.WithRedirectURI(redirect), openaicodex.WithOAuthState("state"))
		if err != nil {
			t.Fatal(err)
		}
		done := make(chan error, 1)
		go func() {
			_, callbackErr := flow.ExchangeCodeFromCallback(t.Context())
			done <- callbackErr
		}()
		getCallback(t, redirect+"?state=state")
		if err := <-done; err == nil || !strings.Contains(err.Error(), "unknown") {
			t.Fatalf("unexpected unknown denial: %v", err)
		}
	})
	t.Run("canceled", func(t *testing.T) {
		redirect := fmt.Sprintf("http://127.0.0.1:%d/callback", freePort(t))
		flow, err := openaicodex.NewOAuthFlow(http.DefaultClient, openaicodex.WithRedirectURI(redirect))
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := flow.ExchangeCodeFromCallback(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("unexpected callback cancellation: %v", err)
		}
	})
	t.Run("listener busy", func(t *testing.T) {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = listener.Close() }()
		flow, err := openaicodex.NewOAuthFlow(http.DefaultClient,
			openaicodex.WithRedirectURI("http://"+listener.Addr().String()+"/callback"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := flow.ExchangeCodeFromCallback(t.Context()); err == nil || !strings.Contains(err.Error(), "listen") {
			t.Fatalf("unexpected busy listener error: %v", err)
		}
	})
}

func TestOAuthExchangeFailure(t *testing.T) {
	failure := errors.New("exchange unavailable")
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, failure
	})}
	flow, err := openaicodex.NewOAuthFlow(client)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := flow.ExchangeCode(t.Context(), "code"); !errors.Is(err, failure) {
		t.Fatalf("unexpected exchange failure: %v", err)
	}
	client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return response(http.StatusOK, `{"access_token":"access","refresh_token":"refresh","id_token":`+
			fmt.Sprintf("%q", jwt(map[string]any{}))+`}`), nil
	})}
	missingAccount, err := openaicodex.NewOAuthFlow(client)
	if err != nil {
		t.Fatal(err)
	}
	_, err = missingAccount.ExchangeCode(t.Context(), "code")
	if err == nil || !strings.Contains(err.Error(), "account_id") {
		t.Fatalf("unexpected missing account error: %v", err)
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = listener.Close() }()
	return listener.Addr().(*net.TCPAddr).Port
}

func freeIPv6Port(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "[::1]:0")
	if err != nil {
		t.Skip("IPv6 loopback is unavailable")
	}
	defer func() { _ = listener.Close() }()
	return listener.Addr().(*net.TCPAddr).Port
}

func getCallback(t *testing.T, address string) string {
	t.Helper()
	client := &http.Client{Timeout: time.Second}
	var lastErr error
	for range 50 {
		response, err := client.Get(address)
		if err == nil {
			defer func() { _ = response.Body.Close() }()
			data := make([]byte, response.ContentLength)
			if response.ContentLength < 0 {
				data = make([]byte, 256)
			}
			count, _ := response.Body.Read(data)
			return string(data[:count])
		}
		lastErr = err
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("callback server did not start: %v", lastErr)
	return ""
}
