package githubcopilot_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/models/githubcopilot"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (failingBody) Close() error             { return nil }

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func oauthClient(handler roundTripFunc) *http.Client {
	return &http.Client{Transport: handler}
}

func oauthResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestOAuthFlowApproval(t *testing.T) {
	var mu sync.Mutex
	requests := 0
	client := oauthClient(func(request *http.Request) (*http.Response, error) {
		mu.Lock()
		defer mu.Unlock()
		requests++
		if request.URL.Host != "github.com" || request.Header.Get("Accept") != "application/json" ||
			request.Header.Get("Content-Type") != "application/x-www-form-urlencoded" {
			t.Fatalf("unexpected request: %s %#v", request.URL, request.Header)
		}
		if request.URL.Path == "/login/device/code" {
			if err := request.ParseForm(); err != nil {
				t.Fatal(err)
			}
			if request.Form.Get("client_id") != "client" || request.Form.Get("scope") != "read:user" {
				t.Fatalf("unexpected form: %#v", request.Form)
			}
			return oauthResponse(http.StatusOK, `{"device_code":"device-secret","user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/device","expires_in":10,"interval":1}`), nil
		}
		if err := request.ParseForm(); err != nil {
			t.Fatal(err)
		}
		if request.Form.Get("device_code") != "device-secret" ||
			request.Form.Get("grant_type") != "urn:ietf:params:oauth:grant-type:device_code" {
			t.Fatalf("unexpected token form: %#v", request.Form)
		}
		return oauthResponse(http.StatusOK, `{"access_token":"access-secret","token_type":"bearer","scope":"read:user","refresh_token":"refresh-secret","expires_in":3600,"refresh_token_expires_in":7200}`), nil
	})
	flow, err := githubcopilot.NewOAuthFlow(
		"client", githubcopilot.WithOAuthScope("read:user"), githubcopilot.WithOAuthHTTPClient(client),
	)
	if err != nil {
		t.Fatal(err)
	}
	authorization, err := flow.Start(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if authorization.UserCode != "ABCD-EFGH" || authorization.Interval != 1 ||
		strings.Contains(fmt.Sprint(authorization), "device-secret") {
		t.Fatalf("unexpected authorization: %v", authorization)
	}
	credentials, err := flow.WaitForAuthorization(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if credentials.AccessToken != "access-secret" || credentials.RefreshToken != "refresh-secret" ||
		credentials.ExpiresIn != 3600 || credentials.RefreshTokenExpiresIn != 7200 ||
		strings.Contains(fmt.Sprint(credentials), "secret") {
		t.Fatalf("unexpected credentials: %v", credentials)
	}
	if requests != 2 {
		t.Fatalf("got %d OAuth requests", requests)
	}
	if _, err := flow.WaitForAuthorization(t.Context()); err == nil {
		t.Fatal("consumed challenge remained available")
	}
}

func TestOAuthFlowPollingAndCancellation(t *testing.T) {
	for _, test := range []struct {
		name      string
		pollReply string
	}{
		{name: "pending", pollReply: `{"error":"authorization_pending"}`},
		{name: "slow down", pollReply: `{"error":"slow_down"}`},
		{name: "server interval", pollReply: `{"error":"slow_down","interval":20}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			polls := 0
			client := oauthClient(func(request *http.Request) (*http.Response, error) {
				if request.URL.Path == "/login/device/code" {
					return oauthResponse(http.StatusOK, `{"device_code":"device","user_code":"CODE","verification_uri":"https://github.com/login/device","expires_in":30,"interval":1}`), nil
				}
				polls++
				cancel()
				return oauthResponse(http.StatusOK, test.pollReply), nil
			})
			flow, err := githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(client))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := flow.Start(ctx); err != nil {
				t.Fatal(err)
			}
			if _, err := flow.WaitForAuthorization(ctx); !errors.Is(err, context.Canceled) {
				t.Fatalf("unexpected cancellation: %v", err)
			}
			if polls != 1 {
				t.Fatalf("got %d polls", polls)
			}
		})
	}
}

func TestOAuthFlowFailures(t *testing.T) {
	if _, err := githubcopilot.NewOAuthFlow(" "); err == nil {
		t.Fatal("empty client ID accepted")
	}
	if panicValue := capturePanic(func() { githubcopilot.WithOAuthHTTPClient(nil) }); panicValue == nil {
		t.Fatal("nil client accepted")
	}

	tests := []struct {
		name, device, token, match string
		status                     int
		cancel                     bool
	}{
		{name: "device error", device: `{"error":"device_flow_disabled"}`, match: "device_flow_disabled"},
		{name: "invalid JSON", device: `not-json`, match: "invalid device authorization"},
		{name: "invalid device", device: `{"device_code":"secret"}`, match: "invalid device authorization"},
		{name: "trailing device JSON", device: validDevice() + `{}`, match: "invalid device authorization"},
		{name: "invalid interval", device: `{"device_code":"secret","user_code":"CODE","verification_uri":"https://github.com/login/device","expires_in":10,"interval":0}`, match: "invalid device authorization"},
		{name: "HTTP", status: http.StatusFound, device: "secret", match: "HTTP 302"},
		{name: "token error", device: validDevice(), token: `{"error":"access_denied"}`, match: "access_denied"},
		{name: "invalid token", device: validDevice(), token: `{"access_token":"secret","token_type":"mac","scope":""}`, match: "invalid device token"},
		{name: "cancel", device: validDevice(), cancel: true, match: "context canceled"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			calls := 0
			client := oauthClient(func(*http.Request) (*http.Response, error) {
				calls++
				if calls == 1 {
					status := test.status
					if status == 0 {
						status = http.StatusOK
					}
					return oauthResponse(status, test.device), nil
				}
				return oauthResponse(http.StatusOK, test.token), nil
			})
			flow, err := githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(client))
			if err != nil {
				t.Fatal(err)
			}
			_, err = flow.Start(ctx)
			if err == nil && test.cancel {
				cancel()
				_, err = flow.WaitForAuthorization(ctx)
			} else if err == nil && test.token != "" {
				_, err = flow.WaitForAuthorization(ctx)
			}
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("got %v, want %q", err, test.match)
			}
		})
	}
}

func TestOAuthFlowRequestFailures(t *testing.T) {
	redirects := 0
	flow, _ := githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(oauthClient(
		func(request *http.Request) (*http.Response, error) {
			redirects++
			response := oauthResponse(http.StatusFound, "")
			response.Header.Set("Location", "https://example.com/secret")
			return response, nil
		},
	)))
	if _, err := flow.Start(t.Context()); err == nil || !strings.Contains(err.Error(), "HTTP 302") || redirects != 1 {
		t.Fatalf("unexpected redirect result: calls=%d err=%v", redirects, err)
	}

	flow, _ = githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(oauthClient(
		func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: failingBody{}}, nil
		},
	)))
	if _, err := flow.Start(t.Context()); err == nil || !strings.Contains(err.Error(), "read failed") {
		t.Fatalf("unexpected read error: %v", err)
	}

	pollError := errors.New("poll offline")
	calls := 0
	flow, _ = githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(oauthClient(
		func(*http.Request) (*http.Response, error) {
			calls++
			if calls == 1 {
				return oauthResponse(http.StatusOK, validDevice()), nil
			}
			return nil, pollError
		},
	)))
	if _, err := flow.Start(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := flow.WaitForAuthorization(t.Context()); !errors.Is(err, pollError) {
		t.Fatalf("unexpected polling error: %v", err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	calls = 0
	entered := make(chan struct{})
	flow, _ = githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(oauthClient(
		func(request *http.Request) (*http.Response, error) {
			calls++
			if calls == 1 {
				return oauthResponse(http.StatusOK, validDevice()), nil
			}
			close(entered)
			<-request.Context().Done()
			return nil, context.Cause(request.Context())
		},
	)))
	if _, err := flow.Start(ctx); err != nil {
		t.Fatal(err)
	}
	go func() {
		<-entered
		cancel()
	}()
	if _, err := flow.WaitForAuthorization(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected in-flight cancellation: %v", err)
	}
}

func TestOAuthFlowExpiryTransportAndDefaultClient(t *testing.T) {
	client := oauthClient(func(*http.Request) (*http.Response, error) {
		return oauthResponse(http.StatusOK, `{"device_code":"device","user_code":"CODE","verification_uri":"https://github.com/login/device","expires_in":1}`), nil
	})
	flow, err := githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := flow.Start(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := flow.WaitForAuthorization(t.Context()); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("unexpected expiry: %v", err)
	}

	transportError := errors.New("offline")
	flow, _ = githubcopilot.NewOAuthFlow("client", githubcopilot.WithOAuthHTTPClient(oauthClient(
		func(*http.Request) (*http.Response, error) { return nil, transportError },
	)))
	if _, err := flow.Start(t.Context()); !errors.Is(err, transportError) {
		t.Fatalf("unexpected transport error: %v", err)
	}

	original := http.DefaultTransport
	http.DefaultTransport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return oauthResponse(http.StatusOK, validDevice()), nil
	})
	defer func() { http.DefaultTransport = original }()
	flow, _ = githubcopilot.NewOAuthFlow("client")
	if authorization, err := flow.Start(t.Context()); err != nil || authorization.Interval != 1 {
		t.Fatalf("default client failed: %v, %v", authorization, err)
	}
}

func validDevice() string {
	return `{"device_code":"device","user_code":"CODE","verification_uri":"https://github.com/login/device","expires_in":30,"interval":1}`
}

func capturePanic(function func()) (recovered any) {
	defer func() { recovered = recover() }()
	function()
	return nil
}
