package openaicodex

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ExchangeCodeFromCallback serves one local callback and exchanges its code.
// It accepts only an HTTP loopback redirect and stops when ctx is canceled.
func (flow *OAuthFlow) ExchangeCodeFromCallback(ctx context.Context) (Credentials, error) {
	redirect, _ := url.Parse(flow.redirectURI)
	if redirect.Scheme != "http" || !isLoopbackHost(redirect.Hostname()) {
		return Credentials{}, errors.New("openai-codex: OAuth callback server requires an HTTP loopback redirect URI")
	}
	if redirect.Port() == "" {
		return Credentials{}, errors.New("openai-codex: OAuth callback redirect URI must include a port")
	}
	listener, err := net.Listen("tcp", redirect.Host)
	if err != nil {
		return Credentials{}, fmt.Errorf("openai-codex: listen for OAuth callback: %w", err)
	}
	defer func() { _ = listener.Close() }()

	type callback struct {
		code string
		err  error
	}
	result := make(chan callback, 1)
	handler := http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != redirect.Path || request.URL.Query().Get("state") != flow.state {
			response.WriteHeader(http.StatusOK)
			_, _ = response.Write([]byte("You can close this tab."))
			return
		}
		value := callback{code: request.URL.Query().Get("code")}
		if value.code == "" {
			reason := request.URL.Query().Get("error")
			if reason == "" {
				reason = "unknown"
			}
			value.err = fmt.Errorf("openai-codex: authorization failed: %s", reason)
		}
		result <- value
		response.WriteHeader(http.StatusOK)
		_, _ = response.Write([]byte("You can close this tab."))
	})
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second}
	serveErr := make(chan error, 1)
	go func() { serveErr <- server.Serve(listener) }()

	var received callback
	select {
	case <-ctx.Done():
		_ = server.Close()
		<-serveErr
		return Credentials{}, context.Cause(ctx)
	case received = <-result:
		shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		_ = server.Shutdown(shutdownCtx)
		cancel()
		<-serveErr
	}
	if received.err != nil {
		return Credentials{}, received.err
	}
	return flow.ExchangeCode(ctx, received.code)
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	address := net.ParseIP(host)
	return address != nil && address.IsLoopback()
}
