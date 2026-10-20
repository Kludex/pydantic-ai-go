package openaichatgpt

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"time"
)

// ExchangeCodeFromCallback serves one callback on 127.0.0.1 and exchanges its
// code. It stops when ctx is canceled.
func (flow *OAuthFlow) ExchangeCodeFromCallback(ctx context.Context) (Credentials, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(flow.redirectPort))
	if err != nil {
		return Credentials{}, fmt.Errorf("openai-chatgpt: listen for OAuth callback: %w", err)
	}
	defer func() { _ = listener.Close() }()

	type callback struct {
		code, clientID string
		err            error
	}
	result := make(chan callback, 1)
	handler := http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		query := request.URL.Query()
		if request.URL.Path == "/auth/callback" && query.Get("state") == flow.state {
			value := callback{code: query.Get("code"), clientID: query.Get("client_id")}
			if value.code == "" {
				reason := query.Get("error")
				if reason == "" {
					reason = "unknown"
				}
				value.err = fmt.Errorf("openai-chatgpt: authorization failed: %s", reason)
			}
			select {
			case result <- value:
			default:
			}
		}
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
	return flow.ExchangeCode(ctx, received.code, received.clientID)
}
