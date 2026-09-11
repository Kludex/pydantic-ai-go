package openaicodex

import (
	"bytes"
	"io"
	"net/http"
)

type codexTransport struct {
	base    http.RoundTripper
	manager *credentialManager
}

func (transport *codexTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.URL.Scheme != "https" || request.URL.Hostname() != "chatgpt.com" {
		return transport.base.RoundTrip(request)
	}
	body := readRequestBody(request)
	credentials, revision, failures, err := transport.manager.prepare(request.Context())
	if err != nil {
		return nil, err
	}
	response, err := transport.base.RoundTrip(authenticatedRequest(request, body, credentials))
	if err != nil || response.StatusCode != http.StatusUnauthorized {
		return response, err
	}
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()
	if err := transport.manager.refresh(request.Context(), revision, failures); err != nil {
		return nil, err
	}
	credentials, _ = transport.manager.current()
	return transport.base.RoundTrip(authenticatedRequest(request, body, credentials))
}

func readRequestBody(request *http.Request) []byte {
	if request.Body == nil {
		return nil
	}
	body, _ := io.ReadAll(request.Body)
	_ = request.Body.Close()
	return body
}

func authenticatedRequest(request *http.Request, body []byte, credentials Credentials) *http.Request {
	cloned := request.Clone(request.Context())
	cloned.Header = request.Header.Clone()
	cloned.Header.Set("Authorization", "Bearer "+credentials.AccessToken)
	cloned.Header.Set("chatgpt-account-id", credentials.AccountID)
	cloned.Header.Set("originator", "pydantic-ai")
	if body != nil {
		cloned.Body = io.NopCloser(bytes.NewReader(body))
		cloned.ContentLength = int64(len(body))
	}
	return cloned
}
