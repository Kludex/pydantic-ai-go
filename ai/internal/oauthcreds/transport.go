package oauthcreds

import (
	"bytes"
	"io"
	"net/http"
)

// Transport injects a manager's credentials into HTTPS requests to Host and
// replays a request once after a 401 with refreshed credentials. Requests to
// any other destination, or over plaintext, pass through untouched, so a
// caller-owned client reused elsewhere never leaks the token.
type Transport[C comparable] struct {
	Base    http.RoundTripper
	Host    string
	Manager *Manager[C]
	Headers func(credentials C) map[string]string
}

// RoundTrip implements http.RoundTripper.
func (transport *Transport[C]) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.URL.Scheme != "https" || request.URL.Hostname() != transport.Host {
		return transport.Base.RoundTrip(request)
	}
	var body []byte
	if request.Body != nil {
		// Buffered so the 401 replay can resend it.
		body, _ = io.ReadAll(request.Body)
		_ = request.Body.Close()
	}
	credentials, revision, failures, err := transport.Manager.Prepare(request.Context())
	if err != nil {
		return nil, err
	}
	response, err := transport.Base.RoundTrip(transport.authenticated(request, body, credentials))
	if err != nil || response.StatusCode != http.StatusUnauthorized {
		return response, err
	}
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()
	if err := transport.Manager.Refresh(request.Context(), revision, failures); err != nil {
		return nil, err
	}
	credentials, _ = transport.Manager.Current()
	if err := transport.Manager.hooks.Validate(credentials); err != nil {
		return nil, err
	}
	return transport.Base.RoundTrip(transport.authenticated(request, body, credentials))
}

func (transport *Transport[C]) authenticated(request *http.Request, body []byte, credentials C) *http.Request {
	cloned := request.Clone(request.Context())
	cloned.Header = request.Header.Clone()
	for name, value := range transport.Headers(credentials) {
		cloned.Header.Set(name, value)
	}
	if body != nil {
		cloned.Body = io.NopCloser(bytes.NewReader(body))
		cloned.ContentLength = int64(len(body))
	}
	return cloned
}
