package openaichatgpt_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/models/openaichatgpt"
)

const (
	clientID  = "oaiapp_issued"
	hostID    = "urn:uuid:6f1c3a52-2b0e-4b0f-9a35-6d2e4f1a7b9c"
	planScope = "chatgpt.tokens.use.direct email offline_access openid profile resource.invoke"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func jwt(claims map[string]any) string {
	encode := func(value any) string {
		data, _ := json.Marshal(value)
		return base64.RawURLEncoding.EncodeToString(data)
	}
	return encode(map[string]any{"alg": "none"}) + "." + encode(claims) + ".signature"
}

func response(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

// tokenEndpoint answers token requests with *reply and records their forms.
// The reply is read on each request, so a test can set it after creating a flow.
type tokenEndpoint struct {
	reply func(form url.Values) *http.Response
	forms []url.Values
}

func (endpoint *tokenEndpoint) client(t *testing.T) *http.Client {
	t.Helper()
	return &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://auth.openai.com/api/accounts/oauth/token" {
			t.Errorf("unexpected token URL %s", request.URL)
		}
		_ = request.ParseForm()
		endpoint.forms = append(endpoint.forms, request.PostForm)
		return endpoint.reply(request.PostForm), nil
	})}
}

func signedIn(flow *openaichatgpt.OAuthFlow, scope any, claims map[string]any) func(url.Values) *http.Response {
	return func(form url.Values) *http.Response {
		base := map[string]any{
			"iss": "https://auth.openai.com", "aud": form.Get("client_id"), "sub": "user-1",
			"nonce": flow.Nonce(), "exp": time.Now().Add(time.Hour).Unix(),
		}
		for key, value := range claims {
			base[key] = value
		}
		body := map[string]any{"access_token": "access-1", "refresh_token": "refresh-1", "id_token": jwt(base)}
		if scope != nil {
			body["scope"] = scope
		}
		data, _ := json.Marshal(body)
		return response(http.StatusOK, string(data))
	}
}

func TestAuthorizationURLRegistersANewClient(t *testing.T) {
	flow, err := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App", openaichatgpt.WithHostID(hostID))
	if err != nil {
		t.Fatal(err)
	}
	authorization, err := flow.AuthorizationURL("", nil)
	if err != nil {
		t.Fatal(err)
	}
	parsed, _ := url.Parse(authorization)
	query := parsed.Query()
	want := map[string]string{
		"response_type": "code", "client_id": "dynamic_agent_client",
		"redirect_uri":          "http://127.0.0.1:1455/auth/callback",
		"scope":                 "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
		"resource":              "https://api.openai.com/v1",
		"code_challenge_method": "S256", "agent_name_hint": "My App", "ext_agent_host_id": hostID,
		"state": flow.State(), "nonce": flow.Nonce(), "code_challenge": flow.CodeChallenge(),
	}
	if parsed.Scheme+"://"+parsed.Host+parsed.Path != "https://auth.openai.com/api/accounts/authorize" ||
		len(query) != len(want) {
		t.Fatalf("unexpected authorization URL %s", authorization)
	}
	for name, value := range want {
		if query.Get(name) != value {
			t.Fatalf("%s = %q, want %q", name, query.Get(name), value)
		}
	}
	if flow.ClientID() != "" || len(flow.CodeVerifier()) < 43 {
		t.Fatalf("unexpected flow state %q %q", flow.ClientID(), flow.CodeVerifier())
	}
	custom, err := flow.AuthorizationURL("openid", nil)
	if err != nil || !strings.Contains(custom, "scope=openid&") {
		t.Fatalf("custom scope %s: %v", custom, err)
	}
}

func TestAuthorizationURLReusesARegistration(t *testing.T) {
	hint := jwt(map[string]any{"sub": "user-1"})
	flow, err := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App",
		openaichatgpt.WithClientID(clientID), openaichatgpt.WithIDTokenHint(hint))
	if err != nil {
		t.Fatal(err)
	}
	authorization, _ := flow.AuthorizationURL("", url.Values{"prompt": {"consent"}})
	parsed, _ := url.Parse(authorization)
	query := parsed.Query()
	if query.Get("client_id") != clientID || query.Get("id_token_hint") != hint || query.Get("prompt") != "consent" ||
		query.Has("agent_name_hint") || query.Has("ext_agent_host_id") {
		t.Fatalf("unexpected authorization URL %s", authorization)
	}
	for _, name := range []string{"client_id", "redirect_uri", "state", "code_challenge", "nonce", "resource"} {
		if _, err := flow.AuthorizationURL("", url.Values{name: {"other"}}); err == nil ||
			!strings.Contains(err.Error(), "cannot override "+name) {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

func TestOAuthFlowValidation(t *testing.T) {
	for name, options := range map[string][]openaichatgpt.OAuthOption{
		"low port":     {openaichatgpt.WithRedirectPort(0)},
		"high port":    {openaichatgpt.WithRedirectPort(99999)},
		"host ID":      {openaichatgpt.WithHostID("user@example.com")},
		"hint":         {openaichatgpt.WithIDTokenHint("not-a-jwt")},
		"hint subject": {openaichatgpt.WithIDTokenHint(jwt(map[string]any{}))},
		"state":        {openaichatgpt.WithOAuthState("")},
	} {
		if _, err := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App", options...); err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
	if _, err := openaichatgpt.NewOAuthFlow(nil, "My App"); err == nil {
		t.Fatal("nil client accepted")
	}
	flow, err := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App", openaichatgpt.WithRedirectPort(54321))
	if err != nil || flow.RedirectURI() != "http://127.0.0.1:54321/auth/callback" {
		t.Fatalf("port did not vary: %v", err)
	}
}

func TestExchangeCode(t *testing.T) {
	endpoint := &tokenEndpoint{}
	client := endpoint.client(t)
	flow, _ := openaichatgpt.NewOAuthFlow(client, "My App")
	endpoint.reply = signedIn(flow, planScope, nil)
	credentials, err := flow.ExchangeCode(t.Context(), "the-code", clientID)
	if err != nil {
		t.Fatal(err)
	}
	if credentials.ClientID != clientID || credentials.AccessToken != "access-1" || credentials.RefreshToken != "refresh-1" ||
		credentials.Scope != planScope || !credentials.PlanUsageGranted() || flow.ClientID() != clientID {
		t.Fatalf("unexpected credentials %+v", credentials)
	}
	if fmt.Sprint(credentials) != `OpenAIChatGPTCredentials{ClientID:"oaiapp_issued", Scope:"`+planScope+`"}` ||
		fmt.Sprintf("%#v", credentials) != fmt.Sprint(credentials) || strings.Contains(fmt.Sprint(credentials), "access-1") {
		t.Fatalf("credentials leak tokens: %v", credentials)
	}
	form := endpoint.forms[0]
	if form.Get("grant_type") != "authorization_code" || form.Get("code") != "the-code" ||
		form.Get("code_verifier") != flow.CodeVerifier() || form.Get("redirect_uri") != flow.RedirectURI() ||
		form.Get("client_id") != clientID || form.Get("resource") != "https://api.openai.com/v1" {
		t.Fatalf("unexpected form %v", form)
	}

	declined, _ := openaichatgpt.NewOAuthFlow(client, "My App")
	endpoint.reply = signedIn(declined, nil, nil)
	credentials, err = declined.ExchangeCode(t.Context(), "the-code", clientID)
	if err != nil || credentials.PlanUsageGranted() || credentials.ClientID != clientID {
		t.Fatalf("declined plan usage: %+v %v", credentials, err)
	}

	again, _ := openaichatgpt.NewOAuthFlow(client, "My App", openaichatgpt.WithClientID(clientID),
		openaichatgpt.WithIDTokenHint(jwt(map[string]any{"sub": "user-1"})))
	endpoint.reply = signedIn(again, planScope, map[string]any{"aud": []string{clientID, "other"}})
	if credentials, err := again.ExchangeCode(t.Context(), "the-code", ""); err != nil || credentials.ClientID != clientID {
		t.Fatalf("returning sign-in: %+v %v", credentials, err)
	}
}

func TestExchangeCodeRejections(t *testing.T) {
	for _, test := range []struct {
		name, flowClient, callbackClient, message string
	}{
		{"registration without issued ID", "", "", "registration is incomplete"},
		{"registration entrypoint ID", "", "dynamic_agent_client", "registration is incomplete"},
		{"different registration", clientID, "oaiapp_other", `carried client "oaiapp_other"`},
	} {
		flow, _ := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App", openaichatgpt.WithClientID(test.flowClient))
		if _, err := flow.ExchangeCode(t.Context(), "the-code", test.callbackClient); err == nil ||
			!strings.Contains(err.Error(), test.message) {
			t.Fatalf("%s: %v", test.name, err)
		}
	}
	flow, _ := openaichatgpt.NewOAuthFlow(http.DefaultClient, "My App", openaichatgpt.WithClientID(clientID))
	if _, err := flow.ExchangeCode(t.Context(), "", ""); err == nil {
		t.Fatal("empty code accepted")
	}

	for name, test := range map[string]struct {
		claims  map[string]any
		message string
	}{
		"issuer":       {map[string]any{"iss": "https://evil.example"}, `issued by "https://evil.example"`},
		"audience":     {map[string]any{"aud": "oaiapp_other"}, `not issued for client "oaiapp_issued"`},
		"no audience":  {map[string]any{"aud": nil}, `not issued for client "oaiapp_issued"`},
		"expired":      {map[string]any{"exp": time.Now().Add(-time.Hour).Unix()}, "has expired"},
		"beyond skew":  {map[string]any{"exp": time.Now().Add(-2 * time.Minute).Unix()}, "has expired"},
		"no expiry":    {map[string]any{"exp": nil}, "has expired"},
		"nonce":        {map[string]any{"nonce": "other-attempt"}, "nonce"},
		"no subject":   {map[string]any{"sub": nil}, "does not identify an account"},
		"other person": {map[string]any{"sub": "user-2"}, "different ChatGPT account"},
		"within skew":  {map[string]any{"exp": time.Now().Add(-30 * time.Second).Unix()}, ""},
	} {
		endpoint := &tokenEndpoint{}
		flow, _ := openaichatgpt.NewOAuthFlow(endpoint.client(t), "My App", openaichatgpt.WithClientID(clientID),
			openaichatgpt.WithIDTokenHint(jwt(map[string]any{"sub": "user-1"})))
		endpoint.reply = signedIn(flow, planScope, test.claims)
		_, err := flow.ExchangeCode(t.Context(), "the-code", "")
		if test.message == "" && err != nil || test.message != "" && (err == nil || !strings.Contains(err.Error(), test.message)) {
			t.Fatalf("%s: %v", name, err)
		}
		var refresh *openaichatgpt.CredentialsRefreshError
		if test.message != "" && name != "other person" && (!errors.As(err, &refresh) || !refresh.IsModelAPIError()) {
			t.Fatalf("%s: not a refresh error: %v", name, err)
		}
	}

	for name, body := range map[string]string{
		"missing":   `{"access_token":"a","refresh_token":"r"}`,
		"malformed": `{"access_token":"a","refresh_token":"r","id_token":"not-a-jwt"}`,
		"rejected":  `{"error":"invalid_grant"}`,
		"empty":     `{"access_token":"","refresh_token":"r"}`,
		"invalid":   `not json`,
	} {
		status := http.StatusOK
		if name == "rejected" {
			status = http.StatusBadRequest
		}
		endpoint := &tokenEndpoint{reply: func(url.Values) *http.Response { return response(status, body) }}
		flow, _ := openaichatgpt.NewOAuthFlow(endpoint.client(t), "My App")
		if _, err := flow.ExchangeCode(t.Context(), "the-code", clientID); err == nil {
			t.Fatalf("%s: accepted", name)
		} else if name == "rejected" && (!strings.Contains(err.Error(), "the grant was rejected") || flow.ClientID() != clientID) {
			t.Fatalf("a failed exchange must keep the issued client: %v %q", err, flow.ClientID())
		}
	}
	failing := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("offline")
	})}
	flow, _ = openaichatgpt.NewOAuthFlow(failing, "My App")
	if _, err := flow.ExchangeCode(t.Context(), "the-code", clientID); err == nil || !strings.Contains(err.Error(), "offline") {
		t.Fatalf("transport failure: %v", err)
	}
	for status, body := range map[int]string{http.StatusBadGateway: `not json`, http.StatusUnauthorized: `{"error":"denied"}`} {
		endpoint := &tokenEndpoint{reply: func(url.Values) *http.Response { return response(status, body) }}
		flow, _ := openaichatgpt.NewOAuthFlow(endpoint.client(t), "My App")
		if _, err := flow.ExchangeCode(t.Context(), "the-code", clientID); err == nil ||
			strings.Contains(err.Error(), "the grant was rejected") {
			t.Fatalf("status %d: %v", status, err)
		}
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

func TestExchangeCodeFromCallback(t *testing.T) {
	port := freePort(t)
	endpoint := &tokenEndpoint{}
	client := endpoint.client(t)
	flow, err := openaichatgpt.NewOAuthFlow(client, "My App", openaichatgpt.WithRedirectPort(port))
	if err != nil {
		t.Fatal(err)
	}
	endpoint.reply = signedIn(flow, planScope, nil)
	result := make(chan error, 1)
	var credentials openaichatgpt.Credentials
	go func() {
		var err error
		credentials, err = flow.ExchangeCodeFromCallback(t.Context())
		result <- err
	}()
	callback := func(query url.Values) string {
		for {
			reply, err := http.Get(flow.RedirectURI() + "?" + query.Encode())
			if err == nil {
				body, _ := io.ReadAll(reply.Body)
				_ = reply.Body.Close()
				return string(body)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	callback(url.Values{"state": {"forged"}, "code": {"stolen"}})
	if body := callback(url.Values{"state": {flow.State()}, "code": {"the-code"}, "client_id": {clientID}}); !strings.Contains(body, "close this tab") {
		t.Fatalf("unexpected callback page %q", body)
	}
	if err := <-result; err != nil || credentials.ClientID != clientID || len(endpoint.forms) != 1 || endpoint.forms[0].Get("code") != "the-code" {
		t.Fatalf("callback exchange: %+v %v %v", credentials, err, endpoint.forms)
	}

	flow, _ = openaichatgpt.NewOAuthFlow(client, "My App", openaichatgpt.WithRedirectPort(port))
	go func() { _, err := flow.ExchangeCodeFromCallback(t.Context()); result <- err }()
	callback(url.Values{"state": {flow.State()}, "error": {"access_denied"}})
	if err := <-result; err == nil || !strings.Contains(err.Error(), "access_denied") {
		t.Fatalf("denied callback: %v", err)
	}
	flow, _ = openaichatgpt.NewOAuthFlow(client, "My App", openaichatgpt.WithRedirectPort(port))
	go func() { _, err := flow.ExchangeCodeFromCallback(t.Context()); result <- err }()
	callback(url.Values{"state": {flow.State()}})
	if err := <-result; err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("callback without code: %v", err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := flow.ExchangeCodeFromCallback(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled callback: %v", err)
	}
	busy, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = busy.Close() }()
	flow, _ = openaichatgpt.NewOAuthFlow(client, "My App",
		openaichatgpt.WithRedirectPort(busy.Addr().(*net.TCPAddr).Port))
	if _, err := flow.ExchangeCodeFromCallback(t.Context()); err == nil {
		t.Fatal("listened on a busy port")
	}
}
