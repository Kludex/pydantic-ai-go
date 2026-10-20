package openaichatgpt_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaichatgpt"
)

func credentials(options ...func(*openaichatgpt.Credentials)) openaichatgpt.Credentials {
	value := openaichatgpt.Credentials{
		ClientID: clientID, AccessToken: "access-1", RefreshToken: "refresh-1", IDToken: "id-1", Scope: planScope,
	}
	for _, option := range options {
		option(&value)
	}
	return value
}

type memorySource struct {
	credentials openaichatgpt.Credentials
	saved       []openaichatgpt.Credentials
	afterLoad   func(*memorySource)
	saveErr     error
}

func (source *memorySource) Load(context.Context) (openaichatgpt.Credentials, error) {
	loaded := source.credentials
	if source.afterLoad != nil {
		source.afterLoad(source)
	}
	return loaded, nil
}

func (source *memorySource) Save(_ context.Context, credentials openaichatgpt.Credentials) error {
	if source.saveErr != nil {
		return source.saveErr
	}
	source.credentials = credentials
	source.saved = append(source.saved, credentials)
	return nil
}

func completed(id string, output ...map[string]any) string {
	response := map[string]any{
		"id": id, "object": "response", "model": "gpt-6.1-sol", "status": "completed", "output": output,
		"usage": map[string]any{"input_tokens": 10, "output_tokens": 5},
	}
	var events []string
	for index, item := range output {
		events = append(events, fmt.Sprintf(`{"type":"response.output_item.added","output_index":%d,"item":%s}`, index, encode(item)))
		if item["type"] == "message" {
			events = append(events, fmt.Sprintf(
				`{"type":"response.output_text.delta","item_id":%q,"output_index":%d,"content_index":0,"delta":"Moo!"}`,
				item["id"], index,
			))
		}
		events = append(events, fmt.Sprintf(`{"type":"response.output_item.done","output_index":%d,"item":%s}`, index, encode(item)))
	}
	events = append(events, `{"type":"response.completed","response":`+encode(response)+`}`)
	return "data: " + strings.Join(events, "\n\ndata: ") + "\n\n"
}

func encode(value any) string {
	data, _ := json.Marshal(value)
	return string(data)
}

func TestAgentRunUsesThePlanUsageDialect(t *testing.T) {
	var bodies []map[string]any
	replies := []string{
		completed("resp_1", map[string]any{
			"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "moo", "arguments": "{}", "status": "completed",
		}),
		completed("resp_2", map[string]any{
			"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
			"content": []any{map[string]any{"type": "output_text", "text": "Moo!", "annotations": []any{}}},
		}),
	}
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://api.openai.com/v1/responses" || request.Header.Get("Authorization") != "Bearer access-1" {
			t.Errorf("unexpected request %s %s", request.URL, request.Header)
		}
		var body map[string]any
		_ = json.NewDecoder(request.Body).Decode(&body)
		bodies = append(bodies, body)
		reply := response(http.StatusOK, replies[len(bodies)-1])
		reply.Header.Set("Content-Type", "text/event-stream")
		return reply, nil
	})}
	model, err := openaichatgpt.NewModel("gpt-6.1-sol",
		openaichatgpt.WithCredentials(credentials()), openaichatgpt.WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	temperature, topP := 0.5, 0.5
	agent := ai.NewAgent[struct{}, string](model, ai.WithSystemPrompt("Be a cow."), ai.WithModelSettings(ai.ModelSettings{
		Temperature: &temperature, TopP: &topP, MaxTokens: 100, ServiceTier: ai.ServiceTierFlex,
		Cache: &ai.CacheConfig{},
	}))
	ai.AddSimpleTool(agent, "moo", func(context.Context, struct{}) (string, error) { return "Moo!", nil },
		ai.WithDescription("Return a cow sound."))
	result, err := agent.Run(t.Context(), "Moo once.", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Moo!" || len(bodies) != 2 {
		t.Fatalf("unexpected run: %q %d", result.Output, len(bodies))
	}
	for _, body := range bodies {
		instructions := body["input"].([]any)[0].(map[string]any)
		if body["stream"] != true || body["store"] != false || instructions["role"] != "developer" ||
			instructions["content"] != "Be a cow." {
			t.Fatalf("unexpected dialect %v", body)
		}
		for _, key := range []string{"temperature", "top_p", "max_output_tokens", "service_tier", "prompt_cache_key", "prompt_cache_retention"} {
			if _, sent := body[key]; sent {
				t.Fatalf("%s was sent: %v", key, body)
			}
		}
	}
	if model.ProviderName() != "openai-chatgpt" || model.ProviderURL() != "https://api.openai.com/v1" ||
		model.ModelProfile().SupportsCache {
		t.Fatalf("unexpected identity %q %q %+v", model.ProviderName(), model.ProviderURL(), model.ModelProfile())
	}
}

func TestModelSupportsOnlyWebSearchAndNoTokenCounting(t *testing.T) {
	model, err := openaichatgpt.NewModel("gpt-6.1-sol", openaichatgpt.WithCredentials(credentials()),
		openaichatgpt.WithDefaultSettings(ai.ModelSettings{MaxTokens: 1}))
	if err != nil {
		t.Fatal(err)
	}
	if !model.SupportsNativeTool(ai.WebSearchTool{}) || model.SupportsNativeTool(ai.CodeExecutionTool{}) ||
		model.SupportsNativeTool(ai.WebSearchTool{MaxUses: 1}) || model.DefaultModelSettings().MaxTokens != 1 {
		t.Fatal("unexpected native tool support")
	}
	if _, err := model.CountTokens(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, ai.ErrTokenCountingUnsupported) {
		t.Fatalf("token counting: %v", err)
	}
	suspended := []ai.ModelMessage{ai.ModelResponse{State: ai.ModelResponseStateSuspended}}
	if _, err := model.Request(t.Context(), suspended, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "store=false") {
		t.Fatalf("suspended resume: %v", err)
	}
	if current, err := model.Credentials(); err != nil || current.AccessToken != "access-1" {
		t.Fatalf("credentials %v %v", current, err)
	}
}

func TestNewModelValidation(t *testing.T) {
	source := &memorySource{credentials: credentials()}
	for name, test := range map[string]struct {
		model   string
		options []openaichatgpt.Option
		message string
	}{
		"name":         {"", []openaichatgpt.Option{openaichatgpt.WithCredentials(credentials())}, "name"},
		"none":         {"m", nil, "sign in with NewOAuthFlow first"},
		"both":         {"m", []openaichatgpt.Option{openaichatgpt.WithCredentials(credentials()), openaichatgpt.WithCredentialSource(source)}, "mutually exclusive"},
		"client":       {"m", []openaichatgpt.Option{openaichatgpt.WithCredentials(credentials()), openaichatgpt.WithHTTPClient(nil)}, "HTTP client"},
		"no plan":      {"m", []openaichatgpt.Option{openaichatgpt.WithCredentials(credentials(func(c *openaichatgpt.Credentials) { c.Scope = "openid" }))}, `client "oaiapp_issued"`},
		"no client":    {"m", []openaichatgpt.Option{openaichatgpt.WithCredentials(credentials(func(c *openaichatgpt.Credentials) { c.ClientID = "" }))}, "client_id"},
		"no access":    {"m", []openaichatgpt.Option{openaichatgpt.WithCredentials(credentials(func(c *openaichatgpt.Credentials) { c.AccessToken = "" }))}, "access_token"},
		"no refreshes": {"m", []openaichatgpt.Option{openaichatgpt.WithCredentials(credentials(func(c *openaichatgpt.Credentials) { c.RefreshToken = "" }))}, "refresh_token"},
	} {
		if _, err := openaichatgpt.NewModel(test.model, test.options...); err == nil || !strings.Contains(err.Error(), test.message) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentialSource(source))
	if _, err := model.Credentials(); err == nil {
		t.Fatal("credentials before load")
	}
}

func TestProviderRefusesStoredCredentialsWithoutPlanUsage(t *testing.T) {
	var apiRequests int
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		apiRequests++
		return response(http.StatusOK, `{}`), nil
	})}
	source := &memorySource{credentials: credentials(func(c *openaichatgpt.Credentials) { c.Scope = "openid" })}
	model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentialSource(source), openaichatgpt.WithHTTPClient(client))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, openaichatgpt.ErrPlanUsageNotGranted) {
		t.Fatalf("unexpected error %v", err)
	}
	if apiRequests != 0 {
		t.Fatal("credentials without plan usage reached the API")
	}
}

func TestProviderRefreshesAgainstTheIssuedClient(t *testing.T) {
	for name, test := range map[string]struct {
		refreshed      string
		idToken, scope string
	}{
		"grant kept":     {``, "id-1", planScope},
		"grant replaced": {`,"id_token":"id-2","scope":"chatgpt.tokens.use.direct"`, "id-2", "chatgpt.tokens.use.direct"},
	} {
		var forms []url.Values
		var seen []string
		client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if request.URL.Hostname() == "auth.openai.com" {
				_ = request.ParseForm()
				forms = append(forms, request.PostForm)
				return response(http.StatusOK, `{"access_token":"access-2","refresh_token":"refresh-2"`+test.refreshed+`}`), nil
			}
			seen = append(seen, request.Header.Get("Authorization"))
			if request.Header.Get("Authorization") == "Bearer access-1" {
				return response(http.StatusUnauthorized, `{}`), nil
			}
			reply := response(http.StatusOK, completed("resp"))
			reply.Header.Set("Content-Type", "text/event-stream")
			return reply, nil
		})}
		source := &memorySource{credentials: credentials()}
		model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentialSource(source), openaichatgpt.WithHTTPClient(client))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want := credentials(func(c *openaichatgpt.Credentials) {
			c.AccessToken, c.RefreshToken, c.IDToken, c.Scope = "access-2", "refresh-2", test.idToken, test.scope
		})
		if !slices.Equal(seen, []string{"Bearer access-1", "Bearer access-2"}) || len(source.saved) != 1 ||
			source.saved[0] != want || len(forms) != 1 || forms[0].Get("grant_type") != "refresh_token" ||
			forms[0].Get("refresh_token") != "refresh-1" || forms[0].Get("client_id") != clientID ||
			forms[0].Get("resource") != "https://api.openai.com/v1" {
			t.Fatalf("%s: seen=%v saved=%v forms=%v", name, seen, source.saved, forms)
		}
	}
}

func TestProviderRefusesCredentialsThatLosePlanUsage(t *testing.T) {
	withoutPlan := credentials(func(c *openaichatgpt.Credentials) { c.Scope, c.AccessToken = "openid", "access-2" })
	for _, storedElsewhere := range []bool{false, true} {
		client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if request.URL.Hostname() == "auth.openai.com" {
				return response(http.StatusOK, `{"access_token":"access-2","refresh_token":"refresh-1","scope":"openid"}`), nil
			}
			return response(http.StatusUnauthorized, `{}`), nil
		})}
		source := &memorySource{credentials: credentials()}
		if storedElsewhere {
			loads := 0
			source.afterLoad = func(source *memorySource) {
				if loads++; loads == 1 {
					source.credentials = withoutPlan
				}
			}
		}
		model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentialSource(source), openaichatgpt.WithHTTPClient(client))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, openaichatgpt.ErrPlanUsageNotGranted) {
			t.Fatalf("stored elsewhere=%v: %v", storedElsewhere, err)
		}
		// A refresh that loses plan usage is still saved: its refresh token is the only usable one.
		if wantSaved := !storedElsewhere; (len(source.saved) == 1) != wantSaved {
			t.Fatalf("stored elsewhere=%v: saved %v", storedElsewhere, source.saved)
		}
	}
}

func TestUnusableRefreshTokensSayToSignInAgain(t *testing.T) {
	for _, code := range []string{
		"invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
		"refresh_token_invalidated", "refresh_token_reused",
	} {
		client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if request.URL.Hostname() == "auth.openai.com" {
				return response(http.StatusBadRequest, `{"error":"`+code+`"}`), nil
			}
			return response(http.StatusUnauthorized, `{}`), nil
		})}
		model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentials(credentials()), openaichatgpt.WithHTTPClient(client))
		var refresh *openaichatgpt.CredentialsRefreshError
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if !errors.As(err, &refresh) || !strings.Contains(err.Error(), "the grant was rejected, rerun the authorization flow") {
			t.Fatalf("%s: %v", code, err)
		}
	}
}

func TestProviderStopsUsingCredentialsThatLostPlanUsage(t *testing.T) {
	var tokenPosts, apiRequests int
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Hostname() == "auth.openai.com" {
			tokenPosts++
			return response(http.StatusOK, `{"access_token":"access-2","refresh_token":"refresh-2","scope":"openid"}`), nil
		}
		apiRequests++
		return response(http.StatusOK, `{}`), nil
	})}
	expired := credentials(func(c *openaichatgpt.Credentials) {
		c.AccessToken = jwt(map[string]any{"exp": time.Now().Add(-time.Hour).Unix()})
	})
	model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentials(expired), openaichatgpt.WithHTTPClient(client))
	for range 2 {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, openaichatgpt.ErrPlanUsageNotGranted) {
			t.Fatalf("unexpected error %v", err)
		}
	}
	if tokenPosts != 1 || apiRequests != 0 {
		t.Fatalf("token posts %d, API requests %d", tokenPosts, apiRequests)
	}
}

func TestProviderKeepsAnUnsavedRotationOverTheSetItSpent(t *testing.T) {
	var grants []string
	rotation := 1
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Hostname() == "auth.openai.com" {
			_ = request.ParseForm()
			grants = append(grants, request.PostForm.Get("refresh_token"))
			rotation++
			return response(http.StatusOK, fmt.Sprintf(`{"access_token":"access-%d","refresh_token":"refresh-%d"}`, rotation, rotation)), nil
		}
		if request.Header.Get("Authorization") != "Bearer access-3" {
			return response(http.StatusUnauthorized, `{}`), nil
		}
		reply := response(http.StatusOK, completed("resp"))
		reply.Header.Set("Content-Type", "text/event-stream")
		return reply, nil
	})}
	source := &memorySource{credentials: credentials(func(c *openaichatgpt.Credentials) {
		c.AccessToken = jwt(map[string]any{"exp": time.Now().Add(-time.Hour).Unix()})
	}), saveErr: errors.New("disk full")}
	model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentialSource(source), openaichatgpt.WithHTTPClient(client))
	var persistence *openaichatgpt.CredentialsPersistenceError
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.As(err, &persistence) ||
		!persistence.IsModelAPIError() || !strings.Contains(err.Error(), "disk full") || !errors.Is(err, source.saveErr) {
		t.Fatalf("unexpected first error %v", err)
	}
	source.saveErr = nil
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	current, _ := model.Credentials()
	if !slices.Equal(grants, []string{"refresh-1", "refresh-2"}) || current.RefreshToken != "refresh-3" ||
		source.credentials.RefreshToken != "refresh-3" {
		t.Fatalf("spent credentials were adopted: grants=%v current=%v stored=%v", grants, current, source.credentials)
	}
}

func TestProviderSendsCredentialsOnlyToTheOpenAIAPI(t *testing.T) {
	for _, location := range []string{
		"https://example.com/final", "http://api.openai.com/final", "https://api.openai.com/final",
	} {
		var redirected http.Header
		client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if request.URL.Path != "/final" {
				redirect := response(http.StatusTemporaryRedirect, "")
				redirect.Header.Set("Location", location)
				return redirect, nil
			}
			redirected = request.Header.Clone()
			reply := response(http.StatusOK, completed("resp"))
			reply.Header.Set("Content-Type", "text/event-stream")
			return reply, nil
		})}
		model, _ := openaichatgpt.NewModel("m", openaichatgpt.WithCredentials(credentials()), openaichatgpt.WithHTTPClient(client))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
			t.Fatalf("%s: %v", location, err)
		}
		if strings.HasPrefix(location, "https://api.openai.com/") != (redirected.Get("Authorization") == "Bearer access-1") {
			t.Fatalf("%s: authorization %q", location, redirected.Get("Authorization"))
		}
	}
}
