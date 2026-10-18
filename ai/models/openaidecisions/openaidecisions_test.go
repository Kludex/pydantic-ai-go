package openaidecisions_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/infer"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaidecisions"
)

type ticket struct {
	Urgent bool   `json:"urgent" jsonschema:"description=Does this need a reply within the hour?"`
	Area   string `json:"area" jsonschema:"enum=billing,enum=bug,description=Which team owns it?"`
}

type urgency struct {
	Urgent bool `json:"urgent" jsonschema:"description=Is this urgent?"`
}

const ticketReply = `{"model":"gpt-6-luna-2026","answers":[` +
	`{"type":"predicate","name":"urgent","probability":0.75},` +
	`{"type":"choice","name":"area","choice":"billing","confidence":0.94,` +
	`"probabilities":[{"value":"billing","probability":0.94},{"value":"bug","probability":0.06}]}],` +
	`"usage":{"input_tokens":312,"input_tokens_details":{"cached_tokens":12,"cache_write_tokens":3},` +
	`"output_tokens":2,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":314}}`

type exchange struct {
	request *http.Request
	body    map[string]any
}

type fake struct {
	t         *testing.T
	mu        sync.Mutex
	replies   []string
	exchanges []exchange
}

func newFake(t *testing.T, replies ...string) *fake { return &fake{t: t, replies: replies} }

func (f *fake) RoundTrip(request *http.Request) (*http.Response, error) {
	var body map[string]any
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		f.t.Errorf("decode request: %v", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.exchanges = append(f.exchanges, exchange{request: request, body: body})
	if len(f.exchanges) > len(f.replies) {
		f.t.Errorf("unexpected request %d", len(f.exchanges))
		return nil, errors.New("unexpected request")
	}
	return reply(http.StatusOK, f.replies[len(f.exchanges)-1], http.Header{
		"X-Request-Id": {"req_" + strconv.Itoa(len(f.exchanges))},
	}), nil
}

func (f *fake) model(options ...openaidecisions.Option) *openaidecisions.Model {
	return openaidecisions.NewModel("gpt-6-luna", append([]openaidecisions.Option{
		openaidecisions.WithBaseURL("https://decisions.test/v1/"),
		openaidecisions.WithAPIKey("sk-test"),
		openaidecisions.WithHTTPClient(&http.Client{Transport: f}),
	}, options...)...)
}

func (f *fake) only() exchange {
	f.t.Helper()
	if len(f.exchanges) != 1 {
		f.t.Fatalf("got %d requests, want 1", len(f.exchanges))
	}
	return f.exchanges[0]
}

type transport func(*http.Request) (*http.Response, error)

func (fn transport) RoundTrip(request *http.Request) (*http.Response, error) { return fn(request) }

func reply(status int, body string, header http.Header) *http.Response {
	if header == nil {
		header = http.Header{}
	}
	return &http.Response{StatusCode: status, Header: header, Body: io.NopCloser(strings.NewReader(body))}
}

func prompt(text string) []ai.ModelMessage {
	return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: text}}}}
}

func output(schema string) ai.ModelRequestParams {
	var decoded map[string]any
	if err := json.Unmarshal([]byte(schema), &decoded); err != nil {
		panic(err)
	}
	return ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: decoded}}
}

func urgentParams() ai.ModelRequestParams {
	return output(`{"type":"boolean","description":"Is this urgent?"}`)
}

func urgentReply(probability float64) string {
	return `{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"response","probability":` +
		strconv.FormatFloat(probability, 'f', -1, 64) + `}],"usage":{"input_tokens":5}}`
}

func floatPointer(value float64) *float64 { return &value }

func assertJSON(t *testing.T, got any, want string) {
	t.Helper()
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var gotValue, wantValue any
	if err := json.Unmarshal(encoded, &gotValue); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatalf("invalid expected JSON: %v", err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("got  %s\nwant %s", encoded, want)
	}
}

func toolCall(t *testing.T, response *ai.ModelResponse) ai.ToolCallPart {
	t.Helper()
	if len(response.Parts) != 1 {
		t.Fatalf("parts=%#v", response.Parts)
	}
	call, ok := response.Parts[0].(ai.ToolCallPart)
	if !ok {
		t.Fatalf("part=%#v", response.Parts[0])
	}
	return call
}

func TestAgentRun(t *testing.T) {
	fake := newFake(t, ticketReply)
	model := fake.model()
	agent := ai.NewAgent[struct{}, ticket](model, ai.WithInstructions("Triage a support ticket."))
	result, err := agent.Run(t.Context(), "My invoice was charged twice!", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Urgent || result.Output.Area != "billing" {
		t.Fatalf("output=%+v", result.Output)
	}
	sent := fake.only()
	if sent.request.Method != http.MethodPost || sent.request.URL.String() != "https://decisions.test/v1/decisions" {
		t.Fatalf("request=%s %s", sent.request.Method, sent.request.URL)
	}
	for header, want := range map[string]string{
		"Authorization": "Bearer sk-test", "Content-Type": "application/json", "Accept": "application/json",
	} {
		if got := sent.request.Header.Get(header); got != want {
			t.Fatalf("%s=%q", header, got)
		}
	}
	if sent.body["model"] != "gpt-6-luna" || sent.body["input"] != "My invoice was charged twice!" {
		t.Fatalf("body=%v", sent.body)
	}
	questions := sent.body["questions"].([]any)
	area, urgent := questions[0].(map[string]any), questions[1].(map[string]any)
	if area["type"] != "choice" || area["name"] != "area" || urgent["type"] != "predicate" || urgent["name"] != "urgent" {
		t.Fatalf("questions=%v", questions)
	}
	assertJSON(t, area["choices"], `[{"value":"billing"},{"value":"bug"}]`)
	var instructions map[string]any
	if err := json.Unmarshal([]byte(urgent["instructions"].(string)), &instructions); err != nil ||
		instructions["question"] != "Does this need a reply within the hour?" ||
		instructions["background"] != "Triage a support ticket." {
		t.Fatalf("instructions=%v error=%v", urgent["instructions"], err)
	}

	response := result.NewMessages()[1].(ai.ModelResponse)
	call := toolCall(t, &response)
	if call.ToolCallID != "openai-decisions-1" || call.ProviderName != "openai-decisions" {
		t.Fatalf("call=%+v", call)
	}
	if response.ModelName != "gpt-6-luna-2026" || response.ProviderName != "openai-decisions" ||
		response.ProviderURL != "https://decisions.test/v1" || response.ProviderResponseID != "req_1" ||
		response.FinishReason != ai.FinishReasonToolCall {
		t.Fatalf("response=%+v", response)
	}
	want := ai.Usage{
		Requests: 1, InputTokens: 312, OutputTokens: 2, CacheReadTokens: 12, CacheWriteTokens: 3, ReasoningTokens: 1,
	}
	if !reflect.DeepEqual(response.Usage, want) || result.Usage().InputTokens != 312 {
		t.Fatalf("usage=%+v run=%+v", response.Usage, result.Usage())
	}
	assertJSON(t, response.ProviderDetails, `{
		"confidence":{"area":0.94,"urgent":0.5},
		"probabilities":{"area":{"billing":0.94,"bug":0.06}},
		"scores":{}
	}`)
}

func TestIdentityAndEnvironment(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("OPENAI_API_KEY", "")
	model := openaidecisions.NewModel("gpt-6-luna")
	profile := model.ModelProfile()
	if model.Name() != "gpt-6-luna" || model.ProviderName() != "openai-decisions" ||
		model.ProviderURL() != "https://api.openai.com/v1" || profile.DefaultOutputMode != ai.OutputModeTool ||
		profile.SupportsTextOutput == nil || *profile.SupportsTextOutput {
		t.Fatalf("model=%q %q %q profile=%+v", model.Name(), model.ProviderName(), model.ProviderURL(), profile)
	}

	t.Setenv("OPENAI_BASE_URL", "https://proxy.test/v1/")
	t.Setenv("OPENAI_API_KEY", "sk-env")
	var authorization string
	model = openaidecisions.NewModel("gpt-6-luna", openaidecisions.WithHTTPClient(&http.Client{
		Transport: transport(func(request *http.Request) (*http.Response, error) {
			authorization = request.Header.Get("Authorization")
			return reply(http.StatusOK, urgentReply(.9), nil), nil
		}),
	}))
	if model.ProviderURL() != "https://proxy.test/v1" {
		t.Fatalf("url=%q", model.ProviderURL())
	}
	if _, err := model.Request(t.Context(), prompt("Down?"), urgentParams()); err != nil || authorization != "Bearer sk-env" {
		t.Fatalf("authorization=%q error=%v", authorization, err)
	}

	inferred, err := infer.Model("openai-decisions:gpt-6-luna")
	if err != nil {
		t.Fatal(err)
	}
	decisions, ok := inferred.(*openaidecisions.Model)
	if !ok || decisions.Name() != "gpt-6-luna" || decisions.ProviderURL() != "https://proxy.test/v1" {
		t.Fatalf("inferred=%T %v", inferred, inferred)
	}
}

func TestProvider(t *testing.T) {
	var sent *http.Request
	provider := openai.ProviderConfig{
		Name:    "azure-decisions",
		BaseURL: "https://azure.test/openai/v1/",
		APIKey:  "sk-ignored",
		HTTPClient: &http.Client{Transport: transport(func(request *http.Request) (*http.Response, error) {
			sent = request
			return reply(http.StatusOK, urgentReply(.9), nil), nil
		})},
		Headers: http.Header{"Authorization": {"Bearer header"}, "X-Team": {"support"}, "X-Static": {"kept"}},
		Query:   url.Values{"api-version": {"2026-10-01"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Prepared", request.Header.Get("X-Team"))
			return nil
		},
	}
	option := openaidecisions.WithProvider(provider)
	provider.Headers["X-Team"][0] = "mutated"
	provider.Query["api-version"][0] = "mutated"
	model := openaidecisions.NewModel("gpt-6-luna", option)
	if model.ProviderName() != "azure-decisions" || model.ProviderURL() != "https://azure.test/openai/v1" {
		t.Fatalf("identity=%q %q", model.ProviderName(), model.ProviderURL())
	}
	params := urgentParams()
	params.Settings.ExtraHeaders = map[string]string{"x-team": "override"}
	response, err := model.Request(t.Context(), prompt("Down?"), params)
	if err != nil {
		t.Fatal(err)
	}
	if sent.URL.String() != "https://azure.test/openai/v1/decisions?api-version=2026-10-01" {
		t.Fatalf("url=%s", sent.URL)
	}
	for header, want := range map[string]string{
		"Authorization": "Bearer header", "X-Team": "override", "X-Prepared": "support", "X-Static": "kept",
	} {
		if got := sent.Header.Values(header); len(got) != 1 || got[0] != want {
			t.Fatalf("%s=%q", header, got)
		}
	}
	if response.ProviderName != "azure-decisions" || toolCall(t, response).ProviderName != "azure-decisions" {
		t.Fatalf("response=%+v", response)
	}

	failure := errors.New("token expired")
	provider.PrepareRequest = func(*http.Request) error { return failure }
	provider.Name = ""
	model = openaidecisions.NewModel("gpt-6-luna", openaidecisions.WithProvider(provider))
	if model.ProviderName() != "openai-decisions" {
		t.Fatalf("provider=%q", model.ProviderName())
	}
	if _, err := model.Request(t.Context(), prompt("Down?"), urgentParams()); !errors.Is(err, failure) {
		t.Fatalf("error=%v", err)
	}
}

func TestSettings(t *testing.T) {
	agentReply := `{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"urgent","probability":0.6}],"usage":{}}`
	fake := newFake(t, agentReply, agentReply, urgentReply(.6))
	common := ai.ModelSettings{ExtraBody: map[string]any{"service_tier": "flex"}, ExtraHeaders: map[string]string{"X-Run": "default"}}
	defaults, err := openaidecisions.Settings{Common: common, BooleanThreshold: floatPointer(.9)}.Build()
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := common.ExtraBody["openai_decisions_settings"]; exists {
		t.Fatal("Build mutated the common settings")
	}
	model := fake.model(openaidecisions.WithDefaultSettings(defaults))
	defaults.ExtraBody["service_tier"] = "mutated"
	exposed := model.DefaultModelSettings()
	exposed.ExtraBody["service_tier"] = "mutated"
	if model.DefaultModelSettings().ExtraBody["service_tier"] != "flex" {
		t.Fatal("default settings were not detached")
	}

	strict := ai.NewAgent[struct{}, urgency](model)
	result, err := strict.Run(t.Context(), "Down?", struct{}{})
	if err != nil || result.Output.Urgent {
		t.Fatalf("default threshold output=%v error=%v", result, err)
	}
	if body := fake.exchanges[0].body; body["service_tier"] != "flex" || body["openai_decisions_settings"] != nil ||
		fake.exchanges[0].request.Header.Get("X-Run") != "default" {
		t.Fatalf("body=%v", body)
	}

	lenient, err := openaidecisions.Settings{BooleanThreshold: floatPointer(.5)}.Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err = strict.Run(t.Context(), "Down?", struct{}{}, ai.WithRunModelSettings(lenient))
	if err != nil || !result.Output.Urgent {
		t.Fatalf("run threshold output=%v error=%v", result, err)
	}

	params := urgentParams()
	params.Settings = defaults
	if response, err := model.Request(t.Context(), prompt("Down?"), params); err != nil ||
		string(toolCall(t, response).Args) != "false" {
		t.Fatalf("response=%v error=%v", response, err)
	}
	if _, exists := params.Settings.ExtraBody["openai_decisions_settings"]; !exists {
		t.Fatal("Request mutated the caller's settings")
	}
}

func TestSettingsValidation(t *testing.T) {
	for _, value := range []float64{-.1, 1.1, math.NaN(), math.Inf(1)} {
		for _, settings := range []openaidecisions.Settings{{BooleanThreshold: &value}, {RouteThreshold: &value}} {
			if _, err := settings.Build(); err == nil || !strings.Contains(err.Error(), "between 0 and 1") {
				t.Fatalf("threshold %v: error=%v", value, err)
			}
		}
	}
	reserved := openaidecisions.Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{"openai_decisions_settings": 1}}}
	if _, err := reserved.Build(); err == nil || !strings.Contains(err.Error(), "reserved") {
		t.Fatalf("error=%v", err)
	}
	fake := newFake(t)
	params := urgentParams()
	params.Settings.ExtraBody = map[string]any{"openai_decisions_settings": map[string]any{"route": .5}}
	if _, err := fake.model().Request(t.Context(), prompt("Down?"), params); err == nil ||
		!strings.Contains(err.Error(), "Settings.Build") {
		t.Fatalf("error=%v", err)
	}
	if len(fake.exchanges) != 0 {
		t.Fatal("invalid settings reached the transport")
	}
}

func TestRequestTimeout(t *testing.T) {
	model := openaidecisions.NewModel("gpt-6-luna", openaidecisions.WithHTTPClient(&http.Client{
		Transport: transport(func(request *http.Request) (*http.Response, error) {
			if _, ok := request.Context().Deadline(); !ok {
				t.Error("missing request deadline")
			}
			<-request.Context().Done()
			return nil, request.Context().Err()
		}),
	}))
	params := urgentParams()
	params.Settings.RequestTimeout = 10 * time.Millisecond
	_, err := model.Request(t.Context(), prompt("Down?"), params)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error=%v", err)
	}
}
