package googledecisions_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
	"github.com/Kludex/pydantic-ai-go/ai/models/googledecisions"
	"github.com/Kludex/pydantic-ai-go/ai/models/infer"
)

// The answers and usage are those Gemini returned for the recorded upstream request.
const ticketReply = `{"answers":{
	"q1":{"confidence":0.9830831764080006,"noul":0.9830831764080006,"type":"noul"},
	"q2":{"confidence":0.7177949412677748,"score":1.1171219130010348,"type":"score",
		"probabilities":{"0":0.0825415728655953,"1":0.7177949412677748,"2":0.19966348586662996},
		"legend":{"0":"Can wait a week","1":"Should be handled today","2":"Needs a reply within the hour"}},
	"q0":{"confidence":0.8778519632794022,"choice":"product defect","type":"choice",
		"probabilities":{"billing":0.12214803672059774,"product defect":0.8778519632794022}}
},"usage":{"total_tokens":303,"total_input_tokens":303,"total_output_tokens":0,
	"input_tokens_by_modality":[{"modality":"text","tokens":303}]}}`

type exchange struct {
	request *http.Request
	body    map[string]any
}

type fake struct {
	t         *testing.T
	replies   []string
	exchanges []exchange
}

func (f *fake) RoundTrip(request *http.Request) (*http.Response, error) {
	var body map[string]any
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		f.t.Errorf("decode request: %v", err)
	}
	f.exchanges = append(f.exchanges, exchange{request: request, body: body})
	if len(f.exchanges) > len(f.replies) {
		return nil, errors.New("unexpected request")
	}
	return reply(http.StatusOK, f.replies[len(f.exchanges)-1]), nil
}

func (f *fake) model(options ...googledecisions.Option) *googledecisions.Model {
	return googledecisions.NewModel("gemini-decision-1", append([]googledecisions.Option{
		googledecisions.WithBaseURL("https://decisions.test/v1beta/"),
		googledecisions.WithAPIKey("key"),
		googledecisions.WithHTTPClient(&http.Client{Transport: f}),
	}, options...)...)
}

type transport func(*http.Request) (*http.Response, error)

func (fn transport) RoundTrip(request *http.Request) (*http.Response, error) { return fn(request) }

func reply(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}
}

func prompt(text string) []ai.ModelMessage {
	return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: text}}}}
}

func output(schema string) ai.ModelRequestParams {
	var decoded map[string]any
	if err := json.Unmarshal([]byte(schema), &decoded); err != nil {
		panic(err)
	}
	return ai.ModelRequestParams{
		Instructions: "Triage.", OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: decoded},
	}
}

func urgentParams() ai.ModelRequestParams {
	return output(`{"type":"boolean","description":"Is this urgent?"}`)
}

const urgentReply = `{"answers":{"q0":{"type":"noul","noul":0.9,"confidence":0.9}},"usage":{"total_input_tokens":5}}`

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

func TestRecordedTicket(t *testing.T) {
	f := &fake{t: t, replies: []string{ticketReply}}
	params := output(`{"type":"object","properties":{
		"refund":{"type":"boolean","description":"Do they want a refund?","anyOf":[
			{"const":true,"description":"They ask for their money back."},{"const":false,"description":"They do not."}]},
		"area":{"type":"string","enum":["billing","product defect"],"description":"Which team owns it?"},
		"urgency":{"type":"integer","description":"How urgent is it?","anyOf":[
			{"const":0,"description":"Can wait a week"},{"const":1,"description":"Should be handled today"},
			{"const":2,"description":"Needs a reply within the hour"}]}
	},"required":["refund","area","urgency"]}`)
	response, err := f.model().Request(t.Context(), prompt("The blender arrived with a cracked jar."), params)
	if err != nil {
		t.Fatal(err)
	}
	sent := f.exchanges[0]
	if sent.request.URL.String() != "https://decisions.test/v1beta/decisions" ||
		sent.request.Header.Get("x-goog-api-key") != "key" || sent.request.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("request=%s %v", sent.request.URL, sent.request.Header)
	}
	if sent.body["model"] != "gemini-decision-1" || sent.body["input"] != "The blender arrived with a cracked jar." {
		t.Fatalf("body=%v", sent.body)
	}
	questions := sent.body["questions"].(map[string]any)
	assertJSON(t, questions["q0"].(map[string]any)["criteria"], `{"billing":null,"product defect":null}`)
	assertJSON(t, questions["q2"].(map[string]any)["criteria"],
		`["Can wait a week","Should be handled today","Needs a reply within the hour"]`)
	assertJSON(t, questions["q1"].(map[string]any)["criteria"],
		`{"true_description":"They ask for their money back.","false_description":"They do not."}`)
	var instructions map[string]any
	if err := json.Unmarshal([]byte(questions["q1"].(map[string]any)["instructions"].(string)), &instructions); err != nil ||
		instructions["question"] != "Do they want a refund?" {
		t.Fatalf("instructions=%v error=%v", questions["q2"], err)
	}
	if len(response.Parts) != 1 {
		t.Fatalf("parts=%#v", response.Parts)
	}
	call := response.Parts[0].(ai.ToolCallPart)
	assertJSON(t, json.RawMessage(call.Args), `{"area":"product defect","refund":true,"urgency":1}`)
	if call.ToolCallID != "google-decisions-1" || response.ModelName != "gemini-decision-1" ||
		response.ProviderName != "google-decisions" || response.ProviderURL != "https://decisions.test/v1beta" {
		t.Fatalf("response=%+v", response)
	}
	want := ai.Usage{Requests: 1, InputTokens: 303, Details: map[string]int{"text_prompt_tokens": 303}}
	if !reflect.DeepEqual(response.Usage, want) {
		t.Fatalf("usage=%+v", response.Usage)
	}
}

func TestAgentOutput(t *testing.T) {
	f := &fake{t: t, replies: []string{urgentReply}}
	type urgency struct {
		Urgent bool `json:"urgent" jsonschema:"description=Is this urgent?"`
	}
	agent := ai.NewAgent[struct{}, urgency](f.model(), ai.WithInstructions("Triage."))
	result, err := agent.Run(t.Context(), "Server is down", struct{}{})
	if err != nil || !result.Output.Urgent {
		t.Fatalf("output=%+v error=%v", result, err)
	}
}

func TestIdentityEnvironmentAndInference(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	model := googledecisions.NewModel("gemini-decision-1")
	profile := model.ModelProfile()
	if model.Name() != "gemini-decision-1" || model.ProviderName() != "google-decisions" ||
		model.ProviderURL() != "https://generativelanguage.googleapis.com/v1beta" ||
		profile.DefaultOutputMode != ai.OutputModeTool || profile.SupportsTextOutput == nil || *profile.SupportsTextOutput {
		t.Fatalf("model=%q %q %q profile=%+v", model.Name(), model.ProviderName(), model.ProviderURL(), profile)
	}
	var key string
	model = googledecisions.NewModel("gemini-decision-1", googledecisions.WithHTTPClient(&http.Client{
		Transport: transport(func(request *http.Request) (*http.Response, error) {
			key = request.Header.Get("x-goog-api-key")
			return reply(http.StatusOK, urgentReply), nil
		}),
	}))
	if _, err := model.Request(t.Context(), prompt("Down?"), urgentParams()); err != nil || key != "gemini-key" {
		t.Fatalf("key=%q error=%v", key, err)
	}
	t.Setenv("GOOGLE_API_KEY", "google-key")
	inferred, err := infer.Model("google-decisions:gemini-decision-1")
	if err != nil {
		t.Fatal(err)
	}
	if decisions, ok := inferred.(*googledecisions.Model); !ok || decisions.Name() != "gemini-decision-1" {
		t.Fatalf("inferred=%T", inferred)
	}
}

func TestProvider(t *testing.T) {
	var sent *http.Request
	model := googledecisions.NewModel("gemini-decision-1", googledecisions.WithProvider(google.ProviderConfig{
		Name: "gateway", BaseURL: "https://gateway.test/v1beta/",
		HTTPClient: &http.Client{Transport: transport(func(request *http.Request) (*http.Response, error) {
			sent = request
			return reply(http.StatusOK, urgentReply), nil
		})},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("Authorization", "Bearer token")
			return nil
		},
	}))
	params := urgentParams()
	params.Settings.ExtraHeaders = map[string]string{"X-Team": "support"}
	response, err := model.Request(t.Context(), prompt("Down?"), params)
	if err != nil || response.ProviderName != "gateway" || sent.URL.String() != "https://gateway.test/v1beta/decisions" ||
		sent.Header.Get("Authorization") != "Bearer token" || sent.Header.Get("x-goog-api-key") != "" ||
		sent.Header.Get("X-Team") != "support" {
		t.Fatalf("response=%+v request=%v error=%v", response, sent, err)
	}
	failing := googledecisions.NewModel("gemini-decision-1", googledecisions.WithProvider(google.ProviderConfig{
		PrepareRequest: func(*http.Request) error { return errors.New("no token") },
	}))
	if _, err := failing.Request(t.Context(), prompt("Down?"), urgentParams()); err == nil ||
		!strings.Contains(err.Error(), "prepare request: no token") || failing.ProviderURL() != "https://generativelanguage.googleapis.com/v1beta" {
		t.Fatalf("error=%v", err)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("Vertex provider was accepted")
		}
	}()
	googledecisions.WithProvider(google.ProviderConfig{Transport: google.TransportVertexAI})
}

func TestSettings(t *testing.T) {
	threshold := .95
	settings, err := googledecisions.Settings{
		Common: ai.ModelSettings{ExtraBody: map[string]any{"labels": map[string]any{"team": "support"}}}, BooleanThreshold: &threshold,
	}.Build()
	if err != nil {
		t.Fatal(err)
	}
	f := &fake{t: t, replies: []string{urgentReply}}
	params := urgentParams()
	params.Settings = settings
	response, err := f.model(googledecisions.WithDefaultSettings(settings)).Request(t.Context(), prompt("Down?"), params)
	if err != nil || string(response.Parts[0].(ai.ToolCallPart).Args) != "false" {
		t.Fatalf("response=%+v error=%v", response, err)
	}
	assertJSON(t, f.exchanges[0].body["labels"], `{"team":"support"}`)
	if f.model(googledecisions.WithDefaultSettings(settings)).DefaultModelSettings().ExtraBody == nil {
		t.Fatal("default settings were dropped")
	}

	invalid := 2.0
	for _, test := range []googledecisions.Settings{
		{RouteThreshold: &invalid},
		{Common: ai.ModelSettings{ExtraBody: map[string]any{"google_decisions_settings": 1}}},
	} {
		if _, err := test.Build(); err == nil {
			t.Fatalf("settings %+v were accepted", test)
		}
	}
	for name, body := range map[string]map[string]any{
		"forged thresholds": {"google_decisions_settings": 1},
		"questions":         {"Questions": map[string]any{}},
	} {
		t.Run(name, func(t *testing.T) {
			params := urgentParams()
			params.Settings.ExtraBody = body
			if _, err := (&fake{t: t}).model().Request(t.Context(), prompt("Down?"), params); err == nil {
				t.Fatal("expected settings error")
			}
		})
	}
}

func TestRequestTimeout(t *testing.T) {
	model := googledecisions.NewModel("gemini-decision-1", googledecisions.WithHTTPClient(&http.Client{
		Transport: transport(func(request *http.Request) (*http.Response, error) {
			<-request.Context().Done()
			return nil, request.Context().Err()
		}),
	}))
	params := urgentParams()
	params.Settings.RequestTimeout = time.Millisecond
	if _, err := model.Request(t.Context(), prompt("Down?"), params); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error=%v", err)
	}
}

func TestRouteHandOff(t *testing.T) {
	routeParams := output(`{"type":"object","title":"Ticket","properties":{
		"urgent":{"type":"boolean","description":"Is this urgent?"}},"required":["urgent"]}`)
	routeParams.Tools = []ai.ToolDefinition{{
		Name: "escalate", Description: "Hand the ticket to a human.", Schema: map[string]any{"type": "object"},
	}}
	threshold := .9
	settings, err := googledecisions.Settings{RouteThreshold: &threshold}.Build()
	if err != nil {
		t.Fatal(err)
	}
	routeParams.Settings = settings
	f := &fake{t: t, replies: []string{`{"answers":{
		"q0":{"type":"noul","noul":0.9,"confidence":0.9},
		"q1":{"type":"choice","choice":"Ticket","confidence":0.6,"probabilities":{"Ticket":0.6,"escalate":0.4}}
	},"usage":{}}`}}
	_, err = f.model().Request(t.Context(), prompt("Charged twice."), routeParams)
	var unsure *googledecisions.UnsureRoute
	var handOff googledecisions.DecisionHandOff
	if !errors.As(err, &unsure) || !errors.As(err, &handOff) || !handOff.IsModelAPIError() ||
		!strings.Contains(err.Error(), `picked "Ticket" with probability 0.60`) {
		t.Fatalf("error=%v", err)
	}
	if route, probability := handOff.DecisionRoute(); route != "Ticket" || probability != .6 {
		t.Fatalf("route=%q %v", route, probability)
	}

	unfillable := output(`{"type":"object","title":"Ticket","properties":{
		"urgent":{"type":"boolean","description":"Is this urgent?"}},"required":["urgent"]}`)
	unfillable.Tools = []ai.ToolDefinition{{Name: "note", Description: "Leave a note.", Schema: map[string]any{
		"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}, "required": []any{"text"},
	}}}
	f = &fake{t: t, replies: []string{`{"answers":{
		"q0":{"type":"noul","noul":0.9,"confidence":0.9},
		"q1":{"type":"choice","choice":"note","confidence":0.9,"probabilities":{"Ticket":0.1,"note":0.9}}
	},"usage":{}}`}}
	_, err = f.model().Request(t.Context(), prompt("Note it."), unfillable)
	var cannot *googledecisions.UnfillableRoute
	if !errors.As(err, &cannot) || !cannot.IsModelAPIError() || !strings.Contains(err.Error(), "cannot fill it") {
		t.Fatalf("error=%v", err)
	}
	if route, _ := cannot.DecisionRoute(); route != "note" {
		t.Fatalf("route=%q", route)
	}
}

func TestImages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "image/gif")
		_, _ = writer.Write([]byte("gif"))
	}))
	defer server.Close()
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "Is it broken?"},
		ai.BinaryContent{Data: []byte("png"), MediaType: "image/png", VendorMetadata: map[string]any{
			"media_resolution": map[string]any{"level": "MEDIA_RESOLUTION_LOW"},
		}},
		ai.BinaryContent{Data: []byte("jpeg"), MediaType: "image/jpeg", VendorMetadata: map[string]any{
			"media_resolution": "MEDIA_RESOLUTION_UNSPECIFIED",
		}},
		ai.ImageURL{URL: server.URL + "/cat", ForceDownload: ai.FileDownloadAllowLocal, VendorMetadata: map[string]any{
			"media_resolution": "MEDIA_RESOLUTION_HIGH",
		}},
	}}}}}
	f := &fake{t: t, replies: []string{urgentReply}}
	if _, err := f.model().Request(t.Context(), messages, urgentParams()); err != nil {
		t.Fatal(err)
	}
	input := f.exchanges[0].body["input"].([]any)
	data := func(body string) string { return base64.StdEncoding.EncodeToString([]byte(body)) }
	state, _ := json.Marshal(input[0].(map[string]any)["text"])
	assertJSON(t, input, `[
		{"type":"text","text":`+string(state)+`},
		{"type":"text","text":"<image 1>:"},
		{"type":"image","data":"`+data("png")+`","mime_type":"image/png","resolution":"low"},
		{"type":"text","text":"<image 2>:"},
		{"type":"image","data":"`+data("jpeg")+`","mime_type":"image/jpeg"},
		{"type":"text","text":"<image 3>:"},
		{"type":"image","data":"`+data("gif")+`","mime_type":"image/gif","resolution":"high"}
	]`)
}

func TestResponseValidation(t *testing.T) {
	for name, body := range map[string]string{
		"not JSON":         `{`,
		"unknown ID":       `{"answers":{"q1":{"type":"noul","noul":0.9}},"usage":{}}`,
		"missing answer":   `{"answers":{},"usage":{}}`,
		"wrong type":       `{"answers":{"q0":{"type":"choice","choice":"a"}},"usage":{}}`,
		"out of range":     `{"answers":{"q0":{"type":"noul","noul":1.5}},"usage":{}}`,
		"rounded sum":      `{"answers":{"q0":{"type":"choice","choice":"a","confidence":0.67,"probabilities":{"a":0.67,"b":0.32}}},"usage":{}}`,
		"inconsistent sum": `{"answers":{"q0":{"type":"choice","choice":"a","confidence":0.5,"probabilities":{"a":0.5,"b":0.6}}},"usage":{}}`,
	} {
		t.Run(name, func(t *testing.T) {
			params := urgentParams()
			if strings.Contains(body, "choice\",\"choice") {
				params = output(`{"type":"string","enum":["a","b"],"description":"Which?"}`)
			}
			f := &fake{t: t, replies: []string{body}}
			_, err := f.model().Request(t.Context(), prompt("Down?"), params)
			var unexpected *ai.UnexpectedModelBehaviorError
			if !errors.As(err, &unexpected) || !strings.Contains(err.Error(), "google decisions: invalid response") {
				t.Fatalf("error=%v", err)
			}
		})
	}
	f := &fake{t: t, replies: []string{`{"answers":{"q0":{"type":"choice","choice":"a","confidence":0.67,
		"probabilities":{"a":0.6700000001,"b":0.3299999999}}},"usage":{}}`}}
	if _, err := f.model().Request(t.Context(), prompt("Down?"), output(`{"type":"string","enum":["a","b"],"description":"Which?"}`)); err != nil {
		t.Fatalf("unrounded distribution rejected: %v", err)
	}
}

func TestErrors(t *testing.T) {
	model := func(roundTrip transport) *googledecisions.Model {
		return googledecisions.NewModel("gemini-decision-0", googledecisions.WithAPIKey("key"),
			googledecisions.WithHTTPClient(&http.Client{Transport: roundTrip}))
	}
	_, err := model(func(*http.Request) (*http.Response, error) {
		return reply(http.StatusNotFound, `{"error":{"message":"Requested model not found.","code":"not_found"}}`), nil
	}).Request(t.Context(), prompt("Server is down"), urgentParams())
	var apiError *googledecisions.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusNotFound || !apiError.IsModelAPIError() ||
		!strings.Contains(err.Error(), `status 404 for "gemini-decision-0"`) {
		t.Fatalf("error=%v", err)
	}
	_, err = model(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("connection reset")
	}).Request(t.Context(), prompt("Down?"), urgentParams())
	var transportError *ai.ModelTransportError
	if !errors.As(err, &transportError) || transportError.Operation != "request" {
		t.Fatalf("error=%v", err)
	}
	_, err = model(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(failingReader{})}, nil
	}).Request(t.Context(), prompt("Down?"), urgentParams())
	if !errors.As(err, &transportError) || transportError.Operation != "read response" {
		t.Fatalf("error=%v", err)
	}
	params := urgentParams()
	params.Settings.ExtraBody = map[string]any{"bad": func() {}}
	if _, err := model(nil).Request(t.Context(), prompt("Down?"), params); err == nil ||
		!strings.Contains(err.Error(), "encode request") {
		t.Fatalf("error=%v", err)
	}
	bad := googledecisions.NewModel("gemini-decision-1", googledecisions.WithBaseURL("://bad"))
	if _, err := bad.Request(t.Context(), prompt("Down?"), urgentParams()); err == nil ||
		!strings.HasPrefix(err.Error(), "google decisions:") {
		t.Fatalf("error=%v", err)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }
