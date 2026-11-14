package bedrockmantle_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/bedrockmantle"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestRoutesInterfacesAndQualifiesToolIDs(t *testing.T) {
	requests := 0
	var replayInput []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		if request.URL.Path != "/openai/v1/responses" {
			t.Errorf("unexpected path %q", request.URL.Path)
		}
		if request.Header.Get("Authorization") != "Bearer token" {
			t.Errorf("unexpected authorization %q", request.Header.Get("Authorization"))
		}
		var body struct {
			Input []map[string]any `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if requests == 2 {
			replayInput = body.Input
			_, _ = io.WriteString(response, `{"id":"resp_2","model":"openai.gpt-5.6-luna","status":"completed","output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}]}`)
			return
		}
		_, _ = io.WriteString(response, `{"id":"resp_1","model":"openai.gpt-5.6-luna","status":"completed","output":[{"id":"fc_1","type":"function_call","name":"lookup","call_id":"call_0","arguments":"{}"}]}`)
	}))
	defer server.Close()

	temperature := 0.5
	defaults := ai.ModelSettings{Temperature: &temperature}
	model := bedrockmantle.NewModel(
		"openai.gpt-5.6-luna", bedrockmantle.WithBaseURL(server.URL+"/v1"),
		bedrockmantle.WithAPIKey("token"), bedrockmantle.WithHTTPClient(server.Client()),
		bedrockmantle.WithDefaultSettings(defaults),
	)
	response, err := model.Request(t.Context(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.ToolCallPart)
	if call.ToolCallID != "call_0_7QfPUJ9y" || response.ProviderName != "bedrock-mantle" ||
		model.Name() != "openai.gpt-5.6-luna" || model.ProviderName() != "bedrock-mantle" ||
		model.ProviderURL() != server.URL+"/openai/v1" || model.ModelProfile().SupportsImageOutput ||
		model.SupportsNativeTool(ai.WebSearchTool{}) || *model.DefaultModelSettings().Temperature != 0.5 {
		t.Fatalf("unexpected model response or contract: %#v", response)
	}
	response.Parts = append(response.Parts, ai.TextPart{Content: "", ProviderName: "bedrock-mantle"})
	history := []ai.ModelMessage{
		*response,
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "lookup", ToolCallID: call.ToolCallID, Content: "result"},
			ai.RetryPromptPart{ToolName: "lookup", ToolCallID: call.ToolCallID, Content: "retry"},
			ai.ToolAvailabilityDeltaPart{ToolCallID: call.ToolCallID, ToolsAdded: []string{"later"}},
		}},
	}
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if history[0].(ai.ModelResponse).Parts[0].(ai.ToolCallPart).ToolCallID != "call_0_7QfPUJ9y" {
		t.Fatal("request preparation mutated caller history")
	}
	if callIDs := replayedCallIDs(replayInput); len(callIDs) != 3 || callIDs[0] != "call_0" || callIDs[2] != "call_0" {
		t.Fatalf("unexpected replay input: %#v", replayInput)
	}

	for name, test := range map[string]struct {
		providerName, providerURL, callID, want string
	}{
		"legacy prefix":  {"bedrock-mantle", "", "resp_1:call_0", "call_0"},
		"other endpoint": {"bedrock-mantle", server.URL + "/v1", "call_0_7QfPUJ9y", "call_0_7QfPUJ9y"},
		"other provider": {"openai", "", "call_0_7QfPUJ9y", "call_0_7QfPUJ9y"},
	} {
		t.Run(name, func(t *testing.T) {
			requests = 1
			history := []ai.ModelMessage{
				ai.ModelResponse{
					Parts:        []ai.ResponsePart{ai.ToolCallPart{ToolName: "lookup", ToolCallID: test.callID, Args: json.RawMessage(`{}`)}},
					ProviderName: test.providerName, ProviderURL: test.providerURL, ProviderResponseID: "resp_1",
				},
				ai.ModelRequest{Parts: []ai.RequestPart{
					ai.ToolReturnPart{ToolName: "lookup", ToolCallID: test.callID, Content: "result"},
				}},
			}
			if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
				t.Fatal(err)
			}
			if callIDs := replayedCallIDs(replayInput); len(callIDs) != 2 || callIDs[0] != test.want || callIDs[1] != test.want {
				t.Fatalf("unexpected replay input: %#v", replayInput)
			}
		})
	}
}

func replayedCallIDs(input []map[string]any) []any {
	var callIDs []any
	for _, item := range input {
		if callID, ok := item["call_id"]; ok {
			callIDs = append(callIDs, callID)
		}
	}
	return callIDs
}

func TestGPTOSSAndSafeguardRouting(t *testing.T) {
	paths := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		paths <- request.URL.Path
		if strings.HasSuffix(request.URL.Path, "/chat/completions") {
			_, _ = io.WriteString(response, `{"id":"chat","model":"openai.gpt-oss-safeguard-20b","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"safe"}}],"usage":{}}`)
			return
		}
		_, _ = io.WriteString(response, `{"id":"response","model":"openai.gpt-oss-120b","status":"completed","output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"oss"}]}]}`)
	}))
	defer server.Close()
	for _, name := range []string{"openai.gpt-oss-120b", "openai.gpt-oss-safeguard-20b"} {
		model := bedrockmantle.NewModel(name, bedrockmantle.WithBaseURL(server.URL+"/openai/v1"),
			bedrockmantle.WithAPIKey("token"), bedrockmantle.WithHTTPClient(server.Client()))
		if _, err := model.Request(t.Context(), promptMessages(), ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
	}
	if first, second := <-paths, <-paths; first != "/v1/responses" || second != "/v1/chat/completions" {
		t.Fatalf("unexpected paths %q and %q", first, second)
	}
}

func TestResponseWithoutIDScopesToolIDByTimestamp(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{"model":"openai.gpt-5.5","status":"completed","output":[{"id":"fc","type":"function_call","name":"lookup","call_id":"call_0","arguments":"{}"}]}`)
	}))
	defer server.Close()
	model := bedrockmantle.NewModel("openai.gpt-5.5", bedrockmantle.WithBaseURL(server.URL),
		bedrockmantle.WithAPIKey("token"), bedrockmantle.WithHTTPClient(server.Client()))
	response, err := model.Request(t.Context(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if call := response.Parts[0].(ai.ToolCallPart); !strings.HasPrefix(call.ToolCallID, "call_0_") ||
		len(call.ToolCallID) != len("call_0_")+8 {
		t.Fatalf("unexpected tool ID: %#v", call)
	}
	encoded, err := ai.MarshalMessages([]ai.ModelMessage{*response})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	var replayed []map[string]any
	server.Config.Handler = http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body struct {
			Input []map[string]any `json:"input"`
		}
		_ = json.NewDecoder(request.Body).Decode(&body)
		replayed = body.Input
		_, _ = io.WriteString(response, `{"model":"openai.gpt-5.5","status":"completed","output":[]}`)
	})
	if _, err := model.Request(t.Context(), decoded, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if callIDs := replayedCallIDs(replayed); len(callIDs) != 1 || callIDs[0] != "call_0" {
		t.Fatalf("unexpected replay input: %#v", replayed)
	}
}

func TestChatStreamPassesThrough(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: [DONE]\n\n")
	}))
	defer server.Close()
	model := bedrockmantle.NewModel("openai.gpt-oss-safeguard-20b", bedrockmantle.WithBaseURL(server.URL),
		bedrockmantle.WithAPIKey("token"), bedrockmantle.WithHTTPClient(server.Client()))
	stream, err := model.StreamRequest(t.Context(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
	}
}

func TestStreamQualifiesToolIDs(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"resp_stream","model":"openai.gpt-5.5","status":"in_progress"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc","type":"function_call","name":"lookup","call_id":"call_0","arguments":"{}"}}`,
		`{"type":"response.completed","response":{"id":"resp_stream","model":"openai.gpt-5.5","status":"completed","output":[{"id":"fc","type":"function_call","name":"lookup","call_id":"call_0","arguments":"{}"}]}}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(response, "data: %s\n\n", event)
		}
		_, _ = io.WriteString(response, "data: [DONE]\n\n")
	}))
	defer server.Close()
	model := bedrockmantle.NewModel("openai.gpt-5.5", bedrockmantle.WithBaseURL(server.URL),
		bedrockmantle.WithAPIKey("token"), bedrockmantle.WithHTTPClient(server.Client()))
	stream, err := model.StreamRequest(t.Context(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	startID, finishID := "", ""
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			startID = event.ToolCallID
		case ai.FinishEvent:
			finishID = event.Parts[0].(ai.ToolCallPart).ToolCallID
		}
	}
	if startID != "call_0_vkLtF0AF" || finishID != "call_0_vkLtF0AF" {
		t.Fatalf("unexpected streamed IDs %q and %q", startID, finishID)
	}
	stream, err = model.StreamRequest(t.Context(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	stream(func(ai.ModelStreamEvent, error) bool { return false })
}

func TestStreamKeepsToolIDsSentBeforeResponseID(t *testing.T) {
	events := []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc","type":"function_call","name":"lookup","call_id":"call_0","arguments":"{}"}}`,
		`{"type":"response.completed","response":{"id":"resp_stream","model":"openai.gpt-5.5","status":"completed","output":[{"id":"fc","type":"function_call","name":"lookup","call_id":"call_0","arguments":"{}"},{"id":"fc_1","type":"function_call","name":"lookup","call_id":"call_1","arguments":"{}"}]}}`,
	}
	failed := true
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if failed {
			response.WriteHeader(http.StatusInternalServerError)
			return
		}
		response.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(response, "data: %s\n\n", event)
		}
		_, _ = io.WriteString(response, "data: [DONE]\n\n")
	}))
	defer server.Close()
	model := bedrockmantle.NewModel("openai.gpt-5.5", bedrockmantle.WithBaseURL(server.URL),
		bedrockmantle.WithAPIKey("token"), bedrockmantle.WithHTTPClient(server.Client()))
	if _, err := model.StreamRequest(t.Context(), promptMessages(), ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected stream open error")
	}
	failed = false
	stream, err := model.StreamRequest(t.Context(), promptMessages(), ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var finishIDs []string
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if finish, ok := event.(ai.FinishEvent); ok {
			for _, part := range finish.Parts {
				finishIDs = append(finishIDs, part.(ai.ToolCallPart).ToolCallID)
			}
		}
	}
	if len(finishIDs) != 2 || finishIDs[0] != "call_0" || finishIDs[1] != "call_1_vkLtF0AF" {
		t.Fatalf("unexpected streamed IDs %q", finishIDs)
	}
}

func TestBearerProviderAndSigV4Authentication(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_REGION", "")
	requests := make(chan *http.Request, 2)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests <- request.Clone(request.Context())
		_, _ = io.WriteString(response, `{"id":"response","model":"openai.gpt-5.4","status":"completed","output":[]}`)
	}))
	defer server.Close()
	provider := openai.ProviderConfig{
		BaseURL: server.URL + "/v1", APIKey: "gateway", HTTPClient: server.Client(),
		Headers: http.Header{"X-Gateway": []string{"yes"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "yes")
			return nil
		},
	}
	gateway := bedrockmantle.NewModel("openai.gpt-5.4", bedrockmantle.WithProvider(provider))
	if _, err := gateway.Request(t.Context(), promptMessages(), ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	gatewayRequest := <-requests
	if gatewayRequest.Header.Get("Authorization") != "Bearer gateway" ||
		gatewayRequest.Header.Get("X-Gateway") != "yes" || gatewayRequest.Header.Get("X-Dynamic") != "yes" {
		t.Fatalf("unexpected gateway headers: %#v", gatewayRequest.Header)
	}
	loaded := aws.Config{
		Region: "us-east-1", Credentials: credentials.NewStaticCredentialsProvider("access", "secret", "session"),
	}
	signed := bedrockmantle.NewModel("openai.gpt-5.4", bedrockmantle.WithBaseURL(server.URL),
		bedrockmantle.WithHTTPClient(server.Client()), bedrockmantle.WithAWSConfig(loaded))
	if _, err := signed.Request(t.Context(), promptMessages(), ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	signedRequest := <-requests
	if !strings.HasPrefix(signedRequest.Header.Get("Authorization"), "AWS4-HMAC-SHA256 ") ||
		signedRequest.Header.Get("X-Amz-Security-Token") != "session" {
		t.Fatalf("request was not signed: %#v", signedRequest.Header)
	}
}

func TestDefaultAWSAuthenticationAndErrors(t *testing.T) {
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_REGION", "")
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{"id":"response","model":"openai.gpt-5.4","status":"completed","output":[]}`)
	}))
	defer server.Close()
	model := bedrockmantle.NewModel(
		"global.openai.gpt-5.4", bedrockmantle.WithBaseURL(server.URL),
		bedrockmantle.WithHTTPClient(server.Client()), bedrockmantle.WithAWSLoadOptions(
			awsconfig.WithRegion("us-east-1"),
			awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("access", "secret", "")),
		),
	)
	if _, err := model.Request(t.Context(), promptMessages(), ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}

	failure := errors.New("failed")
	models := []*bedrockmantle.Model{
		bedrockmantle.NewModel(
			"openai.gpt-5.4", bedrockmantle.WithBaseURL(server.URL), bedrockmantle.WithRegion("us-east-1"),
			bedrockmantle.WithHTTPClient(server.Client()),
			bedrockmantle.WithAWSLoadOptions(func(*awsconfig.LoadOptions) error { return failure }),
		),
		bedrockmantle.NewModel(
			"openai.gpt-5.4", bedrockmantle.WithBaseURL(server.URL), bedrockmantle.WithHTTPClient(server.Client()),
			bedrockmantle.WithAWSConfig(aws.Config{
				Region: "us-east-1", Credentials: aws.CredentialsProviderFunc(
					func(context.Context) (aws.Credentials, error) { return aws.Credentials{}, failure },
				),
			}),
		),
	}
	for _, model := range models {
		if _, err := model.Request(t.Context(), promptMessages(), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected AWS authentication error")
		}
	}
}

func TestRegionEndpoint(t *testing.T) {
	model := bedrockmantle.NewModel(
		"openai.gpt-5.4", bedrockmantle.WithRegion("eu-west-1"), bedrockmantle.WithAPIKey("token"),
	)
	if model.ProviderURL() != "https://bedrock-mantle.eu-west-1.api.aws/openai/v1" {
		t.Fatalf("unexpected region URL %q", model.ProviderURL())
	}
}

func TestUnsupportedOutputAndNativeTools(t *testing.T) {
	model := bedrockmantle.NewModel(
		"openai.gpt-5.4", bedrockmantle.WithRegion("us-east-1"), bedrockmantle.WithAPIKey("token"),
	)
	for _, params := range []ai.ModelRequestParams{
		{AllowImageOutput: true},
		{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}},
	} {
		if _, err := model.Request(t.Context(), nil, params); err == nil {
			t.Fatal("expected unsupported feature error")
		}
		if _, err := model.StreamRequest(t.Context(), nil, params); err == nil {
			t.Fatal("expected streamed unsupported feature error")
		}
	}
}

func TestConfigurationErrors(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	for _, model := range []*bedrockmantle.Model{
		bedrockmantle.NewModel("gpt5"),
		bedrockmantle.NewModel("anthropic.claude"),
		bedrockmantle.NewModel("openai.gpt-5.5"),
	} {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected configuration error")
		}
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected streamed configuration error")
		}
	}
}

func promptMessages() []ai.ModelMessage {
	return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}
}
