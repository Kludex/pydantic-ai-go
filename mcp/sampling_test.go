package mcp_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
	"github.com/google/jsonschema-go/jsonschema"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func inputRequestServer(
	name string,
	requests mcpsdk.InputRequestMap,
	finish func(mcpsdk.InputResponseMap) (*mcpsdk.CallToolResult, error),
) *mcpsdk.Server {
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: name, Version: "1"}, nil)
	server.AddTool(&mcpsdk.Tool{Name: "request", InputSchema: map[string]any{"type": "object"}}, func(
		_ context.Context, request *mcpsdk.CallToolRequest,
	) (*mcpsdk.CallToolResult, error) {
		if len(request.Params.InputResponses) == 0 {
			return &mcpsdk.CallToolResult{InputRequests: requests, RequestState: "waiting"}, nil
		}
		return finish(request.Params.InputResponses)
	})
	return server
}

func TestSamplingModelAndElicitationRoundTrip(t *testing.T) {
	modelCalls := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		modelCalls++
		if len(messages) != 3 {
			t.Fatalf("unexpected sampling messages: %#v", messages)
		}
		first := messages[0].(ai.ModelRequest)
		if len(first.Parts) != 3 || first.Parts[0].(ai.SystemPromptPart).Content != "Be concise." ||
			first.Parts[1].(ai.UserPromptPart).Content != "Describe this." {
			t.Fatalf("unexpected first sampling request: %#v", first)
		}
		image := first.Parts[2].(ai.UserPromptPart).Contents[0].(ai.BinaryContent)
		if string(image.Data) != "image" || image.MediaType != "image/png" {
			t.Fatalf("unexpected sampling image: %+v", image)
		}
		if messages[1].(ai.ModelResponse).Text() != "Earlier answer." {
			t.Fatalf("unexpected sampling response history: %#v", messages[1])
		}
		audio := messages[2].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents[0].(ai.BinaryContent)
		if string(audio.Data) != "audio" || audio.MediaType != "audio/wav" {
			t.Fatalf("unexpected sampling audio: %+v", audio)
		}
		if params.Settings.MaxTokens != 42 || params.Settings.Temperature == nil ||
			*params.Settings.Temperature != 0.25 || len(params.Settings.StopSequences) != 1 ||
			params.Settings.StopSequences[0] != "STOP" {
			t.Fatalf("unexpected sampling settings: %+v", params.Settings)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ThinkingPart{Content: "private"},
			ai.TextPart{Content: "Sampled "},
			ai.TextPart{Content: "answer."},
		}}, nil
	})

	elicitCalls := 0
	samplingParams := &mcpsdk.CreateMessageParams{
		SystemPrompt: "Be concise.", MaxTokens: 42, Temperature: 0.25, StopSequences: []string{"STOP"},
		Messages: []*mcpsdk.SamplingMessage{
			{Role: mcpsdk.Role("user"), Content: &mcpsdk.TextContent{Text: "Describe this."}},
			{Role: mcpsdk.Role("user"), Content: &mcpsdk.ImageContent{Data: []byte("image"), MIMEType: "image/png"}},
			{Role: mcpsdk.Role("assistant"), Content: &mcpsdk.TextContent{Text: "Earlier answer."}},
			{Role: mcpsdk.Role("user"), Content: &mcpsdk.AudioContent{Data: []byte("audio"), MIMEType: "audio/wav"}},
		},
	}
	server := inputRequestServer("sampling", mcpsdk.InputRequestMap{
		"sample": samplingParams,
		"identify": &mcpsdk.ElicitParams{
			Message: "Who are you?",
			RequestedSchema: &jsonschema.Schema{
				Type: "object", Properties: map[string]*jsonschema.Schema{"name": {Type: "string"}},
				Required: []string{"name"},
			},
		},
	}, func(responses mcpsdk.InputResponseMap) (*mcpsdk.CallToolResult, error) {
		sampled := responses["sample"].(*mcpsdk.CreateMessageWithToolsResult)
		elicited := responses["identify"].(*mcpsdk.ElicitResult)
		if sampled.Role != mcpsdk.Role("assistant") || sampled.Model != model.Name() || len(sampled.Content) != 1 ||
			elicited.Action != "accept" || elicited.Content["name"] != "Ada" {
			t.Fatalf("unexpected input responses: %+v", responses)
		}
		return &mcpsdk.CallToolResult{Content: sampled.Content}, nil
	})
	session := connectSession(t, server,
		aimcp.WithSamplingModel(model),
		aimcp.WithElicitationHandler(func(
			_ context.Context, request *aimcp.ElicitationRequest,
		) (*aimcp.ElicitationResult, error) {
			elicitCalls++
			if request.Params.Message != "Who are you?" {
				t.Fatalf("unexpected elicitation: %+v", request.Params)
			}
			request.Params.Message = "mutated detached request"
			return &aimcp.ElicitationResult{Action: "accept", Content: map[string]any{"name": "Ada"}}, nil
		}),
	)
	result, err := session.CallTool(t.Context(), "request", map[string]any{})
	if err != nil || result.Content[0].(*mcpsdk.TextContent).Text != "Sampled answer." ||
		modelCalls != 1 || elicitCalls != 1 {
		t.Fatalf("unexpected sampling result: %+v model=%d elicit=%d err=%v", result, modelCalls, elicitCalls, err)
	}
	if samplingParams.SystemPrompt != "Be concise." ||
		samplingParams.Messages[0].Content.(*mcpsdk.TextContent).Text != "Describe this." {
		t.Fatal("sampling handler mutated server parameters")
	}
}

func TestCustomSamplingHandlerIsDetached(t *testing.T) {
	var ownedResult *aimcp.SamplingResult
	params := &mcpsdk.CreateMessageParams{SystemPrompt: "original", Messages: []*mcpsdk.SamplingMessage{}}
	server := inputRequestServer("custom", mcpsdk.InputRequestMap{"sample": params}, func(
		responses mcpsdk.InputResponseMap,
	) (*mcpsdk.CallToolResult, error) {
		result := responses["sample"].(*mcpsdk.CreateMessageWithToolsResult)
		result.Content[0].(*mcpsdk.TextContent).Text = "changed by server"
		return &mcpsdk.CallToolResult{}, nil
	})
	session := connectSession(t, server, aimcp.WithSamplingHandler(func(
		_ context.Context, request *aimcp.SamplingRequest,
	) (*aimcp.SamplingResult, error) {
		request.Params.SystemPrompt = "changed by handler"
		ownedResult = &aimcp.SamplingResult{
			Role: mcpsdk.Role("assistant"), Model: "custom", Content: &mcpsdk.TextContent{Text: "owned"},
		}
		return ownedResult, nil
	}))
	if _, err := session.CallTool(t.Context(), "request", map[string]any{}); err != nil {
		t.Fatal(err)
	}
	if params.SystemPrompt != "original" {
		t.Fatalf("sampling handler mutated server parameters: %+v", params)
	}
	if ownedResult.Content.(*mcpsdk.TextContent).Text != "owned" {
		t.Fatal("sampling result returned to server aliased handler state")
	}
}

func TestServerInitiatedHandlerErrors(t *testing.T) {
	handlerError := errors.New("handler failed")
	tests := []struct {
		name    string
		request mcpsdk.InputRequest
		options []aimcp.Option
	}{
		{
			name: "sampling error", request: &mcpsdk.CreateMessageParams{Messages: []*mcpsdk.SamplingMessage{}},
			options: []aimcp.Option{aimcp.WithSamplingHandler(func(
				context.Context, *aimcp.SamplingRequest,
			) (*aimcp.SamplingResult, error) {
				return nil, handlerError
			})},
		},
		{
			name: "sampling nil result", request: &mcpsdk.CreateMessageParams{Messages: []*mcpsdk.SamplingMessage{}},
			options: []aimcp.Option{aimcp.WithSamplingHandler(func(
				context.Context, *aimcp.SamplingRequest,
			) (*aimcp.SamplingResult, error) {
				return nil, nil
			})},
		},
		{
			name: "elicitation error", request: &mcpsdk.ElicitParams{Message: "Confirm?"},
			options: []aimcp.Option{aimcp.WithElicitationHandler(func(
				context.Context, *aimcp.ElicitationRequest,
			) (*aimcp.ElicitationResult, error) {
				return nil, handlerError
			})},
		},
		{
			name: "elicitation nil result", request: &mcpsdk.ElicitParams{Message: "Confirm?"},
			options: []aimcp.Option{aimcp.WithElicitationHandler(func(
				context.Context, *aimcp.ElicitationRequest,
			) (*aimcp.ElicitationResult, error) {
				return nil, nil
			})},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := inputRequestServer(test.name, mcpsdk.InputRequestMap{"input": test.request}, func(
				mcpsdk.InputResponseMap,
			) (*mcpsdk.CallToolResult, error) {
				t.Fatal("failed handler unexpectedly returned a response")
				return nil, nil
			})
			session := connectSession(t, server, test.options...)
			if _, err := session.CallTool(t.Context(), "request", map[string]any{}); err == nil {
				t.Fatal("server-initiated handler failure was ignored")
			}
		})
	}
}

type valueSamplingModel struct{}

func (valueSamplingModel) Name() string { return "value" }

func (valueSamplingModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "value"}}}, nil
}

func TestSamplingErrorsAndOptionValidation(t *testing.T) {
	modelError := errors.New("model failed")
	failingModel := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, modelError
	})
	unsupportedModel := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "bad"}}}, nil
	})

	tests := []struct {
		name   string
		model  ai.Model
		params *mcpsdk.CreateMessageParams
		want   string
	}{
		{name: "role", model: fakes.NewTestModel(), params: &mcpsdk.CreateMessageParams{
			Messages: []*mcpsdk.SamplingMessage{{Role: "admin", Content: &mcpsdk.TextContent{Text: "bad"}}},
		}, want: `unsupported sampling role "admin"`},
		{name: "user content", model: fakes.NewTestModel(), params: &mcpsdk.CreateMessageParams{
			Messages: []*mcpsdk.SamplingMessage{{Role: "user", Content: &mcpsdk.ToolUseContent{}}},
		}, want: "unsupported user sampling content"},
		{name: "assistant content", model: fakes.NewTestModel(), params: &mcpsdk.CreateMessageParams{
			Messages: []*mcpsdk.SamplingMessage{{Role: "assistant", Content: &mcpsdk.ImageContent{}}},
		}, want: "unsupported assistant sampling content"},
		{name: "negative tokens", model: fakes.NewTestModel(), params: &mcpsdk.CreateMessageParams{
			MaxTokens: -1, Messages: []*mcpsdk.SamplingMessage{},
		}, want: "outside the supported range"},
		{name: "model", model: failingModel, params: &mcpsdk.CreateMessageParams{
			Messages: []*mcpsdk.SamplingMessage{},
		}, want: "model failed"},
		{name: "response content", model: unsupportedModel, params: &mcpsdk.CreateMessageParams{
			Messages: []*mcpsdk.SamplingMessage{},
		}, want: "sampling model returned unsupported part"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := inputRequestServer(test.name, mcpsdk.InputRequestMap{"sample": test.params}, func(
				mcpsdk.InputResponseMap,
			) (*mcpsdk.CallToolResult, error) {
				t.Fatal("sampling error unexpectedly returned a response")
				return nil, nil
			})
			session := connectSession(t, server, aimcp.WithSamplingModel(test.model))
			_, err := session.CallTool(t.Context(), "request", map[string]any{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected sampling error: %v", err)
			}
		})
	}

	factory := func(context.Context, *ai.RunContext[struct{}]) (mcpsdk.Transport, error) { return nil, nil }
	aimcp.NewToolset[struct{}](factory, aimcp.WithClientOptions(nil), aimcp.WithSamplingModel(valueSamplingModel{}))
	aimcp.NewToolset[struct{}](factory, aimcp.WithClientOptions(&mcpsdk.ClientOptions{
		MultiRoundTrip: &mcpsdk.MultiRoundTripOptions{},
	}))

	assertMCPPanic(t, "nil sampling model", "sampling model must not be nil", func() {
		aimcp.WithSamplingModel(nil)
	})
	assertMCPPanic(t, "typed nil sampling model", "sampling model must not be nil", func() {
		var model *fakes.TestModel
		aimcp.WithSamplingModel(model)
	})
	assertMCPPanic(t, "nil sampling handler", "sampling handler must not be nil", func() {
		aimcp.WithSamplingHandler(nil)
	})
	assertMCPPanic(t, "nil elicitation handler", "elicitation handler must not be nil", func() {
		aimcp.WithElicitationHandler(nil)
	})

	dummySampling := func(context.Context, *mcpsdk.CreateMessageRequest) (*mcpsdk.CreateMessageResult, error) {
		return nil, nil
	}
	dummySamplingWithTools := func(
		context.Context, *mcpsdk.CreateMessageWithToolsRequest,
	) (*mcpsdk.CreateMessageWithToolsResult, error) {
		return nil, nil
	}
	dummyElicitation := func(context.Context, *mcpsdk.ElicitRequest) (*mcpsdk.ElicitResult, error) {
		return nil, nil
	}
	conflicts := []struct {
		name    string
		options []aimcp.Option
		want    string
	}{
		{
			name: "client sampling handlers", options: []aimcp.Option{aimcp.WithClientOptions(&mcpsdk.ClientOptions{
				CreateMessageHandler: dummySampling, CreateMessageWithToolsHandler: dummySamplingWithTools,
			})}, want: "cannot set both sampling handlers",
		},
		{
			name: "model and convenience handler", options: []aimcp.Option{
				aimcp.WithSamplingModel(fakes.NewTestModel()), aimcp.WithSamplingHandler(func(
					context.Context, *aimcp.SamplingRequest,
				) (*aimcp.SamplingResult, error) {
					return nil, nil
				}),
			}, want: "sampling model and sampling handler cannot both be set",
		},
		{
			name: "model and client handler", options: []aimcp.Option{
				aimcp.WithSamplingModel(fakes.NewTestModel()),
				aimcp.WithClientOptions(&mcpsdk.ClientOptions{CreateMessageHandler: dummySampling}),
			}, want: "sampling model and sampling handler cannot both be set",
		},
		{
			name: "duplicate sampling handler", options: []aimcp.Option{
				aimcp.WithSamplingHandler(func(context.Context, *aimcp.SamplingRequest) (*aimcp.SamplingResult, error) {
					return nil, nil
				}), aimcp.WithClientOptions(&mcpsdk.ClientOptions{CreateMessageHandler: dummySampling}),
			}, want: "sampling handler is already set",
		},
		{
			name: "duplicate elicitation handler", options: []aimcp.Option{
				aimcp.WithElicitationHandler(func(
					context.Context, *aimcp.ElicitationRequest,
				) (*aimcp.ElicitationResult, error) {
					return nil, nil
				}),
				aimcp.WithClientOptions(&mcpsdk.ClientOptions{ElicitationHandler: dummyElicitation}),
			}, want: "elicitation handler is already set",
		},
	}
	for _, conflict := range conflicts {
		assertMCPPanic(t, conflict.name, conflict.want, func() {
			aimcp.NewToolset[struct{}](func(
				context.Context, *ai.RunContext[struct{}],
			) (mcpsdk.Transport, error) {
				return nil, nil
			}, conflict.options...)
		})
	}
}

func assertMCPPanic(t *testing.T, name, want string, call func()) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		defer func() {
			value := recover()
			if value == nil || !strings.Contains(value.(string), want) {
				t.Fatalf("unexpected panic: %v", value)
			}
		}()
		call()
	})
}
