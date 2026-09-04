//nolint:staticcheck // MCP sampling remains supported during its protocol deprecation window.
package mcp_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

type fakeSamplingSession struct {
	createMessage          func(context.Context, *mcpsdk.CreateMessageParams) (*mcpsdk.CreateMessageResult, error)
	createMessageWithTools func(
		context.Context, *mcpsdk.CreateMessageWithToolsParams,
	) (*mcpsdk.CreateMessageWithToolsResult, error)
}

func (session fakeSamplingSession) CreateMessage(
	ctx context.Context, params *mcpsdk.CreateMessageParams,
) (*mcpsdk.CreateMessageResult, error) {
	return session.createMessage(ctx, params)
}

func (session fakeSamplingSession) CreateMessageWithTools(
	ctx context.Context, params *mcpsdk.CreateMessageWithToolsParams,
) (*mcpsdk.CreateMessageWithToolsResult, error) {
	return session.createMessageWithTools(ctx, params)
}

func TestSamplingServerModelWithoutTools(t *testing.T) {
	temperature := 0.3
	preferences := &aimcp.ModelPreferences{
		CostPriority: 0.8,
		Hints:        []*mcpsdk.ModelHint{{Name: "fast-model"}},
	}
	settings, err := (aimcp.SamplingSettings{
		Common: ai.ModelSettings{
			Temperature: &temperature, StopSequences: []string{"STOP"}, ExtraBody: map[string]any{"custom": true},
		},
		ModelPreferences: preferences,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	preferences.Hints[0].Name = "mutated"

	samplingCalls := 0
	serverSession := &fakeSamplingSession{createMessage: func(
		_ context.Context, params *mcpsdk.CreateMessageParams,
	) (*mcpsdk.CreateMessageResult, error) {
		samplingCalls++
		if params.MaxTokens != 99 || params.SystemPrompt != "Current system.\n\nStored system." ||
			params.Temperature != 0.3 || len(params.StopSequences) != 1 || params.StopSequences[0] != "STOP" ||
			params.ModelPreferences.Hints[0].Name != "fast-model" || len(params.Messages) != 6 {
			t.Fatalf("unexpected sampling params: %+v", params)
		}
		if params.Messages[0].Content.(*mcpsdk.TextContent).Text != "Describe the files." ||
			params.Messages[1].Content.(*mcpsdk.TextContent).Text != "Context" ||
			string(params.Messages[2].Content.(*mcpsdk.ImageContent).Data) != "image" ||
			string(params.Messages[3].Content.(*mcpsdk.AudioContent).Data) != "audio" ||
			params.Messages[4].Content.(*mcpsdk.TextContent).Text != "Earlier." ||
			string(params.Messages[5].Content.(*mcpsdk.ImageContent).Data) != "old image" {
			t.Fatalf("unexpected sampling messages: %+v", params.Messages)
		}
		if samplingCalls == 1 {
			return &mcpsdk.CreateMessageResult{
				Role: "assistant", Model: "client-model", StopReason: "maxTokens", Meta: mcpsdk.Meta{"trace": "one"},
				Content: &mcpsdk.ImageContent{Data: []byte("generated"), MIMEType: "image/png"},
			}, nil
		}
		return &mcpsdk.CreateMessageResult{
			Role: "assistant", Model: "client-model", StopReason: "stopSequence",
			Content: &mcpsdk.AudioContent{Data: []byte("speech"), MIMEType: "audio/wav"},
		}, nil
	}}
	model := aimcp.NewSamplingModel(serverSession, aimcp.WithSamplingDefaultMaxTokens(99))
	if model.Name() != "mcp-sampling" || model.ProviderName() != "mcp" || model.ProviderURL() != "" {
		t.Fatalf("unexpected sampling identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "Stored system."},
			ai.UserPromptPart{Content: "Describe the files."},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "Context"},
				ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
				ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/wav"},
				ai.CachePoint{},
			}},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"later"}},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ThinkingPart{Content: "private"}, ai.CompactionPart{Content: "summary"},
			ai.TextPart{Content: "Earlier."},
			ai.FilePart{Content: ai.BinaryContent{Data: []byte("old image"), MediaType: "image/jpeg"}},
		}},
	}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		Instructions: "Current system.", Settings: settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	file := response.Parts[0].(ai.FilePart)
	if response.ModelName != "client-model" || response.ProviderName != "mcp" ||
		response.FinishReason != ai.FinishReasonLength || string(file.Content.Data) != "generated" ||
		file.Content.MediaType != "image/png" || response.ProviderDetails["mcp_stop_reason"] != "maxTokens" {
		t.Fatalf("unexpected sampled response: %+v", response)
	}
	response, err = model.Request(t.Context(), messages, ai.ModelRequestParams{
		Instructions: "Current system.", Settings: settings,
	})
	if err != nil || response.FinishReason != ai.FinishReasonStop ||
		string(response.Parts[0].(ai.FilePart).Content.Data) != "speech" || samplingCalls != 2 {
		t.Fatalf("unexpected audio response=%+v calls=%d err=%v", response, samplingCalls, err)
	}
}

func TestSamplingServerModelRunsAgentTools(t *testing.T) {
	calls := 0
	serverSession := &fakeSamplingSession{createMessageWithTools: func(
		_ context.Context, params *mcpsdk.CreateMessageWithToolsParams,
	) (*mcpsdk.CreateMessageWithToolsResult, error) {
		calls++
		if len(params.Tools) != 1 || params.Tools[0].Name != "lookup" ||
			params.Tools[0].Description != "Look up the answer" || params.ToolChoice.Mode != "auto" {
			t.Fatalf("unexpected sampled tools: %+v", params)
		}
		if calls == 1 {
			return &mcpsdk.CreateMessageWithToolsResult{
				Role: "assistant", Model: "client", StopReason: "toolUse", Content: []mcpsdk.Content{
					&mcpsdk.ToolUseContent{ID: "lookup-1", Name: "lookup", Input: map[string]any{}},
				},
			}, nil
		}
		if len(params.Messages) != 3 {
			t.Fatalf("unexpected tool history: %+v", params.Messages)
		}
		toolUse := params.Messages[1].Content[0].(*mcpsdk.ToolUseContent)
		toolResult := params.Messages[2].Content[0].(*mcpsdk.ToolResultContent)
		if toolUse.ID != "lookup-1" || toolResult.ToolUseID != "lookup-1" || toolResult.IsError ||
			toolResult.Content[0].(*mcpsdk.TextContent).Text != "tool result" {
			t.Fatalf("unexpected sampled tool history: %+v", params.Messages)
		}
		return &mcpsdk.CreateMessageWithToolsResult{
			Role: "assistant", Model: "client", StopReason: "endTurn",
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "agent done"}},
		}, nil
	}}
	agent := ai.NewAgent[struct{}, string](aimcp.NewSamplingModel(serverSession))
	ai.AddSimpleTool(agent, "lookup", func(context.Context, struct{}) (string, error) {
		return "tool result", nil
	}, ai.WithDescription("Look up the answer"))
	result, err := agent.Run(t.Context(), "Use lookup.", struct{}{})
	if err != nil || result.Output != "agent done" || result.Usage().ToolCalls != 1 || calls != 2 {
		t.Fatalf("unexpected agent result=%+v calls=%d err=%v", result, calls, err)
	}
}

func TestSamplingServerModelStructuredOutput(t *testing.T) {
	type answer struct {
		Value string `json:"value"`
	}
	serverSession := &fakeSamplingSession{createMessageWithTools: func(
		_ context.Context, params *mcpsdk.CreateMessageWithToolsParams,
	) (*mcpsdk.CreateMessageWithToolsResult, error) {
		if len(params.Tools) != 1 || params.ToolChoice.Mode != "required" {
			t.Fatalf("unexpected output tool request: %+v", params)
		}
		if params.Tools[0].Name == "direct" && (params.Tools[0].OutputSchema == nil || params.Temperature != 0.4) {
			t.Fatalf("missing direct output schema or temperature: %+v", params)
		}
		return &mcpsdk.CreateMessageWithToolsResult{
			Role: "assistant", Model: "client", StopReason: "toolUse", Content: []mcpsdk.Content{
				&mcpsdk.ToolUseContent{
					ID: "final-1", Name: params.Tools[0].Name, Input: map[string]any{"value": "done"},
				},
			},
		}, nil
	}}
	model := aimcp.NewSamplingModel(serverSession)
	includeReturn := true
	temperature := 0.4
	direct, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{
			Name: "direct", Schema: map[string]any{"type": "object"}, ReturnSchema: map[string]any{"type": "object"},
			IncludeReturnSchema: &includeReturn,
		},
		Settings: ai.ModelSettings{Temperature: &temperature},
	})
	if err != nil || direct.Parts[0].(ai.ToolCallPart).ToolName != "direct" {
		t.Fatalf("unexpected direct structured response=%+v err=%v", direct, err)
	}
	agent := ai.NewAgent[struct{}, answer](model)
	result, err := agent.Run(t.Context(), "Answer.", struct{}{})
	if err != nil || result.Output.Value != "done" {
		t.Fatalf("unexpected structured result=%+v err=%v", result, err)
	}
}

func TestSamplingServerModelHistoryVariants(t *testing.T) {
	calls := 0
	session := &fakeSamplingSession{createMessageWithTools: func(
		_ context.Context, params *mcpsdk.CreateMessageWithToolsParams,
	) (*mcpsdk.CreateMessageWithToolsResult, error) {
		calls++
		if len(params.Messages) != 2 {
			t.Fatalf("unexpected history: %+v", params.Messages)
		}
		requestContent := params.Messages[0].Content
		if requestContent[0].(*mcpsdk.TextContent).Text != "retry plain" ||
			requestContent[1].(*mcpsdk.ToolResultContent).ToolUseID != "call-1" ||
			!requestContent[1].(*mcpsdk.ToolResultContent).IsError ||
			string(requestContent[2].(*mcpsdk.ToolResultContent).Content[0].(*mcpsdk.ImageContent).Data) != "image" ||
			requestContent[3].(*mcpsdk.ToolResultContent).StructuredContent.(map[string]any)["ok"] != true {
			t.Fatalf("unexpected request history: %+v", requestContent)
		}
		return &mcpsdk.CreateMessageWithToolsResult{
			Role: "assistant", Model: "client", Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "done"}},
		}, nil
	}}
	model := aimcp.NewSamplingModel(session)
	response, err := model.Request(t.Context(), []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{Content: "retry plain"},
			ai.RetryPromptPart{Content: "retry tool", ToolCallID: "call-1"},
			ai.ToolReturnPart{
				ToolCallID: "call-2", Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
			},
			ai.ToolReturnPart{ToolCallID: "call-3", Content: map[string]any{"ok": true}, Outcome: ai.ToolReturnOutcomeFailed},
		}},
		ai.ModelRequest{},
		ai.ModelResponse{},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "earlier"}}},
	}, ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}}})
	if err != nil || response.Text() != "done" || calls != 1 {
		t.Fatalf("unexpected history response=%+v calls=%d err=%v", response, calls, err)
	}
}

func TestSamplingServerModelErrors(t *testing.T) {
	clientCalls := 0
	serverSession := &fakeSamplingSession{createMessage: func(
		context.Context, *mcpsdk.CreateMessageParams,
	) (*mcpsdk.CreateMessageResult, error) {
		clientCalls++
		switch clientCalls {
		case 1:
			return &mcpsdk.CreateMessageResult{
				Role: "user", Model: "client", Content: &mcpsdk.TextContent{Text: "bad"},
			}, nil
		case 2:
			return &mcpsdk.CreateMessageResult{
				Role: "assistant", Model: "client", StopReason: "unknown",
				Content: &mcpsdk.ResourceLink{URI: "file:///bad", Name: "bad"},
			}, nil
		default:
			return nil, errors.New("client failed")
		}
	}}
	model := aimcp.NewSamplingModel(serverSession)
	invalidRequests := []struct {
		messages []ai.ModelMessage
		params   ai.ModelRequestParams
		want     string
	}{
		{params: ai.ModelRequestParams{Settings: ai.ModelSettings{MaxTokens: -1}}, want: "must not be negative"},
		{params: ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{
			"mcp_model_preferences": "bad",
		}}}, want: "must be a non-nil"},
		{messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
			Contents: []ai.UserContent{ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"}},
		}}}}, want: "binary media type"},
		{messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
			Contents: []ai.UserContent{ai.ImageURL{URL: "https://example.com/image.png"}},
		}}}}, want: "unsupported sampling user content"},
		{messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{}}}},
			want: "unsupported sampling request part"},
		{messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{
			Content: ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"},
		}}}}, want: "binary media type"},
		{messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.NativeToolCallPart{
			ToolName: "native",
		}}}}, want: "unsupported sampling response part"},
		{messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "bad", Args: json.RawMessage(`{`),
		}}}}, want: "decode tool call"},
		{messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
			ToolCallID: "bad", Content: make(chan int),
		}}}}, want: "encode tool result"},
		{messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
			ToolCallID: "bad-binary", Content: ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"},
		}}}}, want: "binary media type"},
	}
	for _, test := range invalidRequests {
		if _, err := model.Request(t.Context(), test.messages, test.params); err == nil ||
			!strings.Contains(err.Error(), test.want) {
			t.Fatalf("expected %q, got %v", test.want, err)
		}
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected invalid-role error")
	} else {
		var behavior *ai.UnexpectedModelBehaviorError
		if !errors.As(err, &behavior) || !strings.Contains(err.Error(), "role") {
			t.Fatalf("unexpected role error: %v", err)
		}
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected unsupported-content error")
	} else {
		var behavior *ai.UnexpectedModelBehaviorError
		if !errors.As(err, &behavior) || !strings.Contains(err.Error(), "unsupported content") {
			t.Fatalf("unexpected content error: %v", err)
		}
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected transport error")
	} else {
		var transport *ai.ModelTransportError
		if !errors.As(err, &transport) || transport.ProviderName != "mcp" {
			t.Fatalf("unexpected transport error: %v", err)
		}
	}
	toolParams := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{}}}}
	toolTransportModel := aimcp.NewSamplingModel(&fakeSamplingSession{createMessageWithTools: func(
		context.Context, *mcpsdk.CreateMessageWithToolsParams,
	) (*mcpsdk.CreateMessageWithToolsResult, error) {
		return nil, errors.New("tool client failed")
	}})
	if _, err := toolTransportModel.Request(t.Context(), nil, toolParams); err == nil {
		t.Fatal("expected tool transport error")
	} else {
		var transport *ai.ModelTransportError
		if !errors.As(err, &transport) || transport.Operation != "sampling request with tools" {
			t.Fatalf("unexpected tool transport error: %v", err)
		}
	}
	nilResultModel := aimcp.NewSamplingModel(&fakeSamplingSession{
		createMessage: func(context.Context, *mcpsdk.CreateMessageParams) (*mcpsdk.CreateMessageResult, error) {
			return nil, nil
		},
		createMessageWithTools: func(
			context.Context, *mcpsdk.CreateMessageWithToolsParams,
		) (*mcpsdk.CreateMessageWithToolsResult, error) {
			return nil, nil
		},
	})
	if _, err := nilResultModel.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected empty sampling result error")
	}
	if _, err := nilResultModel.Request(t.Context(), nil, toolParams); err == nil {
		t.Fatal("expected empty tool sampling result error")
	}
	badToolModel := aimcp.NewSamplingModel(&fakeSamplingSession{createMessageWithTools: func(
		context.Context, *mcpsdk.CreateMessageWithToolsParams,
	) (*mcpsdk.CreateMessageWithToolsResult, error) {
		return &mcpsdk.CreateMessageWithToolsResult{
			Role: "assistant", Content: []mcpsdk.Content{&mcpsdk.ToolUseContent{
				Name: "bad", Input: map[string]any{"channel": make(chan int)},
			}},
		}, nil
	}})
	if _, err := badToolModel.Request(t.Context(), nil, toolParams); err == nil ||
		!strings.Contains(err.Error(), "encode sampled tool call") {
		t.Fatalf("unexpected sampled tool encoding error: %v", err)
	}

	var nilModel *aimcp.SamplingModel
	if _, err := nilModel.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected disconnected model error")
	}
	if _, err := (aimcp.SamplingSettings{
		Common:           ai.ModelSettings{ExtraBody: map[string]any{"mcp_model_preferences": true}},
		ModelPreferences: &aimcp.ModelPreferences{},
	}).Build(); err == nil {
		t.Fatal("expected settings conflict")
	}
	if settings, err := (aimcp.SamplingSettings{Common: ai.ModelSettings{MaxTokens: 1}}).Build(); err != nil || settings.MaxTokens != 1 {
		t.Fatalf("unexpected common settings: %+v %v", settings, err)
	}
	if settings, err := (aimcp.SamplingSettings{ModelPreferences: &aimcp.ModelPreferences{}}).Build(); err != nil || settings.ExtraBody["mcp_model_preferences"] == nil {
		t.Fatalf("unexpected preference settings: %+v %v", settings, err)
	}
	aimcp.NewSamplingModel(fakeSamplingSession{})
	assertMCPPanic(t, "nil session", "must not be nil", func() { aimcp.NewSamplingModel(nil) })
	assertMCPPanic(t, "typed nil session", "must not be nil", func() {
		var session *fakeSamplingSession
		aimcp.NewSamplingModel(session)
	})
	assertMCPPanic(t, "nil option", "option must not be nil", func() {
		aimcp.NewSamplingModel(&fakeSamplingSession{}, nil)
	})
	assertMCPPanic(t, "invalid max", "must be positive", func() { aimcp.WithSamplingDefaultMaxTokens(0) })
}
