package bedrock_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/bedrock"
)

type converseOnlyClient struct{}

func (converseOnlyClient) Converse(
	context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return completeOutput(types.StopReasonEndTurn), nil
}

type valueClient struct{}

func (valueClient) Converse(
	context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return completeOutput(types.StopReasonEndTurn), nil
}

func (valueClient) CountTokens(
	context.Context, *bedrockruntime.CountTokensInput, ...func(*bedrockruntime.Options),
) (*bedrockruntime.CountTokensOutput, error) {
	return &bedrockruntime.CountTokensOutput{InputTokens: aws.Int32(1)}, nil
}

type fakeClient struct {
	converse func(*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	count    func(*bedrockruntime.CountTokensInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.CountTokensOutput, error)
}

func (client *fakeClient) Converse(
	_ context.Context, input *bedrockruntime.ConverseInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return client.converse(input, options...)
}

func (client *fakeClient) CountTokens(
	_ context.Context, input *bedrockruntime.CountTokensInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.CountTokensOutput, error) {
	return client.count(input, options...)
}

func TestModelRequestAndCountTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", request.URL.Query().Get("type"))
		_, _ = response.Write([]byte("downloaded"))
	}))
	defer server.Close()

	inputData := []byte("inline")
	strict := true
	client := &fakeClient{}
	client.converse = func(
		input *bedrockruntime.ConverseInput, options ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		if *input.ModelId != "us.anthropic.claude-sonnet-4-v1:0" {
			t.Fatalf("unexpected model ID: %q", *input.ModelId)
		}
		if len(input.System) != 2 || len(input.Messages) != 3 {
			t.Fatalf("unexpected conversation shape: system=%d messages=%d", len(input.System), len(input.Messages))
		}
		if input.ToolConfig == nil || len(input.ToolConfig.Tools) != 2 {
			t.Fatalf("unexpected tool config: %#v", input.ToolConfig)
		}
		if _, ok := input.ToolConfig.ToolChoice.(*types.ToolChoiceMemberTool); !ok {
			t.Fatalf("output tool was not forced: %T", input.ToolConfig.ToolChoice)
		}
		if input.InferenceConfig == nil || *input.InferenceConfig.MaxTokens != 123 ||
			*input.InferenceConfig.Temperature != 0.25 || *input.InferenceConfig.TopP != 0.75 {
			t.Fatalf("unexpected inference config: %#v", input.InferenceConfig)
		}
		if input.ServiceTier == nil || input.ServiceTier.Type != types.ServiceTierTypePriority {
			t.Fatalf("unexpected service tier: %#v", input.ServiceTier)
		}
		if input.AdditionalModelRequestFields == nil {
			t.Fatal("additional request fields were omitted")
		}
		if len(options) != 1 {
			t.Fatalf("unexpected option count: %d", len(options))
		}
		settings := bedrockruntime.Options{}
		options[0](&settings)
		if len(settings.APIOptions) != 1 {
			t.Fatalf("extra headers were not configured: %#v", settings.APIOptions)
		}
		return completeOutput(types.StopReasonToolUse), nil
	}
	client.count = func(
		input *bedrockruntime.CountTokensInput, _ ...func(*bedrockruntime.Options),
	) (*bedrockruntime.CountTokensOutput, error) {
		counted, ok := input.Input.(*types.CountTokensInputMemberConverse)
		if !ok || len(counted.Value.Messages) != 3 || counted.Value.ToolConfig == nil ||
			counted.Value.AdditionalModelRequestFields == nil {
			t.Fatalf("unexpected count input: %#v", input)
		}
		return &bedrockruntime.CountTokensOutput{InputTokens: aws.Int32(321)}, nil
	}
	defaults := ai.ModelSettings{MaxTokens: 77, ExtraHeaders: map[string]string{"default": "detached"}}
	model := bedrock.NewModel(
		"us.anthropic.claude-sonnet-4-v1:0", bedrock.WithClient(client),
		bedrock.WithProviderURL("https://bedrock.example/"), bedrock.WithDefaultSettings(defaults),
	)
	defaults.ExtraHeaders["default"] = "mutated"
	if model.Name() != "us.anthropic.claude-sonnet-4-v1:0" || model.ProviderName() != "bedrock" ||
		model.ProviderURL() != "https://bedrock.example" || model.DefaultModelSettings().ExtraHeaders["default"] != "detached" {
		t.Fatalf("unexpected model identity or defaults")
	}

	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "history system"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "hello"},
				ai.BinaryContent{Data: inputData, MediaType: "image/png"},
				ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/mpeg"},
				ai.BinaryContent{Data: []byte("video"), MediaType: "video/mp4"},
				ai.BinaryContent{Data: []byte("document"), MediaType: "application/pdf"},
				ai.ImageURL{URL: server.URL + "?type=image%2Fjpeg", ForceDownload: ai.FileDownloadAllowLocal},
				ai.AudioURL{URL: server.URL + "?type=audio%2Fwav", ForceDownload: ai.FileDownloadAllowLocal},
				ai.VideoURL{URL: server.URL + "?type=video%2Fwebm", ForceDownload: ai.FileDownloadAllowLocal},
				ai.DocumentURL{URL: server.URL + "?type=text%2Fplain", ForceDownload: ai.FileDownloadAllowLocal},
				ai.UploadedFile{FileID: "s3://bucket/image", ProviderName: "bedrock", MediaType: "image/webp"},
				ai.UploadedFile{FileID: "s3://bucket/audio", ProviderName: "bedrock", MediaType: "audio/flac"},
				ai.UploadedFile{FileID: "s3://bucket/video", ProviderName: "bedrock", MediaType: "video/quicktime"},
				ai.UploadedFile{FileID: "s3://bucket/document", ProviderName: "bedrock", MediaType: "text/markdown"},
				ai.CachePoint{TTL: ai.CachePointTTL1Hour},
			}},
			ai.ToolReturnPart{ToolCallID: "call-text", Content: "sunny"},
			ai.ToolReturnPart{ToolCallID: "call-json", Content: map[string]any{"temperature": 20}},
			ai.RetryPromptPart{Content: "try a city"},
			ai.RetryPromptPart{Content: "bad args", ToolCallID: "call-bad"},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: aws.String("spoken")},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &ai.BinaryContent{Data: []byte("speech"), MediaType: "audio/wav"}},
		}},
		ai.ModelResponse{ProviderName: "bedrock", Parts: []ai.ResponsePart{
			ai.TextPart{Content: "working"},
			ai.ToolCallPart{ToolName: "weather", ToolCallID: "call-1", Args: json.RawMessage(`{"city":"London"}`)},
			ai.ThinkingPart{Content: "reason", Signature: "signature", ProviderName: "bedrock"},
			ai.ThinkingPart{Content: "foreign", ProviderName: "other"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "continue"}}},
	}
	params := ai.ModelRequestParams{
		Instructions: "system", Tools: []ai.ToolDefinition{{
			Name: "weather", Description: "Weather", Schema: map[string]any{"type": "object"}, Strict: &strict,
		}},
		OutputTool: &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}},
		Settings: ai.ModelSettings{
			MaxTokens: 123, Temperature: float64Pointer(0.25), TopP: float64Pointer(0.75),
			StopSequences: []string{"STOP"}, ServiceTier: ai.ServiceTierPriority,
			ExtraHeaders: map[string]string{"x-test": "yes"}, ExtraBody: map[string]any{"top_k": 40},
		},
	}
	response, err := model.Request(context.Background(), messages, params)
	if err != nil {
		t.Fatal(err)
	}
	inputData[0] = 'X'
	if response.FinishReason != ai.FinishReasonToolCall || response.Usage.InputTokens != 12 ||
		response.Usage.OutputTokens != 7 || response.Usage.CacheReadTokens != 3 || response.Usage.CacheWriteTokens != 4 ||
		len(response.Parts) != 4 || response.ProviderDetails["latency_ms"] != int64(45) ||
		response.ProviderDetails["service_tier"] != "priority" {
		t.Fatalf("unexpected response: %#v", response)
	}
	call := response.Parts[1].(ai.ToolCallPart)
	if call.ToolName != "weather" || call.ToolCallID != "call-2" || string(call.Args) != `{"city":"Paris"}` {
		t.Fatalf("unexpected tool call: %#v", call)
	}
	if _, ok := response.Parts[3].(ai.ThinkingPart).ProviderDetails["redacted_content"].([]byte); !ok {
		t.Fatalf("redacted reasoning was not retained: %#v", response.Parts[3])
	}

	usage, err := model.CountTokens(context.Background(), messages, params)
	if err != nil || usage.InputTokens != 321 {
		t.Fatalf("unexpected token count: usage=%#v err=%v", usage, err)
	}
}

func completeOutput(reason types.StopReason) *bedrockruntime.ConverseOutput {
	return &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Role: types.ConversationRoleAssistant,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "done"},
				&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
					Name: aws.String("weather"), ToolUseId: aws.String("call-2"),
					Input: document.NewLazyDocument(map[string]any{"city": "Paris"}),
				}},
				&types.ContentBlockMemberReasoningContent{Value: &types.ReasoningContentBlockMemberReasoningText{Value: types.ReasoningTextBlock{Text: aws.String("thought"), Signature: aws.String("signed")}}},
				&types.ContentBlockMemberReasoningContent{Value: &types.ReasoningContentBlockMemberRedactedContent{
					Value: []byte("encrypted"),
				}},
			},
		}},
		StopReason: reason,
		Usage: &types.TokenUsage{
			InputTokens: aws.Int32(12), OutputTokens: aws.Int32(7),
			CacheReadInputTokens: aws.Int32(3), CacheWriteInputTokens: aws.Int32(4),
		},
		Metrics:     &types.ConverseMetrics{LatencyMs: aws.Int64(45)},
		ServiceTier: &types.ServiceTier{Type: types.ServiceTierTypePriority},
	}
}

func TestModelErrorsAndResponseShapes(t *testing.T) {
	tests := []struct {
		name   string
		output *bedrockruntime.ConverseOutput
		err    error
		match  string
	}{
		{name: "nil response", match: "response is nil"},
		{name: "missing output", output: &bedrockruntime.ConverseOutput{}, match: "omitted output message"},
		{name: "unsupported output", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{&types.ContentBlockMemberImage{}},
		}}}, match: "unsupported response content"},
		{name: "nil text", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{(*types.ContentBlockMemberText)(nil)},
		}}}, match: "nil text block"},
		{name: "nil tool use", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{(*types.ContentBlockMemberToolUse)(nil)},
		}}}, match: "nil tool-use block"},
		{name: "nil reasoning", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{&types.ContentBlockMemberReasoningContent{}},
		}}}, match: "nil reasoning block"},
		{name: "nil reasoning text", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{&types.ContentBlockMemberReasoningContent{Value: (*types.ReasoningContentBlockMemberReasoningText)(nil)}},
		}}}, match: "nil reasoning-text block"},
		{name: "nil redacted reasoning", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{&types.ContentBlockMemberReasoningContent{Value: (*types.ReasoningContentBlockMemberRedactedContent)(nil)}},
		}}}, match: "nil redacted-reasoning block"},
		{name: "missing tool input", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{&types.ContentBlockMemberToolUse{}},
		}}}, match: "tool use omitted input"},
		{name: "invalid tool input document", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				Input: document.NewLazyDocument(make(chan int)),
			}}},
		}}}, match: "not valid JSON"},
		{name: "unencodable tool input document", output: &bedrockruntime.ConverseOutput{Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Content: []types.ContentBlock{&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				Input: document.NewLazyDocument(map[string]any{"": 1}),
			}}},
		}}}, match: "encode tool use input"},
		{name: "transport", err: errors.New("offline"), match: "offline"},
		{name: "api", err: &smithyhttp.ResponseError{
			Response: &smithyhttp.Response{Response: &http.Response{StatusCode: http.StatusTooManyRequests}},
			Err:      errors.New("throttled"),
		}, match: "throttled"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{converse: func(
				*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
			) (*bedrockruntime.ConverseOutput, error) {
				return test.output, test.err
			}}
			model := bedrock.NewModel("model", bedrock.WithClient(client))
			_, err := model.Request(context.Background(), []ai.ModelMessage{
				ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
			}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
			if test.name == "api" {
				var apiError *bedrock.APIError
				if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusTooManyRequests ||
					!apiError.IsModelAPIError() || !errors.Is(err, test.err) {
					t.Fatalf("unexpected API error: %#v", err)
				}
			}
		})
	}

	apiError := &bedrock.APIError{}
	if apiError.Error() != "bedrock: API request failed" || apiError.Unwrap() != nil {
		t.Fatalf("unexpected empty API error: %v", apiError)
	}
}

func TestCachePointPlacement(t *testing.T) {
	var captured *bedrockruntime.ConverseInput
	client := &fakeClient{converse: func(
		input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		captured = input
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	model := bedrock.NewModel("model", bedrock.WithClient(client))
	messages := []ai.ModelMessage{
		ai.ModelRequest{},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "response"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.CachePoint{},
		}}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.CachePoint{TTL: ai.CachePointTTL1Hour},
		}}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "one"}, ai.CachePoint{},
			ai.TextContent{Text: "two"}, ai.CachePoint{},
			ai.TextContent{Text: "three"}, ai.CachePoint{},
			ai.TextContent{Text: "four"}, ai.CachePoint{},
			ai.TextContent{Text: "five"}, ai.CachePoint{},
		}}}},
	}
	if _, err := model.Request(context.Background(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	points := 0
	for _, message := range captured.Messages {
		for _, block := range message.Content {
			if _, ok := block.(*types.ContentBlockMemberCachePoint); ok {
				points++
			}
		}
	}
	if points != 4 {
		t.Fatalf("expected newest four cache points, got %d: %#v", points, captured.Messages)
	}

	captured = nil
	_, err := model.Request(context.Background(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"}}},
	}}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	content := captured.Messages[0].Content
	if len(content) != 2 {
		t.Fatalf("document prompt did not receive required text: %#v", content)
	}
	if text, ok := content[0].(*types.ContentBlockMemberText); !ok || text.Value != "See attached document(s)." {
		t.Fatalf("unexpected document preface: %#v", content[0])
	}

	_, err = model.Request(context.Background(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}}},
	}}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCachePointPlacementErrors(t *testing.T) {
	client := &fakeClient{converse: func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		t.Fatal("client should not be called")
		return nil, nil
	}}
	tests := []struct {
		name     string
		contents []ai.UserContent
		match    string
	}{
		{name: "first content", contents: []ai.UserContent{ai.CachePoint{}}, match: "requires preceding user content"},
		{name: "consecutive", contents: []ai.UserContent{
			ai.TextContent{Text: "text"}, ai.CachePoint{}, ai.CachePoint{},
		}, match: "require content between"},
		{name: "after only document", contents: []ai.UserContent{
			ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"}, ai.CachePoint{},
		}, match: "requires preceding non-document content"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := bedrock.NewModel("model", bedrock.WithClient(client)).Request(
				context.Background(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
					ai.UserPromptPart{Contents: test.contents},
				}}}, ai.ModelRequestParams{},
			)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestUnsupportedRequestFeatures(t *testing.T) {
	never := &fakeClient{converse: func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		t.Fatal("client should not be called")
		return nil, nil
	}}
	tests := []struct {
		name   string
		params ai.ModelRequestParams
		match  string
	}{
		{name: "native output", params: ai.ModelRequestParams{
			OutputSchema: map[string]any{"type": "object"}, OutputMode: ai.OutputModeNative,
		}, match: "native JSON output mode"},
		{name: "native tools", params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
		}, match: "provider-native tools"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := bedrock.NewModel("model", bedrock.WithClient(never)).Request(
				context.Background(), nil, test.params,
			)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestRequestValidationErrors(t *testing.T) {
	never := &fakeClient{converse: func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		t.Fatal("client should not be called")
		return nil, nil
	}}
	tests := []struct {
		name    string
		message ai.ModelMessage
		match   string
	}{
		{name: "tool availability", message: ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"later"}},
		}}, match: "tool availability history"},
		{name: "unsupported response part", message: ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.FilePart{},
		}}, match: "unsupported response part"},
		{name: "invalid tool JSON", message: ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "bad", Args: json.RawMessage("{")},
		}}, match: "decode tool call"},
		{name: "unsupported media", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.BinaryContent{Data: []byte("x"), MediaType: "application/zip"},
		}}}}, match: "unsupported binary content"},
		{name: "foreign upload", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.UploadedFile{FileID: "s3://bucket/file", ProviderName: "other", MediaType: "application/pdf"},
		}}}}, match: "belongs to provider"},
		{name: "invalid upload URI", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.UploadedFile{FileID: "file-1", ProviderName: "bedrock", MediaType: "application/pdf"},
		}}}}, match: "must use an s3:// URI"},
		{name: "invalid upload media", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.UploadedFile{FileID: "s3://bucket/file", ProviderName: "bedrock", MediaType: "application/zip"},
		}}}}, match: "unsupported uploaded file media"},
		{name: "invalid uploaded image", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.UploadedFile{FileID: "s3://bucket/file", ProviderName: "bedrock", MediaType: "image/svg+xml"},
		}}}}, match: "unsupported uploaded file media"},
		{name: "invalid uploaded audio", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.UploadedFile{FileID: "s3://bucket/file", ProviderName: "bedrock", MediaType: "audio/midi"},
		}}}}, match: "unsupported uploaded file media"},
		{name: "invalid uploaded video", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.UploadedFile{FileID: "s3://bucket/file", ProviderName: "bedrock", MediaType: "video/avi"},
		}}}}, match: "unsupported uploaded file media"},
		{name: "invalid cache TTL", message: ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.CachePoint{TTL: "forever"},
		}}}}, match: "invalid cache point TTL"},
		{name: "unmarshalable tool result", message: ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolCallID: "call", Content: make(chan int)},
		}}, match: "marshal tool result"},
		{name: "invalid speech audio", message: ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &ai.BinaryContent{MediaType: "application/zip"}},
		}}, match: "unsupported binary content"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := bedrock.NewModel("model", bedrock.WithClient(never))
			_, err := model.Request(context.Background(), []ai.ModelMessage{test.message}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestCountTokenErrors(t *testing.T) {
	_, err := bedrock.NewModel("model", bedrock.WithClient(converseOnlyClient{})).CountTokens(
		context.Background(), nil, ai.ModelRequestParams{},
	)
	if !errors.Is(err, ai.ErrTokenCountingUnsupported) {
		t.Fatalf("unexpected unsupported token-counting error: %v", err)
	}

	tests := []struct {
		name   string
		output *bedrockruntime.CountTokensOutput
		err    error
		match  string
	}{
		{name: "missing", output: &bedrockruntime.CountTokensOutput{}, match: "omitted inputTokens"},
		{name: "nil", match: "omitted inputTokens"},
		{name: "transport", err: errors.New("offline"), match: "offline"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{count: func(
				*bedrockruntime.CountTokensInput, ...func(*bedrockruntime.Options),
			) (*bedrockruntime.CountTokensOutput, error) {
				return test.output, test.err
			}}
			model := bedrock.NewModel("model", bedrock.WithClient(client))
			_, err := model.CountTokens(context.Background(), nil, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestOptionsAndAWSConfiguration(t *testing.T) {
	assertPanic(t, func() { bedrock.WithClient(nil) })
	var typedNil *fakeClient
	assertPanic(t, func() { bedrock.WithClient(typedNil) })
	if model := bedrock.NewModel("model", bedrock.WithClient(valueClient{})); model.Name() != "model" {
		t.Fatal("value client was not accepted")
	}

	baseURL := "https://custom.example/"
	model := bedrock.NewModel("model", bedrock.WithAWSConfig(aws.Config{Region: "us-east-1", BaseEndpoint: &baseURL}))
	baseURL = "https://mutated.example"
	if model.ProviderURL() != "https://custom.example" {
		t.Fatalf("AWS configuration was not detached: %q", model.ProviderURL())
	}
	china := bedrock.NewModel("model", bedrock.WithAWSConfig(aws.Config{Region: "cn-north-1"}))
	if china.ProviderURL() != "https://bedrock-runtime.cn-north-1.amazonaws.com.cn" {
		t.Fatalf("unexpected China endpoint: %q", china.ProviderURL())
	}
	standard := bedrock.NewModel("model", bedrock.WithAWSConfig(aws.Config{Region: "eu-west-1"}))
	if standard.ProviderURL() != "https://bedrock-runtime.eu-west-1.amazonaws.com" {
		t.Fatalf("unexpected standard endpoint: %q", standard.ProviderURL())
	}
	empty := bedrock.NewModel("model", bedrock.WithAWSConfig(aws.Config{}))
	if empty.ProviderURL() != "" {
		t.Fatalf("unexpected empty-region endpoint: %q", empty.ProviderURL())
	}

	loadFailure := bedrock.NewModel("model", bedrock.WithAWSLoadOptions(func(*awsconfig.LoadOptions) error {
		return errors.New("load failed")
	}))
	_, err := loadFailure.CountTokens(context.Background(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "load AWS configuration") {
		t.Fatalf("unexpected load error: %v", err)
	}
	_, err = loadFailure.Request(context.Background(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "load AWS configuration") {
		t.Fatalf("unexpected request load error: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("x-custom") != "present" {
			t.Errorf("custom header was omitted: %v", request.Header)
		}
		response.WriteHeader(http.StatusInternalServerError)
		_, _ = response.Write([]byte(`{"message":"failed"}`))
	}))
	defer server.Close()
	loaded := bedrock.NewModel("model", bedrock.WithAWSLoadOptions(
		awsconfig.WithRegion("us-east-1"),
		awsconfig.WithBaseEndpoint(server.URL),
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("key", "secret", "")),
	))
	_, err = loaded.Request(context.Background(), []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
	}, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraHeaders: map[string]string{"x-custom": "present"}}})
	if err == nil {
		t.Fatal("expected AWS service error")
	}
	if loaded.ProviderURL() != server.URL {
		t.Fatalf("unexpected loaded provider URL: %q", loaded.ProviderURL())
	}
}

func TestToolChoiceAndServiceTiers(t *testing.T) {
	tests := []struct {
		tier ai.ServiceTier
		want types.ServiceTierType
	}{
		{ai.ServiceTierDefault, types.ServiceTierTypeDefault},
		{ai.ServiceTierFlex, types.ServiceTierTypeFlex},
	}
	for _, test := range tests {
		client := &fakeClient{converse: func(
			input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
		) (*bedrockruntime.ConverseOutput, error) {
			if input.ServiceTier == nil || input.ServiceTier.Type != test.want {
				t.Fatalf("unexpected service tier: %#v", input.ServiceTier)
			}
			if _, ok := input.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAuto); !ok {
				t.Fatalf("unexpected tool choice: %T", input.ToolConfig.ToolChoice)
			}
			return completeOutput(types.StopReasonEndTurn), nil
		}}
		_, err := bedrock.NewModel("model", bedrock.WithClient(client)).Request(
			context.Background(), nil, ai.ModelRequestParams{
				Tools:    []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}},
				Settings: ai.ModelSettings{ServiceTier: test.tier},
			},
		)
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestCountTokensRejectsInvalidHistory(t *testing.T) {
	client := &fakeClient{count: func(
		*bedrockruntime.CountTokensInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.CountTokensOutput, error) {
		t.Fatal("client should not be called")
		return nil, nil
	}}
	_, err := bedrock.NewModel("model", bedrock.WithClient(client)).CountTokens(
		context.Background(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{}}}},
		ai.ModelRequestParams{},
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported response part") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFinishReasonsAndSparseResponses(t *testing.T) {
	tests := []struct {
		stop types.StopReason
		want ai.FinishReason
	}{
		{types.StopReasonEndTurn, ai.FinishReasonStop},
		{types.StopReasonStopSequence, ai.FinishReasonStop},
		{types.StopReasonToolUse, ai.FinishReasonToolCall},
		{types.StopReasonMaxTokens, ai.FinishReasonLength},
		{types.StopReasonModelContextWindowExceeded, ai.FinishReasonLength},
		{types.StopReasonGuardrailIntervened, ai.FinishReasonContentFilter},
		{types.StopReasonContentFiltered, ai.FinishReasonContentFilter},
		{types.StopReasonMalformedModelOutput, ai.FinishReasonError},
	}
	for _, test := range tests {
		client := &fakeClient{converse: func(
			input *bedrockruntime.ConverseInput, options ...func(*bedrockruntime.Options),
		) (*bedrockruntime.ConverseOutput, error) {
			configured := bedrockruntime.Options{}
			options[0](&configured)
			if len(configured.APIOptions) != 0 {
				t.Fatalf("unexpected API options without headers: %#v", configured.APIOptions)
			}
			if input.InferenceConfig != nil || input.ToolConfig != nil {
				t.Fatalf("unexpected empty request configuration: %#v", input)
			}
			output := &bedrockruntime.ConverseOutput{
				Output: &types.ConverseOutputMemberMessage{Value: types.Message{Content: []types.ContentBlock{
					&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{Input: document.NewLazyDocument(nil)}},
				}}},
				StopReason: test.stop,
			}
			if test.stop != types.StopReasonEndTurn {
				output.Usage = &types.TokenUsage{}
			}
			return output, nil
		}}
		response, err := bedrock.NewModel("model", bedrock.WithClient(client)).Request(
			context.Background(), nil, ai.ModelRequestParams{},
		)
		if err != nil || response.FinishReason != test.want || response.Usage.Requests != 1 {
			t.Fatalf("stop=%q response=%#v err=%v", test.stop, response, err)
		}
	}
}

func float64Pointer(value float64) *float64 { return &value }

func assertPanic(t *testing.T, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	function()
}

func TestDefaultSettingsClone(t *testing.T) {
	temperature := 0.4
	model := bedrock.NewModel("model", bedrock.WithDefaultSettings(ai.ModelSettings{
		Temperature: &temperature, StopSequences: []string{"stop"},
	}))
	first := model.DefaultModelSettings()
	first.StopSequences[0] = "mutated"
	*first.Temperature = 1
	second := model.DefaultModelSettings()
	if second.StopSequences[0] != "stop" || *second.Temperature != 0.4 {
		t.Fatalf("settings were not detached: %#v", second)
	}
}

func TestContextCancellationIsNotModelAPIError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := &fakeClient{converse: func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		return nil, context.Canceled
	}}
	_, err := bedrock.NewModel("model", bedrock.WithClient(client)).Request(ctx, nil, ai.ModelRequestParams{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected cancellation error: %v", err)
	}
	var apiError ai.ModelAPIError
	if errors.As(err, &apiError) {
		t.Fatalf("cancellation was classified as a model API error: %v", err)
	}
}

func TestDownloadValidation(t *testing.T) {
	client := &fakeClient{converse: func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	model := bedrock.NewModel("model", bedrock.WithClient(client))
	_, err := model.Request(context.Background(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.ImageURL{URL: "http://127.0.0.1/image", ForceDownload: "bad"}}},
	}}}, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "invalid file download mode") {
		t.Fatalf("unexpected mode error: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err = model.Request(ctx, []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.ImageURL{URL: "ftp://example.com/image"}}},
	}}}, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "URL scheme") {
		t.Fatalf("unexpected download error: %v", err)
	}
}
