package bedrock_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/middleware"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/bedrock"
)

type valueStream struct {
	events <-chan types.ConverseStreamOutput
}

func (stream valueStream) Events() <-chan types.ConverseStreamOutput { return stream.events }
func (valueStream) Close() error                                     { return nil }
func (valueStream) Err() error                                       { return nil }

type fakeStream struct {
	events     chan types.ConverseStreamOutput
	err        error
	metadata   middleware.Metadata
	closeCount int
}

func newFakeStream(events ...types.ConverseStreamOutput) *fakeStream {
	stream := &fakeStream{events: make(chan types.ConverseStreamOutput, len(events))}
	for _, event := range events {
		stream.events <- event
	}
	close(stream.events)
	return stream
}

func (stream *fakeStream) Events() <-chan types.ConverseStreamOutput { return stream.events }
func (stream *fakeStream) Close() error {
	stream.closeCount++
	return nil
}
func (stream *fakeStream) Err() error                          { return stream.err }
func (stream *fakeStream) ResultMetadata() middleware.Metadata { return stream.metadata }

type streamingClient struct {
	*fakeClient
	stream func(*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (bedrock.EventStream, error)
}

func (client *streamingClient) ConverseStream(
	_ context.Context, input *bedrockruntime.ConverseStreamInput, options ...func(*bedrockruntime.Options),
) (bedrock.EventStream, error) {
	return client.stream(input, options...)
}

func TestModelStreamRequest(t *testing.T) {
	index0, index1, index2, index3, index4 := int32(0), int32(1), int32(2), int32(3), int32(4)
	stream := newFakeStream(
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
			ContentBlockIndex: &index0, Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
				Name: aws.String("weather"), ToolUseId: aws.String("call-1"),
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index0, Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
				Input: aws.String(`{"city":"Lon`),
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index0, Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
				Input: aws.String(`don"}`),
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index1, Delta: &types.ContentBlockDeltaMemberText{Value: "hello"},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index2, Delta: &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberText{Value: "think"}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index2, Delta: &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberSignature{Value: "signed"}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index2, Delta: &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberRedactedContent{Value: []byte("secret")}},
		}},
		&types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: &index2}},
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
			ContentBlockIndex: &index3, Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
				Name: aws.String("nova_code_interpreter"), ToolUseId: aws.String("code-1"),
				Type: types.ToolUseTypeServerToolUse,
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index3, Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
				Input: aws.String(`{"snippet":"print(1)"}`),
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: &index3}},
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
			ContentBlockIndex: &index4, Start: &types.ContentBlockStartMemberToolResult{Value: types.ToolResultBlockStart{
				ToolUseId: aws.String("code-1"), Type: aws.String("nova_code_interpreter_result"),
				Status: types.ToolResultStatusError,
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index4, Delta: &types.ContentBlockDeltaMemberToolResult{Value: []types.ToolResultBlockDelta{
				&types.ToolResultBlockDeltaMemberText{Value: "failed"},
				&types.ToolResultBlockDeltaMemberJson{Value: document.NewLazyDocument(map[string]any{"exit": 1})},
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: &index4}},
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}},
		&types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{
			Usage:             &types.TokenUsage{InputTokens: aws.Int32(10), OutputTokens: aws.Int32(4)},
			Metrics:           &types.ConverseStreamMetrics{LatencyMs: aws.Int64(25)},
			ServiceTier:       &types.ServiceTier{Type: types.ServiceTierTypeFlex},
			PerformanceConfig: &types.PerformanceConfiguration{Latency: types.PerformanceConfigLatencyOptimized},
			Trace: &types.ConverseStreamTrace{PromptRouter: &types.PromptRouterTrace{
				InvokedModelId: aws.String("stream-routed-model"),
			}},
		}},
	)
	awsmiddleware.SetRequestIDMetadata(&stream.metadata, "stream-request-id")
	client := &streamingClient{fakeClient: &fakeClient{}, stream: func(
		input *bedrockruntime.ConverseStreamInput, options ...func(*bedrockruntime.Options),
	) (bedrock.EventStream, error) {
		if *input.ModelId != "model" || len(input.Messages) != 1 || len(options) != 1 ||
			input.GuardrailConfig == nil || *input.GuardrailConfig.GuardrailIdentifier != "guardrail" {
			t.Fatalf("unexpected stream input: %#v", input)
		}
		return stream, nil
	}}
	model := bedrock.NewModel("model", bedrock.WithClient(client), bedrock.WithProviderURL("https://bedrock.example"))
	sequence, err := model.StreamRequest(context.Background(), []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
	}, ai.ModelRequestParams{Settings: mustBedrockSettings(t, bedrock.Settings{
		Guardrail: &bedrock.GuardrailConfig{Identifier: "guardrail", Version: "1"},
	})})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ModelStreamEvent
	for event, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		events = append(events, event)
	}
	if stream.closeCount != 1 || len(events) != 11 {
		t.Fatalf("unexpected stream lifecycle: closes=%d events=%#v", stream.closeCount, events)
	}
	start := events[0].(ai.ToolCallStartEvent)
	if start.ToolName != "weather" || start.ToolCallID != "call-1" || start.PartID != "0" {
		t.Fatalf("unexpected tool start: %#v", start)
	}
	text := events[3].(ai.TextDeltaEvent)
	if text.Delta != "hello" || text.PartID != "1" {
		t.Fatalf("unexpected text delta: %#v", text)
	}
	redacted := events[6].(ai.ThinkingDeltaEvent)
	if string(redacted.ProviderDetails["redacted_content"].([]byte)) != "secret" {
		t.Fatalf("unexpected reasoning delta: %#v", redacted)
	}
	nativeStart := events[7].(ai.ToolCallStartEvent)
	nativeResult := events[9].(ai.NativeToolReturnEvent)
	if !nativeStart.Native || nativeStart.ToolKind != ai.ToolPartKindCodeExecution ||
		nativeResult.Part.Outcome != ai.ToolReturnOutcomeFailed || len(nativeResult.Part.Content.([]any)) != 2 {
		t.Fatalf("unexpected native stream events: start=%#v result=%#v", nativeStart, nativeResult)
	}
	finish := events[10].(ai.FinishEvent)
	if finish.FinishReason != ai.FinishReasonToolCall || finish.Usage.InputTokens != 10 ||
		finish.Usage.OutputTokens != 4 || finish.ProviderDetails["latency_ms"] != int64(25) ||
		finish.ProviderDetails["service_tier"] != "flex" || finish.ProviderURL != "https://bedrock.example" ||
		finish.ProviderResponseID != "stream-request-id" || finish.ProviderDetails["performance_latency"] != "optimized" ||
		finish.ProviderDetails["trace"].(map[string]any)["promptRouter"].(map[string]any)["invokedModelId"] != "stream-routed-model" {
		t.Fatalf("unexpected finish event: %#v", finish)
	}
}

func TestModelStreamRequestFailures(t *testing.T) {
	staticClient := &fakeClient{converse: func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	model := bedrock.NewModel("model", bedrock.WithClient(staticClient))
	sequence, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var fallback ai.FinishEvent
	for event, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		fallback = event.(ai.FinishEvent)
	}
	if len(fallback.Parts) != 4 || fallback.FinishReason != ai.FinishReasonStop {
		t.Fatalf("unexpected static stream fallback: %#v", fallback)
	}

	staticErr := errors.New("static failed")
	staticClient.converse = func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		return nil, staticErr
	}
	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if !errors.Is(err, staticErr) {
		t.Fatalf("unexpected static fallback error: %v", err)
	}

	streamErr := errors.New("stream open failed")
	client := &streamingClient{fakeClient: &fakeClient{}, stream: func(
		*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options),
	) (bedrock.EventStream, error) {
		return nil, streamErr
	}}
	model = bedrock.NewModel("model", bedrock.WithClient(client))
	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if !errors.Is(err, streamErr) {
		t.Fatalf("unexpected stream-open error: %v", err)
	}

	client.stream = func(*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (bedrock.EventStream, error) {
		return nil, nil
	}
	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "returned nil stream") {
		t.Fatalf("unexpected nil-stream error: %v", err)
	}
	var nilStream *fakeStream
	client.stream = func(*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (bedrock.EventStream, error) {
		return nilStream, nil
	}
	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "returned nil stream") {
		t.Fatalf("unexpected typed nil-stream error: %v", err)
	}

	loadFailure := bedrock.NewModel("model", bedrock.WithAWSLoadOptions(func(*awsconfig.LoadOptions) error {
		return errors.New("load failed")
	}))
	_, err = loadFailure.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "load AWS configuration") {
		t.Fatalf("unexpected stream load error: %v", err)
	}

	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{
		OutputSchema: map[string]any{"bad": make(chan int)}, OutputMode: ai.OutputModeNative,
	})
	if err == nil || !strings.Contains(err.Error(), "marshal output schema") {
		t.Fatalf("unexpected stream validation error: %v", err)
	}
}

func TestStreamConsumerStopsProviderIteration(t *testing.T) {
	index := int32(0)
	tests := []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
			ContentBlockIndex: &index, Start: &types.ContentBlockStartMemberToolUse{},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberText{Value: "text"},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
				Input: aws.String("{}"),
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberText{Value: "thinking"}},
		}},
	}
	for _, event := range tests {
		stream := newFakeStream(event)
		client := &streamingClient{fakeClient: &fakeClient{}, stream: func(
			*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options),
		) (bedrock.EventStream, error) {
			return stream, nil
		}}
		sequence, err := bedrock.NewModel("model", bedrock.WithClient(client)).StreamRequest(
			context.Background(), nil, ai.ModelRequestParams{},
		)
		if err != nil {
			t.Fatal(err)
		}
		calls := 0
		sequence(func(ai.ModelStreamEvent, error) bool {
			calls++
			return false
		})
		if calls != 1 || stream.closeCount != 1 {
			t.Fatalf("unexpected detached iteration: calls=%d closes=%d", calls, stream.closeCount)
		}
	}

	nativeEvents := nativeResultStreamEvents(&index, &types.ToolResultBlockDeltaMemberText{Value: "ok"})
	nativeEvents = append(nativeEvents, &types.ConverseStreamOutputMemberContentBlockStop{
		Value: types.ContentBlockStopEvent{ContentBlockIndex: &index},
	})
	stream := newFakeStream(nativeEvents...)
	client := &streamingClient{fakeClient: &fakeClient{}, stream: func(
		*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options),
	) (bedrock.EventStream, error) {
		return stream, nil
	}}
	sequence, err := bedrock.NewModel("model", bedrock.WithClient(client)).StreamRequest(
		context.Background(), nil, ai.ModelRequestParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	sequence(func(ai.ModelStreamEvent, error) bool { return false })
	if stream.closeCount != 1 {
		t.Fatalf("native-result stream was not closed: %d", stream.closeCount)
	}
}

func TestValueStream(t *testing.T) {
	index := int32(0)
	events := make(chan types.ConverseStreamOutput, 4)
	events <- &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: &index, Start: &types.ContentBlockStartMemberToolResult{Value: types.ToolResultBlockStart{
			ToolUseId: aws.String("code"), Type: aws.String("nova_code_interpreter_result"),
		}},
	}}
	events <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberToolResult{Value: []types.ToolResultBlockDelta{
			&types.ToolResultBlockDeltaMemberText{Value: "ok"},
		}},
	}}
	events <- &types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: &index}}
	events <- &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn}}
	close(events)
	client := &streamingClient{fakeClient: &fakeClient{}, stream: func(
		*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options),
	) (bedrock.EventStream, error) {
		return valueStream{events: events}, nil
	}}
	sequence, err := bedrock.NewModel("model", bedrock.WithClient(client)).StreamRequest(
		context.Background(), nil, ai.ModelRequestParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
	}
}

func nativeResultStreamEvents(
	index *int32, delta types.ToolResultBlockDelta,
) []types.ConverseStreamOutput {
	return []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
			ContentBlockIndex: index, Start: &types.ContentBlockStartMemberToolResult{Value: types.ToolResultBlockStart{
				Type: aws.String("nova_code_interpreter_result"),
			}},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: index,
			Delta:             &types.ContentBlockDeltaMemberToolResult{Value: []types.ToolResultBlockDelta{delta}},
		}},
	}
}

func TestMalformedStreamEvents(t *testing.T) {
	index := int32(0)
	tests := []struct {
		name   string
		events []types.ConverseStreamOutput
		err    error
		match  string
	}{
		{name: "start index", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockStart{},
		}, match: "start omitted index"},
		{name: "nil tool start", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
				ContentBlockIndex: &index, Start: (*types.ContentBlockStartMemberToolUse)(nil),
			}},
		}, match: "nil tool-use start"},
		{name: "unsupported start", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
				ContentBlockIndex: &index, Start: &types.ContentBlockStartMemberImage{},
			}},
		}, match: "unsupported stream content start"},
		{name: "nil result start", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
				ContentBlockIndex: &index, Start: (*types.ContentBlockStartMemberToolResult)(nil),
			}},
		}, match: "unsupported stream native tool result start"},
		{name: "unknown result start", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
				ContentBlockIndex: &index, Start: &types.ContentBlockStartMemberToolResult{},
			}},
		}, match: "unsupported stream native tool result start"},
		{name: "stop index", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockStop{},
		}, match: "stop omitted index"},
		{name: "delta index", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{},
		}, match: "delta omitted index"},
		{name: "nil text delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: (*types.ContentBlockDeltaMemberText)(nil),
			}},
		}, match: "nil text delta"},
		{name: "nil tool delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: (*types.ContentBlockDeltaMemberToolUse)(nil),
			}},
		}, match: "tool-use delta omitted input"},
		{name: "empty tool delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberToolUse{},
			}},
		}, match: "tool-use delta omitted input"},
		{name: "nil reasoning delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: (*types.ContentBlockDeltaMemberReasoningContent)(nil),
			}},
		}, match: "nil reasoning delta"},
		{name: "nil reasoning text delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberReasoningContent{Value: (*types.ReasoningContentBlockDeltaMemberText)(nil)},
			}},
		}, match: "nil reasoning-text delta"},
		{name: "nil reasoning signature delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberReasoningContent{Value: (*types.ReasoningContentBlockDeltaMemberSignature)(nil)},
			}},
		}, match: "nil reasoning-signature delta"},
		{name: "nil redacted reasoning delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberReasoningContent{Value: (*types.ReasoningContentBlockDeltaMemberRedactedContent)(nil)},
			}},
		}, match: "nil redacted-reasoning delta"},
		{name: "unsupported reasoning delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberReasoningContent{},
			}},
		}, match: "unsupported stream reasoning delta"},
		{name: "native result without start", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberToolResult{},
			}},
		}, match: "has no matching start"},
		{name: "nil result text", events: nativeResultStreamEvents(&index, (*types.ToolResultBlockDeltaMemberText)(nil)),
			match: "nil text"},
		{name: "nil result JSON", events: nativeResultStreamEvents(&index, (*types.ToolResultBlockDeltaMemberJson)(nil)),
			match: "nil JSON"},
		{name: "empty result JSON", events: nativeResultStreamEvents(&index, &types.ToolResultBlockDeltaMemberJson{}),
			match: "nil JSON"},
		{name: "unencodable result JSON", events: nativeResultStreamEvents(
			&index, &types.ToolResultBlockDeltaMemberJson{Value: document.NewLazyDocument(map[string]any{"": 1})},
		), match: "encode streamed native tool result"},
		{name: "invalid result JSON", events: nativeResultStreamEvents(
			&index, &types.ToolResultBlockDeltaMemberJson{Value: document.NewLazyDocument(make(chan int))},
		), match: "decode streamed native tool result"},
		{name: "unsupported result content", events: nativeResultStreamEvents(&index, nil),
			match: "unsupported stream native tool result content"},
		{name: "unsupported delta", events: []types.ConverseStreamOutput{
			&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: &index, Delta: &types.ContentBlockDeltaMemberImage{},
			}},
		}, match: "unsupported stream content delta"},
		{name: "nil stop", events: []types.ConverseStreamOutput{
			(*types.ConverseStreamOutputMemberMessageStop)(nil),
		}, match: "nil message stop"},
		{name: "nil metadata", events: []types.ConverseStreamOutput{
			(*types.ConverseStreamOutputMemberMetadata)(nil),
		}, match: "nil metadata"},
		{name: "unsupported event", events: []types.ConverseStreamOutput{nil}, match: "unsupported stream event"},
		{name: "reader error", err: errors.New("read failed"), match: "read failed"},
		{name: "missing stop", match: "without message stop"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := newFakeStream(test.events...)
			stream.err = test.err
			client := &streamingClient{fakeClient: &fakeClient{}, stream: func(
				*bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options),
			) (bedrock.EventStream, error) {
				return stream, nil
			}}
			sequence, err := bedrock.NewModel("model", bedrock.WithClient(client)).StreamRequest(
				context.Background(), nil, ai.ModelRequestParams{},
			)
			if err != nil {
				t.Fatal(err)
			}
			var got error
			for _, eventErr := range sequence {
				if eventErr != nil {
					got = eventErr
				}
			}
			if got == nil || !strings.Contains(got.Error(), test.match) {
				t.Fatalf("unexpected stream error: %v", got)
			}
			if stream.closeCount != 1 {
				t.Fatalf("stream was not closed: %d", stream.closeCount)
			}
		})
	}
}
