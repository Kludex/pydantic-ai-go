package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type requestModel struct {
	name    string
	request func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error)
}

func (model requestModel) Name() string { return model.name }

func (model requestModel) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return model.request(ctx, messages, params)
}

type streamingRequestModel struct {
	ai.Model
	stream func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error)
}

func (model streamingRequestModel) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return model.stream(ctx, messages, params)
}

type lifecycleRequestModel struct {
	ai.Model
	open func(context.Context) (ai.ModelCloseFunc, error)
}

type nativeHistoryRequestModel struct{ ai.Model }

func (nativeHistoryRequestModel) NativeToolSearchProvider() string { return "openai" }

func (model lifecycleRequestModel) OpenModel(ctx context.Context) (ai.ModelCloseFunc, error) {
	return model.open(ctx)
}

type continuationRequestModel struct {
	ai.Model
	delay       time.Duration
	cancelCount int
	cancelCtx   context.Context
}

func (model *continuationRequestModel) ContinuationDelay(ai.ModelResponse) time.Duration {
	return model.delay
}

func (model *continuationRequestModel) CancelSuspendedResponse(ctx context.Context, _ ai.ModelResponse) error {
	model.cancelCount++
	model.cancelCtx = ctx
	return errors.New("ignored cancellation failure")
}

func TestRequestModelUsesDirectParametersAndDetachedInput(t *testing.T) {
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}, Instructions: "Be helpful."},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "lookup", ToolCallID: "call", Args: json.RawMessage(`{}`),
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
			ToolName: "lookup", ToolCallID: "call", Content: "done",
		}}},
	}
	params := ai.ModelRequestParams{
		Tools:    []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		Settings: ai.ModelSettings{ExtraHeaders: map[string]string{"X-Test": "original"}},
	}
	model := requestModel{name: "direct-test", request: func(
		_ context.Context, got []ai.ModelMessage, gotParams ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if gotParams.Instructions != "Be helpful." ||
			!reflect.DeepEqual(gotParams.InstructionParts, []ai.InstructionPart{{Content: "Be helpful."}}) {
			t.Fatalf("instructions were not restored: %+v", gotParams)
		}
		got[0].(ai.ModelRequest).Metadata["mutated"] = true
		gotParams.Tools[0].Schema["type"] = "string"
		gotParams.Settings.ExtraHeaders["X-Test"] = "changed"
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	}}
	first := messages[0].(ai.ModelRequest)
	first.Metadata = map[string]any{"source": "caller"}
	messages[0] = first
	response, err := ai.RequestModel(t.Context(), model, messages, params)
	if err != nil {
		t.Fatal(err)
	}
	if response.ModelName != "direct-test" || response.Timestamp.IsZero() || response.State != ai.ModelResponseStateComplete {
		t.Fatalf("response was not normalized: %+v", response)
	}
	if messages[0].(ai.ModelRequest).Metadata["mutated"] != nil ||
		params.Tools[0].Schema["type"] != "object" || params.Settings.ExtraHeaders["X-Test"] != "original" {
		t.Fatal("direct request mutated caller input")
	}

	explicit := requestModel{name: "explicit", request: func(
		_ context.Context, _ []ai.ModelMessage, got ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if got.Instructions != "Explicit." || len(got.InstructionParts) != 1 || got.InstructionParts[0].Dynamic {
			t.Fatalf("explicit instruction parts were not preserved: %+v", got)
		}
		return &ai.ModelResponse{}, nil
	}}
	_, err = ai.RequestModel(t.Context(), explicit, messages, ai.ModelRequestParams{
		InstructionParts: []ai.InstructionPart{{Content: "Explicit."}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDirectModelDefaultsAndValidation(t *testing.T) {
	temperature := 0.2
	model := modelWithDefaults{
		Model: requestModel{name: "defaults", request: func(
			_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			if params.Settings.MaxTokens != 50 || params.Settings.Temperature == nil ||
				*params.Settings.Temperature != temperature {
				t.Fatalf("model defaults were not merged: %+v", params.Settings)
			}
			return &ai.ModelResponse{}, nil
		}},
		defaults: ai.ModelSettings{MaxTokens: 100, Temperature: &temperature},
	}
	if _, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{MaxTokens: 50},
	}); err != nil {
		t.Fatal(err)
	}

	invalid := ai.ModelRequestParams{Settings: ai.ModelSettings{RequestTimeout: -time.Second}}
	if _, err := ai.RequestModel(t.Context(), fakes.NewTestModel(), nil, invalid); err == nil {
		t.Fatal("expected direct request settings validation error")
	}
	stream := ai.StreamModel(t.Context(), fakes.NewTestModel(), nil, invalid)
	for _, err := range stream.Events() {
		if err == nil {
			t.Fatal("expected direct stream settings validation error")
		}
	}
	if stream.Err() == nil || !stream.Done() {
		t.Fatalf("invalid stream settings were not retained: done=%v err=%v", stream.Done(), stream.Err())
	}
}

func TestRequestModelLifecycleAndErrors(t *testing.T) {
	if response, err := ai.RequestModel(t.Context(), nil, nil, ai.ModelRequestParams{}); response != nil ||
		!errors.Is(err, ai.ErrNoModel) {
		t.Fatalf("unexpected nil model result: response=%+v err=%v", response, err)
	}
	openErr := errors.New("open")
	model := lifecycleRequestModel{
		Model: fakes.NewTestModel(),
		open:  func(context.Context) (ai.ModelCloseFunc, error) { return nil, openErr },
	}
	if _, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{}); !errors.Is(err, openErr) {
		t.Fatalf("unexpected open error: %v", err)
	}

	requestErr := errors.New("request")
	closeErr := errors.New("close")
	ctx, cancel := context.WithCancel(t.Context())
	model = lifecycleRequestModel{
		Model: requestModel{name: "lifecycle", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			cancel()
			return nil, requestErr
		}},
		open: func(context.Context) (ai.ModelCloseFunc, error) {
			return func(ctx context.Context) error {
				if ctx.Err() != nil {
					t.Fatal("close context was canceled")
				}
				return closeErr
			}, nil
		},
	}
	if _, err := ai.RequestModel(ctx, model, nil, ai.ModelRequestParams{}); !errors.Is(err, requestErr) ||
		!errors.Is(err, closeErr) {
		t.Fatalf("request and close errors were not joined: %v", err)
	}

	closed := false
	model = lifecycleRequestModel{
		Model: fakes.NewTestModel(),
		open: func(context.Context) (ai.ModelCloseFunc, error) {
			return func(context.Context) error { closed = true; return nil }, nil
		},
	}
	if _, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{}); err != nil || !closed {
		t.Fatalf("successful lifecycle was not closed: closed=%v err=%v", closed, err)
	}
}

func TestRequestModelContinuesSuspendedResponses(t *testing.T) {
	requests := 0
	base := requestModel{name: "continuation", request: func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.TextPart{Content: "first"}},
				Usage: ai.Usage{Requests: 1, InputTokens: 2}, State: ai.ModelResponseStateSuspended,
			}, nil
		}
		if len(messages) != 2 || messages[1].(ai.ModelResponse).State != ai.ModelResponseStateSuspended {
			t.Fatalf("suspended response was not replayed: %+v", messages)
		}
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "second"}},
			Usage: ai.Usage{Requests: 1, OutputTokens: 3}, State: ai.ModelResponseStateComplete,
		}, nil
	}}
	model := &continuationRequestModel{Model: base, delay: time.Nanosecond}
	response, err := ai.RequestModel(t.Context(), model, []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
	}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if requests != 2 || len(response.Parts) != 2 || response.Parts[1].(ai.TextPart).Content != "second" ||
		response.Usage.Requests != 2 || response.Usage.TotalTokens() != 5 {
		t.Fatalf("unexpected continued response: requests=%d response=%+v", requests, response)
	}
}

func TestRequestModelResumesHistoryAndCancelsDelay(t *testing.T) {
	base := requestModel{name: "continuation", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{State: ai.ModelResponseStateSuspended, ProviderResponseID: "job"}, nil
	}}
	model := &continuationRequestModel{Model: base, delay: time.Hour}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	seed := ai.ModelResponse{State: ai.ModelResponseStateSuspended, ProviderResponseID: "job"}
	response, err := ai.RequestModel(ctx, model, []ai.ModelMessage{seed}, ai.ModelRequestParams{})
	if !errors.Is(err, context.Canceled) || response == nil || model.cancelCount != 1 || model.cancelCtx.Err() != nil {
		t.Fatalf("unexpected canceled continuation: response=%+v count=%d err=%v", response, model.cancelCount, err)
	}
}

func TestRequestModelTimeoutAndNativeHistory(t *testing.T) {
	t.Run("timeout", func(t *testing.T) {
		model := requestModel{name: "timeout", request: func(
			ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			<-ctx.Done()
			return nil, context.Cause(ctx)
		}}
		_, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{
			Settings: ai.ModelSettings{RequestTimeout: time.Millisecond},
		})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("unexpected request timeout: %v", err)
		}
	})

	t.Run("foreign native history", func(t *testing.T) {
		base := requestModel{name: "history", request: func(
			_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			response := messages[0].(ai.ModelResponse)
			request := messages[1].(ai.ModelRequest)
			if _, ok := response.Parts[0].(ai.ToolCallPart); !ok {
				t.Fatalf("native call was not adapted: %+v", response.Parts)
			}
			if _, ok := request.Parts[0].(ai.ToolReturnPart); !ok {
				t.Fatalf("native return was not adapted: %+v", request.Parts)
			}
			return &ai.ModelResponse{}, nil
		}}
		model := nativeHistoryRequestModel{Model: base}
		_, err := ai.RequestModel(t.Context(), model, []ai.ModelMessage{
			ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.NativeToolCallPart{
					ToolName: "tool_search", ToolCallID: "call", ToolKind: ai.ToolPartKindToolSearch,
					ProviderName: "anthropic", Args: json.RawMessage(`{"query":"x"}`),
				},
				ai.NativeToolReturnPart{
					ToolName: "tool_search", ToolCallID: "call", ToolKind: ai.ToolPartKindToolSearch,
					ProviderName: "anthropic", Content: map[string]any{"tools": []any{}},
				},
			}},
		}, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
	})
}

func TestRequestModelContinuationLimitsAndReplacement(t *testing.T) {
	t.Run("generation limit", func(t *testing.T) {
		requests := 0
		base := requestModel{name: "limit", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			requests++
			return &ai.ModelResponse{
				State:              ai.ModelResponseStateSuspended,
				ProviderResponseID: fmt.Sprintf("job-%d", requests),
			}, nil
		}}
		model := &continuationRequestModel{Model: base}
		response, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{})
		if err == nil || !strings.Contains(err.Error(), "maximum of 10") || response == nil || model.cancelCount != 1 {
			t.Fatalf("unexpected continuation limit: requests=%d response=%+v count=%d err=%v", requests, response, model.cancelCount, err)
		}
	})

	t.Run("fresh replacement", func(t *testing.T) {
		requests := 0
		model := requestModel{name: "replacement", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			requests++
			if requests == 1 {
				return &ai.ModelResponse{
					Parts: []ai.ResponsePart{ai.TextPart{Content: "stale"}},
					State: ai.ModelResponseStateSuspended, Usage: ai.Usage{Requests: 1},
				}, nil
			}
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.TextPart{Content: "fresh"}}, Usage: ai.Usage{Requests: 1},
				Metadata: map[string]any{"__pydantic_ai__": map[string]any{"replace_previous_response": true}},
			}, nil
		}}
		response, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{})
		if err != nil || len(response.Parts) != 1 || response.Parts[0].(ai.TextPart).Content != "fresh" ||
			response.Usage.Requests != 2 {
			t.Fatalf("unexpected replacement response: %+v err=%v", response, err)
		}
	})
}

func TestStreamModelReturnsNormalizedEventsAndSnapshots(t *testing.T) {
	base := requestModel{name: "stream", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		t.Fatal("non-streaming request called")
		return nil, nil
	}}
	model := streamingRequestModel{Model: base, stream: func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (iter.Seq2[ai.ModelStreamEvent, error], error) {
		if params.Instructions != "Stream instruction." {
			t.Fatalf("missing stream instructions: %+v", params)
		}
		return func(yield func(ai.ModelStreamEvent, error) bool) {
			if !yield(ai.TextDeltaEvent{
				PartID: "answer", Delta: "hello", ProviderDetails: map[string]any{"nested": map[string]any{"a": "b"}},
			}, nil) {
				return
			}
			if !yield(ai.TextDeltaEvent{PartID: "answer", Delta: " world"}, nil) {
				return
			}
			yield(ai.FinishEvent{
				Usage: ai.Usage{Requests: 1, InputTokens: 2, OutputTokens: 1}, ProviderName: "provider",
			}, nil)
		}, nil
	}}
	stream := ai.StreamModel(t.Context(), model, []ai.ModelMessage{
		ai.ModelRequest{Instructions: "Stream instruction."},
	}, ai.ModelRequestParams{AllowText: true})
	if stream.Response() != nil || !stream.Usage().IsZero() || stream.Done() || stream.Err() != nil {
		t.Fatal("fresh stream had terminal state")
	}
	var kinds []string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		kinds = append(kinds, reflect.TypeOf(event).Name())
		if _, ok := event.(ai.PartStartEvent); ok {
			response := stream.Response()
			if len(response.Parts) != 1 || !strings.HasPrefix(response.Parts[0].(ai.TextPart).Content, "hello") {
				t.Fatalf("live response is incomplete: %+v", response)
			}
			response.Parts[0] = ai.TextPart{Content: "mutated"}
		}
	}
	if !stream.Done() || stream.Err() != nil || stream.Response().ModelName != "stream" ||
		stream.Response().State != ai.ModelResponseStateComplete || stream.Usage().TotalTokens() != 3 {
		t.Fatalf("unexpected terminal stream: response=%+v usage=%+v err=%v", stream.Response(), stream.Usage(), stream.Err())
	}
	if got := strings.Join(kinds, ","); got != "PartStartEvent,FinalResultEvent,PartDeltaEvent,PartEndEvent,FinishEvent" {
		t.Fatalf("unexpected events: %s", got)
	}
	for _, err := range stream.Events() {
		if !errors.Is(err, ai.ErrModelResponseStreamConsumed) {
			t.Fatalf("unexpected second-consumption error: %v", err)
		}
	}
}

func TestStreamModelFallsBackAndStopsEarly(t *testing.T) {
	model := requestModel{name: "fallback", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "fallback"}}}, nil
	}}
	stream := ai.StreamModel(t.Context(), model, nil, ai.ModelRequestParams{AllowText: true})
	seen := 0
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		seen++
		break
	}
	if seen != 1 || !stream.Done() || stream.Err() != nil || stream.Response() == nil {
		t.Fatalf("early stop did not retain response: seen=%d done=%v response=%+v err=%v", seen, stream.Done(), stream.Response(), stream.Err())
	}
}

func TestStreamModelSurfacesModelAndLifecycleErrors(t *testing.T) {
	for name, test := range map[string]struct {
		model ai.Model
		want  error
	}{
		"nil model": {want: ai.ErrNoModel},
		"stream start": {
			model: streamingRequestModel{
				Model: fakes.NewTestModel(),
				stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
					return nil, errors.New("stream start")
				},
			},
			want: errors.New("stream start"),
		},
		"nil response": {
			model: requestModel{name: "nil", request: func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return nil, nil
			}},
			want: errors.New("model returned no response"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			stream := ai.StreamModel(t.Context(), test.model, nil, ai.ModelRequestParams{})
			var got error
			for _, err := range stream.Events() {
				got = err
			}
			if got == nil || stream.Err() == nil || !stream.Done() {
				t.Fatalf("expected terminal error: got=%v state=%v", got, stream.Err())
			}
			if errors.Is(test.want, ai.ErrNoModel) {
				if !errors.Is(got, ai.ErrNoModel) {
					t.Fatalf("unexpected nil-model error: %v", got)
				}
			} else if !strings.Contains(got.Error(), test.want.Error()) {
				t.Fatalf("unexpected stream error: %v", got)
			}
		})
	}

	openErr := errors.New("open")
	stream := ai.StreamModel(t.Context(), lifecycleRequestModel{
		Model: fakes.NewTestModel(), open: func(context.Context) (ai.ModelCloseFunc, error) { return nil, openErr },
	}, nil, ai.ModelRequestParams{})
	for _, err := range stream.Events() {
		if !errors.Is(err, openErr) {
			t.Fatalf("unexpected open error: %v", err)
		}
	}

	closeErr := errors.New("close")
	stream = ai.StreamModel(t.Context(), lifecycleRequestModel{
		Model: fakes.NewTestModel(), open: func(context.Context) (ai.ModelCloseFunc, error) {
			return func(context.Context) error { return closeErr }, nil
		},
	}, nil, ai.ModelRequestParams{AllowText: true})
	for _, err := range stream.Events() {
		if err != nil && !errors.Is(err, closeErr) {
			t.Fatalf("unexpected close error: %v", err)
		}
	}
	if !errors.Is(stream.Err(), closeErr) {
		t.Fatalf("close error not retained: %v", stream.Err())
	}
}

func TestStreamModelRequestTimeout(t *testing.T) {
	model := streamingRequestModel{
		Model: fakes.NewTestModel(),
		stream: func(ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				<-ctx.Done()
				yield(nil, context.Cause(ctx))
			}, nil
		},
	}
	stream := ai.StreamModel(t.Context(), model, nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{RequestTimeout: time.Millisecond},
	})
	for _, err := range stream.Events() {
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("unexpected timeout: %v", err)
		}
	}
}

func TestStreamModelResumesBackgroundResponse(t *testing.T) {
	base := requestModel{name: "background", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("not used")
	}}
	model := streamingRequestModel{Model: base, stream: func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (iter.Seq2[ai.ModelStreamEvent, error], error) {
		if len(messages) != 1 || messages[0].(ai.ModelResponse).ProviderResponseID != "job" {
			t.Fatalf("background seed was not replayed: %+v", messages)
		}
		return func(yield func(ai.ModelStreamEvent, error) bool) {
			yield(ai.TextDeltaEvent{PartID: "text", Delta: "complete"}, nil)
			yield(ai.FinishEvent{
				State: ai.ModelResponseStateComplete, ProviderResponseID: "job", Usage: ai.Usage{Requests: 2},
			}, nil)
		}, nil
	}}
	seed := ai.ModelResponse{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "partial"}},
		State: ai.ModelResponseStateSuspended, ProviderResponseID: "job",
		ProviderDetails: map[string]any{"background": true}, Usage: ai.Usage{Requests: 1},
	}
	stream := ai.StreamModel(t.Context(), model, []ai.ModelMessage{seed}, ai.ModelRequestParams{AllowText: true})
	var starts []int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.PartStartEvent); ok {
			starts = append(starts, event.Index)
		}
	}
	if !reflect.DeepEqual(starts, []int{0}) || stream.Response().Parts[0].(ai.TextPart).Content != "complete" {
		t.Fatalf("background continuation was not replaced: starts=%v response=%+v", starts, stream.Response())
	}
}

func TestStreamModelRetainsPartialContinuationOnError(t *testing.T) {
	requests := 0
	streamErr := errors.New("stream failed")
	base := requestModel{name: "partial", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("not used")
	}}
	model := streamingRequestModel{Model: base, stream: func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (iter.Seq2[ai.ModelStreamEvent, error], error) {
		requests++
		return func(yield func(ai.ModelStreamEvent, error) bool) {
			yield(ai.TextDeltaEvent{PartID: "text", Delta: fmt.Sprintf("part-%d", requests)}, nil)
			if requests == 1 {
				yield(ai.FinishEvent{State: ai.ModelResponseStateSuspended, Usage: ai.Usage{Requests: 1}}, nil)
				return
			}
			yield(nil, streamErr)
		}, nil
	}}
	stream := ai.StreamModel(t.Context(), model, nil, ai.ModelRequestParams{AllowText: true})
	for _, err := range stream.Events() {
		if err != nil && !errors.Is(err, streamErr) {
			t.Fatalf("unexpected stream error: %v", err)
		}
	}
	response := stream.Response()
	if !errors.Is(stream.Err(), streamErr) || response == nil || len(response.Parts) != 2 {
		t.Fatalf("partial continuation was not retained: response=%+v err=%v", response, stream.Err())
	}
}

func TestStreamModelContinuationReindexesEvents(t *testing.T) {
	requests := 0
	base := requestModel{name: "continuation", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("not used")
	}}
	streaming := streamingRequestModel{Model: base, stream: func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (iter.Seq2[ai.ModelStreamEvent, error], error) {
		requests++
		current := requests
		return func(yield func(ai.ModelStreamEvent, error) bool) {
			yield(ai.TextDeltaEvent{PartID: "text", Delta: string(rune('0' + current))}, nil)
			state := ai.ModelResponseStateSuspended
			if current == 2 {
				state = ai.ModelResponseStateComplete
			}
			yield(ai.FinishEvent{
				State: state, ProviderResponseID: "job-" + string(rune('0'+current)),
				Usage: ai.Usage{Requests: 1},
			}, nil)
		}, nil
	}}
	stream := ai.StreamModel(t.Context(), streaming, nil, ai.ModelRequestParams{AllowText: true})
	var starts []int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.PartStartEvent); ok {
			starts = append(starts, event.Index)
		}
	}
	if !reflect.DeepEqual(starts, []int{0, 1}) || requests != 2 || len(stream.Response().Parts) != 2 {
		t.Fatalf("continuation events were not reindexed: starts=%v requests=%d response=%+v", starts, requests, stream.Response())
	}
}

var _ ai.StreamingModel = streamingRequestModel{}
var _ ai.ModelOpener = lifecycleRequestModel{}
var _ ai.ModelContinuationDelayer = (*continuationRequestModel)(nil)
var _ ai.SuspendedResponseCanceler = (*continuationRequestModel)(nil)
