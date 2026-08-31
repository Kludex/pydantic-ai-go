package ai_test

import (
	"context"
	"errors"
	"iter"
	"reflect"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type fullyOptionalModel struct {
	opened    bool
	closed    bool
	cancelled bool
}

func (*fullyOptionalModel) Name() string { return "optional" }

func (*fullyOptionalModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "request"}}}, nil
}

func (*fullyOptionalModel) StreamRequest(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		yield(ai.FinishEvent{Parts: []ai.ResponsePart{ai.TextPart{Content: "stream"}}}, nil)
	}, nil
}

func (model *fullyOptionalModel) OpenModel(context.Context) (ai.ModelCloseFunc, error) {
	model.opened = true
	return func(context.Context) error { model.closed = true; return nil }, nil
}

func (*fullyOptionalModel) DefaultModelSettings() ai.ModelSettings {
	return ai.ModelSettings{MaxTokens: 42, StopSequences: []string{"stop"}}
}

func (*fullyOptionalModel) SupportsToolSearchStrategy(strategy ai.ToolSearchStrategy) bool {
	return strategy == ai.ToolSearchStrategyRegex
}

func (*fullyOptionalModel) NativeToolSearchProvider() string { return "provider" }

func (*fullyOptionalModel) ContinuationDelay(ai.ModelResponse) time.Duration { return time.Second }

func (model *fullyOptionalModel) CancelSuspendedResponse(context.Context, ai.ModelResponse) error {
	model.cancelled = true
	return nil
}

type customModelUnwrapper struct {
	ai.Model
	wrapped ai.Model
}

func (wrapper *customModelUnwrapper) UnwrapModel() ai.Model { return wrapper.wrapped }

func TestModelWrapperDelegatesModelContract(t *testing.T) {
	underlying := &fullyOptionalModel{}
	wrapper := ai.WrapModel(underlying)
	if wrapper.Name() != "optional" || wrapper.UnwrapModel() != underlying || ai.UnwrapModel(wrapper) != underlying {
		t.Fatalf("unexpected wrapper identity: name=%q wrapped=%T", wrapper.Name(), ai.UnwrapModel(wrapper))
	}
	response, err := wrapper.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || response.Text() != "request" {
		t.Fatalf("request was not delegated: response=%+v err=%v", response, err)
	}
	events, err := wrapper.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var finish ai.FinishEvent
	for event, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		finish = event.(ai.FinishEvent)
	}
	if finish.Parts[0].(ai.TextPart).Content != "stream" {
		t.Fatalf("stream was not delegated: %+v", finish)
	}
	closeModel, err := wrapper.OpenModel(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := closeModel(t.Context()); err != nil {
		t.Fatal(err)
	}
	settings := wrapper.DefaultModelSettings()
	settings.StopSequences[0] = "changed"
	if !underlying.opened || !underlying.closed || wrapper.DefaultModelSettings().StopSequences[0] != "stop" ||
		!wrapper.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) ||
		wrapper.SupportsToolSearchStrategy(ai.ToolSearchStrategyBM25) ||
		wrapper.NativeToolSearchProvider() != "provider" ||
		wrapper.ContinuationDelay(ai.ModelResponse{}) != time.Second {
		t.Fatalf("optional interfaces were not delegated: settings=%+v", settings)
	}
	if err := wrapper.CancelSuspendedResponse(t.Context(), ai.ModelResponse{}); err != nil || !underlying.cancelled {
		t.Fatalf("cancellation was not delegated: cancelled=%v err=%v", underlying.cancelled, err)
	}
}

func TestModelWrapperProvidesSafeOptionalFallbacks(t *testing.T) {
	wrapper := ai.WrapModel(fakes.NewTestModel())
	if closeModel, err := wrapper.OpenModel(t.Context()); err != nil || closeModel != nil {
		t.Fatalf("unexpected lifecycle fallback: close=%v err=%v", closeModel, err)
	}
	if settings := wrapper.DefaultModelSettings(); !reflect.DeepEqual(settings, ai.ModelSettings{}) {
		t.Fatalf("unexpected default settings: %+v", settings)
	}
	if wrapper.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) || wrapper.NativeToolSearchProvider() != "" ||
		wrapper.ContinuationDelay(ai.ModelResponse{}) != 0 {
		t.Fatal("unexpected optional capability support")
	}
	if err := wrapper.CancelSuspendedResponse(t.Context(), ai.ModelResponse{}); err != nil {
		t.Fatal(err)
	}
	events, err := wrapper.StreamRequest(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		seen++
	}
	if seen != 2 {
		t.Fatalf("non-streaming response was not replayed: %d events", seen)
	}
}

func TestModelWrapperPropagatesRequestAndStreamErrors(t *testing.T) {
	requestErr := errors.New("request")
	wrapper := ai.WrapModel(requestModel{name: "error", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, requestErr
	}})
	if _, err := wrapper.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, requestErr) {
		t.Fatalf("request error was not delegated: %v", err)
	}
	if events, err := wrapper.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); events != nil || !errors.Is(err, requestErr) {
		t.Fatalf("stream fallback error was not delegated: events=%v err=%v", events, err)
	}

	nilResponse := ai.WrapModel(requestModel{name: "nil", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, nil
	}})
	if events, err := nilResponse.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); events != nil || err != nil {
		t.Fatalf("nil response fallback changed result: events=%v err=%v", events, err)
	}
}

func TestUnwrapModelHandlesNestedInvalidAndCyclicWrappers(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != "ai: wrapped model must not be nil" {
			t.Fatalf("unexpected panic: %v", recovered)
		}
	}()

	underlying := fakes.NewTestModel()
	nested := ai.WrapModel(ai.WrapModel(underlying))
	if ai.UnwrapModel(nested) != underlying {
		t.Fatalf("nested wrapper did not unwrap: %T", ai.UnwrapModel(nested))
	}
	if ai.UnwrapModel(nil) != nil {
		t.Fatal("nil model changed while unwrapping")
	}
	first := &customModelUnwrapper{Model: underlying}
	second := &customModelUnwrapper{Model: underlying, wrapped: first}
	first.wrapped = second
	if ai.UnwrapModel(first) != first {
		t.Fatal("cyclic wrappers should return the original model")
	}
	invalid := &customModelUnwrapper{Model: underlying}
	if ai.UnwrapModel(invalid) != invalid {
		t.Fatal("nil wrapped model should return the original model")
	}
	ai.WrapModel(nil)
}

var _ ai.Model = (*ai.ModelWrapper)(nil)
var _ ai.StreamingModel = (*ai.ModelWrapper)(nil)
var _ ai.ModelOpener = (*ai.ModelWrapper)(nil)
var _ ai.ModelDefaultSettings = (*ai.ModelWrapper)(nil)
var _ ai.ToolSearchStrategyModel = (*ai.ModelWrapper)(nil)
var _ ai.NativeToolSearchHistoryModel = (*ai.ModelWrapper)(nil)
var _ ai.ModelContinuationDelayer = (*ai.ModelWrapper)(nil)
var _ ai.SuspendedResponseCanceler = (*ai.ModelWrapper)(nil)
var _ ai.ModelUnwrapper = (*ai.ModelWrapper)(nil)
