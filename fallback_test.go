package ai_test

import (
	"context"
	"errors"
	"iter"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

type fallbackAPIError struct{ message string }

func (err *fallbackAPIError) Error() string     { return err.message }
func (*fallbackAPIError) IsModelAPIError() bool { return true }

type fallbackStreamingContinuationModel struct {
	streamingRequestModel
	cancelCount int
}

func (model *fallbackStreamingContinuationModel) CancelSuspendedResponse(
	context.Context, ai.ModelResponse,
) error {
	model.cancelCount++
	return nil
}

type fallbackCapabilityModel struct {
	requestModel
	provider  string
	strategy  ai.ToolSearchStrategy
	delay     time.Duration
	cancelErr error
	cancelled int
}

func (model *fallbackCapabilityModel) SupportsToolSearchStrategy(strategy ai.ToolSearchStrategy) bool {
	return strategy == model.strategy
}

func (model *fallbackCapabilityModel) NativeToolSearchProvider() string { return model.provider }
func (model *fallbackCapabilityModel) ContinuationDelay(ai.ModelResponse) time.Duration {
	return model.delay
}
func (model *fallbackCapabilityModel) CancelSuspendedResponse(context.Context, ai.ModelResponse) error {
	model.cancelled++
	return model.cancelErr
}

func fallbackTextModel(name, text string, calls *int) ai.Model {
	return requestModel{name: name, request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if calls != nil {
			*calls++
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: text}}}, nil
	}}
}

func TestFallbackModelUsesDefaultAPIPolicy(t *testing.T) {
	primaryCalls, backupCalls := 0, 0
	primary := requestModel{name: "primary", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		primaryCalls++
		return nil, &fallbackAPIError{message: "unavailable"}
	}}
	fallback := ai.NewFallbackModel(primary, ai.WithFallbackModels(fallbackTextModel("backup", "ok", &backupCalls)))
	response, err := ai.RequestModel(t.Context(), fallback, nil, ai.ModelRequestParams{})
	if err != nil || response.Text() != "ok" || primaryCalls != 1 || backupCalls != 1 ||
		fallback.Name() != "fallback:primary,backup" {
		t.Fatalf("fallback failed: response=%+v calls=%d/%d name=%q err=%v", response, primaryCalls, backupCalls, fallback.Name(), err)
	}
	models := fallback.Models()
	models[0] = nil
	if fallback.Models()[0] == nil {
		t.Fatal("Models returned shared storage")
	}

	ordinaryErr := errors.New("invalid request")
	noFallback := ai.NewFallbackModel(requestModel{name: "primary", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, ordinaryErr
	}}, ai.WithFallbackModels(fallbackTextModel("backup", "unused", nil)))
	if _, err := ai.RequestModel(t.Context(), noFallback, nil, ai.ModelRequestParams{}); !errors.Is(err, ordinaryErr) {
		t.Fatalf("ordinary error unexpectedly fell back: %v", err)
	}
}

func TestFallbackModelCustomErrorAndResponsePolicies(t *testing.T) {
	firstErr := errors.New("retry me")
	primary := requestModel{name: "primary", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, firstErr
	}}
	fallback := ai.NewFallbackModel(
		primary,
		ai.WithFallbackModels(fallbackTextModel("backup", "accepted", nil)),
		ai.WithFallbackOnError(func(context.Context, error) (bool, error) { return false, nil }),
		ai.WithFallbackOnError(func(_ context.Context, err error) (bool, error) { return errors.Is(err, firstErr), nil }),
	)
	response, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || response.Text() != "accepted" {
		t.Fatalf("custom error fallback failed: response=%+v err=%v", response, err)
	}

	firstCost, secondCost := 0.25, 0.5
	fallback = ai.NewFallbackModel(
		requestModel{name: "rejected", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.TextPart{Content: "reject"}}, Usage: ai.Usage{CostUSD: &firstCost},
			}, nil
		}},
		ai.WithFallbackModels(requestModel{name: "accepted", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.TextPart{Content: "accept"}}, Usage: ai.Usage{CostUSD: &secondCost},
			}, nil
		}}),
		ai.WithFallbackOnResponse(func(_ context.Context, response *ai.ModelResponse) (bool, error) {
			reject := response.Text() == "reject"
			response.Parts[0] = ai.TextPart{Content: "mutated"}
			return reject, nil
		}),
	)
	response, err = fallback.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || response.Text() != "accept" || response.Usage.CostUSD == nil || *response.Usage.CostUSD != 0.75 {
		t.Fatalf("response fallback failed: response=%+v err=%v", response, err)
	}
}

func TestFallbackModelPredicateErrorsAndExhaustion(t *testing.T) {
	predicateErr := errors.New("predicate")
	requestErr := errors.New("request")
	model := requestModel{name: "failure", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, requestErr
	}}
	fallback := ai.NewFallbackModel(model, ai.WithFallbackOnError(func(context.Context, error) (bool, error) {
		return false, predicateErr
	}))
	if _, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, predicateErr) {
		t.Fatalf("unexpected error predicate failure: %v", err)
	}

	responsePredicateErr := errors.New("response predicate")
	fallback = ai.NewFallbackModel(fallbackTextModel("response", "value", nil), ai.WithFallbackOnResponse(
		func(context.Context, *ai.ModelResponse) (bool, error) { return false, responsePredicateErr },
	))
	if _, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, responsePredicateErr) {
		t.Fatalf("unexpected response predicate failure: %v", err)
	}

	first := errors.New("first")
	second := errors.New("second")
	fallback = ai.NewFallbackModel(
		requestModel{name: "first", request: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return nil, first
		}},
		ai.WithFallbackModels(requestModel{name: "second", request: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return nil, second
		}}),
		ai.WithFallbackOnError(func(context.Context, error) (bool, error) { return true, nil }),
	)
	_, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{})
	var exhausted *ai.FallbackExhaustedError
	if !errors.Is(err, ai.ErrFallbackExhausted) || !errors.As(err, &exhausted) ||
		!errors.Is(err, first) || !errors.Is(err, second) || len(exhausted.Failures()) != 2 ||
		len(exhausted.RejectedResponses()) != 0 {
		t.Fatalf("unexpected exhausted error: %T %v", err, err)
	}
	failures := exhausted.Failures()
	failures[0] = nil
	if exhausted.Failures()[0] == nil {
		t.Fatal("failures were not detached")
	}

	fallback = ai.NewFallbackModel(fallbackTextModel("one", "reject", nil), ai.WithFallbackOnResponse(
		func(context.Context, *ai.ModelResponse) (bool, error) { return true, nil },
	))
	_, err = fallback.Request(t.Context(), nil, ai.ModelRequestParams{})
	if !errors.As(err, &exhausted) || len(exhausted.RejectedResponses()) != 1 ||
		!strings.Contains(err.Error(), "1 rejected response") {
		t.Fatalf("unexpected response exhaustion: %v", err)
	}
	rejected := exhausted.RejectedResponses()
	rejected[0].Parts[0] = ai.TextPart{Content: "mutated"}
	if exhausted.RejectedResponses()[0].Text() != "reject" {
		t.Fatal("rejected responses were not detached")
	}
}

func TestFallbackModelRejectsNilResponsesAndInvalidSettings(t *testing.T) {
	fallback := ai.NewFallbackModel(requestModel{name: "nil", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, nil
	}})
	if _, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "returned no response") {
		t.Fatalf("unexpected nil response error: %v", err)
	}
	if _, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{RequestTimeout: -time.Second},
	}); err == nil || !strings.Contains(err.Error(), "must be non-negative") {
		t.Fatalf("unexpected settings error: %v", err)
	}
	invalidProfile := invalidProfileModel{Model: requestModel{name: "invalid", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, nil
	}}}
	fallback = ai.NewFallbackModel(invalidProfile)
	if _, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{
		OutputMode: ai.OutputModeAuto, OutputSchema: map[string]any{"type": "object"},
	}); err == nil || !strings.Contains(err.Error(), "invalid default output mode") {
		t.Fatalf("unexpected profile error: %v", err)
	}
}

func TestFallbackModelPinsAndRewindsContinuations(t *testing.T) {
	primaryCalls, backupCalls, cancelCalls := 0, 0, 0
	apiErr := &fallbackAPIError{message: "continue elsewhere"}
	primary := requestModel{name: "primary", request: func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		primaryCalls++
		if len(messages) != 1 {
			t.Fatalf("rewound history retained suspended response: %+v", messages)
		}
		if primaryCalls == 1 {
			return nil, apiErr
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "fresh"}}}, nil
	}}
	backupBase := requestModel{name: "backup", request: func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		backupCalls++
		if backupCalls == 1 {
			return &ai.ModelResponse{State: ai.ModelResponseStateSuspended, ProviderResponseID: "job"}, nil
		}
		if messages[len(messages)-1].(ai.ModelResponse).ProviderResponseID != "job" {
			t.Fatalf("continuation was not pinned: %+v", messages)
		}
		return nil, apiErr
	}}
	backup := &continuationRequestModel{Model: backupBase}
	fallback := ai.NewFallbackModel(primary, ai.WithFallbackModels(backup))
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}}}
	suspended, err := fallback.Request(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil || suspended.State != ai.ModelResponseStateSuspended {
		t.Fatalf("initial fallback failed: response=%+v err=%v", suspended, err)
	}
	namespace := suspended.Metadata["__pydantic_ai__"].(map[string]any)
	if namespace["fallback_model_id"] != "backup" {
		t.Fatalf("continuation was not pinned: %+v", suspended.Metadata)
	}
	backup.cancelCount = 0
	fresh, err := fallback.Request(t.Context(), append(messages, *suspended), ai.ModelRequestParams{})
	cancelCalls = backup.cancelCount
	if err != nil || fresh.Text() != "fresh" || primaryCalls != 2 || backupCalls != 2 || cancelCalls != 1 {
		t.Fatalf("pinned fallback failed: response=%+v calls=%d/%d cancel=%d err=%v", fresh, primaryCalls, backupCalls, cancelCalls, err)
	}
	namespace = fresh.Metadata["__pydantic_ai__"].(map[string]any)
	if namespace["replace_previous_response"] != true {
		t.Fatalf("fresh response was not marked as a replacement: %+v", fresh.Metadata)
	}
}

func TestFallbackModelPinnedContinuationOutcomes(t *testing.T) {
	history := func(modelName string) []ai.ModelMessage {
		return []ai.ModelMessage{ai.ModelResponse{
			State:    ai.ModelResponseStateSuspended,
			Metadata: map[string]any{"__pydantic_ai__": map[string]any{"fallback_model_id": modelName}},
		}}
	}

	t.Run("success", func(t *testing.T) {
		calls := 0
		model := fallbackTextModel("pinned", "done", &calls)
		fallback := ai.NewFallbackModel(model)
		response, err := fallback.Request(t.Context(), history("pinned"), ai.ModelRequestParams{})
		if err != nil || response.Text() != "done" || calls != 1 {
			t.Fatalf("pinned continuation failed: response=%+v calls=%d err=%v", response, calls, err)
		}
	})

	t.Run("non-fallback error", func(t *testing.T) {
		requestErr := errors.New("stop")
		model := requestModel{name: "pinned", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, requestErr
		}}
		fallback := ai.NewFallbackModel(model)
		if _, err := fallback.Request(t.Context(), history("pinned"), ai.ModelRequestParams{}); !errors.Is(err, requestErr) {
			t.Fatalf("unexpected pinned error: %v", err)
		}
	})

	t.Run("nil response", func(t *testing.T) {
		model := requestModel{name: "pinned", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, nil
		}}
		fallback := ai.NewFallbackModel(model)
		if _, err := fallback.Request(t.Context(), history("pinned"), ai.ModelRequestParams{}); err == nil ||
			!strings.Contains(err.Error(), "returned no response") {
			t.Fatalf("unexpected pinned nil response error: %v", err)
		}
	})

	t.Run("predicate error", func(t *testing.T) {
		predicateErr := errors.New("policy")
		model := requestModel{name: "pinned", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, errors.New("request")
		}}
		fallback := ai.NewFallbackModel(model, ai.WithFallbackOnError(
			func(context.Context, error) (bool, error) { return false, predicateErr },
		))
		if _, err := fallback.Request(t.Context(), history("pinned"), ai.ModelRequestParams{}); !errors.Is(err, predicateErr) {
			t.Fatalf("unexpected policy error: %v", err)
		}
	})
}

func TestFallbackModelStreamsAndDoesNotRetryMidstream(t *testing.T) {
	apiErr := &fallbackAPIError{message: "open stream"}
	primary := streamingRequestModel{
		Model: fallbackTextModel("primary", "unused", nil),
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return nil, apiErr
		},
	}
	backupCalls := 0
	fallback := ai.NewFallbackModel(primary, ai.WithFallbackModels(fallbackTextModel("backup", "streamed", &backupCalls)))
	events, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var finish ai.FinishEvent
	for event, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.FinishEvent); ok {
			finish = event
		}
	}
	if backupCalls != 1 || finish.ModelName != "backup" {
		t.Fatalf("stream did not fall back: calls=%d finish=%+v", backupCalls, finish)
	}

	midstreamErr := errors.New("midstream")
	backupCalls = 0
	primary = streamingRequestModel{
		Model: fallbackTextModel("primary", "unused", nil),
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				yield(ai.TextDeltaEvent{Delta: "partial"}, nil)
				yield(nil, midstreamErr)
			}, nil
		},
	}
	fallback = ai.NewFallbackModel(
		primary,
		ai.WithFallbackModels(fallbackTextModel("backup", "unused", &backupCalls)),
		ai.WithFallbackOnError(func(context.Context, error) (bool, error) { return true, nil }),
	)
	events, err = fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var got error
	for _, err := range events {
		got = err
	}
	if !errors.Is(got, midstreamErr) || backupCalls != 0 {
		t.Fatalf("midstream error triggered fallback: calls=%d err=%v", backupCalls, got)
	}
}

func TestFallbackModelStreamOpeningFailures(t *testing.T) {
	history := []ai.ModelMessage{ai.ModelResponse{
		State:    ai.ModelResponseStateSuspended,
		Metadata: map[string]any{"__pydantic_ai__": map[string]any{"fallback_model_id": "pinned"}},
	}}
	streamErrorModel := func(err error) ai.Model {
		return streamingRequestModel{
			Model: fallbackTextModel("pinned", "unused", nil),
			stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
				return nil, err
			},
		}
	}

	t.Run("pinned success", func(t *testing.T) {
		model := streamingRequestModel{
			Model: fallbackTextModel("pinned", "unused", nil),
			stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
				return func(yield func(ai.ModelStreamEvent, error) bool) {
					yield(ai.FinishEvent{ModelName: "pinned"}, nil)
				}, nil
			},
		}
		fallback := ai.NewFallbackModel(model)
		if _, err := fallback.StreamRequest(t.Context(), history, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("pinned non-fallback", func(t *testing.T) {
		requestErr := errors.New("stop")
		fallback := ai.NewFallbackModel(streamErrorModel(requestErr))
		if _, err := fallback.StreamRequest(t.Context(), history, ai.ModelRequestParams{}); !errors.Is(err, requestErr) {
			t.Fatalf("unexpected pinned stream error: %v", err)
		}
	})

	t.Run("pinned predicate error", func(t *testing.T) {
		predicateErr := errors.New("predicate")
		fallback := ai.NewFallbackModel(streamErrorModel(errors.New("request")), ai.WithFallbackOnError(
			func(context.Context, error) (bool, error) { return false, predicateErr },
		))
		if _, err := fallback.StreamRequest(t.Context(), history, ai.ModelRequestParams{}); !errors.Is(err, predicateErr) {
			t.Fatalf("unexpected stream predicate error: %v", err)
		}
	})

	t.Run("pinned fallback and replacement", func(t *testing.T) {
		pinned := &fallbackStreamingContinuationModel{
			streamingRequestModel: streamErrorModel(&fallbackAPIError{message: "retry"}).(streamingRequestModel),
		}
		backup := streamingRequestModel{
			Model: fallbackTextModel("backup", "unused", nil),
			stream: func(_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
				if len(messages) != 0 {
					t.Fatalf("stream history was not rewound: %+v", messages)
				}
				return func(yield func(ai.ModelStreamEvent, error) bool) {
					yield(ai.ResponseMetadataEvent{}, nil)
					yield(ai.FinishEvent{}, nil)
				}, nil
			},
		}
		fallback := ai.NewFallbackModel(pinned, ai.WithFallbackModels(backup))
		events, err := fallback.StreamRequest(t.Context(), history, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for event, err := range events {
			if err != nil {
				t.Fatal(err)
			}
			switch event := event.(type) {
			case ai.ResponseMetadataEvent:
				if event.Metadata["__pydantic_ai__"].(map[string]any)["replace_previous_response"] != true {
					t.Fatalf("metadata event lacks replacement marker: %+v", event)
				}
			case ai.FinishEvent:
				if event.Metadata["__pydantic_ai__"].(map[string]any)["replace_previous_response"] != true {
					t.Fatalf("finish event lacks replacement marker: %+v", event)
				}
			}
		}
		if pinned.cancelCount != 1 {
			t.Fatalf("pinned stream was not canceled: %d", pinned.cancelCount)
		}
	})

	t.Run("chain predicate error", func(t *testing.T) {
		predicateErr := errors.New("predicate")
		fallback := ai.NewFallbackModel(streamErrorModel(errors.New("request")), ai.WithFallbackOnError(
			func(context.Context, error) (bool, error) { return false, predicateErr },
		))
		if _, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, predicateErr) {
			t.Fatalf("unexpected stream predicate error: %v", err)
		}
	})

	t.Run("chain non-fallback", func(t *testing.T) {
		requestErr := errors.New("stop")
		fallback := ai.NewFallbackModel(streamErrorModel(requestErr))
		if _, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, requestErr) {
			t.Fatalf("unexpected stream error: %v", err)
		}
	})

	t.Run("exhausted", func(t *testing.T) {
		fallback := ai.NewFallbackModel(
			streamErrorModel(errors.New("one")),
			ai.WithFallbackModels(streamErrorModel(errors.New("two"))),
			ai.WithFallbackOnError(func(context.Context, error) (bool, error) { return true, nil }),
		)
		if _, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, ai.ErrFallbackExhausted) {
			t.Fatalf("unexpected exhausted stream error: %v", err)
		}
	})

	t.Run("non-streaming opening error", func(t *testing.T) {
		fallback := ai.NewFallbackModel(
			requestModel{name: "primary", request: func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return nil, &fallbackAPIError{message: "retry"}
			}},
			ai.WithFallbackModels(fallbackTextModel("backup", "ok", nil)),
		)
		if _, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err != nil {
			t.Fatalf("non-streaming model did not fall back: %v", err)
		}
	})

	t.Run("invalid settings", func(t *testing.T) {
		fallback := ai.NewFallbackModel(fallbackTextModel("model", "unused", nil))
		_, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
			Settings: ai.ModelSettings{RequestTimeout: -time.Second},
		})
		if err == nil || !strings.Contains(err.Error(), "must be non-negative") {
			t.Fatalf("unexpected settings error: %v", err)
		}
	})

	t.Run("invalid profile", func(t *testing.T) {
		fallback := ai.NewFallbackModel(invalidProfileModel{Model: fallbackTextModel("model", "unused", nil)})
		_, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
			OutputMode: ai.OutputModeAuto, OutputSchema: map[string]any{"type": "object"},
		})
		if err == nil || !strings.Contains(err.Error(), "invalid default output mode") {
			t.Fatalf("unexpected profile error: %v", err)
		}
	})

	t.Run("nil fallback response", func(t *testing.T) {
		fallback := ai.NewFallbackModel(requestModel{name: "nil", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, nil
		}})
		if _, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
			!strings.Contains(err.Error(), "returned no response") {
			t.Fatalf("unexpected nil stream response error: %v", err)
		}
	})
}

func TestFallbackModelStreamingContinuationMetadata(t *testing.T) {
	model := streamingRequestModel{
		Model: fallbackTextModel("stream", "unused", nil),
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				if !yield(ai.ResponseMetadataEvent{State: ai.ModelResponseStateSuspended}, nil) {
					return
				}
				yield(ai.FinishEvent{State: ai.ModelResponseStateSuspended}, nil)
			}, nil
		},
	}
	fallback := ai.NewFallbackModel(model)
	events, err := fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.ResponseMetadataEvent:
			if event.Metadata["__pydantic_ai__"].(map[string]any)["fallback_model_id"] != "stream" {
				t.Fatalf("metadata event was not pinned: %+v", event)
			}
		case ai.FinishEvent:
			if event.Metadata["__pydantic_ai__"].(map[string]any)["fallback_model_id"] != "stream" {
				t.Fatalf("finish event was not pinned: %+v", event)
			}
		}
	}
	events, err = fallback.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
		break
	}
}

func TestFallbackModelLifecycleAndCapabilities(t *testing.T) {
	order := []string{}
	newLifecycle := func(name string, openErr, closeErr error) ai.Model {
		return lifecycleRequestModel{
			Model: fallbackTextModel(name, name, nil),
			open: func(context.Context) (ai.ModelCloseFunc, error) {
				order = append(order, "open "+name)
				if openErr != nil {
					return nil, openErr
				}
				return func(ctx context.Context) error {
					if ctx.Err() != nil {
						t.Fatal("fallback close context was canceled")
					}
					order = append(order, "close "+name)
					return closeErr
				}, nil
			},
		}
	}
	fallback := ai.NewFallbackModel(
		newLifecycle("one", nil, nil),
		ai.WithFallbackModels(fallbackTextModel("plain", "plain", nil), newLifecycle("two", nil, nil)),
	)
	closeModels, err := fallback.OpenModel(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := closeModels(t.Context()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"open one", "open two", "close two", "close one"}) {
		t.Fatalf("unexpected lifecycle order: %v", order)
	}

	order = nil
	openErr := errors.New("open")
	rollbackErr := errors.New("rollback")
	fallback = ai.NewFallbackModel(
		newLifecycle("one", nil, rollbackErr),
		ai.WithFallbackModels(newLifecycle("two", openErr, nil)),
	)
	if _, err := fallback.OpenModel(t.Context()); !errors.Is(err, openErr) || !errors.Is(err, rollbackErr) ||
		!reflect.DeepEqual(order, []string{"open one", "open two", "close one"}) {
		t.Fatalf("partial open was not rolled back: order=%v err=%v", order, err)
	}

	closeErr := errors.New("close")
	fallback = ai.NewFallbackModel(newLifecycle("one", nil, closeErr))
	closeModels, err = fallback.OpenModel(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := closeModels(t.Context()); !errors.Is(err, closeErr) {
		t.Fatalf("close failure was not propagated: %v", err)
	}

	first := &fallbackCapabilityModel{
		requestModel: requestModel{name: "one", request: fallbackTextModel("one", "one", nil).Request},
		provider:     "same", strategy: ai.ToolSearchStrategyRegex, delay: time.Second,
	}
	second := &fallbackCapabilityModel{
		requestModel: requestModel{name: "two", request: fallbackTextModel("two", "two", nil).Request},
		provider:     "same", strategy: ai.ToolSearchStrategyRegex,
	}
	fallback = ai.NewFallbackModel(first, ai.WithFallbackModels(second))
	if !fallback.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) ||
		fallback.SupportsToolSearchStrategy(ai.ToolSearchStrategyBM25) ||
		fallback.NativeToolSearchProvider() != "same" || fallback.ContinuationDelay(ai.ModelResponse{}) != time.Second {
		t.Fatal("fallback capabilities were not combined")
	}
	if err := fallback.CancelSuspendedResponse(t.Context(), ai.ModelResponse{}); err != nil ||
		first.cancelled != 1 || second.cancelled != 1 {
		t.Fatalf("unrouted cancellation was not broadcast: %d/%d err=%v", first.cancelled, second.cancelled, err)
	}

	pinned := ai.ModelResponse{Metadata: map[string]any{
		"__pydantic_ai__": map[string]any{"fallback_model_id": "two"},
	}}
	second.delay = 2 * time.Second
	second.cancelErr = errors.New("cancel")
	if fallback.ContinuationDelay(pinned) != 2*time.Second {
		t.Fatal("pinned continuation delay was not delegated")
	}
	if err := fallback.CancelSuspendedResponse(t.Context(), pinned); !errors.Is(err, second.cancelErr) {
		t.Fatalf("pinned cancellation error was not propagated: %v", err)
	}
	mixed := &fallbackCapabilityModel{
		requestModel: requestModel{name: "mixed", request: fallbackTextModel("mixed", "mixed", nil).Request},
		provider:     "other", strategy: ai.ToolSearchStrategyRegex,
	}
	if ai.NewFallbackModel(first, ai.WithFallbackModels(mixed)).NativeToolSearchProvider() != "" {
		t.Fatal("mixed providers reported a native history provider")
	}
	plain := ai.NewFallbackModel(fallbackTextModel("plain", "plain", nil))
	if plain.NativeToolSearchProvider() != "" || plain.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) ||
		plain.ContinuationDelay(ai.ModelResponse{}) != 0 {
		t.Fatal("plain fallback reported optional capabilities")
	}
	plainPinned := ai.ModelResponse{Metadata: map[string]any{
		"__pydantic_ai__": map[string]any{"fallback_model_id": "plain"},
	}}
	if plain.ContinuationDelay(plainPinned) != 0 || plain.CancelSuspendedResponse(t.Context(), plainPinned) != nil {
		t.Fatal("plain pinned model reported continuation behavior")
	}
	unknownPinned := ai.ModelResponse{Metadata: map[string]any{
		"__pydantic_ai__": map[string]any{"fallback_model_id": "unknown"},
	}}
	if fallback.ContinuationDelay(unknownPinned) != time.Second {
		t.Fatal("unknown pin did not fall back to the first available delay")
	}
}

func TestFallbackModelConstructorValidation(t *testing.T) {
	for name, test := range map[string]struct {
		call func()
		want string
	}{
		"primary": {call: func() { ai.NewFallbackModel(nil) }, want: "primary fallback model"},
		"fallback": {call: func() {
			ai.NewFallbackModel(fallbackTextModel("one", "one", nil), ai.WithFallbackModels(nil))
		}, want: "fallback model"},
		"error predicate": {call: func() {
			ai.NewFallbackModel(fallbackTextModel("one", "one", nil), ai.WithFallbackOnError(nil))
		}, want: "error predicate"},
		"response predicate": {call: func() {
			ai.NewFallbackModel(fallbackTextModel("one", "one", nil), ai.WithFallbackOnResponse(nil))
		}, want: "response predicate"},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered == nil || !strings.Contains(recovered.(string), test.want) {
					t.Fatalf("unexpected panic: %v", recovered)
				}
			}()
			test.call()
		})
	}
}

var _ ai.ModelAPIError = (*fallbackAPIError)(nil)
var _ ai.StreamingModel = (*ai.FallbackModel)(nil)
var _ ai.ModelOpener = (*ai.FallbackModel)(nil)
var _ ai.ToolSearchStrategyModel = (*ai.FallbackModel)(nil)
var _ ai.NativeToolSearchHistoryModel = (*ai.FallbackModel)(nil)
var _ ai.ModelContinuationDelayer = (*ai.FallbackModel)(nil)
var _ ai.SuspendedResponseCanceler = (*ai.FallbackModel)(nil)
