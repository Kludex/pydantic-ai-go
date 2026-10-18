package ai_test

import (
	"context"
	"errors"
	"iter"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type continuationWrapper struct {
	calls         int
	clearMessages bool
}

func (*continuationWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (w *continuationWrapper) WrapModelRequest(
	ctx context.Context,
	_ *ai.RunInfo,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	w.calls++
	if w.clearMessages {
		messages = nil
	}
	return next(ctx, messages, params)
}

type responseErrorContinuationModel struct {
	calls int
	seen  []ai.ModelResponse
}

func (m *responseErrorContinuationModel) Name() string { return "response-error" }

func (m *responseErrorContinuationModel) Request(
	_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	m.calls++
	if m.calls == 1 {
		return &ai.ModelResponse{
			ModelName: "model", ProviderResponseID: "first", State: ai.ModelResponseStateSuspended,
		}, nil
	}
	return &ai.ModelResponse{
		ModelName: "model", ProviderResponseID: "second", State: ai.ModelResponseStateSuspended,
	}, errors.New("stream failed after response")
}

func (m *responseErrorContinuationModel) CancelSuspendedResponse(
	_ context.Context, response ai.ModelResponse,
) error {
	m.seen = append(m.seen, response)
	return nil
}

type immediateContinuationModel struct {
	responses  []*ai.ModelResponse
	requestErr error
}

func (m *immediateContinuationModel) Name() string { return "immediate" }

func (m *immediateContinuationModel) Request(
	_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	if len(m.responses) == 0 {
		return nil, m.requestErr
	}
	response := m.responses[0]
	m.responses = m.responses[1:]
	return response, nil
}

type continuationModel struct {
	mu          sync.Mutex
	responses   []*ai.ModelResponse
	requests    [][]ai.ModelMessage
	requestErr  error
	delay       time.Duration
	delayCalled chan struct{}
	canceled    []ai.ModelResponse
}

func (m *continuationModel) Name() string { return "continuation" }

func (m *continuationModel) Request(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, slices.Clone(messages))
	if len(m.responses) == 0 {
		return nil, m.requestErr
	}
	response := m.responses[0]
	m.responses = m.responses[1:]
	return response, nil
}

func (m *continuationModel) StreamRequest(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	m.mu.Lock()
	m.requests = append(m.requests, slices.Clone(messages))
	if len(m.responses) == 0 {
		err := m.requestErr
		m.mu.Unlock()
		return nil, err
	}
	response := m.responses[0]
	m.responses = m.responses[1:]
	m.mu.Unlock()
	return continuationResponseEvents(response), nil
}

func (m *continuationModel) ContinuationDelay(ai.ModelResponse) time.Duration {
	if m.delayCalled != nil {
		select {
		case m.delayCalled <- struct{}{}:
		default:
		}
	}
	return m.delay
}

func (m *continuationModel) CancelSuspendedResponse(ctx context.Context, response ai.ModelResponse) error {
	if ctx.Err() != nil {
		return errors.New("cancellation context is canceled")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.canceled = append(m.canceled, response)
	return errors.New("best-effort cancellation failed")
}

func continuationResponseEvents(response *ai.ModelResponse) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		if !yield(ai.ResponseMetadataEvent{
			Usage: response.Usage, ModelName: response.ModelName, ProviderName: response.ProviderName,
			ProviderDetails: response.ProviderDetails, ProviderResponseID: response.ProviderResponseID,
			State: response.State,
		}, nil) {
			return
		}
		for index, part := range response.Parts {
			partID := string(rune('a' + index))
			switch part := part.(type) {
			case ai.TextPart:
				middle := len(part.Content) / 2
				if middle == 0 {
					middle = len(part.Content)
				}
				if !yield(ai.TextDeltaEvent{PartID: partID, Delta: part.Content[:middle]}, nil) {
					return
				}
				if middle < len(part.Content) && !yield(ai.TextDeltaEvent{
					PartID: partID, Delta: part.Content[middle:],
				}, nil) {
					return
				}
			case ai.ThinkingPart:
				if !yield(ai.ThinkingDeltaEvent{PartID: partID, Delta: part.Content}, nil) {
					return
				}
			}
		}
		yield(ai.FinishEvent{
			Usage: response.Usage, ModelName: response.ModelName, ProviderName: response.ProviderName,
			ProviderDetails: response.ProviderDetails, ProviderResponseID: response.ProviderResponseID,
			State: response.State,
		}, nil)
	}
}

func TestResumeContinuesSuspendedHistoryWithoutNewPrompt(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "original"}}, RunID: "old-run"},
		ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "partial"}}, Usage: ai.Usage{Requests: 1},
			ModelName: "model", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		},
	}
	model := &continuationModel{responses: []*ai.ModelResponse{{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Usage: ai.Usage{Requests: 1},
		ModelName: "model", ProviderResponseID: "job", State: ai.ModelResponseStateComplete,
	}}}
	agent := ai.NewAgent[deps, string](model)
	result, err := agent.Resume(
		t.Context(), history, deps{}, ai.WithMessageHistory([]ai.ModelMessage{ai.ModelRequest{}}),
	)
	if err != nil || result.Output != "done" || result.Usage().Requests != 1 {
		t.Fatalf("unexpected resumed result=%+v err=%v", result, err)
	}
	if len(model.requests) != 1 || len(model.requests[0]) != 2 {
		t.Fatalf("resume added a request instead of echoing the seed: %+v", model.requests)
	}
	seed := model.requests[0][1].(ai.ModelResponse)
	if seed.State != ai.ModelResponseStateSuspended || seed.ProviderResponseID != "job" {
		t.Fatalf("unexpected resume seed: %+v", seed)
	}
	if newMessages := result.NewMessages(); len(newMessages) != 1 ||
		newMessages[0].(ai.ModelResponse).State != ai.ModelResponseStateComplete {
		t.Fatalf("unexpected resumed new messages: %+v", newMessages)
	}
	if len(result.Messages()) != 2 {
		t.Fatalf("suspended seed remained in completed history: %+v", result.Messages())
	}
	if original := result.Messages()[0].(ai.ModelRequest); original.RunID != "old-run" {
		t.Fatalf("resume mutated historical request context: %+v", original)
	}
}

func TestResumeStreamContinuesSuspendedHistory(t *testing.T) {
	history := []ai.ModelMessage{ai.ModelResponse{
		ProviderResponseID: "job", ModelName: "model", State: ai.ModelResponseStateSuspended,
	}}
	model := &continuationModel{responses: []*ai.ModelResponse{{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, ProviderResponseID: "job",
		ModelName: "model", State: ai.ModelResponseStateComplete,
	}}}
	stream := ai.NewAgent[deps, string](model).ResumeStream(t.Context(), history, deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if result := stream.Result(); result == nil || result.Output != "done" || len(result.NewMessages()) != 1 {
		t.Fatalf("unexpected resumed stream result=%+v", result)
	}
}

func TestResumeRequiresSuspendedHistory(t *testing.T) {
	for name, history := range map[string][]ai.ModelMessage{
		"empty":    nil,
		"request":  {ai.ModelRequest{}},
		"complete": {ai.ModelResponse{State: ai.ModelResponseStateComplete}},
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, string](&immediateContinuationModel{})
			if _, err := agent.Resume(t.Context(), history, deps{}); !errors.Is(err, ai.ErrNoSuspendedResponse) {
				t.Fatalf("unexpected resume error: %v", err)
			}
			stream := agent.ResumeStream(t.Context(), history, deps{})
			for _, err := range stream.Events() {
				if !errors.Is(err, ai.ErrNoSuspendedResponse) {
					t.Fatalf("unexpected stream resume error: %v", err)
				}
			}
			if stream.Result() != nil {
				t.Fatalf("invalid resumed stream completed: %+v", stream.Result())
			}
		})
	}
}

func TestModelRequestMiddlewareMayReplaceMessagesWithEmptyHistory(t *testing.T) {
	wrapper := &continuationWrapper{clearMessages: true}
	model := &immediateContinuationModel{responses: []*ai.ModelResponse{{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
	}}}
	result, err := ai.NewAgent[deps, string](model, ai.WithCapabilities(wrapper)).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || wrapper.calls != 1 {
		t.Fatalf("unexpected empty middleware history result=%+v err=%v calls=%d", result, err, wrapper.calls)
	}
}

func TestContinuationRunsModelRequestMiddlewareOnce(t *testing.T) {
	wrapper := &continuationWrapper{}
	model := &immediateContinuationModel{responses: []*ai.ModelResponse{
		{ModelName: "model", State: ai.ModelResponseStateSuspended},
		{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, ModelName: "model"},
	}}
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(wrapper))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || wrapper.calls != 1 {
		t.Fatalf("continuation escaped request wrapper result=%+v err=%v calls=%d", result, err, wrapper.calls)
	}
}

func TestSuspendedResponsesAccumulateAsOneTurn(t *testing.T) {
	firstMetadata := map[string]any{"first": true}
	model := &continuationModel{responses: []*ai.ModelResponse{
		{
			Parts: []ai.ResponsePart{ai.ThinkingPart{Content: "working"}, ai.TextPart{Content: "hel"}},
			Usage: ai.Usage{Requests: 1, InputTokens: 2}, ModelName: "same", ProviderName: "provider",
			ProviderResponseID: "segment-1", ProviderDetails: map[string]any{"container": "one"},
			Metadata: firstMetadata, State: ai.ModelResponseStateSuspended,
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "lo"}},
			Usage: ai.Usage{Requests: 1, OutputTokens: 3}, ModelName: "same", ProviderName: "provider",
			ProviderResponseID: "segment-2", ProviderDetails: map[string]any{"latest": true},
			Metadata: map[string]any{"second": true}, State: ai.ModelResponseStateComplete,
		},
	}}
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "hello" {
		t.Fatalf("unexpected continuation result=%+v err=%v", result, err)
	}
	if usage := result.Usage(); usage.Requests != 2 || usage.InputTokens != 2 || usage.OutputTokens != 3 {
		t.Fatalf("continuation usage was not accumulated: %+v", usage)
	}
	if len(model.requests) != 2 || len(model.requests[1]) != len(model.requests[0])+1 {
		t.Fatalf("suspended response was not echoed: %+v", model.requests)
	}
	echoed := model.requests[1][len(model.requests[1])-1].(ai.ModelResponse)
	if echoed.State != ai.ModelResponseStateSuspended || echoed.ProviderResponseID != "segment-1" {
		t.Fatalf("unexpected continuation seed: %+v", echoed)
	}
	response := result.NewMessages()[1].(ai.ModelResponse)
	if len(response.Parts) != 3 || response.ProviderResponseID != "segment-2" ||
		response.ProviderDetails["container"] != "one" || response.ProviderDetails["latest"] != true ||
		response.Metadata["first"] != true || response.Metadata["second"] != true {
		t.Fatalf("unexpected merged response: %+v", response)
	}
	firstMetadata["mutated"] = true
	if response.Metadata["mutated"] != nil {
		t.Fatal("merged continuation metadata aliases a provider response")
	}
}

func TestContinuationPreservesLastKnownResponseID(t *testing.T) {
	firstDetails := map[string]any{"first": true}
	model := &immediateContinuationModel{responses: []*ai.ModelResponse{
		{
			ModelName: "model", ProviderResponseID: "known", ProviderDetails: firstDetails,
			State: ai.ModelResponseStateSuspended,
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, ModelName: "model",
			State: ai.ModelResponseStateComplete,
		},
	}}
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected immediate continuation result=%+v err=%v", result, err)
	}
	response := result.NewMessages()[1].(ai.ModelResponse)
	if response.ProviderResponseID != "known" || response.ProviderDetails["first"] != true {
		t.Fatalf("continuation lost provider metadata: %+v", response)
	}
	firstDetails["mutated"] = true
	if response.ProviderDetails["mutated"] != nil {
		t.Fatal("merged provider metadata aliases a prior segment")
	}
}

func TestBackgroundContinuationReplacesCumulativeSnapshots(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "partial"}}, Usage: ai.Usage{Requests: 1, InputTokens: 4},
			ModelName: "model", ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
			State: ai.ModelResponseStateSuspended,
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "final"}}, Usage: ai.Usage{Requests: 1, InputTokens: 4, OutputTokens: 2},
			ModelName: "model", ProviderResponseID: "job", State: ai.ModelResponseStateComplete,
		},
	}}
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "final" {
		t.Fatalf("unexpected background continuation result=%+v err=%v", result, err)
	}
	if usage := result.Usage(); usage.Requests != 1 || usage.InputTokens != 4 || usage.OutputTokens != 2 {
		t.Fatalf("background cumulative usage was added twice: %+v", usage)
	}
}

func TestContinuationReplacementMarkerSupersedesPriorSegments(t *testing.T) {
	marker := map[string]any{"__pydantic_ai__": map[string]any{
		"replace_previous_response": true,
		"continuation_pin":          "pin",
	}}
	model := &continuationModel{responses: []*ai.ModelResponse{
		{
			Parts: []ai.ResponsePart{
				ai.TextPart{Content: "old"},
				ai.ToolCallPart{ToolName: "discarded", Args: []byte(`{}`), ProviderDetails: map[string]any{"old": true}},
			},
			Usage: ai.Usage{Requests: 1}, ModelName: "first", ProviderResponseID: "old",
			State: ai.ModelResponseStateSuspended,
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "new"}}, Usage: ai.Usage{Requests: 1},
			ModelName: "first", ProviderResponseID: "new", Metadata: marker, State: ai.ModelResponseStateComplete,
		},
	}}
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "new" || result.Usage().Requests != 2 {
		t.Fatalf("unexpected replacement result=%+v err=%v", result, err)
	}
	response := result.NewMessages()[1].(ai.ModelResponse)
	namespace := response.Metadata["__pydantic_ai__"].(map[string]any)
	if namespace["replace_previous_response"] != nil || namespace["continuation_pin"] != "pin" {
		t.Fatalf("transient replacement marker survived: %+v", response.Metadata)
	}
}

func TestContinuationReplacementMarkerIsRemovedWhenNamespaceBecomesEmpty(t *testing.T) {
	model := &immediateContinuationModel{responses: []*ai.ModelResponse{
		{ModelName: "model", State: ai.ModelResponseStateSuspended},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, ModelName: "model",
			Metadata: map[string]any{"__pydantic_ai__": map[string]any{"replace_previous_response": true}},
			State:    ai.ModelResponseStateComplete,
		},
	}}
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected empty-marker result=%+v err=%v", result, err)
	}
	response := result.NewMessages()[1].(ai.ModelResponse)
	if response.Metadata["__pydantic_ai__"] != nil {
		t.Fatalf("empty marker namespace survived: %+v", response.Metadata)
	}
}

func TestContinuationModelChangeReplacesPriorSegments(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "old"}}, Usage: ai.Usage{Requests: 1},
			ModelName: "first", State: ai.ModelResponseStateSuspended,
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "new"}}, Usage: ai.Usage{Requests: 1},
			ModelName: "second", State: ai.ModelResponseStateComplete,
		},
	}}
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "new" || result.Usage().Requests != 2 {
		t.Fatalf("unexpected model-change continuation result=%+v err=%v", result, err)
	}
}

func TestContinuationLimitsAndFailuresCancelSuspendedJobs(t *testing.T) {
	t.Run("fresh generations", func(t *testing.T) {
		responses := make([]*ai.ModelResponse, ai.MaxGenerationContinuations+1)
		for index := range responses {
			responses[index] = &ai.ModelResponse{
				ModelName: "model", ProviderResponseID: strconv.Itoa(index), State: ai.ModelResponseStateSuspended,
			}
		}
		model := &continuationModel{responses: responses}
		_, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
		if err == nil || !strings.Contains(err.Error(), "suspended more than the maximum of 10 times") ||
			len(model.canceled) != 1 {
			t.Fatalf("unexpected generation limit err=%v canceled=%d", err, len(model.canceled))
		}
	})

	t.Run("background polls", func(t *testing.T) {
		responses := make([]*ai.ModelResponse, ai.MaxBackgroundPolls+2)
		for index := range responses {
			responses[index] = &ai.ModelResponse{
				ModelName: "model", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
			}
		}
		model := &continuationModel{responses: responses}
		_, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
		if err == nil || !strings.Contains(err.Error(), "polling the maximum of 1000 times") ||
			len(model.canceled) != 1 {
			t.Fatalf("unexpected poll limit err=%v canceled=%d", err, len(model.canceled))
		}
	})

	t.Run("next request fails", func(t *testing.T) {
		model := &continuationModel{
			responses:  []*ai.ModelResponse{{ProviderResponseID: "job", State: ai.ModelResponseStateSuspended}},
			requestErr: errors.New("network failed"),
		}
		_, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
		if err == nil || !strings.Contains(err.Error(), "network failed") || len(model.canceled) != 1 {
			t.Fatalf("unexpected continuation failure err=%v canceled=%d", err, len(model.canceled))
		}
	})

	t.Run("usage limit", func(t *testing.T) {
		model := &continuationModel{responses: []*ai.ModelResponse{{
			Usage: ai.Usage{Requests: 1, OutputTokens: 4}, ProviderResponseID: "job",
			State: ai.ModelResponseStateSuspended,
		}}}
		agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{OutputTokenLimit: 3}))
		_, err := agent.Run(t.Context(), "go", deps{})
		if !errors.Is(err, ai.ErrUsageLimitExceeded) || len(model.requests) != 1 || len(model.canceled) != 1 {
			t.Fatalf("unexpected continuation usage limit err=%v requests=%d canceled=%d", err, len(model.requests), len(model.canceled))
		}
	})
}

func TestContinuationWithoutOptionalLifecycleInterfaces(t *testing.T) {
	model := &immediateContinuationModel{
		responses:  []*ai.ModelResponse{{State: ai.ModelResponseStateSuspended}},
		requestErr: errors.New("continuation failed"),
	}
	_, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "continuation failed") {
		t.Fatalf("unexpected immediate continuation error: %v", err)
	}
}

func TestContinuationDelayCompletes(t *testing.T) {
	model := &continuationModel{
		responses: []*ai.ModelResponse{
			{State: ai.ModelResponseStateSuspended},
			{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, State: ai.ModelResponseStateComplete},
		},
		delay: time.Millisecond,
	}
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected delayed continuation result=%+v err=%v", result, err)
	}
}

func TestContinuationMergesPartialResponseBeforeCancellation(t *testing.T) {
	model := &responseErrorContinuationModel{}
	_, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "stream failed after response") || len(model.seen) != 1 ||
		model.seen[0].ProviderResponseID != "second" {
		t.Fatalf("unexpected partial response cancellation err=%v seen=%+v", err, model.seen)
	}
}

func TestNilContinuationSegmentCancelsSuspendedResponse(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{ProviderResponseID: "job", State: ai.ModelResponseStateSuspended},
		nil,
	}}
	_, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "model returned no response") || len(model.canceled) != 1 {
		t.Fatalf("unexpected nil continuation err=%v canceled=%d", err, len(model.canceled))
	}
}

func TestContinuationDelayRespectsCancellation(t *testing.T) {
	called := make(chan struct{}, 1)
	model := &continuationModel{
		responses:   []*ai.ModelResponse{{ProviderResponseID: "job", State: ai.ModelResponseStateSuspended}},
		delay:       time.Hour,
		delayCalled: called,
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		_, err := ai.NewAgent[deps, string](model).Run(ctx, "go", deps{})
		done <- err
	}()
	<-called
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) || len(model.canceled) != 1 {
		t.Fatalf("unexpected canceled delay err=%v canceled=%d", err, len(model.canceled))
	}
}

func TestStreamedBackgroundContinuationReusesPartIndexes(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "old"}}, ModelName: "model",
			ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
			State: ai.ModelResponseStateSuspended,
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, ModelName: "model",
			ProviderResponseID: "job", State: ai.ModelResponseStateComplete,
		},
	}}
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var starts []int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.PartStartEvent); ok {
			starts = append(starts, event.Index)
		}
	}
	if result := stream.Result(); result == nil || result.Output != "done" || !slices.Equal(starts, []int{0, 0}) {
		t.Fatalf("unexpected streamed replacement result=%+v starts=%v", result, starts)
	}
}

func TestStoppingAfterSuspendedFinishDetachesJob(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{{
		ProviderResponseID: "one", State: ai.ModelResponseStateSuspended,
	}}}
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for range stream.Events() {
		break
	}
	if stream.Result() != nil || len(model.canceled) != 0 || stream.Suspended() == nil ||
		stream.Suspended().Response().ProviderResponseID != "one" {
		t.Fatalf("suspended finish was not detached: result=%+v canceled=%+v snapshot=%+v", stream.Result(), model.canceled, stream.Suspended())
	}
}

func TestStoppingDuringSuspendedContinuationDetachesJob(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{ProviderResponseID: "one", State: ai.ModelResponseStateSuspended},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "pending"}}, ModelName: "model",
			ProviderResponseID: "two", State: ai.ModelResponseStateSuspended,
		},
	}}
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	events := 0
	for range stream.Events() {
		events++
		if events == 2 {
			break
		}
	}
	if stream.Result() != nil || len(model.canceled) != 0 || stream.Suspended() == nil ||
		stream.Suspended().Response().ProviderResponseID != "two" {
		t.Fatalf("stopped continuation was not detached: result=%+v canceled=%+v snapshot=%+v", stream.Result(), model.canceled, stream.Suspended())
	}
}

func TestStoppingDuringCompleteContinuationPreservesPriorSuspendedJob(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{ProviderResponseID: "one", State: ai.ModelResponseStateSuspended},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "pending"}}, ModelName: "model",
			ProviderResponseID: "two", State: ai.ModelResponseStateComplete,
		},
	}}
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	events := 0
	for range stream.Events() {
		events++
		if events == 2 {
			break
		}
	}
	if stream.Result() != nil || len(model.canceled) != 0 || stream.Suspended() == nil ||
		stream.Suspended().Response().ProviderResponseID != "one" {
		t.Fatalf(
			"complete continuation did not preserve prior job: result=%+v canceled=%+v snapshot=%+v",
			stream.Result(), model.canceled, stream.Suspended(),
		)
	}
}

func TestStreamedContinuationsReindexAccumulatedParts(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{
			Parts: []ai.ResponsePart{ai.ThinkingPart{Content: "first"}}, Usage: ai.Usage{Requests: 1},
			ModelName: "model", ProviderResponseID: "one", State: ai.ModelResponseStateSuspended,
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Usage: ai.Usage{Requests: 1},
			ModelName: "model", ProviderResponseID: "two", State: ai.ModelResponseStateComplete,
		},
	}}
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var starts []int
	finishes := 0
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			starts = append(starts, event.Index)
		case ai.FinishEvent:
			finishes++
		}
	}
	result := stream.Result()
	if result == nil || result.Output != "done" || !slices.Equal(starts, []int{0, 1}) || finishes != 2 ||
		result.Usage().Requests != 2 {
		t.Fatalf("unexpected streamed continuation result=%+v starts=%v finishes=%d", result, starts, finishes)
	}
}

func TestResumeCountsOnlyUsageBilledBeyondSuspendedResponse(t *testing.T) {
	cost := func(value float64) *float64 { return &value }
	seed := func() ai.ModelResponse {
		return ai.ModelResponse{
			Parts:     []ai.ResponsePart{ai.TextPart{Content: "a"}},
			ModelName: "model", ProviderResponseID: "r1", State: ai.ModelResponseStateSuspended,
			Usage: ai.Usage{
				Requests: 1, InputTokens: 5, OutputTokens: 2, CostUSD: cost(0.02), Details: map[string]int{"searches": 1},
			},
			FailedAttempts: []ai.ModelRequestAttempt{{
				ModelName: "rejected", Outcome: ai.ModelRequestAttemptRejected,
				Usage: &ai.Usage{InputTokens: 100, OutputTokens: 10},
			}},
		}
	}
	segment := func(id string, state ai.ModelResponseState, input, output int, usd float64) *ai.ModelResponse {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: id}}, ModelName: "model", ProviderResponseID: id,
			State: state, Usage: ai.Usage{Requests: 1, InputTokens: input, OutputTokens: output, CostUSD: cost(usd)},
		}
	}
	shapes := map[string]struct {
		segments func() []*ai.ModelResponse
		billed   int
		turn     int
	}{
		"pause turn": {
			segments: func() []*ai.ModelResponse {
				return []*ai.ModelResponse{segment("r2", ai.ModelResponseStateComplete, 8, 6, 0.003)}
			},
			billed: 14, turn: 21,
		},
		"background poll": {
			segments: func() []*ai.ModelResponse {
				return []*ai.ModelResponse{segment("r1", ai.ModelResponseStateComplete, 8, 6, 0.023)}
			},
			billed: 7, turn: 14,
		},
		"three segments": {
			segments: func() []*ai.ModelResponse {
				return []*ai.ModelResponse{
					segment("r2", ai.ModelResponseStateSuspended, 4, 1, 0.001),
					segment("r3", ai.ModelResponseStateComplete, 3, 1, 0.002),
				}
			},
			billed: 9, turn: 16,
		},
	}
	for name, shape := range shapes {
		for _, streamed := range []bool{false, true} {
			t.Run(name+"/stream="+strconv.FormatBool(streamed), func(t *testing.T) {
				history := []ai.ModelMessage{
					ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}}, seed(),
				}
				model := &continuationModel{responses: shape.segments()}
				agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{
					TotalTokenLimit: shape.billed, PerRequestInputTokenLimit: shape.billed, CostLimitUSD: cost(0.01),
				}))
				var result *ai.RunResult[string]
				var err error
				if streamed {
					stream := agent.ResumeStream(t.Context(), history, deps{})
					for _, eventErr := range stream.Events() {
						if eventErr != nil {
							err = eventErr
						}
						if live := stream.Usage().TotalTokens(); live > shape.billed {
							t.Fatalf("live usage %d counts the resumed response", live)
						}
					}
					result = stream.Result()
				} else {
					result, err = agent.Resume(t.Context(), history, deps{})
				}
				if err != nil {
					t.Fatal(err)
				}
				usage := result.Usage()
				if usage.TotalTokens() != shape.billed || usage.CostUSD == nil || *usage.CostUSD > 0.0031 ||
					*usage.CostUSD < 0.0029 {
					t.Fatalf("resumed run usage = %+v, want %d tokens and $0.003", usage, shape.billed)
				}
				messages := result.Messages()
				response := messages[len(messages)-1].(ai.ModelResponse)
				if usage.Details["searches"] != response.Usage.Details["searches"]-1 {
					t.Fatalf("resumed run details = %v, turn details = %v", usage.Details, response.Usage.Details)
				}
				if response.Usage.TotalTokens() != shape.turn || len(response.FailedAttempts) != 1 {
					t.Fatalf("committed response no longer covers the whole turn: %+v", response)
				}
			})
		}
	}
}

func TestResumingWithCarriedUsageCapsTheConversationTotal(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
		ai.ModelResponse{
			ModelName: "model", ProviderResponseID: "r1", State: ai.ModelResponseStateSuspended,
			Usage: ai.Usage{Requests: 1, InputTokens: 5, OutputTokens: 2},
		},
	}
	model := &continuationModel{responses: []*ai.ModelResponse{{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, ModelName: "model", ProviderResponseID: "r2",
		Usage: ai.Usage{Requests: 1, InputTokens: 8, OutputTokens: 6},
	}}}
	carried := ai.Conversation{Messages: history, Usage: ai.Usage{Requests: 1, InputTokens: 5, OutputTokens: 2}}
	agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{TotalTokenLimit: 21}))
	result, err := agent.Resume(t.Context(), nil, deps{}, ai.WithConversation(carried))
	if err != nil || result.Usage().TotalTokens() != 21 {
		t.Fatalf("carried usage = %+v err=%v, want 21 tokens", result, err)
	}
}

func TestWalkAwayAndResumeCountWhatTheProviderBilled(t *testing.T) {
	model := &continuationModel{responses: []*ai.ModelResponse{
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "a"}}, ModelName: "model", ProviderResponseID: "r1",
			State: ai.ModelResponseStateSuspended, Usage: ai.Usage{Requests: 1, InputTokens: 5, OutputTokens: 2},
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "b"}}, ModelName: "model", ProviderResponseID: "r2",
			State: ai.ModelResponseStateSuspended, Usage: ai.Usage{Requests: 1, InputTokens: 4, OutputTokens: 1},
		},
		{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "c"}}, ModelName: "model", ProviderResponseID: "r3",
			State: ai.ModelResponseStateComplete, Usage: ai.Usage{Requests: 1, InputTokens: 3, OutputTokens: 1},
		},
	}}
	agent := ai.NewAgent[deps, string](model)
	first := agent.RunStream(t.Context(), "go", deps{})
	for range first.Events() {
		break
	}
	walkedAway := first.Suspended()
	if walkedAway == nil || walkedAway.Usage().TotalTokens() != 7 {
		t.Fatalf("walk-away snapshot = %+v", walkedAway)
	}
	second := agent.ResumeStream(t.Context(), walkedAway.Messages(), deps{})
	events := 0
	for range second.Events() {
		events++
		if events == 3 {
			break
		}
	}
	resumed := second.Suspended()
	if resumed == nil || resumed.Response().ProviderResponseID != "r2" || resumed.Usage().TotalTokens() != 5 {
		t.Fatalf("second walk-away snapshot = %+v", resumed)
	}
	result, err := agent.Resume(t.Context(), resumed.Messages(), deps{})
	if err != nil {
		t.Fatal(err)
	}
	if total := walkedAway.Usage().TotalTokens() + resumed.Usage().TotalTokens() + result.Usage().TotalTokens(); total != 16 {
		t.Fatalf("runs counted %d tokens, provider billed 16", total)
	}
}
