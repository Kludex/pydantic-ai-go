package ai_test

import (
	"context"
	"errors"
	"iter"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestConcurrencyLimiterCountsAndBackpressure(t *testing.T) {
	limiter := ai.NewConcurrencyLimiter(
		1, ai.WithMaxQueued(1), ai.WithConcurrencyLimiterName("shared"),
	)
	if limiter.Name() != "shared" || limiter.MaxRunning() != 1 || limiter.Running() != 0 ||
		limiter.Waiting() != 0 || limiter.Available() != 1 {
		t.Fatalf("unexpected initial limiter state")
	}
	if err := limiter.Acquire(t.Context(), "source"); err != nil {
		t.Fatal(err)
	}
	acquired := make(chan error, 1)
	go func() { acquired <- limiter.Acquire(t.Context(), "source") }()
	for limiter.Waiting() != 1 {
		time.Sleep(time.Millisecond)
	}
	if limiter.Running() != 1 || limiter.Available() != 0 {
		t.Fatalf("unexpected active limiter state: running=%d available=%d", limiter.Running(), limiter.Available())
	}
	if err := limiter.Acquire(t.Context(), "source"); err == nil {
		t.Fatal("expected queue limit error")
	} else {
		var limitErr *ai.ConcurrencyLimitExceededError
		if !errors.Is(err, ai.ErrConcurrencyLimitExceeded) || !errors.As(err, &limitErr) ||
			limitErr.Name != "shared" || limitErr.QueueDepth != 2 || limitErr.MaxQueued != 1 {
			t.Fatalf("unexpected queue limit error: %v", err)
		}
	}
	limiter.Release()
	if err := <-acquired; err != nil {
		t.Fatal(err)
	}
	limiter.Release()
}

func TestConcurrencyLimiterCancellationAndSourceName(t *testing.T) {
	limiter := ai.NewConcurrencyLimiter(1, ai.WithMaxQueued(1))
	canceled, cancelImmediately := context.WithCancel(t.Context())
	cancelImmediately()
	if err := limiter.Acquire(canceled, "canceled"); !errors.Is(err, context.Canceled) || limiter.Running() != 0 {
		t.Fatalf("already canceled acquisition succeeded: running=%d err=%v", limiter.Running(), err)
	}
	if err := limiter.Acquire(t.Context(), "holder"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	waiting := make(chan error, 1)
	go func() { waiting <- limiter.Acquire(ctx, "queued-source") }()
	for limiter.Waiting() != 1 {
		time.Sleep(time.Millisecond)
	}
	if err := limiter.Acquire(t.Context(), "overflow-source"); err == nil ||
		!strings.Contains(err.Error(), "overflow-source") {
		t.Fatalf("source name missing from queue error: %v", err)
	}
	cancel()
	if err := <-waiting; !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected waiting error: %v", err)
	}
	if limiter.Waiting() != 0 {
		t.Fatalf("canceled waiter was retained: %d", limiter.Waiting())
	}
	limiter.Release()
}

func TestConcurrencyLimitedModelSerializesRequests(t *testing.T) {
	var running atomic.Int32
	var maximum atomic.Int32
	release := make(chan struct{})
	entered := make(chan struct{}, 2)
	underlying := &requestModel{name: "limited", request: func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		current := running.Add(1)
		defer running.Add(-1)
		for {
			prior := maximum.Load()
			if current <= prior || maximum.CompareAndSwap(prior, current) {
				break
			}
		}
		entered <- struct{}{}
		select {
		case <-release:
			return &ai.ModelResponse{}, nil
		case <-ctx.Done():
			return nil, context.Cause(ctx)
		}
	}}
	limiter := ai.NewConcurrencyLimiter(1)
	model := ai.NewConcurrencyLimitedModel(underlying, limiter)
	if ai.UnwrapModel(model) != underlying || model.Name() != "limited" {
		t.Fatal("limited model did not preserve wrapper identity")
	}
	var wait sync.WaitGroup
	errorsFound := make(chan error, 2)
	for range 2 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
			errorsFound <- err
		}()
	}
	<-entered
	select {
	case <-entered:
		t.Fatal("second request entered before the first released")
	case <-time.After(10 * time.Millisecond):
	}
	close(release)
	wait.Wait()
	close(errorsFound)
	for err := range errorsFound {
		if err != nil {
			t.Fatal(err)
		}
	}
	if maximum.Load() != 1 || limiter.Running() != 0 || limiter.Waiting() != 0 {
		t.Fatalf("requests were not serialized: max=%d running=%d waiting=%d", maximum.Load(), limiter.Running(), limiter.Waiting())
	}
}

func TestConcurrencyLimitedModelHoldsStreamingSlot(t *testing.T) {
	started := make(chan struct{})
	continueStream := make(chan struct{})
	underlying := streamingRequestModel{
		Model: fallbackTextModel("stream", "fallback", nil),
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				close(started)
				if !yield(ai.TextDeltaEvent{Delta: "first"}, nil) {
					return
				}
				<-continueStream
				yield(ai.FinishEvent{}, nil)
			}, nil
		},
	}
	limiter := ai.NewConcurrencyLimiter(1, ai.WithMaxQueued(0))
	model := ai.NewConcurrencyLimitedModel(underlying, limiter)
	events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for _, err := range events {
			if err != nil {
				t.Errorf("unexpected stream error: %v", err)
			}
		}
	}()
	<-started
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, ai.ErrConcurrencyLimitExceeded) {
		t.Fatalf("stream did not hold slot for request: %v", err)
	}
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, ai.ErrConcurrencyLimitExceeded) {
		t.Fatalf("stream did not hold slot for another stream: %v", err)
	}
	close(continueStream)
	<-done
	if limiter.Running() != 0 {
		t.Fatalf("stream slot was not released: %d", limiter.Running())
	}
}

func TestConcurrencyLimitedModelReleasesOnStreamFailureAndBreak(t *testing.T) {
	streamErr := errors.New("stream")
	underlying := streamingRequestModel{
		Model: fallbackTextModel("stream", "unused", nil),
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return nil, streamErr
		},
	}
	limiter := ai.NewConcurrencyLimiter(1)
	model := ai.NewConcurrencyLimitedModel(&underlying, limiter)
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, streamErr) || limiter.Running() != 0 {
		t.Fatalf("opening failure leaked slot: running=%d err=%v", limiter.Running(), err)
	}

	underlying.stream = func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
		return func(yield func(ai.ModelStreamEvent, error) bool) {
			if !yield(ai.TextDeltaEvent{Delta: "one"}, nil) {
				return
			}
			yield(ai.FinishEvent{}, nil)
		}, nil
	}
	events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range events {
		break
	}
	if limiter.Running() != 0 {
		t.Fatalf("consumer break leaked slot: %d", limiter.Running())
	}

	underlying.stream = func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
		return func(yield func(ai.ModelStreamEvent, error) bool) { yield(nil, streamErr) }, nil
	}
	events, err = model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range events {
		if !errors.Is(err, streamErr) {
			t.Fatalf("unexpected stream event error: %v", err)
		}
	}
	if limiter.Running() != 0 {
		t.Fatalf("failed stream leaked slot: %d", limiter.Running())
	}
}

func TestConcurrencyCapabilityLimitsWholeRuns(t *testing.T) {
	limiter := ai.NewConcurrencyLimiter(1, ai.WithMaxQueued(0))
	if err := limiter.Acquire(t.Context(), "holder"); err != nil {
		t.Fatal(err)
	}
	calls := 0
	agent := ai.NewAgent[struct{}, string](
		fallbackTextModel("model", "ok", &calls),
		ai.WithCapabilities(ai.NewConcurrencyCapability(limiter, "agent:shared")),
	)
	if _, err := agent.Run(t.Context(), "blocked", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "agent:shared") || calls != 0 {
		t.Fatalf("run was not rejected before model execution: calls=%d err=%v", calls, err)
	}
	limiter.Release()
	result, err := agent.Run(t.Context(), "allowed", struct{}{})
	if err != nil || result.Output != "ok" || limiter.Running() != 0 {
		t.Fatalf("successful run leaked slot: result=%+v running=%d err=%v", result, limiter.Running(), err)
	}

	requestErr := errors.New("model failed")
	limiter = ai.NewConcurrencyLimiter(1)
	agent = ai.NewAgent[struct{}, string](requestModel{name: "error", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, requestErr
	}}, ai.WithCapabilities(ai.NewConcurrencyCapability(limiter, "")))
	if _, err := agent.Run(t.Context(), "fail", struct{}{}); !errors.Is(err, requestErr) || limiter.Running() != 0 {
		t.Fatalf("failed run leaked slot: running=%d err=%v", limiter.Running(), err)
	}
}

func TestConcurrencyLimiterValidationAndOptionalWrapper(t *testing.T) {
	model := fallbackTextModel("model", "ok", nil)
	model = &requestModel{name: "model", request: model.Request}
	if ai.LimitModelConcurrency(model, nil) != model {
		t.Fatal("nil limiter should preserve the model")
	}
	for name, test := range map[string]struct {
		call func()
		want string
	}{
		"running":     {call: func() { ai.NewConcurrencyLimiter(0) }, want: "max running"},
		"queued":      {call: func() { ai.WithMaxQueued(-1) }, want: "max queued"},
		"nil limiter": {call: func() { ai.NewConcurrencyLimitedModel(model, nil) }, want: "must not be nil"},
		"nil capability limiter": {call: func() {
			ai.NewConcurrencyCapability(nil, "")
		}, want: "must not be nil"},
		"nil model": {call: func() {
			ai.NewConcurrencyLimitedModel(nil, ai.NewConcurrencyLimiter(1))
		}, want: "wrapped model"},
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

	limiter := ai.NewConcurrencyLimiter(1)
	defer func() {
		if recovered := recover(); recovered != "ai: release called without an acquired concurrency slot" {
			t.Fatalf("unexpected release panic: %v", recovered)
		}
	}()
	limiter.Release()
}

func TestConcurrencyLimiterCanBeShared(t *testing.T) {
	limiter := ai.NewConcurrencyLimiter(2)
	first := ai.LimitModelConcurrency(fallbackTextModel("one", "one", nil), limiter)
	second := ai.LimitModelConcurrency(fallbackTextModel("two", "two", nil), limiter)
	if reflect.TypeOf(first) != reflect.TypeOf(second) {
		t.Fatalf("shared limiter wrappers differ: %T %T", first, second)
	}
}

var _ ai.ConcurrencyGate = (*ai.ConcurrencyLimiter)(nil)
var _ ai.StreamingModel = (*ai.ConcurrencyLimitedModel)(nil)
