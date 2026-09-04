package ai_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type fullRunCapability struct {
	name string
	log  *[]string
}

func (*fullRunCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *fullRunCapability) BeforeRun(context.Context, *ai.RunInfo) error {
	*capability.log = append(*capability.log, capability.name+":before")
	return nil
}

func (capability *fullRunCapability) WrapRun(
	ctx context.Context, _ *ai.RunInfo, next ai.RunFunc,
) (ai.RunOutcome, error) {
	*capability.log = append(*capability.log, capability.name+":wrap-before")
	outcome, err := next(ctx)
	*capability.log = append(*capability.log, capability.name+":wrap-after")
	return outcome, err
}

func (capability *fullRunCapability) AfterRun(
	_ context.Context, ri *ai.RunInfo, outcome ai.RunOutcome,
) (ai.RunOutcome, error) {
	*capability.log = append(*capability.log, capability.name+":after")
	if ri.Usage().Requests != 1 || len(ri.Messages()) != 2 || outcome.Deferred != nil {
		return ai.RunOutcome{}, fmt.Errorf("unexpected completed run state")
	}
	outcome.Output = outcome.Output.(string) + ":" + capability.name
	return outcome, nil
}

func TestRunLifecycleHooksUseMiddlewareOrder(t *testing.T) {
	var log []string
	outer := &fullRunCapability{name: "outer", log: &log}
	inner := &fullRunCapability{name: "inner", log: &log}
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		log = append(log, "model")
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Usage: ai.Usage{Requests: 1},
		}, nil
	})
	result, err := ai.NewAgent[deps, string](model, ai.WithCapabilities(outer, inner)).Run(
		t.Context(), "go", deps{},
	)
	if err != nil || result.Output != "done:inner:outer" {
		t.Fatalf("unexpected run lifecycle result=%+v err=%v", result, err)
	}
	want := []string{
		"outer:wrap-before", "inner:wrap-before", "outer:before", "inner:before", "model",
		"inner:wrap-after", "outer:wrap-after", "inner:after", "outer:after",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected run lifecycle order:\n got %v\nwant %v", log, want)
	}
}

type shortCircuitRunCapability struct {
	output string
	calls  *int
}

func (*shortCircuitRunCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *shortCircuitRunCapability) WrapRun(
	context.Context, *ai.RunInfo, ai.RunFunc,
) (ai.RunOutcome, error) {
	*capability.calls++
	return ai.CompletedRunOutcome(capability.output), nil
}

func TestRunWrapperCanShortCircuitRunAndStream(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("streaming=%t", streaming), func(t *testing.T) {
			modelCalls := 0
			beforeCalls := 0
			wrapperCalls := 0
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				modelCalls++
				return nil, errors.New("model must not run")
			})
			before := ai.BeforeRunFunc(func(context.Context, *ai.RunInfo) error {
				beforeCalls++
				return nil
			})
			agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(
				&shortCircuitRunCapability{output: "short", calls: &wrapperCalls}, before,
			))
			var output string
			if streaming {
				stream := agent.RunStream(t.Context(), "go", deps{})
				for _, err := range stream.Events() {
					if err != nil {
						t.Fatal(err)
					}
				}
				if stream.Result() == nil {
					t.Fatal("short-circuit stream has no result")
				}
				output = stream.Result().Output
			} else {
				result, err := agent.Run(t.Context(), "go", deps{})
				if err != nil {
					t.Fatal(err)
				}
				output = result.Output
			}
			if output != "short" || wrapperCalls != 1 || beforeCalls != 0 || modelCalls != 0 {
				t.Fatalf(
					"unexpected short circuit output=%q wrappers=%d before=%d model=%d",
					output, wrapperCalls, beforeCalls, modelCalls,
				)
			}
		})
	}
}

type recoveringRunWrapper struct {
	seen error
}

func (*recoveringRunWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (wrapper *recoveringRunWrapper) WrapRun(
	ctx context.Context, _ *ai.RunInfo, next ai.RunFunc,
) (ai.RunOutcome, error) {
	outcome, err := next(ctx)
	if err == nil {
		return outcome, nil
	}
	wrapper.seen = err
	return ai.CompletedRunOutcome("wrapper recovered"), nil
}

func TestRunWrapperRecoversBeforeErrorHooks(t *testing.T) {
	wrapper := &recoveringRunWrapper{}
	errorHookCalls := 0
	errorHook := ai.RunErrorFunc(func(
		context.Context, *ai.RunInfo, error,
	) (ai.RunOutcome, error) {
		errorHookCalls++
		return ai.RunOutcome{}, errors.New("must not run")
	})
	modelErr := errors.New("model exploded")
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, modelErr
	})
	result, err := ai.NewAgent[deps, string](model, ai.WithCapabilities(wrapper, errorHook)).Run(
		t.Context(), "go", deps{},
	)
	if err != nil || result.Output != "wrapper recovered" || !errors.Is(wrapper.seen, modelErr) || errorHookCalls != 0 {
		t.Fatalf(
			"unexpected wrapper recovery result=%+v seen=%v error_hooks=%d err=%v",
			result, wrapper.seen, errorHookCalls, err,
		)
	}
}

func TestRunErrorHooksTransformAndRecoverInsideOut(t *testing.T) {
	var log []string
	inner := ai.RunErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, err error,
	) (ai.RunOutcome, error) {
		log = append(log, "inner:"+err.Error())
		return ai.RunOutcome{}, fmt.Errorf("inner: %w", err)
	})
	outer := ai.RunErrorFunc(func(
		_ context.Context, ri *ai.RunInfo, err error,
	) (ai.RunOutcome, error) {
		log = append(log, "outer:"+err.Error())
		if len(ri.Messages()) != 1 {
			return ai.RunOutcome{}, errors.New("run history unavailable during recovery")
		}
		return ai.CompletedRunOutcome("recovered"), nil
	})
	after := ai.AfterRunFunc(func(
		_ context.Context, _ *ai.RunInfo, outcome ai.RunOutcome,
	) (ai.RunOutcome, error) {
		outcome.Output = outcome.Output.(string) + ":after"
		return outcome, nil
	})
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("model failed")
	})
	result, err := ai.NewAgent[deps, string](model, ai.WithCapabilities(outer, inner, after)).Run(
		t.Context(), "go", deps{},
	)
	wantLog := []string{"inner:model failed", "outer:inner: model failed"}
	if err != nil || result.Output != "recovered:after" || !slices.Equal(log, wantLog) {
		t.Fatalf("unexpected run error recovery result=%+v log=%v err=%v", result, log, err)
	}
}

func TestBeforeAndAfterRunErrorsPropagate(t *testing.T) {
	for name, test := range map[string]struct {
		capability ai.Capability
		want       string
	}{
		"before": {
			capability: ai.BeforeRunFunc(func(context.Context, *ai.RunInfo) error {
				return errors.New("before failed")
			}),
			want: "before failed",
		},
		"after": {
			capability: ai.AfterRunFunc(func(
				context.Context, *ai.RunInfo, ai.RunOutcome,
			) (ai.RunOutcome, error) {
				return ai.RunOutcome{}, errors.New("after failed")
			}),
			want: "after failed",
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ai.NewAgent[deps, string](
				fakes.NewTestModel(), ai.WithCapabilities(test.capability),
			).Run(t.Context(), "go", deps{})
			if err == nil || err.Error() != test.want {
				t.Fatalf("unexpected %s hook error: %v", name, err)
			}
		})
	}
}

func TestRunOutcomeSupportsDeferredAndNilOutputs(t *testing.T) {
	requests := ai.DeferredToolRequests{
		Calls:    []ai.ToolCallPart{{ToolName: "external", ToolCallID: "call"}},
		Metadata: map[string]map[string]any{"call": {"nested": map[string]any{"value": 1}}},
	}
	pending := ai.PendingRunOutcome(requests)
	cloned := pending.Clone()
	cloned.Deferred.Calls[0].ToolName = "changed"
	cloned.Deferred.Metadata["call"]["nested"].(map[string]any)["value"] = 2
	if pending.Deferred.Calls[0].ToolName != "external" ||
		pending.Deferred.Metadata["call"]["nested"].(map[string]any)["value"] != 1 {
		t.Fatal("run outcome clone mutated deferred requests")
	}
	calls := 0
	wrapper := &pendingRunWrapper{outcome: pending, calls: &calls}
	result, err := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(wrapper)).Run(
		t.Context(), "go", deps{},
	)
	if err != nil || result.Deferred() == nil || result.Deferred().Calls[0].ToolCallID != "call" || calls != 1 {
		t.Fatalf("unexpected deferred short circuit result=%+v err=%v", result, err)
	}

	nilWrapper := &pendingRunWrapper{outcome: ai.CompletedRunOutcome(nil), calls: &calls}
	nilResult, err := ai.NewAgent[deps, *hookedOutput](
		fakes.NewTestModel(), ai.WithCapabilities(nilWrapper),
	).Run(t.Context(), "go", deps{})
	if err != nil || nilResult.Output != nil {
		t.Fatalf("unexpected nil pointer output result=%+v err=%v", nilResult, err)
	}

	_, err = ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(nilWrapper),
	).Run(t.Context(), "go", deps{})
	if err == nil || err.Error() != "ai: run outcome has output type <nil>, expected string" {
		t.Fatalf("unexpected nil string outcome error: %v", err)
	}
}

type pendingRunWrapper struct {
	outcome ai.RunOutcome
	calls   *int
}

func (*pendingRunWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (wrapper *pendingRunWrapper) WrapRun(
	context.Context, *ai.RunInfo, ai.RunFunc,
) (ai.RunOutcome, error) {
	*wrapper.calls++
	return wrapper.outcome.Clone(), nil
}

func TestRunCancellationCannotBeRecovered(t *testing.T) {
	hookCalls := 0
	afterCalls := 0
	errorHook := ai.RunErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, err error,
	) (ai.RunOutcome, error) {
		hookCalls++
		if !errors.Is(err, ai.ErrRunCancelled) {
			return ai.RunOutcome{}, fmt.Errorf("unexpected cancellation error: %w", err)
		}
		return ai.CompletedRunOutcome("recovered"), nil
	})
	after := ai.AfterRunFunc(func(
		_ context.Context, _ *ai.RunInfo, outcome ai.RunOutcome,
	) (ai.RunOutcome, error) {
		afterCalls++
		return outcome, nil
	})
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "cancel", ToolCallID: "cancel", Args: []byte(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(errorHook, after))
	ai.AddTool(agent, "cancel", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		rc.Cancel()
		return "canceled", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if result != nil || !errors.Is(err, ai.ErrRunCancelled) || hookCalls != 1 || afterCalls != 1 {
		t.Fatalf(
			"cancellation was recovered result=%+v hooks=%d after=%d err=%v",
			result, hookCalls, afterCalls, err,
		)
	}
}

type observingRunWrapper struct {
	entered int
	exited  int
	err     error
}

func (*observingRunWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (wrapper *observingRunWrapper) WrapRun(
	ctx context.Context, _ *ai.RunInfo, next ai.RunFunc,
) (ai.RunOutcome, error) {
	wrapper.entered++
	outcome, err := next(ctx)
	wrapper.exited++
	wrapper.err = err
	return outcome, err
}

func TestStoppingStreamExitsWrapperWithoutCompletionHooks(t *testing.T) {
	wrapper := &observingRunWrapper{}
	afterCalls := 0
	errorCalls := 0
	after := ai.AfterRunFunc(func(
		_ context.Context, _ *ai.RunInfo, outcome ai.RunOutcome,
	) (ai.RunOutcome, error) {
		afterCalls++
		return outcome, nil
	})
	onError := ai.RunErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, err error,
	) (ai.RunOutcome, error) {
		errorCalls++
		return ai.RunOutcome{}, err
	})
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "one"},
			ai.TextDeltaEvent{PartID: "text", Delta: "two"},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](
		model, ai.WithCapabilities(wrapper, after, onError),
	).RunStream(t.Context(), "go", deps{})
	for range stream.Events() {
		break
	}
	if wrapper.entered != 1 || wrapper.exited != 1 || wrapper.err == nil || afterCalls != 0 || errorCalls != 0 {
		t.Fatalf(
			"unexpected stopped lifecycle entered=%d exited=%d wrapper_err=%v after=%d errors=%d",
			wrapper.entered, wrapper.exited, wrapper.err, afterCalls, errorCalls,
		)
	}
}

func TestParentCancellationCannotBeRecovered(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	started := make(chan struct{})
	model := fakes.NewFunctionModel(func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	errorCalls := 0
	afterCalls := 0
	onError := ai.RunErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ error,
	) (ai.RunOutcome, error) {
		errorCalls++
		return ai.CompletedRunOutcome("recovered"), nil
	})
	after := ai.AfterRunFunc(func(
		_ context.Context, _ *ai.RunInfo, outcome ai.RunOutcome,
	) (ai.RunOutcome, error) {
		afterCalls++
		return outcome, nil
	})
	done := make(chan error, 1)
	go func() {
		_, err := ai.NewAgent[deps, string](model, ai.WithCapabilities(onError, after)).Run(
			ctx, "go", deps{},
		)
		done <- err
	}()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) || errorCalls != 1 || afterCalls != 1 {
		t.Fatalf("unexpected parent cancellation err=%v errors=%d after=%d", err, errorCalls, afterCalls)
	}
}

func TestAfterRunErrorBypassesRunErrorHooks(t *testing.T) {
	errorCalls := 0
	after := ai.AfterRunFunc(func(
		context.Context, *ai.RunInfo, ai.RunOutcome,
	) (ai.RunOutcome, error) {
		return ai.RunOutcome{}, errors.New("after failed")
	})
	onError := ai.RunErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, err error,
	) (ai.RunOutcome, error) {
		errorCalls++
		return ai.RunOutcome{}, err
	})
	_, err := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(onError, after),
	).Run(t.Context(), "go", deps{})
	if err == nil || err.Error() != "after failed" || errorCalls != 0 {
		t.Fatalf("unexpected after-run failure err=%v error_hooks=%d", err, errorCalls)
	}
}
