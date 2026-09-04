package ai_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestModelRequestHooksTransformInMiddlewareOrder(t *testing.T) {
	var calls []string
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		calls = append(calls, "model")
		if params.Instructions != "base outer inner" || len(messages) != 1 {
			t.Fatalf("before hooks did not transform request: instructions=%q messages=%+v", params.Instructions, messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "model"}}}, nil
	})
	before := func(name string) ai.BeforeModelRequestFunc {
		return func(
			_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
		) (ai.ModelRequestContext, error) {
			calls = append(calls, "before "+name)
			if request.Streaming {
				t.Fatal("ordinary request reported streaming")
			}
			request.Params.Instructions += " " + name
			return request, nil
		}
	}
	after := func(name string) ai.AfterModelRequestFunc {
		return func(
			_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext, response *ai.ModelResponse,
		) (*ai.ModelResponse, error) {
			calls = append(calls, "after "+name)
			if request.Params.Instructions != "base outer inner" || response.RunID == "" || response.ModelName == "" {
				t.Fatalf("after hook received incomplete context request=%+v response=%+v", request, response)
			}
			part := response.Parts[0].(ai.TextPart)
			part.Content += " " + name
			response.Parts[0] = part
			return response, nil
		}
	}
	agent := ai.NewAgent[deps, string](model,
		ai.WithInstructions("base"),
		ai.WithCapabilities(before("outer"), after("outer"), before("inner"), after("inner")),
	)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "model inner outer" {
		t.Fatalf("unexpected transformed output %q", result.Output)
	}
	want := []string{"before outer", "before inner", "model", "after inner", "after outer"}
	if !slices.Equal(calls, want) {
		t.Fatalf("unexpected hook order %v, want %v", calls, want)
	}
}

func TestModelRequestErrorHooksRecoverInsideOut(t *testing.T) {
	var calls []string
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("provider unavailable")
	})
	outer := ai.ModelRequestErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ModelRequestContext, err error,
	) (*ai.ModelResponse, error) {
		calls = append(calls, "outer: "+err.Error())
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "recovered"}}}, nil
	})
	inner := ai.ModelRequestErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ModelRequestContext, err error,
	) (*ai.ModelResponse, error) {
		calls = append(calls, "inner: "+err.Error())
		return nil, fmt.Errorf("inner: %w", err)
	})
	after := ai.AfterModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ModelRequestContext, response *ai.ModelResponse,
	) (*ai.ModelResponse, error) {
		calls = append(calls, "after")
		return response, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(outer, after, inner))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "recovered" {
		t.Fatalf("unexpected recovery result=%+v err=%v", result, err)
	}
	want := []string{"inner: provider unavailable", "outer: inner: provider unavailable", "after"}
	if !slices.Equal(calls, want) {
		t.Fatalf("unexpected recovery order %v, want %v", calls, want)
	}
}

func TestModelRequestRetryHooksPreserveResponseHistory(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: fmt.Sprintf("response %d", requests)}}}, nil
	})
	afterCalls := 0
	after := ai.AfterModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ModelRequestContext, response *ai.ModelResponse,
	) (*ai.ModelResponse, error) {
		afterCalls++
		if afterCalls == 1 {
			return response, ai.Retryf("try another response")
		}
		return response, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(after))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "response 2" || requests != 2 {
		t.Fatalf("unexpected retry result=%+v requests=%d err=%v", result, requests, err)
	}
	messages := result.Messages()
	if len(messages) != 4 || messages[1].(ai.ModelResponse).Text() != "response 1" ||
		messages[1].(ai.ModelResponse).RunID == "" ||
		messages[2].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart).Content != "try another response" {
		t.Fatalf("rejected response was not preserved: %+v", messages)
	}
}

func TestBeforeModelRequestRetrySkipsModelErrorHooks(t *testing.T) {
	beforeCalls := 0
	errorCalls := 0
	before := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		beforeCalls++
		if beforeCalls == 1 {
			return request, ai.Retryf("prepare again")
		}
		return request, nil
	})
	onError := ai.ModelRequestErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ModelRequestContext, err error,
	) (*ai.ModelResponse, error) {
		errorCalls++
		return nil, err
	})
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(before, onError))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output == "" || beforeCalls != 2 || errorCalls != 0 {
		t.Fatalf("unexpected before retry result=%+v before=%d errors=%d err=%v", result, beforeCalls, errorCalls, err)
	}
}

func TestBeforeModelRequestReportsStreaming(t *testing.T) {
	streaming := false
	before := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		streaming = request.Streaming
		return request, nil
	})
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(before))
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if !streaming {
		t.Fatal("streaming model request was not identified")
	}
}

func TestModelRequestHooksRejectNilResponses(t *testing.T) {
	modelError := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("failed")
	})
	recoverNil := ai.ModelRequestErrorFunc(func(
		context.Context, *ai.RunInfo, ai.ModelRequestContext, error,
	) (*ai.ModelResponse, error) {
		return nil, nil
	})
	_, err := ai.NewAgent[deps, string](modelError, ai.WithCapabilities(recoverNil)).Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "error hook returned no response") {
		t.Fatalf("unexpected nil error-hook response: %v", err)
	}

	afterNil := ai.AfterModelRequestFunc(func(
		context.Context, *ai.RunInfo, ai.ModelRequestContext, *ai.ModelResponse,
	) (*ai.ModelResponse, error) {
		return nil, nil
	})
	_, err = ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(afterNil)).Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "after model request hook returned no response") {
		t.Fatalf("unexpected nil after-hook response: %v", err)
	}
}

func TestBeforeModelRequestCanResolveModelID(t *testing.T) {
	resolved := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request := messages[len(messages)-1].(ai.ModelRequest)
		if request.RunID == "" || request.ConversationID == "" {
			t.Fatalf("replacement request was not stamped: %+v", request)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "resolved"}}}, nil
	})
	before := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Model = nil
		request.ModelID = "resolved"
		last := request.Messages[len(request.Messages)-1].(ai.ModelRequest)
		last.RunID = ""
		last.ConversationID = ""
		request.Messages[len(request.Messages)-1] = last
		return request, nil
	})
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(before))
	agent.AddModelIDResolver(func(
		context.Context, ai.ModelResolutionContext[deps], string,
	) (ai.Model, error) {
		return resolved, nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "resolved" {
		t.Fatalf("model hook did not resolve model result=%+v err=%v", result, err)
	}
}

func TestBeforeModelRequestValidatesChangedSettings(t *testing.T) {
	for name, change := range map[string]func(*ai.ModelSettings){
		"timeout":      func(settings *ai.ModelSettings) { settings.RequestTimeout = -1 },
		"service tier": func(settings *ai.ModelSettings) { settings.ServiceTier = "expedited" },
	} {
		t.Run(name, func(t *testing.T) {
			before := ai.BeforeModelRequestFunc(func(
				_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
			) (ai.ModelRequestContext, error) {
				change(&request.Params.Settings)
				return request, nil
			})
			_, err := ai.NewAgent[deps, string](
				fakes.NewTestModel(), ai.WithCapabilities(before),
			).Run(t.Context(), "go", deps{})
			if err == nil {
				t.Fatal("expected changed settings validation error")
			}
		})
	}
}

func TestBeforeModelRequestRejectsMissingAndUnknownModels(t *testing.T) {
	for name, modelID := range map[string]string{"missing": "", "unknown": "unknown"} {
		t.Run(name, func(t *testing.T) {
			before := ai.BeforeModelRequestFunc(func(
				_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
			) (ai.ModelRequestContext, error) {
				request.Model = nil
				request.ModelID = modelID
				return request, nil
			})
			_, err := ai.NewAgent[deps, string](
				fakes.NewTestModel(), ai.WithCapabilities(before),
			).Run(t.Context(), "go", deps{})
			if err == nil {
				t.Fatal("expected model hook selection error")
			}
		})
	}
}

func TestBeforeModelRequestReplacementOpenFailure(t *testing.T) {
	var log []string
	failed := &lifecycleModel{name: "failed", log: &log, openErr: errors.New("open failed")}
	before := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Model = failed
		return request, nil
	})
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(before)).Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "open failed") {
		t.Fatalf("unexpected replacement open error: %v", err)
	}
}

func TestModelRequestHookRetryLimit(t *testing.T) {
	after := ai.AfterModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ModelRequestContext, response *ai.ModelResponse,
	) (*ai.ModelResponse, error) {
		return response, ai.Retryf("again")
	})
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(after)).Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("unexpected hook retry limit error: %v", err)
	}
}

func TestBeforeModelRequestCanSelectModelAndCloneContext(t *testing.T) {
	first := fakes.NewTestModel()
	second := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "second"}}}, nil
	})
	before := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		cloned := request.Clone()
		cloned.Messages = nil
		cloned.Params.Settings.ExtraHeaders["x"] = "changed"
		if len(request.Messages) == 0 || request.Params.Settings.ExtraHeaders["x"] != "original" {
			t.Fatalf("request clone was not detached: original=%+v clone=%+v", request, cloned)
		}
		request.Model = second
		request.ModelID = "replacement"
		return request, nil
	})
	agent := ai.NewAgent[deps, string](first,
		ai.WithModelSettings(ai.ModelSettings{ExtraHeaders: map[string]string{"x": "original"}}),
		ai.WithCapabilities(before),
	)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "second" {
		t.Fatalf("model hook did not select replacement result=%+v err=%v", result, err)
	}
	response := result.Messages()[len(result.Messages())-1].(ai.ModelResponse)
	if response.ModelName != second.Name() {
		t.Fatalf("replacement model metadata was lost: %+v", response)
	}
}

func TestBeforeModelRequestReprofilesStructuredOutputAfterModelSwitch(t *testing.T) {
	initial := ai.NewProfiledModel(fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		t.Fatal("initial model should have been replaced")
		return nil, nil
	}), ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool})
	replacement := ai.NewProfiledModel(fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModeNative || params.OutputSchema == nil || params.OutputTool != nil {
			t.Fatalf("replacement model was not reprofiled: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"switched"}`}}}, nil
	}), ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative})
	hook := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Model = replacement
		return request, nil
	})
	result, err := ai.NewAgent[deps, profileOutput](initial, ai.WithCapabilities(hook)).Run(
		t.Context(), "go", deps{},
	)
	if err != nil || result.Output.Value != "switched" {
		t.Fatalf("unexpected reprofiled output: %+v err=%v", result, err)
	}
}

func TestModelHookCanReplaceHistoryAndRecordAdditionalUsage(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) != 1 {
			t.Fatalf("model received %d replaced messages: %#v", len(messages), messages)
		}
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
			Usage: ai.Usage{Requests: 1, InputTokens: 3, OutputTokens: 1},
		}, nil
	})
	hook := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Messages = []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "replacement"},
		}}}
		request.ReplaceHistory = true
		request.AdditionalUsage = ai.Usage{
			Requests: 1, InputTokens: 7, OutputTokens: 2,
			Details: map[string]int{"compaction_tokens": 9},
		}
		clone := request.Clone()
		clone.AdditionalUsage.Details["compaction_tokens"] = 0
		if request.AdditionalUsage.Details["compaction_tokens"] != 9 {
			t.Fatal("additional usage clone aliases the request")
		}
		return request, nil
	})
	history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}}}
	result, err := ai.NewAgent[deps, string](model, ai.WithCapabilities(hook)).Run(
		t.Context(), "new", deps{}, ai.WithMessageHistory(history),
	)
	if err != nil {
		t.Fatal(err)
	}
	messages := result.Messages()
	if len(messages) != 2 || messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "replacement" {
		t.Fatalf("history replacement was not persisted: %#v", messages)
	}
	usage := result.Usage()
	if usage.Requests != 2 || usage.InputTokens != 10 || usage.OutputTokens != 3 ||
		usage.Details["compaction_tokens"] != 9 {
		t.Fatalf("additional usage was not accumulated: %+v", usage)
	}
}

func TestModelHookAdditionalUsageIsCheckedBeforeRequest(t *testing.T) {
	called := false
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		called = true
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unexpected"}}}, nil
	})
	hook := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.AdditionalUsage = ai.Usage{InputTokens: 2}
		return request, nil
	})
	_, err := ai.NewAgent[deps, string](
		model, ai.WithCapabilities(hook), ai.WithUsageLimits(ai.UsageLimits{InputTokenLimit: 1}),
	).Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("unexpected usage error: %v", err)
	}
	if called {
		t.Fatal("model was called after additional usage exceeded the limit")
	}
}
