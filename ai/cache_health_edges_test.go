package ai_test

import (
	"context"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestCacheHealthActiveRunKeepsEvictedMarks(t *testing.T) {
	exporter, provider := cacheExporter(t)
	writer := ai.NewInstrumentedModel(fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return cacheResponse(0, 14000), nil
	}), ai.WithInstrumentationTracerProvider(provider))
	first, second := cacheResponse(0, 14000), cacheResponse(1000, 13000)
	for _, response := range []*ai.ModelResponse{first, second} {
		response.Parts = []ai.ResponsePart{ai.ToolCallPart{ToolName: "continue", ToolCallID: "continue", Args: []byte(`{}`)}}
	}
	model := ai.NewProfiledModel(cacheModel(first, second, cacheResponse(14000, 0), cacheResponse(1000, 0)), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: time.Hour,
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))))
	calls := 0
	ai.AddTool(agent, "continue", func(_ context.Context, info *ai.RunContext[struct{}], _ struct{}) (string, error) {
		calls++
		if calls == 1 {
			fillCacheConversations(t, writer, 4096)
			other := cacheResponse(0, 9000)
			other.ModelName = "other-model"
			fresh := ai.NewInstrumentedModel(cacheModel(other), ai.WithInstrumentationTracerProvider(provider))
			_, err := fresh.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{ConversationID: info.ConversationID}}, ai.ModelRequestParams{})
			if err != nil {
				return "", err
			}
		}
		return "continue", nil
	})
	result, err := agent.Run(t.Context(), "first", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(t.Context(), "second", struct{}{}, ai.WithConversation(result.Conversation())); err != nil {
		t.Fatal(err)
	}
	var collapsed int
	for _, span := range cacheChatSpans(exporter) {
		attributes := cacheAttributes(span)
		if attributes["pydantic_ai.cache.collapse_reason"] == "unexpected" {
			collapsed++
		}
	}
	if collapsed != 2 {
		t.Fatalf("evicted marks did not survive the active run and its continuation: %d collapses", collapsed)
	}
}

func TestCacheHealthNativeUsageDoesNotRearmAlert(t *testing.T) {
	native := func(read, write int) *ai.ModelResponse {
		response := cacheResponse(read, write)
		response.Parts = append(response.Parts, ai.NativeToolCallPart{ToolName: "search"})
		return response
	}
	responses := []*ai.ModelResponse{
		native(32000, 1000), cacheResponse(0, 14000), cacheResponse(0, 14000), native(32000, 0),
		cacheResponse(0, 14000), cacheResponse(14000, 0), native(1000, 1000), native(32000, 0),
		cacheResponse(0, 14000), cacheResponse(14000, 0), cacheResponse(0, 14000),
	}
	model := ai.NewProfiledModel(cacheModel(responses...), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: time.Hour,
	})
	_, spans := runCacheTurns(t, model, len(responses))
	if cacheAttributes(spans[0])["pydantic_ai.cache.established_tokens"] != int64(0) {
		t.Fatal("summed usage must not establish an initial prefix")
	}
	for index, count := range []int{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1} {
		if len(spans[index].Events) != count {
			t.Fatalf("request %d native usage changed alert latch: %+v", index, spans[index].Events)
		}
	}
	if cacheAttributes(spans[6])["pydantic_ai.cache.established_tokens"] != int64(14000) {
		t.Fatal("summed collapse must keep the ordinary prefix")
	}
}

func TestCacheHealthSerializedConversationAndFreshBranch(t *testing.T) {
	exporter, provider := cacheExporter(t)
	model := ai.NewProfiledModel(cacheModel(cacheResponse(0, 14000), cacheResponse(1000, 0), cacheResponse(1000, 0)), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: time.Hour,
	})
	options := []ai.Option{ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider)))}
	first, err := ai.NewAgent[struct{}, string](model, options...).Run(t.Context(), "first", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	data, err := ai.MarshalMessages(first.Messages())
	if err != nil {
		t.Fatal(err)
	}
	history, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	other := ai.NewAgent[struct{}, string](model, options...)
	if _, err := other.Run(t.Context(), "second", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	if _, err := other.Run(t.Context(), "branch", struct{}{}, ai.WithMessageHistory(history), ai.WithConversationID("new")); err != nil {
		t.Fatal(err)
	}
	spans := cacheChatSpans(exporter)
	if cacheAttributes(spans[1])["pydantic_ai.cache.collapse_reason"] != "unexpected" ||
		cacheAttributes(spans[2])["pydantic_ai.cache.collapsed"] != nil {
		t.Fatalf("conversation identity was not respected: %+v", spans)
	}
}
