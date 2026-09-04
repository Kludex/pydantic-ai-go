package ai_test

import (
	"context"
	"fmt"
	"iter"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func BenchmarkAgentLoop(b *testing.B) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := agent.Run(ctx, "benchmark", struct{}{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStreaming(b *testing.B) {
	agent := ai.NewAgent[struct{}, string](benchmarkStreamingModel{})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		stream := agent.RunStream(ctx, "benchmark", struct{}{})
		for _, err := range stream.Events() {
			if err != nil {
				b.Fatal(err)
			}
		}
		if stream.Result() == nil {
			b.Fatal("stream did not produce a result")
		}
	}
}

func BenchmarkSchemaReflection(b *testing.B) {
	tool := func(context.Context, benchmarkToolArgs) (benchmarkToolResult, error) {
		return benchmarkToolResult{}, nil
	}
	model := fakes.NewTestModel()
	b.ReportAllocs()
	b.ResetTimer()
	for index := range b.N {
		agent := ai.NewAgent[struct{}, string](model)
		ai.AddSimpleTool(agent, fmt.Sprintf("benchmark_%d", index), tool)
	}
}

func BenchmarkParallelTools(b *testing.B) {
	calls := make([]ai.ResponsePart, 8)
	for index := range calls {
		calls[index] = ai.ToolCallPart{
			ToolName: fmt.Sprintf("tool_%d", index), ToolCallID: fmt.Sprintf("call_%d", index), Args: []byte(`{"value":1}`),
		}
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			return &ai.ModelResponse{Parts: calls}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	tool := func(context.Context, benchmarkToolArgs) (benchmarkToolResult, error) {
		return benchmarkToolResult{}, nil
	}
	for index := range calls {
		ai.AddSimpleTool(agent, fmt.Sprintf("tool_%d", index), tool)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ReportMetric(float64(len(calls)), "tools/op")
	b.ResetTimer()
	for range b.N {
		if _, err := agent.Run(ctx, "benchmark", struct{}{}); err != nil {
			b.Fatal(err)
		}
	}
}

type benchmarkToolArgs struct {
	Value int `json:"value"`
}

type benchmarkToolResult struct {
	Value int `json:"value"`
}

type benchmarkStreamingModel struct{}

func (benchmarkStreamingModel) Name() string { return "benchmark-stream" }

func (benchmarkStreamingModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func (benchmarkStreamingModel) StreamRequest(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		if !yield(ai.TextDeltaEvent{PartID: "text", Delta: "done"}, nil) {
			return
		}
		yield(ai.FinishEvent{}, nil)
	}, nil
}
