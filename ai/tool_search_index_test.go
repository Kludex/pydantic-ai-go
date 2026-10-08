package ai_test

import (
	"context"
	"encoding/json"
	"slices"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type changingSearchCorpus struct{ corpora [][]ai.ToolDefinition }

func (catalog changingSearchCorpus) Tools(_ context.Context, rc *ai.RunContext[struct{}]) ([]ai.Tool[struct{}], error) {
	index := min(len(completedSearches(rc.Messages())), len(catalog.corpora)-1)
	tools := make([]ai.Tool[struct{}], len(catalog.corpora[index]))
	for index, definition := range catalog.corpora[index] {
		definition.DeferLoading = true
		definition.Schema = map[string]any{"type": "object"}
		tools[index] = ai.NewRawTool[struct{}](definition, func(context.Context, json.RawMessage) (any, error) { return "unused", nil })
	}
	return tools, nil
}

func completedSearches(messages []ai.ModelMessage) []ai.ToolSearchResult {
	var results []ai.ToolSearchResult
	for _, message := range messages {
		if request, ok := message.(ai.ModelRequest); ok {
			for _, part := range request.Parts {
				if returned, ok := part.(ai.ToolReturnPart); ok && returned.ToolName == ai.ToolSearchName {
					results = append(results, returned.Content.(ai.ToolSearchResult))
				}
			}
		}
	}
	return results
}

func TestKeywordIndexTracksCurrentCorpus(t *testing.T) {
	for _, test := range []struct {
		name   string
		second []ai.ToolDefinition
		want   string
	}{
		{name: "description", second: []ai.ToolDefinition{{Name: "one", Description: "beta"}, {Name: "two", Description: "alpha"}}, want: "two"},
		{name: "name", second: []ai.ToolDefinition{{Name: "alpha_two"}, {Name: "one", Description: "beta"}}, want: "alpha_two"},
		{name: "membership", second: []ai.ToolDefinition{{Name: "two", Description: "alpha"}}, want: "two"},
		{name: "order", second: []ai.ToolDefinition{{Name: "two", Description: "alpha"}, {Name: "one", Description: "alpha"}}, want: "two"},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
				if len(completedSearches(messages)) < 2 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: ai.ToolSearchName, Args: []byte(`{"queries":["alpha","alpha"]}`)}}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			catalog := changingSearchCorpus{corpora: [][]ai.ToolDefinition{{{Name: "one", Description: "alpha"}, {Name: "two", Description: "beta"}}, test.second}}
			agent := ai.NewAgent[struct{}, string](model)
			agent.AddToolset(ai.WithToolSearch[struct{}](catalog, ai.ToolSearchConfig[struct{}]{Strategy: ai.ToolSearchStrategyKeywords, MaxResults: 1}))
			result, err := agent.Run(t.Context(), "search", struct{}{})
			if err != nil {
				t.Fatal(err)
			}
			results := completedSearches(result.Messages())
			if len(results) != 2 || results[0].DiscoveredTools[0].Name != "one" || results[1].DiscoveredTools[0].Name != test.want {
				t.Fatalf("results=%+v", results)
			}
		})
	}
}

func TestKeywordIndexConcurrentRanking(t *testing.T) {
	model := fakes.NewFunctionModel(func(_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(completedSearches(messages)) == 0 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: ai.ToolSearchName, Args: []byte(`{"queries":["alpha beta"]}`)}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	catalog := changingSearchCorpus{corpora: [][]ai.ToolDefinition{{{Name: "z", Description: "alpha beta"}, {Name: "a", Description: "alpha"}, {Name: "b", Description: "alpha beta"}, {Name: "k", Description: "unmatched"}}}}
	shared := ai.WithToolSearch[struct{}](catalog, ai.ToolSearchConfig[struct{}]{Strategy: ai.ToolSearchStrategyKeywords, MaxResults: 3})
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			agent := ai.NewAgent[struct{}, string](model)
			agent.AddToolset(shared)
			result, err := agent.Run(t.Context(), "search", struct{}{})
			if err != nil {
				t.Error(err)
				return
			}
			matches := completedSearches(result.Messages())[0].DiscoveredTools
			names := []string{matches[0].Name, matches[1].Name, matches[2].Name}
			if !slices.Equal(names, []string{"z", "b", "a"}) {
				t.Errorf("ranking=%v", names)
			}
		})
	}
	wg.Wait()
}
