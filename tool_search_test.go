package ai_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestDefaultToolSearchDiscoversMostRelevantDeferredTool(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		switch request {
		case 1:
			if !slices.Equal(toolDefinitionNames(params.Tools), []string{"search_tools", "status"}) {
				t.Fatalf("unexpected initial search tools: %+v", params.Tools)
			}
			var search ai.ToolDefinition
			for _, definition := range params.Tools {
				if definition.Name == ai.ToolSearchName {
					search = definition
				}
			}
			if search.ToolKind != ai.ToolPartKindToolSearch || search.Description != "Find GitHub tools." ||
				search.ReturnSchema["type"] != "object" ||
				search.Schema["properties"].(map[string]any)["queries"].(map[string]any)["description"] !=
					"GitHub operations." {
				t.Fatalf("unexpected search definition: %+v", search)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", Args: []byte(`{"queries":["github profile"]}`),
			}}}, nil
		case 2:
			if !slices.Equal(
				toolDefinitionNames(params.Tools), []string{"github_get_me", "search_tools", "status"},
			) {
				t.Fatalf("search result did not reveal the best match: %+v", params.Tools)
			}
			response := messages[len(messages)-2].(ai.ModelResponse)
			call := response.Parts[0].(ai.ToolCallPart)
			parts := messages[len(messages)-1].(ai.ModelRequest).Parts
			returned := parts[0].(ai.ToolReturnPart)
			result := returned.Content.(ai.ToolSearchResult)
			delta := parts[1].(ai.ToolAvailabilityDeltaPart)
			if call.ToolKind != ai.ToolPartKindToolSearch || returned.ToolKind != ai.ToolPartKindToolSearch ||
				len(result.DiscoveredTools) != 1 || result.DiscoveredTools[0].Name != "github_get_me" ||
				!slices.Equal(delta.ToolsAdded, []string{"github_get_me"}) {
				t.Fatalf("unexpected typed search exchange: call=%+v parts=%+v", call, parts)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search-again",
				Args: []byte(`{"queries":["github profile"]}`),
			}}}, nil
		case 3:
			if !slices.Equal(
				toolDefinitionNames(params.Tools),
				[]string{"github_comment", "github_get_me", "search_tools", "status"},
			) {
				t.Fatalf("repeat search did not prioritize an undiscovered match: %+v", params.Tools)
			}
			parts := messages[len(messages)-1].(ai.ModelRequest).Parts
			result := parts[0].(ai.ToolReturnPart).Content.(ai.ToolSearchResult)
			if len(result.DiscoveredTools) != 1 || result.DiscoveredTools[0].Name != "github_comment" ||
				!slices.Equal(parts[1].(ai.ToolAvailabilityDeltaPart).ToolsAdded, []string{"github_comment"}) {
				t.Fatalf("unexpected repeat search result: %+v", parts)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "github_get_me", ToolCallID: "profile", Args: []byte(`{}`),
			}}}, nil
		default:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
	})
	status := ai.NewSimpleTool[deps]("status", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
	profile := ai.NewTool("github_get_me", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		if !slices.Equal(rc.RevealedTools(), []string{"github_comment", "github_get_me"}) {
			t.Fatalf("revealed tools missing from run context: %v", rc.RevealedTools())
		}
		return "profile", nil
	}, ai.WithDescription("Get the GitHub user profile."), ai.WithDeferredLoading())
	comment := ai.NewSimpleTool[deps](
		"github_comment", func(context.Context, struct{}) (string, error) { return "comment", nil },
		ai.WithDescription("Create a GitHub issue comment."), ai.WithDeferredLoading(),
	)
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(status, profile, comment), ai.ToolSearchConfig[deps]{
		MaxResults: 1, ToolDescription: "Find GitHub tools.", QueryDescription: "GitHub operations.",
	}))
	result, err := agent.Run(t.Context(), "find my profile", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected search result=%+v err=%v", result, err)
	}
}

func TestToolSearchNoMatchesAndValidationRetry(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		switch request {
		case 1:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "blank", Args: []byte(`{"queries":["   "]}`),
			}}}, nil
		case 2:
			retry := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
			if retry.Content != "Please provide at least one non-empty search query." {
				t.Fatalf("unexpected blank search retry: %+v", retry)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "punctuation", Args: []byte(`{"queries":["---"]}`),
			}}}, nil
		case 3:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "none", Args: []byte(`{"queries":["quantum"]}`),
			}}}, nil
		default:
			parts := messages[len(messages)-1].(ai.ModelRequest).Parts
			result := parts[0].(ai.ToolReturnPart).Content.(ai.ToolSearchResult)
			if len(parts) != 1 || len(result.DiscoveredTools) != 0 ||
				result.Message != "No matching tools found. The tools you need may not be available." {
				t.Fatalf("unexpected empty search result: %+v", parts)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
	})
	hidden := ai.NewSimpleTool[deps](
		"weather", func(context.Context, struct{}) (string, error) { return "sunny", nil },
		ai.WithDeferredLoading(),
	)
	retries := 2
	toolset := ai.WithToolSearch(ai.NewFunctionToolset(hidden), ai.ToolSearchConfig[deps]{
		MaxRetries: &retries,
	})
	retries = 0
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(toolset)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestCustomToolSearchFiltersResultsAndDetachesDefinitions(t *testing.T) {
	retries := 2
	searchCalls := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			var search ai.ToolDefinition
			for _, definition := range params.Tools {
				if definition.Name == ai.ToolSearchName {
					search = definition
				}
			}
			if search.Name == "" {
				t.Fatal("custom search tool missing")
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", Args: []byte(`{"queries":["tools"]}`),
			}}}, nil
		}
		parts := messages[len(messages)-1].(ai.ModelRequest).Parts
		result := parts[0].(ai.ToolReturnPart).Content.(ai.ToolSearchResult)
		if len(result.DiscoveredTools) != 2 || result.DiscoveredTools[0].Name != "second" ||
			result.DiscoveredTools[1].Name != "first" {
			t.Fatalf("custom results were not filtered: %+v", result)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	first := ai.NewSimpleTool[deps](
		"first", func(context.Context, struct{}) (string, error) { return "first", nil },
		ai.WithDescription("first original"), ai.WithDeferredLoading(),
	)
	second := ai.NewSimpleTool[deps](
		"second", func(context.Context, struct{}) (string, error) { return "second", nil },
		ai.WithDeferredLoading(),
	)
	config := ai.ToolSearchConfig[deps]{
		MaxResults: 2, MaxRetries: &retries,
		Search: func(
			_ context.Context, _ *ai.RunContext[deps], queries []string, tools []ai.ToolDefinition,
		) ([]string, error) {
			searchCalls++
			queries[0] = "changed"
			tools[0].Description = "changed"
			return []string{"second", "unknown", "second", "first"}, nil
		},
	}
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(first, second), config))
	retries = 0
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if searchCalls != 1 || first.Definition().Description != "first original" {
		t.Fatalf("custom search mutated source definitions or call count: calls=%d def=%+v", searchCalls, first.Definition())
	}
}

type failingSearchSource struct {
	err error
}

func (t failingSearchSource) Tools(context.Context, *ai.RunContext[deps]) ([]ai.Tool[deps], error) {
	return nil, t.err
}

func TestToolSearchConfigurationAndResolutionErrors(t *testing.T) {
	for name, build := range map[string]func(){
		"negative results": func() {
			ai.WithToolSearch[deps](failingSearchSource{}, ai.ToolSearchConfig[deps]{MaxResults: -1})
		},
		"negative retries": func() {
			retries := -1
			ai.WithToolSearch[deps](failingSearchSource{}, ai.ToolSearchConfig[deps]{MaxRetries: &retries})
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			build()
		})
	}

	sentinel := errors.New("cannot list corpus")
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	failed := ai.NewAgent[deps, string](model)
	failed.AddToolset(ai.WithToolSearch[deps](failingSearchSource{err: sentinel}, ai.ToolSearchConfig[deps]{}))
	if _, err := failed.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("search source error lost: %v", err)
	}

	reserved := ai.NewAgent[deps, string](model)
	reserved.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(
		ai.NewSimpleTool[deps](ai.ToolSearchName, func(context.Context, struct{}) (string, error) {
			return "reserved", nil
		}),
		ai.NewSimpleTool[deps]("hidden", func(context.Context, struct{}) (string, error) {
			return "hidden", nil
		}, ai.WithDeferredLoading()),
	), ai.ToolSearchConfig[deps]{}))
	if _, err := reserved.Run(t.Context(), "go", deps{}); err == nil ||
		!strings.Contains(err.Error(), `tool name "search_tools" is reserved`) {
		t.Fatalf("unexpected reserved-name error: %v", err)
	}

	searchErr := errors.New("search backend failed")
	searchModel := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", Args: []byte(`{"queries":["hidden"]}`),
		}}}, nil
	})
	searchFailed := ai.NewAgent[deps, string](searchModel)
	searchFailed.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(
		ai.NewSimpleTool[deps]("hidden", func(context.Context, struct{}) (string, error) {
			return "hidden", nil
		}, ai.WithDeferredLoading()),
	), ai.ToolSearchConfig[deps]{
		Search: func(
			context.Context, *ai.RunContext[deps], []string, []ai.ToolDefinition,
		) ([]string, error) {
			return nil, searchErr
		},
	}))
	if _, err := searchFailed.Run(t.Context(), "go", deps{}); !errors.Is(err, searchErr) {
		t.Fatalf("custom search error lost: %v", err)
	}
}
