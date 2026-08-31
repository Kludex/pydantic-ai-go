package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// ToolSearchName is the reserved model-facing name of the tool-search surface.
const ToolSearchName = "search_tools"

// ToolSearchStrategy selects where and how deferred-tool discovery runs.
type ToolSearchStrategy string

const (
	// ToolSearchStrategyAuto prefers provider-managed search and falls back to
	// local keyword matching when the selected model does not support it.
	ToolSearchStrategyAuto ToolSearchStrategy = ""
	// ToolSearchStrategyKeywords always runs local keyword matching. Providers
	// may expose it through a client-executed native search protocol.
	ToolSearchStrategyKeywords ToolSearchStrategy = "keywords"
	// ToolSearchStrategyBM25 requires provider-managed BM25 search.
	ToolSearchStrategyBM25 ToolSearchStrategy = "bm25"
	// ToolSearchStrategyRegex requires provider-managed regular-expression search.
	ToolSearchStrategyRegex ToolSearchStrategy = "regex"
	// ToolSearchStrategyCustom is assigned when Search is set. Configure Search
	// instead of selecting this value directly.
	ToolSearchStrategyCustom ToolSearchStrategy = "custom"
)

const defaultToolSearchDescription = "Search for deferred tools by words from their names and descriptions."
const defaultToolSearchQueryDescription = "Queries containing words likely to appear in tool names or descriptions."
const noToolSearchMatches = "No matching tools found. The tools you need may not be available."

// ToolSearchFunc searches the complete deferred corpus. Return model-facing
// tool names in relevance order. Unknown and duplicate names are ignored.
type ToolSearchFunc[Deps any] func(
	ctx context.Context,
	rc *RunContext[Deps],
	queries []string,
	tools []ToolDefinition,
) ([]string, error)

// ToolSearchConfig configures deferred-tool discovery. The zero value prefers
// provider-managed search, falls back to keyword overlap, and returns at most ten tools locally.
type ToolSearchConfig[Deps any] struct {
	// Strategy defaults to provider-managed search with a local keyword fallback.
	// Set Search for a custom client-executed strategy.
	Strategy         ToolSearchStrategy
	Search           ToolSearchFunc[Deps]
	MaxResults       int
	MaxRetries       *int
	ToolDescription  string
	QueryDescription string
}

// ToolSearchMatch identifies one deferred tool discovered by search.
type ToolSearchMatch struct {
	Name string `json:"name"`
}

// ToolSearchResult is the provider-facing value returned by search_tools.
type ToolSearchResult struct {
	DiscoveredTools []ToolSearchMatch `json:"discovered_tools"`
	Message         string            `json:"message,omitempty"`
}

// WithToolSearch adds a search_tools function to a toolset whenever the
// wrapped collection contains deferred tools.
func WithToolSearch[Deps any](toolset Toolset[Deps], config ToolSearchConfig[Deps]) Toolset[Deps] {
	switch config.Strategy {
	case ToolSearchStrategyAuto, ToolSearchStrategyKeywords, ToolSearchStrategyBM25, ToolSearchStrategyRegex:
	case ToolSearchStrategyCustom:
		panic("ai: configure custom tool search with ToolSearchConfig.Search")
	default:
		panic(fmt.Sprintf("ai: invalid tool search strategy %q", config.Strategy))
	}
	if config.Search != nil && config.Strategy != ToolSearchStrategyAuto {
		panic("ai: tool search strategy and custom search cannot both be set")
	}
	if config.MaxResults < 0 {
		panic(fmt.Sprintf("ai: tool search max results must be non-negative, got %d", config.MaxResults))
	}
	if config.MaxRetries != nil && *config.MaxRetries < 0 {
		panic(fmt.Sprintf("ai: tool search max retries must be non-negative, got %d", *config.MaxRetries))
	}
	if config.MaxResults == 0 {
		config.MaxResults = 10
	}
	if config.MaxRetries != nil {
		retries := *config.MaxRetries
		config.MaxRetries = &retries
	}
	return toolSearchToolset[Deps]{toolset: toolset, config: config}
}

type toolSearchToolset[Deps any] struct {
	toolset Toolset[Deps]
	config  ToolSearchConfig[Deps]
}

func (t toolSearchToolset[Deps]) Tools(
	ctx context.Context, rc *RunContext[Deps],
) ([]Tool[Deps], error) {
	tools, err := resolveToolsetTools(ctx, rc, t.toolset)
	if err != nil {
		return nil, err
	}
	corpus := make([]ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if tool.entry.def.DeferLoading {
			corpus = append(corpus, cloneToolDefinition(tool.entry.def))
		}
	}
	if len(corpus) == 0 {
		return tools, nil
	}
	for _, tool := range tools {
		if tool.entry.def.Name == ToolSearchName {
			return nil, fmt.Errorf("tool name %q is reserved for tool search", ToolSearchName)
		}
	}
	return append(tools, t.searchTool(corpus)), nil
}

func (t toolSearchToolset[Deps]) ToolsetInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	return resolveToolsetInstructions(ctx, rc, t.toolset)
}

func (t toolSearchToolset[Deps]) searchTool(corpus []ToolDefinition) Tool[Deps] {
	description := t.config.ToolDescription
	if description == "" {
		description = defaultToolSearchDescription
	}
	queryDescription := t.config.QueryDescription
	if queryDescription == "" {
		queryDescription = defaultToolSearchQueryDescription
	}
	strategy := t.config.Strategy
	if t.config.Search != nil {
		strategy = ToolSearchStrategyCustom
	}
	definition := ToolDefinition{
		Name: ToolSearchName, Description: description, ToolKind: ToolPartKindToolSearch,
		ToolSearchStrategy: strategy,
		ReturnSchema:       reflectedToolReturnSchema(reflect.TypeFor[ToolSearchResult]()),
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"queries": map[string]any{
					"type": "array", "items": map[string]any{"type": "string"},
					"minItems": 1, "description": queryDescription,
				},
			},
			"required": []string{"queries"}, "additionalProperties": false,
		},
		maxRetries: t.config.MaxRetries,
	}
	call := func(ctx context.Context, rc *RunContext[Deps], raw json.RawMessage) (any, error) {
		var args struct {
			Queries []string `json:"queries"`
		}
		_ = json.Unmarshal(raw, &args)
		if !hasNonBlankQuery(args.Queries) ||
			t.config.Search == nil && len(toolSearchTerms(strings.Join(args.Queries, " "))) == 0 {
			return nil, Retryf("Please provide at least one non-empty search query.")
		}
		definitions := cloneToolDefinitions(corpus)
		var names []string
		var err error
		if t.config.Search == nil {
			names = defaultToolSearch(args.Queries, definitions, rc.RevealedTools())
		} else {
			names, err = t.config.Search(ctx, rc, slices.Clone(args.Queries), definitions)
			if err != nil {
				return nil, err
			}
		}
		matches := validToolSearchMatches(names, definitions, t.config.MaxResults)
		result := ToolSearchResult{DiscoveredTools: make([]ToolSearchMatch, len(matches))}
		for index, name := range matches {
			result.DiscoveredTools[index] = ToolSearchMatch{Name: name}
		}
		if len(matches) == 0 {
			result.Message = noToolSearchMatches
		}
		return ToolReturn{ReturnValue: result, Tools: matches}, nil
	}
	return Tool[Deps]{entry: toolEntry[Deps]{
		def: definition,
		validate: func(_ context.Context, _ *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
			return slices.Clone(rawArgs), nil
		},
		execute: func(ctx context.Context, rc *RunContext[Deps], validated any) (any, error) {
			rawArgs, ok := validated.(json.RawMessage)
			if !ok {
				return nil, fmt.Errorf("validated tool search arguments have type %T, expected json.RawMessage", validated)
			}
			return call(ctx, rc, rawArgs)
		},
	}}
}

func hasNonBlankQuery(queries []string) bool {
	for _, query := range queries {
		if strings.TrimSpace(query) != "" {
			return true
		}
	}
	return false
}
