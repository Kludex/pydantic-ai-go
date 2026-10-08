package ai

import (
	"cmp"
	"regexp"
	"slices"
	"strings"
	"sync"
)

var toolSearchTokenPattern = regexp.MustCompile(`[a-z0-9]+`)

type keywordTool struct{ name, description string }

type keywordToolSearch struct {
	mu       sync.Mutex
	corpus   []keywordTool
	postings map[string][]int
}

func (search *keywordToolSearch) search(queries []string, tools []ToolDefinition, revealed []string) []string {
	search.mu.Lock()
	defer search.mu.Unlock()
	corpus := make([]keywordTool, len(tools))
	for index, tool := range tools {
		corpus[index] = keywordTool{tool.Name, tool.Description}
	}
	if !slices.Equal(corpus, search.corpus) {
		search.corpus = corpus
		search.postings = make(map[string][]int)
		for index, tool := range corpus {
			for term := range toolSearchTerms(tool.name + " " + tool.description) {
				search.postings[term] = append(search.postings[term], index)
			}
		}
	}
	scores := make(map[int]int)
	for term := range toolSearchTerms(strings.Join(queries, " ")) {
		for _, index := range search.postings[term] {
			scores[index]++
		}
	}
	matches := make([]int, 0, len(scores))
	for index := range scores {
		matches = append(matches, index)
	}
	revealedSet := make(map[string]int, len(revealed))
	for _, name := range revealed {
		revealedSet[name] = 1
	}
	slices.SortFunc(matches, func(left, right int) int {
		if difference := revealedSet[corpus[left].name] - revealedSet[corpus[right].name]; difference != 0 {
			return difference
		}
		if difference := scores[right] - scores[left]; difference != 0 {
			return difference
		}
		return cmp.Compare(left, right)
	})
	names := make([]string, len(matches))
	for index, match := range matches {
		names[index] = corpus[match].name
	}
	return names
}

func toolSearchTerms(value string) map[string]struct{} {
	terms := make(map[string]struct{})
	for _, term := range toolSearchTokenPattern.FindAllString(strings.ToLower(value), -1) {
		terms[term] = struct{}{}
	}
	return terms
}

func validToolSearchMatches(names []string, tools []ToolDefinition, limit int) []string {
	known := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		known[tool.Name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(names))
	matches := make([]string, 0, min(len(names), limit))
	for _, name := range names {
		if _, ok := known[name]; !ok {
			continue
		}
		if _, duplicate := seen[name]; duplicate {
			continue
		}
		seen[name] = struct{}{}
		matches = append(matches, name)
		if len(matches) == limit {
			break
		}
	}
	return matches
}

func cloneToolDefinitions(definitions []ToolDefinition) []ToolDefinition {
	cloned := make([]ToolDefinition, len(definitions))
	for index, definition := range definitions {
		cloned[index] = cloneToolDefinition(definition)
	}
	return cloned
}
