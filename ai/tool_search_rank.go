package ai

import (
	"regexp"
	"slices"
	"strings"
)

var toolSearchTokenPattern = regexp.MustCompile(`[a-z0-9]+`)

type scoredToolSearchMatch struct {
	name         string
	score        int
	undiscovered bool
}

func defaultToolSearch(queries []string, tools []ToolDefinition, revealed []string) []string {
	terms := toolSearchTerms(strings.Join(queries, " "))
	revealedSet := make(map[string]struct{}, len(revealed))
	for _, name := range revealed {
		revealedSet[name] = struct{}{}
	}
	var matches []scoredToolSearchMatch
	for _, tool := range tools {
		score := intersectionSize(terms, toolSearchTerms(tool.Name+" "+tool.Description))
		if score == 0 {
			continue
		}
		_, alreadyRevealed := revealedSet[tool.Name]
		matches = append(matches, scoredToolSearchMatch{
			name: tool.Name, score: score, undiscovered: !alreadyRevealed,
		})
	}
	novelty := map[bool]int{false: 0, true: 1}
	slices.SortStableFunc(matches, func(left, right scoredToolSearchMatch) int {
		if difference := novelty[right.undiscovered] - novelty[left.undiscovered]; difference != 0 {
			return difference
		}
		return right.score - left.score
	})
	names := make([]string, len(matches))
	for index, match := range matches {
		names[index] = match.name
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

func intersectionSize(left, right map[string]struct{}) int {
	count := 0
	for value := range left {
		if _, ok := right[value]; ok {
			count++
		}
	}
	return count
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
