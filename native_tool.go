package ai

import (
	"context"
	"fmt"
	"reflect"
	"slices"
)

// NativeTool is executed by a model provider rather than by the agent.
// Implementations must be safe for concurrent inspection.
type NativeTool interface {
	Kind() string
	UniqueID() string
	IsOptional() bool
	CloneNativeTool() NativeTool
}

// NativeToolFunc resolves one provider-native tool before each model request.
// It may be called concurrently by concurrent runs.
type NativeToolFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error)

type nativeToolEntry[Deps any] struct {
	tool NativeTool
	fn   NativeToolFunc[Deps]
}

func cloneNativeToolEntries[Deps any](entries []nativeToolEntry[Deps]) []nativeToolEntry[Deps] {
	cloned := make([]nativeToolEntry[Deps], len(entries))
	for index, entry := range entries {
		cloned[index] = nativeToolEntry[Deps]{tool: cloneNativeTool(entry.tool), fn: entry.fn}
	}
	return cloned
}

func staticNativeTools[Deps any](entries []nativeToolEntry[Deps]) []NativeTool {
	var tools []NativeTool
	for _, entry := range entries {
		if entry.fn == nil {
			tools = append(tools, cloneNativeTool(entry.tool))
		}
	}
	return tools
}

// WebSearchContextSize controls how much search context a provider retrieves.
type WebSearchContextSize string

const (
	WebSearchContextLow    WebSearchContextSize = "low"
	WebSearchContextMedium WebSearchContextSize = "medium"
	WebSearchContextHigh   WebSearchContextSize = "high"
)

// WebSearchUserLocation localizes provider-native web search results.
type WebSearchUserLocation struct {
	City     string
	Country  string
	Region   string
	Timezone string
}

// WebSearchTool asks a compatible provider to perform web searches.
// The zero value uses the provider's default medium context size.
type WebSearchTool struct {
	SearchContextSize WebSearchContextSize
	UserLocation      *WebSearchUserLocation
	BlockedDomains    []string
	AllowedDomains    []string
	MaxUses           int
	ExternalWebAccess *bool
	Optional          bool
}

// Kind returns the stable native-tool discriminator.
func (WebSearchTool) Kind() string { return "web_search" }

// UniqueID identifies this native tool within one model request.
func (WebSearchTool) UniqueID() string { return "web_search" }

// IsOptional reports whether an unsupported model may omit the tool.
func (tool WebSearchTool) IsOptional() bool { return tool.Optional }

// CloneNativeTool returns a detached definition.
func (tool WebSearchTool) CloneNativeTool() NativeTool { return cloneWebSearchTool(tool) }

// CloneNativeTools returns detached native-tool definitions.
func CloneNativeTools(tools []NativeTool) []NativeTool {
	if tools == nil {
		return nil
	}
	cloned := make([]NativeTool, len(tools))
	for index, tool := range tools {
		cloned[index] = cloneNativeTool(tool)
	}
	return cloned
}

func cloneNativeTool(tool NativeTool) NativeTool {
	if tool == nil {
		return nil
	}
	value := reflect.ValueOf(tool)
	if value.Kind() == reflect.Pointer && value.IsNil() {
		return tool
	}
	return tool.CloneNativeTool()
}

func cloneWebSearchTool(tool WebSearchTool) WebSearchTool {
	tool.BlockedDomains = slices.Clone(tool.BlockedDomains)
	tool.AllowedDomains = slices.Clone(tool.AllowedDomains)
	if tool.UserLocation != nil {
		location := *tool.UserLocation
		tool.UserLocation = &location
	}
	if tool.ExternalWebAccess != nil {
		external := *tool.ExternalWebAccess
		tool.ExternalWebAccess = &external
	}
	return tool
}

func validateNativeTools(tools []NativeTool) error {
	ids := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool == nil || (reflect.ValueOf(tool).Kind() == reflect.Pointer && reflect.ValueOf(tool).IsNil()) {
			return fmt.Errorf("ai: native tool must not be nil")
		}
		if tool.Kind() == "" {
			return fmt.Errorf("ai: native tool kind must not be empty")
		}
		if tool.UniqueID() == "" {
			return fmt.Errorf("ai: native tool unique ID must not be empty")
		}
		if _, duplicate := ids[tool.UniqueID()]; duplicate {
			return fmt.Errorf("ai: duplicate native tool ID %q", tool.UniqueID())
		}
		ids[tool.UniqueID()] = struct{}{}
		if webSearch, ok := tool.(WebSearchTool); ok {
			if err := validateWebSearchTool(webSearch); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateWebSearchTool(tool WebSearchTool) error {
	if tool.SearchContextSize != "" && tool.SearchContextSize != WebSearchContextLow &&
		tool.SearchContextSize != WebSearchContextMedium && tool.SearchContextSize != WebSearchContextHigh {
		return fmt.Errorf("ai: invalid web search context size %q", tool.SearchContextSize)
	}
	if tool.MaxUses < 0 {
		return fmt.Errorf("ai: web search max uses must not be negative")
	}
	return nil
}
