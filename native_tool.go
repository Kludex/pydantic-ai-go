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

// CodeExecutionTool asks a compatible provider to execute model-generated code.
type CodeExecutionTool struct {
	Files    []UploadedFile
	Optional bool
}

// Kind returns the stable native-tool discriminator.
func (CodeExecutionTool) Kind() string { return "code_execution" }

// UniqueID identifies this native tool within one model request.
func (CodeExecutionTool) UniqueID() string { return "code_execution" }

// IsOptional reports whether an unsupported model may omit the tool.
func (tool CodeExecutionTool) IsOptional() bool { return tool.Optional }

// CloneNativeTool returns a detached definition.
func (tool CodeExecutionTool) CloneNativeTool() NativeTool {
	tool.Files = slices.Clone(tool.Files)
	for index := range tool.Files {
		tool.Files[index].VendorMetadata = cloneSchemaMap(tool.Files[index].VendorMetadata)
	}
	return tool
}

// WebFetchTool asks a compatible provider to retrieve content from URLs.
type WebFetchTool struct {
	MaxUses          int
	AllowedDomains   []string
	BlockedDomains   []string
	EnableCitations  bool
	MaxContentTokens int
	Optional         bool
}

// Kind returns the stable native-tool discriminator.
func (WebFetchTool) Kind() string { return "web_fetch" }

// UniqueID identifies this native tool within one model request.
func (WebFetchTool) UniqueID() string { return "web_fetch" }

// IsOptional reports whether an unsupported model may omit the tool.
func (tool WebFetchTool) IsOptional() bool { return tool.Optional }

// CloneNativeTool returns a detached definition.
func (tool WebFetchTool) CloneNativeTool() NativeTool {
	tool.AllowedDomains = slices.Clone(tool.AllowedDomains)
	tool.BlockedDomains = slices.Clone(tool.BlockedDomains)
	return tool
}

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
		switch tool := tool.(type) {
		case WebSearchTool:
			if err := validateWebSearchTool(tool); err != nil {
				return err
			}
		case WebFetchTool:
			if tool.MaxUses < 0 {
				return fmt.Errorf("ai: web fetch max uses must not be negative")
			}
			if tool.MaxContentTokens < 0 {
				return fmt.Errorf("ai: web fetch max content tokens must not be negative")
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
