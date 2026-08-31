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

// ImageGenerationAction selects whether a provider generates or edits an image.
type ImageGenerationAction string

const (
	ImageGenerationActionAuto     ImageGenerationAction = "auto"
	ImageGenerationActionGenerate ImageGenerationAction = "generate"
	ImageGenerationActionEdit     ImageGenerationAction = "edit"
)

// ImageGenerationBackground selects the generated image background.
type ImageGenerationBackground string

const (
	ImageGenerationBackgroundAuto        ImageGenerationBackground = "auto"
	ImageGenerationBackgroundOpaque      ImageGenerationBackground = "opaque"
	ImageGenerationBackgroundTransparent ImageGenerationBackground = "transparent"
)

// ImageGenerationInputFidelity controls how closely edits preserve input features.
type ImageGenerationInputFidelity string

const (
	ImageGenerationInputFidelityLow  ImageGenerationInputFidelity = "low"
	ImageGenerationInputFidelityHigh ImageGenerationInputFidelity = "high"
)

// ImageGenerationModeration controls provider image moderation.
type ImageGenerationModeration string

const (
	ImageGenerationModerationAuto ImageGenerationModeration = "auto"
	ImageGenerationModerationLow  ImageGenerationModeration = "low"
)

// ImageGenerationOutputFormat identifies the generated image encoding.
type ImageGenerationOutputFormat string

const (
	ImageGenerationOutputPNG  ImageGenerationOutputFormat = "png"
	ImageGenerationOutputWebP ImageGenerationOutputFormat = "webp"
	ImageGenerationOutputJPEG ImageGenerationOutputFormat = "jpeg"
)

// ImageGenerationQuality controls the provider's image generation effort.
type ImageGenerationQuality string

const (
	ImageGenerationQualityAuto   ImageGenerationQuality = "auto"
	ImageGenerationQualityLow    ImageGenerationQuality = "low"
	ImageGenerationQualityMedium ImageGenerationQuality = "medium"
	ImageGenerationQualityHigh   ImageGenerationQuality = "high"
)

// ImageGenerationSize is a provider-supported image dimension or quality tier.
type ImageGenerationSize string

const (
	ImageGenerationSizeAuto      ImageGenerationSize = "auto"
	ImageGenerationSize1024x1024 ImageGenerationSize = "1024x1024"
	ImageGenerationSize1024x1536 ImageGenerationSize = "1024x1536"
	ImageGenerationSize1536x1024 ImageGenerationSize = "1536x1024"
	ImageGenerationSize512       ImageGenerationSize = "512"
	ImageGenerationSize1K        ImageGenerationSize = "1K"
	ImageGenerationSize2K        ImageGenerationSize = "2K"
	ImageGenerationSize4K        ImageGenerationSize = "4K"
)

// ImageAspectRatio identifies a portable generated-image aspect ratio.
type ImageAspectRatio string

const (
	ImageAspectRatio21x9 ImageAspectRatio = "21:9"
	ImageAspectRatio16x9 ImageAspectRatio = "16:9"
	ImageAspectRatio4x3  ImageAspectRatio = "4:3"
	ImageAspectRatio3x2  ImageAspectRatio = "3:2"
	ImageAspectRatio1x1  ImageAspectRatio = "1:1"
	ImageAspectRatio9x16 ImageAspectRatio = "9:16"
	ImageAspectRatio3x4  ImageAspectRatio = "3:4"
	ImageAspectRatio2x3  ImageAspectRatio = "2:3"
	ImageAspectRatio5x4  ImageAspectRatio = "5:4"
	ImageAspectRatio4x5  ImageAspectRatio = "4:5"
)

// ImageGenerationTool asks a compatible provider to generate or edit images.
// The zero value lets the provider choose portable defaults.
type ImageGenerationTool struct {
	Action            ImageGenerationAction
	Background        ImageGenerationBackground
	InputFidelity     ImageGenerationInputFidelity
	Moderation        ImageGenerationModeration
	Model             string
	OutputCompression *int
	OutputFormat      ImageGenerationOutputFormat
	PartialImages     int
	Quality           ImageGenerationQuality
	Size              ImageGenerationSize
	AspectRatio       ImageAspectRatio
	Optional          bool
}

// Kind returns the stable native-tool discriminator.
func (ImageGenerationTool) Kind() string { return "image_generation" }

// UniqueID identifies this native tool within one model request.
func (ImageGenerationTool) UniqueID() string { return "image_generation" }

// IsOptional reports whether an unsupported model may omit the tool.
func (tool ImageGenerationTool) IsOptional() bool { return tool.Optional }

// CloneNativeTool returns a detached definition.
func (tool ImageGenerationTool) CloneNativeTool() NativeTool {
	if tool.OutputCompression != nil {
		compression := *tool.OutputCompression
		tool.OutputCompression = &compression
	}
	return tool
}

// FileSearchRetrievalMode selects a provider-managed retrieval strategy.
type FileSearchRetrievalMode string

const (
	FileSearchRetrievalHybrid   FileSearchRetrievalMode = "hybrid"
	FileSearchRetrievalSemantic FileSearchRetrievalMode = "semantic"
	FileSearchRetrievalKeyword  FileSearchRetrievalMode = "keyword"
)

// FileSearchTool asks a compatible provider to search managed file stores.
type FileSearchTool struct {
	FileStoreIDs  []string
	MaxNumResults *int
	Instructions  string
	RetrievalMode FileSearchRetrievalMode
	Optional      bool
}

// Kind returns the stable native-tool discriminator.
func (FileSearchTool) Kind() string { return "file_search" }

// UniqueID identifies this native tool within one model request.
func (FileSearchTool) UniqueID() string { return "file_search" }

// IsOptional reports whether an unsupported model may omit the tool.
func (tool FileSearchTool) IsOptional() bool { return tool.Optional }

// CloneNativeTool returns a detached definition.
func (tool FileSearchTool) CloneNativeTool() NativeTool {
	tool.FileStoreIDs = slices.Clone(tool.FileStoreIDs)
	if tool.MaxNumResults != nil {
		maximum := *tool.MaxNumResults
		tool.MaxNumResults = &maximum
	}
	return tool
}

// MCPServerTool asks a compatible provider to connect to a remote MCP server.
// The authorization token and headers are sent by the provider, not by this process.
type MCPServerTool struct {
	ID                 string
	URL                string
	AuthorizationToken string
	Description        string
	AllowedTools       []string
	Headers            map[string]string
	Optional           bool
}

// Kind returns the stable native-tool discriminator.
func (MCPServerTool) Kind() string { return "mcp_server" }

// UniqueID identifies this MCP server within one model request.
func (tool MCPServerTool) UniqueID() string { return "mcp_server:" + tool.ID }

// IsOptional reports whether an unsupported model may omit the tool.
func (tool MCPServerTool) IsOptional() bool { return tool.Optional }

// CloneNativeTool returns a detached definition.
func (tool MCPServerTool) CloneNativeTool() NativeTool {
	tool.AllowedTools = slices.Clone(tool.AllowedTools)
	if tool.Headers != nil {
		headers := make(map[string]string, len(tool.Headers))
		for name, value := range tool.Headers {
			headers[name] = value
		}
		tool.Headers = headers
	}
	return tool
}

// MemoryTool asks a compatible provider to use an application-defined memory tool.
// Providers may require a local function tool named "memory" to execute commands.
type MemoryTool struct {
	Optional bool
}

// Kind returns the stable native-tool discriminator.
func (MemoryTool) Kind() string { return "memory" }

// UniqueID identifies this native tool within one model request.
func (MemoryTool) UniqueID() string { return "memory" }

// IsOptional reports whether an unsupported model may omit the tool.
func (tool MemoryTool) IsOptional() bool { return tool.Optional }

// CloneNativeTool returns a detached definition.
func (tool MemoryTool) CloneNativeTool() NativeTool { return tool }

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

// ValidateNativeTools checks portable definitions and rejects duplicate identities.
// Model implementations can call it before rendering direct requests.
func ValidateNativeTools(tools []NativeTool) error {
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
		case *WebSearchTool:
			if err := validateWebSearchTool(*tool); err != nil {
				return err
			}
		case WebFetchTool:
			if err := validateWebFetchTool(tool); err != nil {
				return err
			}
		case *WebFetchTool:
			if err := validateWebFetchTool(*tool); err != nil {
				return err
			}
		case ImageGenerationTool:
			if err := validateImageGenerationTool(tool); err != nil {
				return err
			}
		case *ImageGenerationTool:
			if err := validateImageGenerationTool(*tool); err != nil {
				return err
			}
		case FileSearchTool:
			if err := validateFileSearchTool(tool); err != nil {
				return err
			}
		case *FileSearchTool:
			if err := validateFileSearchTool(*tool); err != nil {
				return err
			}
		case MCPServerTool:
			if err := validateMCPServerTool(tool); err != nil {
				return err
			}
		case *MCPServerTool:
			if err := validateMCPServerTool(*tool); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateMCPServerTool(tool MCPServerTool) error {
	if tool.ID == "" {
		return fmt.Errorf("ai: MCP server ID must not be empty")
	}
	if tool.URL == "" {
		return fmt.Errorf("ai: MCP server URL must not be empty")
	}
	if tool.URL == "x-openai-connector:" {
		return fmt.Errorf("ai: OpenAI MCP connector ID must not be empty")
	}
	for _, name := range tool.AllowedTools {
		if name == "" {
			return fmt.Errorf("ai: MCP server allowed tool name must not be empty")
		}
	}
	for name := range tool.Headers {
		if name == "" {
			return fmt.Errorf("ai: MCP server header name must not be empty")
		}
	}
	return nil
}

func validateFileSearchTool(tool FileSearchTool) error {
	if len(tool.FileStoreIDs) == 0 {
		return fmt.Errorf("ai: file search requires at least one file store ID")
	}
	for _, id := range tool.FileStoreIDs {
		if id == "" {
			return fmt.Errorf("ai: file search store ID must not be empty")
		}
	}
	if tool.MaxNumResults != nil && *tool.MaxNumResults <= 0 {
		return fmt.Errorf("ai: file search maximum results must be positive")
	}
	if tool.RetrievalMode != "" && tool.RetrievalMode != FileSearchRetrievalHybrid &&
		tool.RetrievalMode != FileSearchRetrievalSemantic && tool.RetrievalMode != FileSearchRetrievalKeyword {
		return fmt.Errorf("ai: invalid file search retrieval mode %q", tool.RetrievalMode)
	}
	return nil
}

func validateWebFetchTool(tool WebFetchTool) error {
	if tool.MaxUses < 0 {
		return fmt.Errorf("ai: web fetch max uses must not be negative")
	}
	if tool.MaxContentTokens < 0 {
		return fmt.Errorf("ai: web fetch max content tokens must not be negative")
	}
	return nil
}

func validateImageGenerationTool(tool ImageGenerationTool) error {
	if tool.Action != "" && tool.Action != ImageGenerationActionAuto &&
		tool.Action != ImageGenerationActionGenerate && tool.Action != ImageGenerationActionEdit {
		return fmt.Errorf("ai: invalid image generation action %q", tool.Action)
	}
	if tool.Background != "" && tool.Background != ImageGenerationBackgroundAuto &&
		tool.Background != ImageGenerationBackgroundOpaque &&
		tool.Background != ImageGenerationBackgroundTransparent {
		return fmt.Errorf("ai: invalid image generation background %q", tool.Background)
	}
	if tool.InputFidelity != "" && tool.InputFidelity != ImageGenerationInputFidelityLow &&
		tool.InputFidelity != ImageGenerationInputFidelityHigh {
		return fmt.Errorf("ai: invalid image generation input fidelity %q", tool.InputFidelity)
	}
	if tool.Moderation != "" && tool.Moderation != ImageGenerationModerationAuto &&
		tool.Moderation != ImageGenerationModerationLow {
		return fmt.Errorf("ai: invalid image generation moderation %q", tool.Moderation)
	}
	if tool.OutputCompression != nil && (*tool.OutputCompression < 0 || *tool.OutputCompression > 100) {
		return fmt.Errorf("ai: image generation output compression must be between 0 and 100")
	}
	if tool.OutputFormat != "" && tool.OutputFormat != ImageGenerationOutputPNG &&
		tool.OutputFormat != ImageGenerationOutputWebP && tool.OutputFormat != ImageGenerationOutputJPEG {
		return fmt.Errorf("ai: invalid image generation output format %q", tool.OutputFormat)
	}
	if tool.PartialImages < 0 || tool.PartialImages > 3 {
		return fmt.Errorf("ai: image generation partial images must be between 0 and 3")
	}
	if tool.Quality != "" && tool.Quality != ImageGenerationQualityAuto &&
		tool.Quality != ImageGenerationQualityLow && tool.Quality != ImageGenerationQualityMedium &&
		tool.Quality != ImageGenerationQualityHigh {
		return fmt.Errorf("ai: invalid image generation quality %q", tool.Quality)
	}
	validSize := tool.Size == "" || tool.Size == ImageGenerationSizeAuto ||
		tool.Size == ImageGenerationSize1024x1024 || tool.Size == ImageGenerationSize1024x1536 ||
		tool.Size == ImageGenerationSize1536x1024 || tool.Size == ImageGenerationSize512 ||
		tool.Size == ImageGenerationSize1K || tool.Size == ImageGenerationSize2K || tool.Size == ImageGenerationSize4K
	if !validSize {
		return fmt.Errorf("ai: invalid image generation size %q", tool.Size)
	}
	validAspectRatio := tool.AspectRatio == "" || tool.AspectRatio == ImageAspectRatio21x9 ||
		tool.AspectRatio == ImageAspectRatio16x9 || tool.AspectRatio == ImageAspectRatio4x3 ||
		tool.AspectRatio == ImageAspectRatio3x2 || tool.AspectRatio == ImageAspectRatio1x1 ||
		tool.AspectRatio == ImageAspectRatio9x16 || tool.AspectRatio == ImageAspectRatio3x4 ||
		tool.AspectRatio == ImageAspectRatio2x3 || tool.AspectRatio == ImageAspectRatio5x4 ||
		tool.AspectRatio == ImageAspectRatio4x5
	if !validAspectRatio {
		return fmt.Errorf("ai: invalid image generation aspect ratio %q", tool.AspectRatio)
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
