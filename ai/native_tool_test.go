package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type nativeToolCapability struct{ tool ai.NativeTool }

func (capability nativeToolCapability) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddNativeTool(capability.tool)
	return nil
}

type customNativeTool struct {
	kind     string
	uniqueID string
	optional bool
	config   map[string]any
}

func (tool customNativeTool) Kind() string     { return tool.kind }
func (tool customNativeTool) UniqueID() string { return tool.uniqueID }
func (tool customNativeTool) IsOptional() bool { return tool.optional }
func (tool customNativeTool) CloneNativeTool() ai.NativeTool {
	cloned := tool
	cloned.config = map[string]any{}
	for key, value := range tool.config {
		switch value := value.(type) {
		case []string:
			cloned.config[key] = append([]string(nil), value...)
		default:
			cloned.config[key] = value
		}
	}
	return cloned
}

func TestNativeToolsAreDetachedAndRunScoped(t *testing.T) {
	external := false
	location := &ai.WebSearchUserLocation{City: "Paris", Country: "FR"}
	configured := ai.WebSearchTool{
		SearchContextSize: ai.WebSearchContextHigh,
		UserLocation:      location, AllowedDomains: []string{"example.com"},
		ExternalWebAccess: &external,
	}
	var captured []ai.NativeTool
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = ai.CloneNativeTools(params.NativeTools)
		web := params.NativeTools[0].(ai.WebSearchTool)
		web.AllowedDomains[0] = "model.example"
		web.UserLocation.City = "Lyon"
		*web.ExternalWebAccess = true
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithNativeTools(configured))
	configured.AllowedDomains[0] = "caller.example"
	location.City = "caller"
	external = true
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
	web := captured[0].(ai.WebSearchTool)
	if web.SearchContextSize != ai.WebSearchContextHigh || web.AllowedDomains[0] != "example.com" ||
		web.UserLocation.City != "Paris" || *web.ExternalWebAccess {
		t.Fatalf("native tool was not detached: %+v", web)
	}
	if _, err := agent.Run(t.Context(), "again", struct{}{}); err != nil {
		t.Fatal(err)
	}
	web = captured[0].(ai.WebSearchTool)
	if web.AllowedDomains[0] != "example.com" || web.UserLocation.City != "Paris" || *web.ExternalWebAccess {
		t.Fatalf("model mutation escaped request: %+v", web)
	}
}

func TestImageGenerationToolIsDetached(t *testing.T) {
	compression := 80
	tool := ai.ImageGenerationTool{OutputCompression: &compression, Optional: true}
	cloned := tool.CloneNativeTool().(ai.ImageGenerationTool)
	*cloned.OutputCompression = 20
	if compression != 80 || *tool.OutputCompression != 80 {
		t.Fatal("image generation compression pointer was not detached")
	}
	if !tool.IsOptional() || tool.Kind() != "image_generation" || tool.UniqueID() != "image_generation" {
		t.Fatalf("unexpected image generation identity: %+v", tool)
	}
	ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{}))
}

func TestFileSearchToolIsDetached(t *testing.T) {
	maximum := 5
	tool := ai.FileSearchTool{
		FileStoreIDs: []string{"store"}, MaxNumResults: &maximum,
		RetrievalMode: ai.FileSearchRetrievalHybrid, Optional: true,
	}
	cloned := tool.CloneNativeTool().(ai.FileSearchTool)
	cloned.FileStoreIDs[0] = "changed"
	*cloned.MaxNumResults = 1
	if tool.FileStoreIDs[0] != "store" || *tool.MaxNumResults != 5 || !tool.IsOptional() ||
		tool.Kind() != "file_search" || tool.UniqueID() != "file_search" {
		t.Fatalf("file search tool was not detached: %+v", tool)
	}
}

func TestAdvisorTool(t *testing.T) {
	maxUses := 2
	maxTokens := 2048
	tool := ai.AdvisorTool{
		Model: "claude-opus-4-8", MaxUses: &maxUses, MaxTokens: &maxTokens,
		Caching: ai.AdvisorCaching1Hour, Optional: true,
	}
	if tool.Kind() != "advisor" || tool.UniqueID() != "advisor" || !tool.IsOptional() {
		t.Fatalf("unexpected advisor identity: %+v", tool)
	}
	clone := tool.CloneNativeTool().(ai.AdvisorTool)
	*clone.MaxUses = 3
	*clone.MaxTokens = 4096
	if *tool.MaxUses != 2 || *tool.MaxTokens != 2048 {
		t.Fatal("advisor clone shares pointer state")
	}
	lowTokens := 512
	for _, test := range []struct {
		name string
		tool ai.AdvisorTool
		want string
	}{
		{name: "model", tool: ai.AdvisorTool{}, want: "model must not be empty"},
		{name: "tokens", tool: ai.AdvisorTool{Model: "advisor", MaxTokens: &lowTokens}, want: "at least 1024"},
		{name: "cache", tool: ai.AdvisorTool{Model: "advisor", Caching: "day"}, want: "caching TTL"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := ai.ValidateNativeTools([]ai.NativeTool{test.tool})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected advisor validation error: %v", err)
			}
		})
	}
	if err := ai.ValidateNativeTools([]ai.NativeTool{&tool}); err != nil {
		t.Fatal(err)
	}
	invalidPointer := &ai.AdvisorTool{}
	if err := ai.ValidateNativeTools([]ai.NativeTool{invalidPointer}); err == nil ||
		!strings.Contains(err.Error(), "model must not be empty") {
		t.Fatalf("unexpected advisor pointer error: %v", err)
	}
}

func TestMCPServerTool(t *testing.T) {
	tool := ai.MCPServerTool{
		ID: "docs", URL: "https://example.com/mcp", AuthorizationToken: "secret",
		Description: "Documentation", AllowedTools: []string{"search"},
		Headers: map[string]string{"X-Tenant": "acme"}, Optional: true,
	}
	if tool.Kind() != "mcp_server" || tool.UniqueID() != "mcp_server:docs" || !tool.IsOptional() {
		t.Fatalf("unexpected MCP server identity: %+v", tool)
	}
	clone := tool.CloneNativeTool().(ai.MCPServerTool)
	clone.AllowedTools[0] = "changed"
	clone.Headers["X-Tenant"] = "changed"
	if tool.AllowedTools[0] != "search" || tool.Headers["X-Tenant"] != "acme" {
		t.Fatal("MCP server clone shares mutable state")
	}

	tests := []struct {
		name string
		tool ai.MCPServerTool
		want string
	}{
		{name: "ID", tool: ai.MCPServerTool{URL: "https://example.com/mcp"}, want: "ID must not be empty"},
		{name: "URL", tool: ai.MCPServerTool{ID: "docs"}, want: "URL must not be empty"},
		{name: "connector", tool: ai.MCPServerTool{ID: "docs", URL: "x-openai-connector:"}, want: "connector ID"},
		{name: "tool", tool: ai.MCPServerTool{ID: "docs", URL: "https://example.com", AllowedTools: []string{""}}, want: "allowed tool name"},
		{name: "header", tool: ai.MCPServerTool{ID: "docs", URL: "https://example.com", Headers: map[string]string{"": "value"}}, want: "header name"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ai.ValidateNativeTools([]ai.NativeTool{test.tool})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected validation error: %v", err)
			}
		})
	}
	if err := ai.ValidateNativeTools([]ai.NativeTool{&tool}); err != nil {
		t.Fatal(err)
	}
	invalidPointer := &ai.MCPServerTool{URL: "https://example.com/mcp"}
	if err := ai.ValidateNativeTools([]ai.NativeTool{invalidPointer}); err == nil ||
		!strings.Contains(err.Error(), "ID must not be empty") {
		t.Fatalf("unexpected pointer validation error: %v", err)
	}
}

func TestMemoryToolIdentity(t *testing.T) {
	tool := ai.MemoryTool{Optional: true}
	if !tool.IsOptional() || tool.Kind() != "memory" || tool.UniqueID() != "memory" ||
		tool.CloneNativeTool() != tool {
		t.Fatalf("unexpected memory tool identity: %+v", tool)
	}
}

func TestNativeToolsFromAgentRunAndCapability(t *testing.T) {
	tests := map[string]func(*ai.Agent[struct{}, string]) []ai.RunOption{
		"agent method": func(agent *ai.Agent[struct{}, string]) []ai.RunOption {
			agent.AddNativeTool(ai.WebSearchTool{})
			return nil
		},
		"run option": func(*ai.Agent[struct{}, string]) []ai.RunOption {
			return []ai.RunOption{ai.WithRunNativeTools(ai.WebSearchTool{})}
		},
		"capability": func(*ai.Agent[struct{}, string]) []ai.RunOption {
			return []ai.RunOption{ai.WithRunCapabilities(nativeToolCapability{tool: ai.WebSearchTool{}})}
		},
	}
	for name, setup := range tests {
		t.Run(name, func(t *testing.T) {
			seen := 0
			model := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				seen = len(params.NativeTools)
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			agent := ai.NewAgent[struct{}, string](model)
			options := setup(agent)
			if _, err := agent.Run(t.Context(), "go", struct{}{}, options...); err != nil {
				t.Fatal(err)
			}
			if seen != 1 {
				t.Fatalf("model received %d native tools", seen)
			}
			if name == "run option" {
				if _, err := agent.Run(t.Context(), "without", struct{}{}); err != nil {
					t.Fatal(err)
				}
				if seen != 0 {
					t.Fatal("per-run native tool mutated the agent")
				}
			}
		})
	}
}

func TestValidateNativeToolPointers(t *testing.T) {
	if err := ai.ValidateNativeTools([]ai.NativeTool{
		&ai.WebSearchTool{}, &ai.WebFetchTool{}, &ai.ImageGenerationTool{},
		&ai.FileSearchTool{FileStoreIDs: []string{"store"}},
	}); err != nil {
		t.Fatal(err)
	}
	for _, tool := range []ai.NativeTool{
		&ai.WebSearchTool{SearchContextSize: "huge"},
		&ai.WebFetchTool{MaxUses: -1},
		&ai.ImageGenerationTool{Quality: "maximum"},
		&ai.FileSearchTool{},
	} {
		if err := ai.ValidateNativeTools([]ai.NativeTool{tool}); err == nil {
			t.Fatalf("expected pointer validation error for %T", tool)
		}
	}
}

func TestNativeToolValidation(t *testing.T) {
	var nilWebSearch *ai.WebSearchTool
	tests := map[string]func(){
		"nil": func() { ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(nilWebSearch)) },
		"empty kind": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(customNativeTool{uniqueID: "id"}))
		},
		"empty ID": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(customNativeTool{kind: "custom"}))
		},
		"duplicate": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(
				ai.WebSearchTool{}, ai.WebSearchTool{},
			))
		},
		"context size": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.WebSearchTool{
				SearchContextSize: "huge",
			}))
		},
		"max uses": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.WebSearchTool{MaxUses: -1}))
		},
		"pointer context size": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(&ai.WebSearchTool{
				SearchContextSize: "huge",
			}))
		},
		"capability duplicate": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(),
				ai.WithNativeTools(ai.WebSearchTool{}),
				ai.WithCapabilities(nativeToolCapability{tool: ai.WebSearchTool{}}),
			)
		},
		"web fetch max uses": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.WebFetchTool{MaxUses: -1}))
		},
		"web fetch max content": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.WebFetchTool{
				MaxContentTokens: -1,
			}))
		},
		"image action": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				Action: "replace",
			}))
		},
		"image background": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				Background: "blue",
			}))
		},
		"image fidelity": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				InputFidelity: "exact",
			}))
		},
		"image moderation": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				Moderation: "none",
			}))
		},
		"image compression low": func() {
			compression := -1
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				OutputCompression: &compression,
			}))
		},
		"image compression high": func() {
			compression := 101
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				OutputCompression: &compression,
			}))
		},
		"image format": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				OutputFormat: "gif",
			}))
		},
		"image partials low": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				PartialImages: -1,
			}))
		},
		"image partials high": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				PartialImages: 4,
			}))
		},
		"image quality": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				Quality: "maximum",
			}))
		},
		"image size": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				Size: "800x600",
			}))
		},
		"image aspect ratio": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.ImageGenerationTool{
				AspectRatio: "7:3",
			}))
		},
		"file search stores": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.FileSearchTool{}))
		},
		"file search empty store": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.FileSearchTool{
				FileStoreIDs: []string{""},
			}))
		},
		"file search max results": func() {
			maximum := 0
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.FileSearchTool{
				FileStoreIDs: []string{"store"}, MaxNumResults: &maximum,
			}))
		},
		"file search retrieval mode": func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(ai.FileSearchTool{
				FileStoreIDs: []string{"store"}, RetrievalMode: "neural",
			}))
		},
		"agent method duplicate": func() {
			agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
			agent.AddNativeTool(ai.WebSearchTool{})
			agent.AddNativeTool(ai.WebSearchTool{})
		},
	}
	for name, operation := range tests {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if value := recover(); value == nil {
					t.Fatal("expected panic")
				}
			}()
			operation()
		})
	}
}

type nativeDeps struct{ Domain string }

func TestDynamicNativeToolsResolvePerStep(t *testing.T) {
	steps := 0
	resolved := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		steps++
		if len(params.NativeTools) != 1 {
			t.Fatalf("step %d received native tools %#v", steps, params.NativeTools)
		}
		web := params.NativeTools[0].(ai.WebSearchTool)
		if web.AllowedDomains[0] != "go.dev" {
			t.Fatalf("unexpected dynamic native tool: %+v", web)
		}
		if steps == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "work", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[nativeDeps, string](model)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		return "worked", nil
	})
	agent.AddNativeToolFunc(func(
		_ context.Context, rc *ai.RunContext[nativeDeps],
	) (ai.NativeTool, error) {
		resolved++
		if rc.Deps.Domain != "go.dev" {
			t.Fatalf("unexpected dependencies: %+v", rc.Deps)
		}
		return ai.WebSearchTool{AllowedDomains: []string{rc.Deps.Domain}}, nil
	})
	if _, err := agent.Run(t.Context(), "go", nativeDeps{Domain: "go.dev"}); err != nil {
		t.Fatal(err)
	}
	if steps != 2 || resolved != 2 {
		t.Fatalf("steps=%d dynamic resolutions=%d", steps, resolved)
	}
}

func TestRunDynamicNativeToolIsolationAndErrors(t *testing.T) {
	t.Run("isolated", func(t *testing.T) {
		seen := 0
		model := fakes.NewFunctionModel(func(
			_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			seen = len(params.NativeTools)
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		})
		agent := ai.NewAgent[nativeDeps, string](model)
		if _, err := agent.Run(t.Context(), "go", nativeDeps{}, ai.WithRunNativeToolFunc(func(
			context.Context, *ai.RunContext[nativeDeps],
		) (ai.NativeTool, error) {
			return ai.WebSearchTool{}, nil
		})); err != nil {
			t.Fatal(err)
		}
		if seen != 1 {
			t.Fatalf("run callback produced %d tools", seen)
		}
		if _, err := agent.Run(t.Context(), "again", nativeDeps{}); err != nil {
			t.Fatal(err)
		}
		if seen != 0 {
			t.Fatal("run native tool callback mutated the agent")
		}
	})

	t.Run("callback error", func(t *testing.T) {
		target := errors.New("unavailable")
		agent := ai.NewAgent[nativeDeps, string](fakes.NewTestModel())
		agent.AddNativeToolFunc(func(context.Context, *ai.RunContext[nativeDeps]) (ai.NativeTool, error) {
			return nil, target
		})
		if _, err := agent.Run(t.Context(), "go", nativeDeps{}); !errors.Is(err, target) ||
			!strings.Contains(err.Error(), "resolve native tool") {
			t.Fatalf("unexpected callback error: %v", err)
		}
	})

	t.Run("nil result", func(t *testing.T) {
		agent := ai.NewAgent[nativeDeps, string](fakes.NewTestModel())
		agent.AddNativeToolFunc(func(context.Context, *ai.RunContext[nativeDeps]) (ai.NativeTool, error) {
			return nil, nil
		})
		if _, err := agent.Run(t.Context(), "go", nativeDeps{}); err == nil ||
			!strings.Contains(err.Error(), "native tool must not be nil") {
			t.Fatalf("unexpected nil tool error: %v", err)
		}
	})

	t.Run("dependency mismatch", func(t *testing.T) {
		agent := ai.NewAgent[nativeDeps, string](fakes.NewTestModel())
		_, err := agent.Run(t.Context(), "go", nativeDeps{}, ai.WithRunNativeToolFunc(func(
			context.Context, *ai.RunContext[struct{}],
		) (ai.NativeTool, error) {
			return ai.WebSearchTool{}, nil
		}))
		if err == nil || !strings.Contains(err.Error(), "dependencies do not match agent") {
			t.Fatalf("unexpected dependency error: %v", err)
		}
	})
}

func TestNilNativeToolFunctionsPanic(t *testing.T) {
	for name, operation := range map[string]func(){
		"agent": func() {
			var fn ai.NativeToolFunc[nativeDeps]
			ai.NewAgent[nativeDeps, string](fakes.NewTestModel()).AddNativeToolFunc(fn)
		},
		"run": func() {
			var fn ai.NativeToolFunc[nativeDeps]
			ai.WithRunNativeToolFunc(fn)
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			operation()
		})
	}
}

func TestNativeToolRequestContextClone(t *testing.T) {
	external := false
	request := ai.ModelRequestContext{Params: ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{
		AllowedDomains: []string{"example.com"}, UserLocation: &ai.WebSearchUserLocation{City: "Paris"},
		ExternalWebAccess: &external,
	}}}}
	cloned := request.Clone()
	web := cloned.Params.NativeTools[0].(ai.WebSearchTool)
	web.AllowedDomains[0] = "changed"
	web.UserLocation.City = "Lyon"
	*web.ExternalWebAccess = true
	original := request.Params.NativeTools[0].(ai.WebSearchTool)
	if original.AllowedDomains[0] != "example.com" || original.UserLocation.City != "Paris" || *original.ExternalWebAccess {
		t.Fatalf("request clone mutated native tool: %+v", original)
	}
}

func TestRunRejectsDuplicateNativeTools(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithNativeTools(&ai.WebSearchTool{}))
	if _, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithRunNativeTools(ai.WebSearchTool{})); err == nil ||
		!strings.Contains(err.Error(), "duplicate native tool ID") {
		t.Fatalf("unexpected duplicate native-tool error: %v", err)
	}
}

func TestModelHookNativeToolsAreRevalidated(t *testing.T) {
	hook := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Params.NativeTools = append(request.Params.NativeTools, ai.WebSearchTool{})
		return request, nil
	})
	agent := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithNativeTools(ai.WebSearchTool{}), ai.WithCapabilities(hook),
	)
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "duplicate native tool ID") {
		t.Fatalf("unexpected hook native-tool error: %v", err)
	}
}

func TestCloneCustomNativeTool(t *testing.T) {
	original := customNativeTool{
		kind: "custom", uniqueID: "custom", optional: true, config: map[string]any{"nested": []string{"value"}},
	}
	cloned := ai.CloneNativeTools([]ai.NativeTool{original})[0].(customNativeTool)
	cloned.config["nested"].([]string)[0] = "changed"
	if original.config["nested"].([]string)[0] != "value" {
		t.Fatal("custom native tool configuration was not detached")
	}
	if ai.CloneNativeTools(nil) != nil {
		t.Fatal("nil native tool slice did not remain nil")
	}
	var nilTool ai.NativeTool
	if cloned := ai.CloneNativeTools([]ai.NativeTool{nilTool}); len(cloned) != 1 || cloned[0] != nil {
		t.Fatalf("nil native tool did not remain nil: %#v", cloned)
	}
	if (ai.WebSearchTool{Optional: true}).IsOptional() != true {
		t.Fatal("web search optional state was not exposed")
	}
	fetch := ai.WebFetchTool{
		AllowedDomains: []string{"go.dev"}, BlockedDomains: []string{"example.com"}, Optional: true,
	}
	clonedFetch := fetch.CloneNativeTool().(ai.WebFetchTool)
	clonedFetch.AllowedDomains[0] = "changed"
	clonedFetch.BlockedDomains[0] = "changed"
	if fetch.AllowedDomains[0] != "go.dev" || fetch.BlockedDomains[0] != "example.com" ||
		!clonedFetch.IsOptional() || clonedFetch.Kind() != "web_fetch" || clonedFetch.UniqueID() != "web_fetch" {
		t.Fatalf("web fetch clone or identity is invalid: %+v", clonedFetch)
	}
	code := ai.CodeExecutionTool{
		Files: []ai.UploadedFile{{
			FileID: "file-1", ProviderName: "anthropic", VendorMetadata: map[string]any{"purpose": "data"},
		}},
		Optional: true,
	}
	clonedCode := code.CloneNativeTool().(ai.CodeExecutionTool)
	clonedCode.Files[0].FileID = "changed"
	clonedCode.Files[0].VendorMetadata["purpose"] = "changed"
	if !clonedCode.IsOptional() || clonedCode.Kind() != "code_execution" ||
		clonedCode.UniqueID() != "code_execution" || code.Files[0].FileID != "file-1" ||
		code.Files[0].VendorMetadata["purpose"] != "data" {
		t.Fatalf("code execution clone or identity is invalid: %+v", clonedCode)
	}
}
