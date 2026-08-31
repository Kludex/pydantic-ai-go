package ai_test

import (
	"context"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
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
}
