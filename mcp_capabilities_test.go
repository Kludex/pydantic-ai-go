package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestMCPServerCapabilitySelectsNativeOrLocal(t *testing.T) {
	localCalls := 0
	local := ai.NewFunctionToolset(webCapabilityTool(t, &localCalls))
	native := ai.MCPServerTool{
		ID: "docs", URL: "https://mcp.example.com", AllowedTools: []string{"local_search"},
	}
	model := &selectiveNativeModel{supported: map[string]bool{"mcp_server:docs": true}}
	capability := ai.NewMCPServerCapability(ai.MCPServerCapabilityConfig[struct{}]{
		Native: native, Local: local,
	})
	native.AllowedTools[0] = "mutated"
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "use MCP", struct{}{}); err != nil || localCalls != 0 ||
		len(model.requests) != 1 || len(model.requests[0].NativeTools) != 1 || len(model.requests[0].Tools) != 0 {
		t.Fatalf("native MCP was not selected: local=%d requests=%#v err=%v", localCalls, model.requests, err)
	}
	resolved := model.requests[0].NativeTools[0].(ai.MCPServerTool)
	if len(resolved.AllowedTools) != 1 || resolved.AllowedTools[0] != "local_search" {
		t.Fatalf("native MCP settings were not detached: %#v", resolved)
	}

	model = &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "use MCP", struct{}{}); err != nil || localCalls != 1 ||
		len(model.requests) != 2 || len(model.requests[0].NativeTools) != 0 || len(model.requests[0].Tools) != 1 {
		t.Fatalf("local MCP was not selected: local=%d requests=%#v err=%v", localCalls, model.requests, err)
	}
}

func TestMCPServerCapabilityFiltersLocalTools(t *testing.T) {
	localCalls := 0
	model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewMCPServerCapability(ai.MCPServerCapabilityConfig[struct{}]{
			Native: ai.MCPServerTool{
				ID: "docs", URL: "https://mcp.example.com", AllowedTools: []string{"other"},
			},
			Local: ai.NewFunctionToolset(webCapabilityTool(t, &localCalls)),
		}),
	))
	if _, err := agent.Run(t.Context(), "use MCP", struct{}{}); err != nil || localCalls != 0 ||
		len(model.requests) != 2 || len(model.requests[0].Tools) != 0 {
		t.Fatalf("MCP allowlist exposed a local tool: local=%d requests=%#v err=%v",
			localCalls, model.requests, err)
	}
}

func TestMCPServerCapabilityRequiresNativeWithoutLocal(t *testing.T) {
	native := ai.MCPServerTool{ID: "docs", URL: "https://mcp.example.com"}
	model := &selectiveNativeModel{supported: map[string]bool{}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewMCPServerCapability(ai.MCPServerCapabilityConfig[struct{}]{Native: native}),
	))
	if _, err := agent.Run(t.Context(), "use MCP", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "have no local fallback") || len(model.requests) != 0 {
		t.Fatalf("missing MCP fallback was ignored: requests=%d err=%v", len(model.requests), err)
	}

	model = &selectiveNativeModel{supported: map[string]bool{"mcp_server:docs": true}}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewMCPServerCapability(ai.MCPServerCapabilityConfig[struct{}]{Native: native}),
	))
	if _, err := agent.Run(t.Context(), "use MCP", struct{}{}); err != nil || len(model.requests) != 1 {
		t.Fatalf("supported native MCP failed: requests=%d err=%v", len(model.requests), err)
	}

	assertNativeOrLocalPanic(t, "requires native support", func() {
		optional := native
		optional.Optional = true
		ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
			ai.NewMCPServerCapability(ai.MCPServerCapabilityConfig[struct{}]{Native: optional}),
		))
	})
}

func TestDynamicMCPServerCapability(t *testing.T) {
	resolveCalls := 0
	allowed := []string{"local_search"}
	model := &selectiveNativeModel{supported: map[string]bool{"mcp_server:docs": true}}
	capability := ai.NewDynamicMCPServerCapability(ai.DynamicMCPServerCapabilityConfig[nativeOrLocalDeps]{
		ID: "docs",
		Resolve: func(_ context.Context, rc *ai.RunContext[nativeOrLocalDeps]) (ai.MCPServerTool, error) {
			resolveCalls++
			return ai.MCPServerTool{ID: "docs", URL: "https://" + rc.Deps.Location + "/mcp"}, nil
		},
		Local:        ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int))),
		AllowedTools: allowed,
	})
	allowed[0] = "mutated"
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "use MCP", nativeOrLocalDeps{Location: "docs.example.com"}); err != nil ||
		resolveCalls != 1 {
		t.Fatalf("dynamic MCP failed: resolves=%d err=%v", resolveCalls, err)
	}
	tool := model.requests[0].NativeTools[0].(ai.MCPServerTool)
	if tool.URL != "https://docs.example.com/mcp" || len(tool.AllowedTools) != 1 ||
		tool.AllowedTools[0] != "local_search" {
		t.Fatalf("unexpected dynamic MCP tool: %#v", tool)
	}

	resolveErr := errors.New("resolve MCP server")
	model = &selectiveNativeModel{supported: map[string]bool{"mcp_server:docs": true}}
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicMCPServerCapability(ai.DynamicMCPServerCapabilityConfig[nativeOrLocalDeps]{
			ID: "docs",
			Resolve: func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.MCPServerTool, error) {
				return ai.MCPServerTool{}, resolveErr
			},
			Local: ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int))),
		}),
	))
	if _, err := agent.Run(t.Context(), "use MCP", nativeOrLocalDeps{}); !errors.Is(err, resolveErr) {
		t.Fatalf("unexpected dynamic MCP resolver error: %v", err)
	}
}

func TestDynamicMCPServerCapabilityValidation(t *testing.T) {
	assertNativeOrLocalPanic(t, "dynamic MCP server ID must not be empty", func() {
		ai.NewDynamicMCPServerCapability(ai.DynamicMCPServerCapabilityConfig[struct{}]{
			Resolve: func(context.Context, *ai.RunContext[struct{}]) (ai.MCPServerTool, error) {
				return ai.MCPServerTool{}, nil
			},
		})
	})
	assertNativeOrLocalPanic(t, "dynamic MCP server resolver must not be nil", func() {
		ai.NewDynamicMCPServerCapability(ai.DynamicMCPServerCapabilityConfig[struct{}]{ID: "docs"})
	})

	model := &selectiveNativeModel{supported: map[string]bool{"mcp_server:docs": true}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewDynamicMCPServerCapability(ai.DynamicMCPServerCapabilityConfig[struct{}]{
			ID: "docs",
			Resolve: func(context.Context, *ai.RunContext[struct{}]) (ai.MCPServerTool, error) {
				return ai.MCPServerTool{ID: "other", URL: "https://example.com/mcp"}, nil
			},
		}),
	))
	if _, err := agent.Run(t.Context(), "use MCP", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), `returned ID "mcp_server:other"`) {
		t.Fatalf("dynamic MCP identity drift succeeded: %v", err)
	}

	model = &selectiveNativeModel{supported: map[string]bool{}}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewDynamicMCPServerCapability(ai.DynamicMCPServerCapabilityConfig[struct{}]{
			ID: "docs",
			Resolve: func(context.Context, *ai.RunContext[struct{}]) (ai.MCPServerTool, error) {
				return ai.MCPServerTool{ID: "docs", URL: "https://example.com/mcp"}, nil
			},
		}),
	))
	if _, err := agent.Run(t.Context(), "use MCP", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "have no local fallback") {
		t.Fatalf("dynamic MCP without fallback succeeded: %v", err)
	}
}
