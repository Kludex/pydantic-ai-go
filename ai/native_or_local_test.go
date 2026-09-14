package ai_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type scalarNativeTool string

func (scalarNativeTool) Kind() string                        { return "scalar" }
func (scalarNativeTool) UniqueID() string                    { return "image_generation" }
func (scalarNativeTool) IsOptional() bool                    { return false }
func (tool scalarNativeTool) CloneNativeTool() ai.NativeTool { return tool }

type pointerNativeTool struct {
	Setting string
}

func (*pointerNativeTool) Kind() string                        { return "pointer" }
func (*pointerNativeTool) UniqueID() string                    { return "image_generation" }
func (*pointerNativeTool) IsOptional() bool                    { return false }
func (tool *pointerNativeTool) CloneNativeTool() ai.NativeTool { clone := *tool; return &clone }

type privateNativeTool struct {
	setting string
}

func (privateNativeTool) Kind() string                        { return "private" }
func (privateNativeTool) UniqueID() string                    { return "image_generation" }
func (privateNativeTool) IsOptional() bool                    { return false }
func (tool privateNativeTool) CloneNativeTool() ai.NativeTool { return tool }

type configurableNativeTool struct {
	Value any
}

func (configurableNativeTool) Kind() string                        { return "configurable" }
func (configurableNativeTool) UniqueID() string                    { return "image_generation" }
func (configurableNativeTool) IsOptional() bool                    { return false }
func (tool configurableNativeTool) CloneNativeTool() ai.NativeTool { return tool }

type alternateImageTool struct{}

func (alternateImageTool) Kind() string                        { return "alternate_image" }
func (alternateImageTool) UniqueID() string                    { return "image_generation" }
func (alternateImageTool) IsOptional() bool                    { return false }
func (tool alternateImageTool) CloneNativeTool() ai.NativeTool { return tool }

func TestNativeOrLocalCapabilityCombinationEdges(t *testing.T) {
	image := ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
		Native: ai.ImageGenerationTool{},
	})
	alternate := ai.NewNativeOrLocalToolset[struct{}](
		alternateImageTool{}, nil, ai.WithNativeRequired("no fallback"),
	)
	deferred := func() (recovered any) {
		defer func() { recovered = recover() }()
		ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(image, alternate))
		return nil
	}()
	if !strings.Contains(fmt.Sprint(deferred), "same type") {
		t.Fatalf("unexpected native type collision: %v", deferred)
	}
	firstScalar := ai.NewNativeOrLocalToolset[struct{}](
		scalarNativeTool("first"), nil, ai.WithNativeRequired("no fallback"),
	)
	secondScalar := ai.NewNativeOrLocalToolset[struct{}](
		scalarNativeTool("second"), nil, ai.WithNativeRequired("no fallback"),
	)
	if _, err := secondScalar.CombineCapabilities([]ai.Capability{firstScalar, secondScalar}); err != nil {
		t.Fatalf("scalar native definitions did not use the later value: %v", err)
	}
	firstPointer := ai.NewNativeOrLocalToolset[struct{}](
		&pointerNativeTool{Setting: "first"}, nil, ai.WithNativeRequired("no fallback"),
	)
	secondPointer := ai.NewNativeOrLocalToolset[struct{}](
		&pointerNativeTool{Setting: "second"}, nil, ai.WithNativeRequired("no fallback"),
	)
	if _, err := secondPointer.CombineCapabilities([]ai.Capability{firstPointer, secondPointer}); err != nil {
		t.Fatalf("pointer native definitions did not merge: %v", err)
	}
	firstPrivate := ai.NewNativeOrLocalToolset[struct{}](
		privateNativeTool{setting: "first"}, nil, ai.WithNativeRequired("no fallback"),
	)
	secondPrivate := ai.NewNativeOrLocalToolset[struct{}](
		privateNativeTool{setting: "second"}, nil, ai.WithNativeRequired("no fallback"),
	)
	if _, err := secondPrivate.CombineCapabilities([]ai.Capability{firstPrivate, secondPrivate}); err == nil ||
		!strings.Contains(err.Error(), "unexported field") {
		t.Fatalf("unexpected private native state error: %v", err)
	}
	firstCollection := ai.NewNativeOrLocalToolset[struct{}](
		configurableNativeTool{Value: []string{"a"}}, nil, ai.WithNativeRequired("no fallback"),
	)
	secondCollection := ai.NewNativeOrLocalToolset[struct{}](
		configurableNativeTool{Value: alternateNames{"b"}}, nil, ai.WithNativeRequired("no fallback"),
	)
	if _, err := secondCollection.CombineCapabilities([]ai.Capability{firstCollection, secondCollection}); err == nil ||
		!strings.Contains(err.Error(), "cannot be rebuilt") {
		t.Fatalf("unexpected native collection error: %v", err)
	}

	resolved := ""
	first := ai.NewDynamicNativeOrLocalToolset(
		"image_generation",
		func(context.Context, *ai.RunContext[struct{}]) (ai.NativeTool, error) {
			resolved = "first"
			return ai.ImageGenerationTool{}, nil
		},
		nil,
		ai.WithNativeRequired("no fallback"),
	)
	second := ai.NewDynamicNativeOrLocalToolset(
		"image_generation",
		func(context.Context, *ai.RunContext[struct{}]) (ai.NativeTool, error) {
			resolved = "second"
			return ai.ImageGenerationTool{}, nil
		},
		nil,
		ai.WithNativeRequired("no fallback"),
	)
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(first, second))
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil || resolved != "second" {
		t.Fatalf("dynamic native merge failed: resolved=%q err=%v", resolved, err)
	}

	local := ai.NewFunctionToolset(ai.NewSimpleTool[struct{}](
		"local_image", func(context.Context, struct{}) (string, error) { return "ok", nil },
	))
	withLocal := ai.NewNativeOrLocalToolset(ai.ImageGenerationTool{}, local)
	required := ai.NewNativeOrLocalToolset[struct{}](
		ai.ImageGenerationTool{}, nil, ai.WithNativeRequired("no fallback"),
	)
	if _, err := required.CombineCapabilities([]ai.Capability{withLocal, required}); err != nil {
		t.Fatalf("earlier local fallback was not retained: %v", err)
	}
	if _, err := withLocal.CombineCapabilities([]ai.Capability{required, withLocal}); err != nil {
		t.Fatalf("earlier native requirement was not retained: %v", err)
	}
	fallback := ai.NewImageGenerationCapabilityWithFallback(ai.ImageGenerationSubagentConfig[struct{}]{
		Model: imageOutputModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return nil, errors.New("unused")
		}),
	})
	if _, err := required.CombineCapabilities([]ai.Capability{fallback, required}); err != nil {
		t.Fatalf("fallback rebuild function was not retained: %v", err)
	}
}

type nativeOrLocalDeps struct {
	Location string
}

func assertNativeOrLocalPanic(t *testing.T, contains string, fn func()) {
	t.Helper()
	defer func() {
		recovered := recover()
		if recovered == nil || !strings.Contains(fmt.Sprint(recovered), contains) {
			t.Fatalf("unexpected panic: %v", recovered)
		}
	}()
	fn()
}

type nilNativeOrLocalToolset struct{}

func (*nilNativeOrLocalToolset) Tools(
	context.Context, *ai.RunContext[nativeOrLocalDeps],
) ([]ai.Tool[nativeOrLocalDeps], error) {
	return nil, nil
}

func nativeOrLocalSearchTool(t *testing.T, called *int) ai.Tool[nativeOrLocalDeps] {
	t.Helper()
	return ai.NewSimpleTool[nativeOrLocalDeps]("local_search", func(context.Context, struct{}) (string, error) {
		*called++
		return "local result", nil
	})
}

func TestNativeOrLocalCapabilitySelectsOnePath(t *testing.T) {
	for _, supported := range []bool{false, true} {
		t.Run(map[bool]string{false: "local", true: "native"}[supported], func(t *testing.T) {
			model := &selectiveNativeModel{
				supported: map[string]bool{"web_search": supported}, callLocal: !supported,
			}
			called := 0
			pair := ai.NewNativeOrLocalTool(ai.WebSearchTool{}, nativeOrLocalSearchTool(t, &called))
			agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(pair))
			result, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{Location: "GB"})
			if err != nil {
				t.Fatal(err)
			}
			if result.Output != "done" || called != map[bool]int{false: 1, true: 0}[supported] {
				t.Fatalf("unexpected result: %#v, local calls=%d", result, called)
			}
			for _, request := range model.requests {
				if supported {
					if len(request.NativeTools) != 1 || len(request.Tools) != 0 {
						t.Fatalf("native path was not selected: %#v", request)
					}
				} else if len(request.NativeTools) != 0 || len(request.Tools) != 1 ||
					request.Tools[0].Name != "local_search" {
					t.Fatalf("local path was not selected: %#v", request)
				}
			}
		})
	}
}

func TestNativeOrLocalSupportsAgentAndRunRegistration(t *testing.T) {
	for _, register := range []string{"agent", "run"} {
		t.Run(register, func(t *testing.T) {
			model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
			called := 0
			pair := ai.NewNativeOrLocalTool(ai.WebSearchTool{}, nativeOrLocalSearchTool(t, &called))
			agent := ai.NewAgent[nativeOrLocalDeps, string](model)
			var options []ai.RunOption
			if register == "agent" {
				agent.AddNativeOrLocal(pair)
			} else {
				options = append(options, ai.WithRunCapabilities(pair))
			}
			if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{}, options...); err != nil {
				t.Fatal(err)
			}
			if called != 1 || len(model.requests) != 2 {
				t.Fatalf("pair was not registered: calls=%d requests=%d", called, len(model.requests))
			}
		})
	}
}

func TestDynamicNativeOrLocalUsesDependenciesAndStableIdentity(t *testing.T) {
	model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	called := 0
	resolved := 0
	pair := ai.NewDynamicNativeOrLocalTool(
		"web_search",
		func(_ context.Context, rc *ai.RunContext[nativeOrLocalDeps]) (ai.NativeTool, error) {
			resolved++
			if rc.Deps.Location != "GB" {
				t.Fatalf("resolver received detached wrong dependencies: %#v", rc.Deps)
			}
			return ai.WebSearchTool{UserLocation: &ai.WebSearchUserLocation{Country: rc.Deps.Location}}, nil
		},
		nativeOrLocalSearchTool(t, &called),
	)
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(pair))
	if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{Location: "GB"}); err != nil {
		t.Fatal(err)
	}
	if called != 1 || resolved != 2 {
		t.Fatalf("unexpected callback counts: local=%d native=%d", called, resolved)
	}
	for _, request := range model.requests {
		if len(request.Tools) != 1 || request.Tools[0].NativeFallbackFor != "web_search" {
			t.Fatalf("dynamic pair lost its stable identity: %#v", request)
		}
	}
}

func TestDynamicNativeOrLocalRejectsIdentityChanges(t *testing.T) {
	pair := ai.NewDynamicNativeOrLocalTool(
		"web_search",
		func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.NativeTool, error) {
			return ai.CodeExecutionTool{}, nil
		},
		nativeOrLocalSearchTool(t, new(int)),
	)
	model := &selectiveNativeModel{supported: map[string]bool{}}
	_, err := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(pair)).Run(
		t.Context(), "search", nativeOrLocalDeps{},
	)
	if err == nil || !strings.Contains(err.Error(), `returned ID "code_execution", expected "web_search"`) {
		t.Fatalf("unexpected identity error: %v", err)
	}
	if len(model.requests) != 0 {
		t.Fatal("identity mismatch reached the model")
	}
}

func TestNativeRequiredSuppressesLocalFallback(t *testing.T) {
	called := 0
	pair := ai.NewNativeOrLocalTool(
		ai.WebSearchTool{}, nativeOrLocalSearchTool(t, &called),
		ai.WithNativeRequired("allowed domains"),
	)
	model := &selectiveNativeModel{supported: map[string]bool{}}
	_, err := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(pair)).Run(
		t.Context(), "search", nativeOrLocalDeps{},
	)
	if err == nil || !strings.Contains(err.Error(), "no local fallback") {
		t.Fatalf("unexpected native-required error: %v", err)
	}
	if called != 0 || len(model.requests) != 0 {
		t.Fatalf("native constraint used local or provider: local=%d requests=%d", called, len(model.requests))
	}

	supported := &selectiveNativeModel{supported: map[string]bool{"web_search": true}}
	if _, err := ai.NewAgent[nativeOrLocalDeps, string](supported, ai.WithCapabilities(pair)).Run(
		t.Context(), "search", nativeOrLocalDeps{},
	); err != nil {
		t.Fatal(err)
	}
	if called != 0 || len(supported.requests) != 1 || len(supported.requests[0].NativeTools) != 1 ||
		len(supported.requests[0].Tools) != 0 {
		t.Fatalf("native constraint did not use native path: local=%d requests=%#v", called, supported.requests)
	}
}

func TestDynamicNativeRequiredRejectsOptionalDefinition(t *testing.T) {
	pair := ai.NewDynamicNativeOrLocalTool(
		"web_search",
		func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.NativeTool, error) {
			return ai.WebSearchTool{Optional: true}, nil
		},
		nativeOrLocalSearchTool(t, new(int)),
		ai.WithNativeRequired("maximum uses"),
	)
	model := &selectiveNativeModel{supported: map[string]bool{}}
	_, err := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(pair)).Run(
		t.Context(), "search", nativeOrLocalDeps{},
	)
	if err == nil || !strings.Contains(err.Error(), "requires native support for maximum uses") {
		t.Fatalf("unexpected optional-native error: %v", err)
	}
}

func TestNativeOrLocalConfigurationErrors(t *testing.T) {
	assertNativeOrLocalPanic(t, "native-required reason", func() { ai.WithNativeRequired("") })
	assertNativeOrLocalPanic(t, "option must not be nil", func() {
		ai.NewNativeOrLocalTool(ai.WebSearchTool{}, nativeOrLocalSearchTool(t, new(int)), nil)
	})
	assertNativeOrLocalPanic(t, "tool ID must not be empty", func() {
		ai.NewDynamicNativeOrLocalTool(
			"", func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.NativeTool, error) {
				return ai.WebSearchTool{}, nil
			}, nativeOrLocalSearchTool(t, new(int)),
		)
	})
	assertNativeOrLocalPanic(t, "resolver must not be nil", func() {
		ai.NewDynamicNativeOrLocalTool("web_search", nil, nativeOrLocalSearchTool(t, new(int)))
	})
	assertNativeOrLocalPanic(t, "native tool must not be nil", func() {
		ai.NewAgent[nativeOrLocalDeps, string](nil, ai.WithCapabilities(
			ai.NewNativeOrLocalTool[nativeOrLocalDeps](nil, nativeOrLocalSearchTool(t, new(int))),
		))
	})
	assertNativeOrLocalPanic(t, "local toolset must not be nil", func() {
		var local *nilNativeOrLocalToolset
		ai.NewAgent[nativeOrLocalDeps, string](nil, ai.WithCapabilities(
			ai.NewNativeOrLocalToolset(ai.WebSearchTool{}, local),
		))
	})
	assertNativeOrLocalPanic(t, "must not be nil", func() {
		var pair *ai.NativeOrLocalTool[nativeOrLocalDeps]
		ai.NewAgent[nativeOrLocalDeps, string](nil, ai.WithCapabilities(pair))
	})
	assertNativeOrLocalPanic(t, "definition is optional", func() {
		ai.NewAgent[nativeOrLocalDeps, string](nil, ai.WithCapabilities(ai.NewNativeOrLocalTool(
			ai.WebSearchTool{Optional: true}, nativeOrLocalSearchTool(t, new(int)),
			ai.WithNativeRequired("allowed domains"),
		)))
	})
}

func TestAddNativeOrLocalCommitsAtomically(t *testing.T) {
	var nilPair *ai.NativeOrLocalTool[nativeOrLocalDeps]
	assertNativeOrLocalPanic(t, "must not be nil", func() {
		ai.NewAgent[nativeOrLocalDeps, string](nil).AddNativeOrLocal(nilPair)
	})

	model := &selectiveNativeModel{supported: map[string]bool{"web_search": true}}
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithNativeTools(ai.WebSearchTool{}))
	pair := ai.NewNativeOrLocalTool(ai.WebSearchTool{}, nativeOrLocalSearchTool(t, new(int)))
	assertNativeOrLocalPanic(t, "duplicate native tool ID", func() {
		agent.AddNativeOrLocal(pair)
	})
	if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{}); err != nil {
		t.Fatal(err)
	}
	if len(model.requests) != 1 || len(model.requests[0].NativeTools) != 1 || len(model.requests[0].Tools) != 0 {
		t.Fatalf("failed registration changed the agent: %#v", model.requests)
	}
}

func TestNativeOrLocalDependencyMismatch(t *testing.T) {
	pair := ai.NewNativeOrLocalTool(
		ai.WebSearchTool{},
		ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
			return "local", nil
		}),
	)
	assertNativeOrLocalPanic(t, "dependencies do not match", func() {
		ai.NewAgent[nativeOrLocalDeps, string](nil, ai.WithCapabilities(pair))
	})

	agent := ai.NewAgent[nativeOrLocalDeps, string](&selectiveNativeModel{supported: map[string]bool{}})
	_, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{}, ai.WithRunCapabilities(pair))
	if err == nil || !strings.Contains(err.Error(), "dependencies do not match") {
		t.Fatalf("unexpected run mismatch error: %v", err)
	}
}

func TestDynamicNativeOrLocalRejectsDuplicateResolvedIdentity(t *testing.T) {
	pair := ai.NewDynamicNativeOrLocalTool(
		"web_search",
		func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.NativeTool, error) {
			return ai.WebSearchTool{}, nil
		},
		nativeOrLocalSearchTool(t, new(int)),
	)
	model := &selectiveNativeModel{supported: map[string]bool{}}
	agent := ai.NewAgent[nativeOrLocalDeps, string](
		model, ai.WithNativeTools(ai.WebSearchTool{}), ai.WithCapabilities(pair),
	)
	_, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{})
	if err == nil || !strings.Contains(err.Error(), `duplicate native tool ID "web_search"`) {
		t.Fatalf("unexpected duplicate identity error: %v", err)
	}
	if len(model.requests) != 0 {
		t.Fatal("duplicate identity reached model")
	}
}

func TestNativeOrLocalResolverErrorsBeforeModelRequest(t *testing.T) {
	sentinel := errors.New("resolver failed")
	for name, resolve := range map[string]ai.NativeToolFunc[nativeOrLocalDeps]{
		"error": func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.NativeTool, error) {
			return nil, sentinel
		},
		"nil": func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.NativeTool, error) {
			return nil, nil
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := &selectiveNativeModel{supported: map[string]bool{}}
			pair := ai.NewDynamicNativeOrLocalTool("web_search", resolve, nativeOrLocalSearchTool(t, new(int)))
			_, err := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(pair)).Run(
				t.Context(), "search", nativeOrLocalDeps{},
			)
			if err == nil || !strings.Contains(err.Error(), "resolve native tool") {
				t.Fatalf("unexpected resolver error: %v", err)
			}
			if len(model.requests) != 0 {
				t.Fatal("resolver failure reached model")
			}
		})
	}
}
