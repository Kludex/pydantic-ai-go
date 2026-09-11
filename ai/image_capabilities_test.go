package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestImageGenerationCapabilitySelectsNativeOrLocal(t *testing.T) {
	localCalls := 0
	local := ai.NewFunctionToolset(webCapabilityTool(t, &localCalls))
	compression := 80
	native := ai.ImageGenerationTool{
		Quality: ai.ImageGenerationQualityHigh, OutputCompression: &compression,
	}
	model := &selectiveNativeModel{supported: map[string]bool{"image_generation": true}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
			Native: native, Local: local,
		}),
	))
	compression = 1
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); err != nil ||
		localCalls != 0 || len(model.requests) != 1 || len(model.requests[0].NativeTools) != 1 ||
		len(model.requests[0].Tools) != 0 {
		t.Fatalf("native image generation was not selected: local=%d requests=%#v err=%v",
			localCalls, model.requests, err)
	}
	resolved := model.requests[0].NativeTools[0].(ai.ImageGenerationTool)
	if resolved.OutputCompression == nil || *resolved.OutputCompression != 80 {
		t.Fatalf("native image settings were not detached: %#v", resolved)
	}

	model = &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
			Native: native, Local: local,
		}),
	))
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); err != nil || localCalls != 1 ||
		len(model.requests) != 2 || len(model.requests[0].NativeTools) != 0 || len(model.requests[0].Tools) != 1 {
		t.Fatalf("local image generation was not selected: local=%d requests=%#v err=%v",
			localCalls, model.requests, err)
	}
}

func TestImageGenerationCapabilityDerivesLocalFromResolvedNative(t *testing.T) {
	resolveCalls := 0
	factoryCalls := 0
	localCalls := 0
	model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	capability := ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[nativeOrLocalDeps]{
		ResolveNative: func(
			_ context.Context, rc *ai.RunContext[nativeOrLocalDeps],
		) (ai.ImageGenerationTool, error) {
			resolveCalls++
			return ai.ImageGenerationTool{Model: rc.Deps.Location}, nil
		},
		LocalForNative: func(tool ai.ImageGenerationTool) ai.Tool[nativeOrLocalDeps] {
			factoryCalls++
			if tool.Model != "dynamic-image" {
				t.Fatalf("unexpected resolved native tool: %#v", tool)
			}
			return nativeOrLocalSearchTool(t, &localCalls)
		},
	})
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "draw", nativeOrLocalDeps{Location: "dynamic-image"}); err != nil ||
		resolveCalls != 2 || factoryCalls != 2 || localCalls != 1 {
		t.Fatalf("resolved local fallback failed: resolves=%d factories=%d calls=%d err=%v",
			resolveCalls, factoryCalls, localCalls, err)
	}

	staticModel := &selectiveNativeModel{supported: map[string]bool{"image_generation": true}}
	static := ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
		Native: ai.ImageGenerationTool{Model: "static-image"},
		LocalForNative: func(tool ai.ImageGenerationTool) ai.Tool[struct{}] {
			if tool.Model != "static-image" {
				t.Fatalf("unexpected static native tool: %#v", tool)
			}
			return ai.NewSimpleTool[struct{}]("unused", func(context.Context, struct{}) (string, error) {
				return "unused", nil
			})
		},
	})
	if _, err := ai.NewAgent[struct{}, string](staticModel, ai.WithCapabilities(static)).Run(
		t.Context(), "draw", struct{}{},
	); err != nil {
		t.Fatal(err)
	}
}

func TestImageGenerationCapabilityLocalFactoryValidation(t *testing.T) {
	assertNativeOrLocalPanic(t, "either Local or LocalForNative", func() {
		ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
			Local: ai.NewFunctionToolset[struct{}](),
			LocalForNative: func(ai.ImageGenerationTool) ai.Tool[struct{}] {
				return ai.NewSimpleTool[struct{}]("unused", func(context.Context, struct{}) (string, error) {
					return "unused", nil
				})
			},
		})
	})
}

func TestDynamicImageGenerationCapability(t *testing.T) {
	resolveCalls := 0
	model := &selectiveNativeModel{supported: map[string]bool{"image_generation": true}}
	capability := ai.NewDynamicImageGenerationCapability(
		func(_ context.Context, rc *ai.RunContext[nativeOrLocalDeps]) (ai.ImageGenerationTool, error) {
			resolveCalls++
			return ai.ImageGenerationTool{Model: rc.Deps.Location}, nil
		},
		ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int))),
	)
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "draw", nativeOrLocalDeps{Location: "image-model"}); err != nil ||
		resolveCalls != 1 {
		t.Fatalf("dynamic image generation failed: resolves=%d err=%v", resolveCalls, err)
	}
	if tool := model.requests[0].NativeTools[0].(ai.ImageGenerationTool); tool.Model != "image-model" {
		t.Fatalf("unexpected dynamic image tool: %#v", tool)
	}

	resolveErr := errors.New("resolve image model")
	model = &selectiveNativeModel{supported: map[string]bool{"image_generation": true}}
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicImageGenerationCapability(
			func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.ImageGenerationTool, error) {
				return ai.ImageGenerationTool{}, resolveErr
			},
			ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int))),
		),
	))
	if _, err := agent.Run(t.Context(), "draw", nativeOrLocalDeps{}); !errors.Is(err, resolveErr) {
		t.Fatalf("unexpected dynamic image resolver error: %v", err)
	}
}

func TestImageGenerationCapabilityRequiresNativeWithoutLocal(t *testing.T) {
	model := &selectiveNativeModel{supported: map[string]bool{}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{}),
	))
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "have no local fallback") || len(model.requests) != 0 {
		t.Fatalf("missing image fallback was ignored: requests=%d err=%v", len(model.requests), err)
	}

	model = &selectiveNativeModel{supported: map[string]bool{"image_generation": true}}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{}),
	))
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); err != nil || len(model.requests) != 1 {
		t.Fatalf("supported native image generation failed: requests=%d err=%v", len(model.requests), err)
	}

	modelWithDeps := &selectiveNativeModel{supported: map[string]bool{}}
	agentWithDeps := ai.NewAgent[nativeOrLocalDeps, string](modelWithDeps, ai.WithCapabilities(
		ai.NewDynamicImageGenerationCapability(
			func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.ImageGenerationTool, error) {
				return ai.ImageGenerationTool{}, nil
			},
			nil,
		),
	))
	if _, err := agentWithDeps.Run(t.Context(), "draw", nativeOrLocalDeps{}); err == nil ||
		!strings.Contains(err.Error(), "have no local fallback") {
		t.Fatalf("dynamic image generation without fallback succeeded: %v", err)
	}

	assertNativeOrLocalPanic(t, "dynamic image-generation resolver must not be nil", func() {
		ai.NewDynamicImageGenerationCapability[nativeOrLocalDeps](nil, nil)
	})
	assertNativeOrLocalPanic(t, "requires native support", func() {
		ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
			ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
				Native: ai.ImageGenerationTool{Optional: true},
			}),
		))
	})
}
