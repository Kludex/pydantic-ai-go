package images_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/fakes"
	modelfakes "github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type noNativeModel struct{ *modelfakes.FunctionModel }

func (*noNativeModel) SupportsNativeTool(ai.NativeTool) bool { return false }

type plainModel struct{ result *images.Result }

func (model plainModel) Generate(context.Context, string, []images.Input, images.Settings) (*images.Result, error) {
	return model.result, nil
}
func (plainModel) Name() string         { return "plain" }
func (plainModel) ProviderName() string { return "test" }
func (plainModel) ProviderURL() string  { return "" }

type valueModel struct{}

func (valueModel) Generate(context.Context, string, []images.Input, images.Settings) (*images.Result, error) {
	return &images.Result{Images: []images.GeneratedImage{{Content: ai.BinaryContent{MediaType: "image/png"}}}}, nil
}
func (valueModel) Name() string                     { return "value" }
func (valueModel) ProviderName() string             { return "test" }
func (valueModel) ProviderURL() string              { return "" }
func (valueModel) DefaultSettings() images.Settings { return images.Settings{} }

type model struct {
	name, provider, url string
	defaults            images.Settings
	result              *images.Result
	err                 error
	prompt              string
	inputs              []images.Input
	settings            images.Settings
}

func (model *model) Generate(
	_ context.Context, prompt string, inputs []images.Input, settings images.Settings,
) (*images.Result, error) {
	model.prompt, model.inputs, model.settings = prompt, inputs, settings
	return model.result, model.err
}
func (model *model) Name() string                     { return model.name }
func (model *model) ProviderName() string             { return model.provider }
func (model *model) ProviderURL() string              { return model.url }
func (model *model) DefaultSettings() images.Settings { return model.defaults.Clone() }

func TestGeneratorPrecedenceOverrideAndDetachment(t *testing.T) {
	defaultDimensions := images.Dimensions{Width: 512, Height: 512}
	base := &model{name: "base", provider: "test", defaults: images.Settings{
		Dimensions: &defaultDimensions, ExtraHeaders: map[string]string{"default": "yes"},
	}, result: &images.Result{
		Images: []images.GeneratedImage{{
			Content:         ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
			ProviderDetails: map[string]any{"nested": []any{"value"}},
		}},
		Usage: ai.Usage{Details: map[string]int{"images": 1}}, ProviderDetails: map[string]any{"ok": true},
	}}
	generator := images.New(base, images.WithSettings(images.Settings{AspectRatio: images.AspectRatio16To9}))
	override := &model{name: "override", provider: "test", result: &images.Result{
		Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("override"), MediaType: "image/webp"}}},
	}}
	result, err := generator.Generate(images.WithModel(t.Context(), override), "draw", []images.Input{
		ai.BinaryContent{Data: []byte("reference"), MediaType: "image/png"},
	}, images.Settings{AspectRatio: images.AspectRatio3To2})
	if err != nil {
		t.Fatal(err)
	}
	if generator.Model() != base || override.prompt != "draw" || override.settings.AspectRatio != images.AspectRatio3To2 {
		t.Fatalf("unexpected request: %#v %#v", override.prompt, override.settings)
	}
	override.inputs[0].(ai.BinaryContent).Data[0] = 'X'
	if result.Image().MediaType != "image/webp" {
		t.Fatalf("unexpected result: %#v", result)
	}
	result.Images[0].Content.Data[0] = 'X'
	if string(override.result.Images[0].Content.Data) != "override" {
		t.Fatal("generator result was not detached")
	}
}

func TestGeneratorWithoutModelDefaults(t *testing.T) {
	plain := plainModel{result: &images.Result{Images: []images.GeneratedImage{{
		Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
	}}}}
	result, err := images.New(plain).Generate(t.Context(), "draw", nil)
	if err != nil || result.ModelName != "" {
		t.Fatalf("unexpected result: %#v %v", result, err)
	}
}

func TestImageTransportErrors(t *testing.T) {
	base := &model{name: "image", provider: "provider"}
	if images.NewModelTransportError(t.Context(), base, "request", nil) != nil {
		t.Fatal("nil transport error was changed")
	}
	cause := errors.New("connection")
	err := images.NewModelTransportError(t.Context(), base, "read response", cause)
	var transport *ai.ModelTransportError
	if !errors.As(err, &transport) || !errors.Is(err, cause) || transport.ModelName != "image" ||
		transport.ProviderName != "provider" || transport.Operation != "read response" {
		t.Fatalf("unexpected transport error: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err = images.NewModelTransportError(ctx, base, "request", context.Canceled)
	if !errors.Is(err, context.Canceled) || errors.As(err, &transport) {
		t.Fatalf("cancellation became a model API error: %v", err)
	}
}

func TestGeneratorValidationAndModelErrors(t *testing.T) {
	requestErr := errors.New("failed")
	base := &model{name: "test", provider: "test", err: requestErr}
	generator := images.New(base)
	if _, err := generator.Generate(t.Context(), "draw", nil); !errors.Is(err, requestErr) {
		t.Fatalf("unexpected model error: %v", err)
	}
	for _, test := range []struct {
		prompt   string
		inputs   []images.Input
		settings []images.Settings
		contains string
	}{
		{prompt: " ", contains: "prompt must not be empty"},
		{prompt: "draw", inputs: []images.Input{ai.BinaryContent{MediaType: "text/plain"}}, contains: "image media type"},
		{prompt: "draw", inputs: []images.Input{ai.UploadedFile{MediaType: "application/pdf"}}, contains: "image media type"},
		{prompt: "draw", settings: []images.Settings{{}, {}}, contains: "at most one"},
		{prompt: "draw", settings: []images.Settings{{Dimensions: &images.Dimensions{Width: 1}, AspectRatio: images.AspectRatio1To1}}, contains: "mutually exclusive"},
		{prompt: "draw", settings: []images.Settings{{Dimensions: &images.Dimensions{Width: -1, Height: 1}}}, contains: "positive"},
	} {
		_, err := generator.Generate(t.Context(), test.prompt, test.inputs, test.settings...)
		if err == nil || !strings.Contains(err.Error(), test.contains) {
			t.Fatalf("expected %q, got %v", test.contains, err)
		}
	}
	base.err = nil
	if _, err := generator.Generate(t.Context(), "draw", nil); err == nil || !strings.Contains(err.Error(), "nil result") {
		t.Fatalf("unexpected nil error: %v", err)
	}
	base.result = &images.Result{}
	if _, err := generator.Generate(t.Context(), "draw", nil); err == nil || !strings.Contains(err.Error(), "no generated images") {
		t.Fatalf("unexpected empty error: %v", err)
	}
}

func TestSettingsMediaResultAndWrapper(t *testing.T) {
	dimensions := images.Dimensions{Width: 10, Height: 20}
	cycle := map[string]any{}
	cycle["self"] = cycle
	cyclicSlice := []any{nil}
	cyclicSlice[0] = cyclicSlice
	settings := images.Settings{
		Dimensions: &dimensions, ExtraHeaders: map[string]string{"x": "one"},
		ExtraBody: map[string]any{
			"nested": map[string]any{"value": "one"}, "slice": []any{[]string{"two"}},
			"array": [1][]string{{"array"}}, "nil_map": map[string]any(nil), "nil_slice": []string(nil),
			"nil": nil, "nil_interface": []any{nil}, "cycle": cycle, "slice_cycle": cyclicSlice, "scalar": "value",
		},
		ProviderSettings: map[string]any{"value": []string{"one"}},
	}
	cloned := settings.Clone()
	settings.Dimensions.Width = 99
	settings.ExtraBody["nested"].(map[string]any)["value"] = "changed"
	if cloned.Dimensions.Width != 10 || cloned.ExtraBody["nested"].(map[string]any)["value"] != "one" ||
		cloned.ExtraBody["slice"].([]any)[0].([]string)[0] != "two" ||
		cloned.ExtraBody["array"].([1][]string)[0][0] != "array" {
		t.Fatalf("settings not detached: %#v", cloned)
	}
	clonedCycle := cloned.ExtraBody["cycle"].(map[string]any)
	if reflect.ValueOf(clonedCycle).Pointer() != reflect.ValueOf(clonedCycle["self"]).Pointer() {
		t.Fatal("map cycle was not retained")
	}
	clonedSlice := cloned.ExtraBody["slice_cycle"].([]any)
	if reflect.ValueOf(clonedSlice).Pointer() != reflect.ValueOf(clonedSlice[0]).Pointer() {
		t.Fatal("slice cycle was not retained")
	}
	merged := images.MergeSettings(cloned, images.Settings{
		AspectRatio: images.AspectRatio1To1, ExtraHeaders: map[string]string{},
		ProviderSettings: map[string]any{"other": true},
	})
	if merged.AspectRatio != images.AspectRatio1To1 || len(merged.ExtraHeaders) != 0 || merged.ProviderSettings["other"] != true {
		t.Fatalf("unexpected merge: %#v", merged)
	}
	if got := images.MediaTypeFromBytes([]byte("\x89PNGrest")); got != "image/png" {
		t.Fatal(got)
	}
	if got := images.MediaTypeFromBytes([]byte("\xff\xd8\xffrest")); got != "image/jpeg" {
		t.Fatal(got)
	}
	if got := images.MediaTypeFromBytes([]byte("RIFFxxxxWEBPrest")); got != "image/webp" {
		t.Fatal(got)
	}
	if images.MediaTypeFromBytes([]byte("no")) != "" || images.OutputFormat("text/plain") != "" ||
		images.OutputFormat("image/webp") != "webp" {
		t.Fatal("unexpected media detection")
	}
	if err := (images.Settings{}).Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (images.Settings{Dimensions: &images.Dimensions{Width: 1, Height: 1}}).Validate(); err != nil {
		t.Fatal(err)
	}
	if merged := images.MergeSettings(images.Settings{}, images.Settings{}); merged.ExtraBody != nil {
		t.Fatal("empty merge changed nil maps")
	}
	if merged := images.MergeSettings(images.Settings{}, images.Settings{
		ExtraBody: map[string]any{}, ProviderSettings: map[string]any{"value": true},
	}); merged.ProviderSettings["value"] != true {
		t.Fatal("provider settings were not initialized")
	}
	if _, _, _, err := images.PrepareRequest("draw", []images.Input{&ai.ImageURL{}}, images.Settings{}); err == nil {
		t.Fatal("pointer input was accepted")
	}
	_, copiedInputs, _, err := images.PrepareRequest("draw", []images.Input{
		ai.ImageURL{URL: "https://example.com/a.png", VendorMetadata: map[string]any{"x": []string{"one"}}},
		ai.UploadedFile{FileID: "file", ProviderName: "test", MediaType: "image/png", VendorMetadata: map[string]any{"x": "one"}},
	}, images.Settings{})
	if err != nil || len(copiedInputs) != 2 {
		t.Fatalf("unexpected copied inputs: %#v %v", copiedInputs, err)
	}

	fake := fakes.NewModel(fakes.WithName("custom"), fakes.WithProviderName("provider"))
	wrapped := images.WrapModel(fake)
	if wrapped.Name() != "custom" || wrapped.ProviderName() != "provider" || wrapped.UnwrapModel() != fake {
		t.Fatalf("wrapper did not delegate: %#v", wrapped)
	}
	result, err := fake.Generate(t.Context(), "two words", nil, images.Settings{})
	if err != nil || result.Usage.InputTokens != 2 || result.ProviderResponseID == "" {
		t.Fatalf("unexpected fake result: %#v %v", result, err)
	}
	first := result.Image()
	first.Data[0] = 0
	if reflect.DeepEqual(first.Data, result.Images[0].Content.Data) {
		t.Fatal("Image exposed result bytes")
	}
	emptyResult := (images.Result{}).Clone()
	if emptyResult.Image().MediaType != "" || emptyResult.Images != nil {
		t.Fatal("empty result returned an image")
	}

	priced := images.Result{
		ModelName: "gpt-image-1", ProviderName: "openai", Timestamp: time.Now(),
		Usage: ai.Usage{InputTokens: 1, OutputTokens: 1},
	}
	if _, err := priced.Price(); err != nil {
		t.Fatalf("price failed: %v", err)
	}
}

func TestGenerationTool(t *testing.T) {
	direct := &model{name: "image", provider: "test", result: &images.Result{
		Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("generated"), MediaType: "image/png"}}},
	}}
	generator := images.New(direct)
	tool := images.NewGenerationTool[struct{}](generator, images.ToolConfig{Description: "Draw it."})
	calls := 0
	outer := &noNativeModel{modelfakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			if len(params.Tools) != 1 || params.Tools[0].Name != "generate_image" || params.Tools[0].Description != "Draw it." {
				t.Fatalf("unexpected tool: %#v", params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "generate_image", ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		returned := request.Parts[0].(ai.ToolReturnPart).Content.(ai.BinaryContent)
		if string(returned.Data) != "generated" {
			t.Fatalf("unexpected generated image: %#v", returned)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	capability := ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
		Native: ai.ImageGenerationTool{}, Local: ai.NewFunctionToolset(tool),
	})
	agent := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(capability))
	result, err := agent.Run(t.Context(), "draw", struct{}{})
	if err != nil || result.Output != "done" || calls != 2 {
		t.Fatalf("unexpected agent result: %#v calls=%d err=%v", result, calls, err)
	}
	if capturePanic(func() { images.NewGenerationTool[struct{}](nil, images.ToolConfig{}) }) == nil {
		t.Fatal("nil generator did not panic")
	}
}

func TestImageGenerationCapabilityDirectFallback(t *testing.T) {
	direct := &model{name: "image", provider: "test", result: &images.Result{
		Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("generated"), MediaType: "image/png"}}},
	}}
	generator := images.New(direct)
	resolverCalls := 0
	outerCalls := 0
	outer := &noNativeModel{modelfakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerCalls++
		if outerCalls == 1 {
			if len(params.NativeTools) != 0 || len(params.Tools) != 1 || params.Tools[0].NativeFallbackFor != "image_generation" {
				t.Fatalf("unexpected direct fallback tools: %#v %#v", params.NativeTools, params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "generate_image", ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		returned := request.Parts[0].(ai.ToolReturnPart).Content.(ai.BinaryContent)
		if string(returned.Data) != "generated" {
			t.Fatalf("unexpected generated image: %#v", returned)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	capability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		ResolveNative: func(context.Context, *ai.RunContext[struct{}]) (ai.ImageGenerationTool, error) {
			resolverCalls++
			return ai.ImageGenerationTool{AspectRatio: ai.ImageAspectRatio3x2}, nil
		},
		Generator: generator,
		Settings:  images.Settings{AspectRatio: images.AspectRatio16To9},
	})
	result, err := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(capability)).Run(t.Context(), "draw", struct{}{})
	if err != nil || result.Output != "done" || resolverCalls != 2 || direct.settings.AspectRatio != images.AspectRatio16To9 {
		t.Fatalf("unexpected direct fallback: result=%#v resolves=%d settings=%#v err=%v",
			result, resolverCalls, direct.settings, err)
	}

	nativeModel := &selectiveImageModel{supported: true}
	nativeCapability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		Native: ai.ImageGenerationTool{}, Generator: generator,
		Settings: images.Settings{AspectRatio: images.AspectRatio16To9},
	})
	if _, err := ai.NewAgent[struct{}, string](nativeModel, ai.WithCapabilities(nativeCapability)).Run(
		t.Context(), "draw", struct{}{},
	); err != nil {
		t.Fatal(err)
	}
	if len(nativeModel.params.NativeTools) != 1 || len(nativeModel.params.Tools) != 0 ||
		nativeModel.params.NativeTools[0].(ai.ImageGenerationTool).AspectRatio != ai.ImageAspectRatio16x9 {
		t.Fatalf("portable aspect ratio did not reach native path: %#v", nativeModel.params)
	}
}

type selectiveImageModel struct {
	supported bool
	params    ai.ModelRequestParams
}

func (model *selectiveImageModel) Request(
	_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.params = params
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func (*selectiveImageModel) Name() string                                { return "selective" }
func (model *selectiveImageModel) SupportsNativeTool(ai.NativeTool) bool { return model.supported }

func TestRepeatedDirectImageCapabilitiesMergeSettingsAndFallback(t *testing.T) {
	for _, settingsFirst := range []bool{false, true} {
		name := "generator first"
		if settingsFirst {
			name = "settings first"
		}
		t.Run(name, func(t *testing.T) {
			direct := &model{name: "image", provider: "test", result: &images.Result{
				Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("generated"), MediaType: "image/png"}}},
			}}
			dimensions := images.Dimensions{Width: 1280, Height: 720}
			compression := 80
			nativeSettings := ai.ImageGenerationTool{
				Action:            ai.ImageGenerationActionGenerate,
				Background:        ai.ImageGenerationBackgroundOpaque,
				InputFidelity:     ai.ImageGenerationInputFidelityHigh,
				Moderation:        ai.ImageGenerationModerationLow,
				Model:             "image-model",
				OutputCompression: &compression,
				OutputFormat:      ai.ImageGenerationOutputPNG,
				PartialImages:     1,
				Quality:           ai.ImageGenerationQualityHigh,
				AspectRatio:       ai.ImageAspectRatio3x2,
				Optional:          true,
			}
			settings := images.Settings{
				Dimensions:       &dimensions,
				ExtraHeaders:     map[string]string{"X-Probe": "present"},
				ExtraBody:        map[string]any{"seed": 42},
				ProviderSettings: map[string]any{"probe": map[string]any{"enabled": true}},
			}
			generatorCapability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Native: nativeSettings, Generator: images.New(direct),
			})
			settingsCapability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Native: ai.ImageGenerationTool{Size: ai.ImageGenerationSize2K}, Settings: settings,
			})
			capabilities := []ai.Capability{generatorCapability, settingsCapability}
			if settingsFirst {
				capabilities[0], capabilities[1] = capabilities[1], capabilities[0]
			}

			nativeModel := &selectiveImageModel{supported: true}
			if _, err := ai.NewAgent[struct{}, string](
				nativeModel, ai.WithCapabilities(capabilities...),
			).Run(t.Context(), "draw", struct{}{}); err != nil {
				t.Fatal(err)
			}
			native := nativeModel.params.NativeTools[0].(ai.ImageGenerationTool)
			if native.Action != nativeSettings.Action || native.Background != nativeSettings.Background ||
				native.InputFidelity != nativeSettings.InputFidelity || native.Moderation != nativeSettings.Moderation ||
				native.Model != nativeSettings.Model || native.OutputCompression == nil || *native.OutputCompression != 80 ||
				native.OutputFormat != nativeSettings.OutputFormat || native.PartialImages != nativeSettings.PartialImages ||
				native.Quality != nativeSettings.Quality || native.Size != ai.ImageGenerationSize2K ||
				native.AspectRatio != nativeSettings.AspectRatio || !native.Optional {
				t.Fatalf("static native image settings were not merged: %#v", native)
			}

			requests := 0
			outer := &noNativeModel{modelfakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				requests++
				if requests == 1 {
					if len(params.NativeTools) != 0 || len(params.Tools) != 1 ||
						params.Tools[0].NativeFallbackFor != "image_generation" {
						t.Fatalf("inherited direct fallback was not selected: %#v", params)
					}
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: params.Tools[0].Name, ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
					}}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})}
			result, err := ai.NewAgent[struct{}, string](
				outer, ai.WithCapabilities(capabilities...),
			).Run(t.Context(), "draw", struct{}{})
			if err != nil || result.Output != "done" || requests != 2 || !reflect.DeepEqual(direct.settings, settings) {
				t.Fatalf("inherited direct fallback failed: result=%#v requests=%d settings=%#v err=%v",
					result, requests, direct.settings, err)
			}
		})
	}
}

func TestRepeatedDirectImageCapabilitySettingsOverrideByRegistrationOrder(t *testing.T) {
	firstDimensions := images.Dimensions{Width: 512, Height: 512}
	secondDimensions := images.Dimensions{Width: 1024, Height: 768}
	direct := &model{name: "image", provider: "test", result: &images.Result{
		Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("generated"), MediaType: "image/png"}}},
	}}
	first := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		Generator: images.New(direct),
		Settings: images.Settings{
			Dimensions:       &firstDimensions,
			ExtraHeaders:     map[string]string{"X-Order": "first"},
			ExtraBody:        map[string]any{"order": "first"},
			ProviderSettings: map[string]any{"retained": true, "order": "first"},
		},
	})
	second := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		Settings: images.Settings{
			Dimensions:       &secondDimensions,
			ExtraHeaders:     map[string]string{"X-Order": "second"},
			ExtraBody:        map[string]any{"order": "second"},
			ProviderSettings: map[string]any{"order": "second"},
		},
	})
	requests := 0
	outer := &noNativeModel{modelfakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: params.Tools[0].Name, ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	if _, err := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(first, second)).Run(
		t.Context(), "draw", struct{}{},
	); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(direct.settings.Dimensions, &secondDimensions) ||
		direct.settings.ExtraHeaders["X-Order"] != "second" || direct.settings.ExtraBody["order"] != "second" ||
		direct.settings.ProviderSettings["order"] != "second" || direct.settings.ProviderSettings["retained"] != true {
		t.Fatalf("later image settings did not override earlier settings: %#v", direct.settings)
	}
}

func TestRepeatedDirectImageCapabilityResolvesMergedNativeOncePerRequest(t *testing.T) {
	direct := &model{name: "image", provider: "test", result: &images.Result{
		Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("generated"), MediaType: "image/png"}}},
	}}
	resolveCalls := 0
	dynamic := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		ResolveNative: func(context.Context, *ai.RunContext[struct{}]) (ai.ImageGenerationTool, error) {
			resolveCalls++
			return ai.ImageGenerationTool{
				Quality: ai.ImageGenerationQualityHigh, AspectRatio: ai.ImageAspectRatio3x2,
			}, nil
		},
	})
	fallback := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		Native: ai.ImageGenerationTool{Size: ai.ImageGenerationSize2K}, Generator: images.New(direct),
	})
	nativeModel := &selectiveImageModel{supported: true}
	if _, err := ai.NewAgent[struct{}, string](nativeModel, ai.WithCapabilities(dynamic, fallback)).Run(
		t.Context(), "draw", struct{}{},
	); err != nil {
		t.Fatal(err)
	}
	native := nativeModel.params.NativeTools[0].(ai.ImageGenerationTool)
	if resolveCalls != 1 || native.Quality != ai.ImageGenerationQualityHigh ||
		native.Size != ai.ImageGenerationSize2K || native.AspectRatio != ai.ImageAspectRatio3x2 {
		t.Fatalf("dynamic native settings were not merged once: calls=%d native=%#v", resolveCalls, native)
	}

	requests := 0
	outer := &noNativeModel{modelfakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: params.Tools[0].Name, ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	if _, err := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(fallback, dynamic)).Run(
		t.Context(), "draw", struct{}{},
	); err != nil {
		t.Fatal(err)
	}
	if resolveCalls != 3 || direct.settings.AspectRatio != images.AspectRatio3To2 {
		t.Fatalf("dynamic fallback did not reuse request settings: calls=%d settings=%#v", resolveCalls, direct.settings)
	}
}

func TestRepeatedDirectImageCapabilityInheritsLocalFallback(t *testing.T) {
	local := ai.NewFunctionToolset(ai.NewSimpleTool[struct{}](
		"local_image", func(context.Context, struct{}) (string, error) { return "image", nil },
	))
	for _, localFirst := range []bool{false, true} {
		localCapability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{Local: local})
		settingsCapability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
			Native:   ai.ImageGenerationTool{Optional: true},
			Settings: images.Settings{ExtraHeaders: map[string]string{"X-Probe": "present"}},
		})
		capabilities := []ai.Capability{settingsCapability, localCapability}
		if localFirst {
			capabilities[0], capabilities[1] = capabilities[1], capabilities[0]
		}
		outer := &noNativeModel{modelfakes.NewFunctionModel(func(
			_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			if len(params.NativeTools) != 0 || len(params.Tools) != 1 || params.Tools[0].Name != "local_image" {
				t.Fatalf("local fallback was not inherited: %#v", params)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		})}
		if _, err := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(capabilities...)).Run(
			t.Context(), "draw", struct{}{},
		); err != nil {
			t.Fatal(err)
		}
	}
}

func TestDirectImageCapabilityCombinerValidationAndNativeRequirement(t *testing.T) {
	capability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{})
	if capability.CapabilityID() != "image_generation" {
		t.Fatalf("unexpected capability ID %q", capability.CapabilityID())
	}
	if _, err := capability.CombineCapabilities(nil); err == nil {
		t.Fatal("empty image capability collection was accepted")
	}
	if _, err := capability.CombineCapabilities([]ai.Capability{
		ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{}),
	}); err == nil {
		t.Fatal("incompatible image capability was accepted")
	}
	var nilCapability *images.ImageGenerationCapability[struct{}]
	if err := nilCapability.Setup(&ai.CapabilityRegistry{}); err == nil {
		t.Fatal("nil image capability was accepted")
	}
	panicValue := capturePanic(func() {
		ai.NewAgent[struct{}, string](&selectiveImageModel{supported: true}, ai.WithCapabilities(
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Native: ai.ImageGenerationTool{Optional: true},
			}),
		))
	})
	if !strings.Contains(fmt.Sprint(panicValue), "requires native support") {
		t.Fatalf("unexpected native-required validation: %q", panicValue)
	}
}

func TestRepeatedDirectImageCapabilityRejectsMergedSettingsConflict(t *testing.T) {
	for _, capabilities := range [][]ai.Capability{
		{
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Settings: images.Settings{Dimensions: &images.Dimensions{Width: 1, Height: 1}},
			}),
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Settings: images.Settings{AspectRatio: images.AspectRatio1To1},
			}),
		},
		{
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Settings: images.Settings{AspectRatio: images.AspectRatio1To1},
			}),
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Settings: images.Settings{Dimensions: &images.Dimensions{Width: 1, Height: 1}},
			}),
		},
	} {
		panicValue := capturePanic(func() {
			ai.NewAgent[struct{}, string](&selectiveImageModel{supported: true}, ai.WithCapabilities(capabilities...))
		})
		if !strings.Contains(fmt.Sprint(panicValue), "dimensions and aspect ratio are mutually exclusive") {
			t.Fatalf("unexpected merged settings conflict: %q", panicValue)
		}
	}
}

func TestImageGenerationCapabilityRejectsUnsupportedEdit(t *testing.T) {
	direct := &model{name: "image", provider: "test", result: &images.Result{
		Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("generated"), MediaType: "image/png"}}},
	}}
	outer := &noNativeModel{modelfakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "generate_image", ToolCallID: "image", Args: []byte(`{"prompt":"edit it"}`),
		}}}, nil
	})}
	capability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		Native: ai.ImageGenerationTool{Action: ai.ImageGenerationActionEdit}, FallbackModel: direct,
	})
	_, err := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(capability)).Run(t.Context(), "edit", struct{}{})
	if err == nil || !strings.Contains(err.Error(), "cannot edit without reference images") || direct.prompt != "" {
		t.Fatalf("unexpected edit fallback result: prompt=%q err=%v", direct.prompt, err)
	}
}

func TestImageGenerationCapabilityResolverError(t *testing.T) {
	resolveErr := errors.New("resolve native")
	capability := images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
		ResolveNative: func(context.Context, *ai.RunContext[struct{}]) (ai.ImageGenerationTool, error) {
			return ai.ImageGenerationTool{}, resolveErr
		},
		Generator: images.New(&model{name: "image", provider: "test"}),
	})
	_, err := ai.NewAgent[struct{}, string](&selectiveImageModel{}, ai.WithCapabilities(capability)).Run(
		t.Context(), "draw", struct{}{},
	)
	if !errors.Is(err, resolveErr) {
		t.Fatalf("unexpected resolver error: %v", err)
	}
}

func TestImageGenerationCapabilityValidation(t *testing.T) {
	generator := images.New(&model{name: "image", provider: "test"})
	for _, function := range []func(){
		func() {
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Local: ai.NewFunctionToolset[struct{}](), Generator: generator,
			})
		},
		func() {
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Generator: generator, FallbackModel: &model{name: "other", provider: "test"},
			})
		},
		func() {
			images.NewImageGenerationCapability(images.CapabilityConfig[struct{}]{
				Generator: generator, Settings: images.Settings{
					Dimensions: &images.Dimensions{Width: 1, Height: 1}, AspectRatio: images.AspectRatio1To1,
				},
			})
		},
	} {
		if capturePanic(function) == nil {
			t.Fatal("invalid direct fallback configuration did not panic")
		}
	}
}

func TestGenerationToolFailures(t *testing.T) {
	filtered := &model{name: "image", provider: "test", err: &ai.ContentFilterError{Message: "blocked"}}
	tool := images.NewGenerationTool[struct{}](images.New(filtered), images.ToolConfig{})
	calls := 0
	outer := modelfakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			if params.Tools[0].Name != "generate_image" {
				t.Fatalf("unexpected default tool: %#v", params.Tools[0])
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "generate_image", ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		if _, ok := request.Parts[0].(ai.RetryPromptPart); !ok {
			t.Fatalf("expected retry prompt: %#v", request.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](outer)
	agent.AddTool(tool)
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); err != nil {
		t.Fatal(err)
	}

	multiple := &model{name: "image", provider: "test", result: &images.Result{Images: []images.GeneratedImage{
		{Content: ai.BinaryContent{Data: []byte("one"), MediaType: "image/png"}},
		{Content: ai.BinaryContent{Data: []byte("two"), MediaType: "image/png"}},
	}}}
	tool = images.NewGenerationTool[struct{}](images.New(multiple), images.ToolConfig{Name: "draw_image"})
	outer = modelfakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "draw_image", ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
		}}}, nil
	})
	agent = ai.NewAgent[struct{}, string](outer)
	agent.AddTool(tool)
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); err == nil || !strings.Contains(err.Error(), "expected exactly one") {
		t.Fatalf("unexpected multiple-image error: %v", err)
	}

	directFailure := errors.New("direct failure")
	failed := &model{name: "image", provider: "test", err: directFailure}
	tool = images.NewGenerationTool[struct{}](images.New(failed), images.ToolConfig{})
	outer = modelfakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "generate_image", ToolCallID: "image", Args: []byte(`{"prompt":"A gopher"}`),
		}}}, nil
	})
	agent = ai.NewAgent[struct{}, string](outer)
	agent.AddTool(tool)
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); !errors.Is(err, directFailure) {
		t.Fatalf("unexpected direct error: %v", err)
	}
}

func TestNilModels(t *testing.T) {
	var typedNil *model
	if images.New(valueModel{}).Model().Name() != "value" {
		t.Fatal("value model was rejected")
	}
	for _, function := range []func(){
		func() { images.New(nil) }, func() { images.New(typedNil) },
		func() { images.WrapModel(nil) }, func() { images.WithModel(context.Background(), nil) },
	} {
		if panicValue := capturePanic(function); panicValue == nil {
			t.Fatal("nil model did not panic")
		}
	}
}

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
