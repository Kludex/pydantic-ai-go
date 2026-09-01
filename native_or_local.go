package ai

import "fmt"

// NativeOrLocalOption configures a NativeOrLocalTool.
type NativeOrLocalOption func(*nativeOrLocalConfig)

type nativeOrLocalConfig struct {
	requiredReason string
}

// WithNativeRequired suppresses the local fallback because the named constraint
// cannot be honored locally. The native tool must not be optional.
func WithNativeRequired(reason string) NativeOrLocalOption {
	if reason == "" {
		panic("ai: native-required reason must not be empty")
	}
	return func(config *nativeOrLocalConfig) { config.requiredReason = reason }
}

// NativeOrLocalTool atomically pairs one provider-native tool with a local
// function-tool fallback. Register it with WithCapabilities or
// WithRunCapabilities, or pass it to Agent.AddNativeOrLocal.
type NativeOrLocalTool[Deps any] struct {
	registration nativeOrLocalRegistration[Deps]
}

type nativeOrLocalRegistration[Deps any] struct {
	native         NativeTool
	resolve        NativeToolFunc[Deps]
	nativeID       string
	local          Toolset[Deps]
	requiredReason string
}

// NewNativeOrLocalTool pairs a static native tool with one local function tool.
func NewNativeOrLocalTool[Deps any](
	native NativeTool, local Tool[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	return NewNativeOrLocalToolset(native, NewFunctionToolset(local), options...)
}

// NewNativeOrLocalToolset pairs a static native tool with a local toolset.
// Toolset lifecycle and instructions are preserved.
func NewNativeOrLocalToolset[Deps any](
	native NativeTool, local Toolset[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	config := applyNativeOrLocalOptions(options)
	return &NativeOrLocalTool[Deps]{registration: nativeOrLocalRegistration[Deps]{
		native: cloneNativeTool(native), local: local, requiredReason: config.requiredReason,
	}}
}

// NewDynamicNativeOrLocalTool pairs a dependency-aware native tool with one
// local function tool. nativeID is the stable identity the resolver must return.
func NewDynamicNativeOrLocalTool[Deps any](
	nativeID string,
	resolve NativeToolFunc[Deps],
	local Tool[Deps],
	options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	return NewDynamicNativeOrLocalToolset(nativeID, resolve, NewFunctionToolset(local), options...)
}

// NewDynamicNativeOrLocalToolset pairs a dependency-aware native tool with a
// local toolset. The resolver runs once before each model request and must be
// safe for concurrent calls.
func NewDynamicNativeOrLocalToolset[Deps any](
	nativeID string,
	resolve NativeToolFunc[Deps],
	local Toolset[Deps],
	options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if nativeID == "" {
		panic("ai: dynamic native-or-local tool ID must not be empty")
	}
	if resolve == nil {
		panic("ai: dynamic native-or-local resolver must not be nil")
	}
	config := applyNativeOrLocalOptions(options)
	return &NativeOrLocalTool[Deps]{registration: nativeOrLocalRegistration[Deps]{
		resolve: resolve, nativeID: nativeID, local: local, requiredReason: config.requiredReason,
	}}
}

// Setup implements Capability. Registration is committed by the typed agent
// only after the native definition and dependency type have been validated.
func (tool *NativeOrLocalTool[Deps]) Setup(registry *CapabilityRegistry) error {
	if tool == nil {
		return fmt.Errorf("ai: native-or-local tool must not be nil")
	}
	registration := tool.registration
	registration.native = cloneNativeTool(registration.native)
	registry.nativeOrLocal = append(registry.nativeOrLocal, registration)
	return nil
}

func applyNativeOrLocalOptions(options []NativeOrLocalOption) nativeOrLocalConfig {
	var config nativeOrLocalConfig
	for _, option := range options {
		if option == nil {
			panic("ai: native-or-local option must not be nil")
		}
		option(&config)
	}
	return config
}
