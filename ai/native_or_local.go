package ai

import (
	"fmt"
	"reflect"
)

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
	rebuildLocal   func(NativeTool) Toolset[Deps]
	requiredReason string
}

// NewNativeOrLocalTool pairs a static native tool with one local function tool.
func NewNativeOrLocalTool[Deps any](
	native NativeTool, local Tool[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	return NewNativeOrLocalToolset(native, NewFunctionToolset(local), options...)
}

// NewNativeOrLocalToolset pairs a static native tool with a local toolset.
// Toolset lifecycle and instructions are preserved. Local may be nil only when
// WithNativeRequired suppresses the fallback.
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
// safe for concurrent calls. Local may be nil only when WithNativeRequired
// suppresses the fallback.
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

// CapabilityID returns the native tool identity shared by repeated declarations.
func (tool *NativeOrLocalTool[Deps]) CapabilityID() string {
	if tool == nil {
		return ""
	}
	if tool.registration.nativeID != "" {
		return tool.registration.nativeID
	}
	if !nativeToolIsNil(tool.registration.native) {
		return tool.registration.native.UniqueID()
	}
	return ""
}

// CombineCapabilities merges repeated native configuration and keeps one local fallback.
func (tool *NativeOrLocalTool[Deps]) CombineCapabilities(capabilities []Capability) (Capability, error) {
	if len(capabilities) == 0 {
		return nil, fmt.Errorf("ai: cannot combine an empty native-or-local capability collection")
	}
	values := make([]*NativeOrLocalTool[Deps], len(capabilities))
	for index, capability := range capabilities {
		value, ok := capability.(*NativeOrLocalTool[Deps])
		if !ok {
			return nil, fmt.Errorf("ai: native-or-local capability has incompatible type %T", capability)
		}
		values[index] = value
	}
	registration := values[len(values)-1].registration
	allStatic := true
	var nativeValues []NativeTool
	for _, value := range values {
		if value.registration.resolve != nil {
			allStatic = false
			break
		}
		nativeValues = append(nativeValues, value.registration.native)
	}
	if allStatic {
		merged, err := mergeNativeToolValues(nativeValues)
		if err != nil {
			return nil, err
		}
		registration.native = merged
		registration.resolve = nil
		registration.nativeID = merged.UniqueID()
	}
	if toolsetIsNil(registration.local) {
		for index := len(values) - 2; index >= 0; index-- {
			if !toolsetIsNil(values[index].registration.local) {
				registration.local = values[index].registration.local
				break
			}
		}
	}
	if registration.requiredReason == "" {
		for index := len(values) - 2; index >= 0; index-- {
			if values[index].registration.requiredReason != "" {
				registration.requiredReason = values[index].registration.requiredReason
				break
			}
		}
	}
	if registration.rebuildLocal == nil {
		for index := len(values) - 2; index >= 0; index-- {
			if values[index].registration.rebuildLocal != nil {
				registration.rebuildLocal = values[index].registration.rebuildLocal
				break
			}
		}
	}
	if registration.rebuildLocal != nil && allStatic {
		registration.local = registration.rebuildLocal(registration.native)
	}
	return &NativeOrLocalTool[Deps]{registration: registration}, nil
}

func nativeToolIsNil(tool NativeTool) bool {
	if tool == nil {
		return true
	}
	value := reflect.ValueOf(tool)
	return value.Kind() == reflect.Pointer && value.IsNil()
}

func mergeNativeToolValues(tools []NativeTool) (NativeTool, error) {
	typ := reflect.TypeOf(tools[0])
	structType := typ
	pointer := typ.Kind() == reflect.Pointer
	if pointer {
		structType = typ.Elem()
	}
	if structType.Kind() != reflect.Struct {
		return cloneNativeTool(tools[len(tools)-1]), nil
	}
	for fieldIndex := 0; fieldIndex < structType.NumField(); fieldIndex++ {
		if structType.Field(fieldIndex).PkgPath != "" {
			return nil, fmt.Errorf(
				"ai: cannot merge native definition %v with unexported field %q",
				typ, structType.Field(fieldIndex).Name,
			)
		}
	}
	values := make([]reflect.Value, len(tools))
	for index, tool := range tools {
		if nativeToolIsNil(tool) || reflect.TypeOf(tool) != typ {
			return nil, fmt.Errorf("ai: native definitions with one capability ID must have the same type")
		}
		values[index] = reflect.ValueOf(tool)
		if pointer {
			values[index] = values[index].Elem()
		}
	}
	merged := reflect.New(structType).Elem()
	merged.Set(values[len(values)-1])
	for fieldIndex := 0; fieldIndex < structType.NumField(); fieldIndex++ {
		stated := make([]reflect.Value, 0, len(values))
		for _, value := range values {
			field := value.Field(fieldIndex)
			if !field.IsZero() {
				stated = append(stated, field)
			}
		}
		if len(stated) == 0 {
			continue
		}
		value, err := mergeCapabilityField(stated)
		if err != nil {
			return nil, fmt.Errorf("ai: merge native tool field %q: %w", structType.Field(fieldIndex).Name, err)
		}
		merged.Field(fieldIndex).Set(value)
	}
	if pointer {
		result := reflect.New(structType)
		result.Elem().Set(merged)
		return cloneNativeTool(result.Interface().(NativeTool)), nil
	}
	return cloneNativeTool(merged.Interface().(NativeTool)), nil
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
