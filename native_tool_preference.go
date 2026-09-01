package ai

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// ResolveNativeToolPreferences selects provider-native tools or their local function fallbacks.
// Models without NativeToolSupportModel retain the request unchanged because support is unknown.
// The returned parameters are detached from params.
func ResolveNativeToolPreferences(model Model, params ModelRequestParams) (ModelRequestParams, error) {
	if err := ValidateNativeTools(params.NativeTools); err != nil {
		return ModelRequestParams{}, err
	}
	preservedNativeTools := cloneNativeToolsPreservingTypes(params.NativeTools)
	params = ModelRequestContext{Params: params}.Clone().Params
	params.NativeTools = preservedNativeTools
	if params.nativeToolPreferencesResolved {
		return params, nil
	}
	support, known := model.(NativeToolSupportModel)
	if !known && !modelIsNil(model) {
		if _, wrapped := model.(ModelUnwrapper); wrapped {
			support, known = UnwrapModel(model).(NativeToolSupportModel)
		}
	}
	if !known || !hasNativeToolPreferences(params) {
		return params, nil
	}

	fallbackIDs := make(map[string]struct{})
	for _, definition := range append(cloneToolDefinitions(params.Tools), params.DeferredTools...) {
		if definition.NativeFallbackFor != "" {
			fallbackIDs[definition.NativeFallbackFor] = struct{}{}
		}
	}

	supportedIDs := make(map[string]struct{}, len(params.NativeTools))
	unsupportedKinds := make([]string, 0)
	nativeTools := make([]NativeTool, 0, len(params.NativeTools))
	for _, nativeTool := range params.NativeTools {
		if support.SupportsNativeTool(cloneNativeTool(nativeTool)) {
			supportedIDs[nativeTool.UniqueID()] = struct{}{}
			nativeTools = append(nativeTools, nativeTool)
			continue
		}
		_, hasFallback := fallbackIDs[nativeTool.UniqueID()]
		if !nativeTool.IsOptional() && !hasFallback {
			unsupportedKinds = append(unsupportedKinds, nativeTool.Kind())
		}
	}
	if len(unsupportedKinds) > 0 {
		quotedKinds := make([]string, len(unsupportedKinds))
		for index, kind := range unsupportedKinds {
			quotedKinds[index] = strconv.Quote(kind)
		}
		return ModelRequestParams{}, fmt.Errorf(
			"ai: native tool(s) %s are not supported by model %q and have no local fallback",
			strings.Join(quotedKinds, ", "), modelName(model),
		)
	}

	params.NativeTools = nativeTools
	params.Tools = resolveNativeFunctionTools(params.Tools, supportedIDs)
	params.DeferredTools = resolveNativeFunctionTools(params.DeferredTools, supportedIDs)
	params.nativeToolPreferencesResolved = true
	return params, nil
}

func resolveNativeFunctionTools(
	definitions []ToolDefinition, supportedIDs map[string]struct{},
) []ToolDefinition {
	resolved := make([]ToolDefinition, 0, len(definitions))
	for _, definition := range definitions {
		if _, supported := supportedIDs[definition.NativeFallbackFor]; definition.NativeFallbackFor != "" && supported {
			continue
		}
		if _, supported := supportedIDs[definition.NativeCompanionFor]; definition.NativeCompanionFor != "" && !supported {
			definition.NativeCompanionFor = ""
		}
		resolved = append(resolved, cloneToolDefinition(definition))
	}
	return resolved
}

func hasNativeToolPreferences(params ModelRequestParams) bool {
	for _, definition := range params.Tools {
		if definition.NativeFallbackFor != "" || definition.NativeCompanionFor != "" {
			return true
		}
	}
	for _, definition := range params.DeferredTools {
		if definition.NativeFallbackFor != "" || definition.NativeCompanionFor != "" {
			return true
		}
	}
	return false
}

func nativeToolPreferenceID(nativeTool NativeTool) string {
	if err := ValidateNativeTools([]NativeTool{nativeTool}); err != nil {
		panic(fmt.Sprintf("ai: invalid native tool preference: %v", err))
	}
	return nativeTool.UniqueID()
}

func cloneNativeToolsPreservingTypes(tools []NativeTool) []NativeTool {
	if tools == nil {
		return nil
	}
	cloned := make([]NativeTool, len(tools))
	for index, tool := range tools {
		copy := cloneNativeTool(tool)
		originalType := reflect.TypeOf(tool)
		copyValue := reflect.ValueOf(copy)
		if originalType.Kind() == reflect.Pointer && copyValue.Type() == originalType.Elem() {
			pointer := reflect.New(copyValue.Type())
			pointer.Elem().Set(copyValue)
			copy = pointer.Interface().(NativeTool)
		}
		cloned[index] = copy
	}
	return cloned
}
