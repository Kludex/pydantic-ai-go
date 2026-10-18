package ai

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// ResolveNativeToolPreferences selects provider-native tools or their local function fallbacks.
// Models without NativeToolSupportModel retain the request unchanged because support is unknown.
// A web search or fetch tool with a restricting setting the model reports unsupported, such as
// AllowedDomains or MaxUses, uses its local fallback, is left out when optional, or fails the
// request, so the provider never receives it without the restriction.
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
	if !known || !hasNativeToolPreferences(params) && !hasUnappliedRestriction(support, params.NativeTools) {
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
	unappliedSettings := make([]string, 0)
	nativeTools := make([]NativeTool, 0, len(params.NativeTools))
	for _, nativeTool := range params.NativeTools {
		if support.SupportsNativeTool(cloneNativeTool(nativeTool)) {
			supportedIDs[nativeTool.UniqueID()] = struct{}{}
			nativeTools = append(nativeTools, nativeTool)
			continue
		}
		_, hasFallback := fallbackIDs[nativeTool.UniqueID()]
		if nativeTool.IsOptional() || hasFallback {
			continue
		}
		if restrictions := unappliedRestrictions(support, nativeTool); restrictions != "" {
			unappliedSettings = append(unappliedSettings, fmt.Sprintf("%q (%s)", nativeTool.Kind(), restrictions))
		} else {
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
	if len(unappliedSettings) > 0 {
		return ModelRequestParams{}, fmt.Errorf(
			"ai: model %q cannot apply native tool setting(s) %s; remove them, use a model that supports them, "+
				"or add a local fallback that applies them",
			modelName(model), strings.Join(unappliedSettings, ", "),
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
	if params.nativeToolSupportRequired {
		return true
	}
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

func hasUnappliedRestriction(support NativeToolSupportModel, tools []NativeTool) bool {
	for _, tool := range tools {
		if !support.SupportsNativeTool(cloneNativeTool(tool)) && unappliedRestrictions(support, tool) != "" {
			return true
		}
	}
	return false
}

// unappliedRestrictions names the restricting settings of a tool the model supports only without them.
func unappliedRestrictions(support NativeToolSupportModel, tool NativeTool) string {
	var restrictions string
	var unrestricted NativeTool
	switch tool := cloneNativeTool(tool).(type) {
	case WebSearchTool:
		restrictions = webSearchNativeRequirement(tool)
		tool.AllowedDomains, tool.BlockedDomains, tool.MaxUses, tool.ExternalWebAccess = nil, nil, 0, nil
		unrestricted = tool
	case WebFetchTool:
		restrictions = webSearchNativeRequirement(WebSearchTool{
			AllowedDomains: tool.AllowedDomains, BlockedDomains: tool.BlockedDomains, MaxUses: tool.MaxUses,
		})
		tool.AllowedDomains, tool.BlockedDomains, tool.MaxUses = nil, nil, 0
		unrestricted = tool
	default:
		return ""
	}
	if restrictions == "" || !support.SupportsNativeTool(unrestricted) {
		return ""
	}
	return restrictions
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
