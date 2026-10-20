package elevenlabs

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	jsonschema "github.com/Kludex/pydantic-ai-go/ai/internal/schema"
)

// preparedParameters inlines definitions and collapses nullable unions onto
// their non-null member, so every node has a plain type. Optionality is
// carried by required alone, as the ElevenLabs dialect stores it.
func preparedParameters(schema map[string]any) map[string]any {
	// Through JSON, so Go slices compare like the decoded remote schema.
	data, _ := json.Marshal(schema)
	var decoded map[string]any
	_ = json.Unmarshal(data, &decoded)
	return jsonschema.Transform(jsonschema.InlineDefinitions(decoded), func(node map[string]any) {
		members, ok := node["anyOf"].([]any)
		if !ok || len(members) != 2 {
			return
		}
		first, _ := members[0].(map[string]any)
		second, _ := members[1].(map[string]any)
		kept := first
		if first["type"] == "null" {
			kept = second
		} else if second["type"] != "null" {
			return
		}
		if kept == nil || kept["type"] == "null" {
			return
		}
		delete(node, "anyOf")
		for key, value := range kept {
			if _, exists := node[key]; !exists {
				node[key] = value
			}
		}
	})
}

// elevenLabsParameters projects a prepared schema onto the client-tool dialect,
// which takes only type, description, enum, items, properties, and required,
// and needs a description on every parameter. Every problem is reported at
// once, so a tool with several costs one connect, not one per parameter.
func elevenLabsParameters(toolName string, schema map[string]any) (map[string]any, error) {
	var problems []string
	result := convertParameters(schema, "", true, &problems)
	if len(problems) > 0 {
		return nil, fmt.Errorf(
			"elevenlabs: cannot sync tool %q: %s. Add parameter descriptions or simplify the parameter types, "+
				"or keep the tool configured in the ElevenLabs dashboard with tool sync off",
			toolName, strings.Join(problems, "; "),
		)
	}
	return result, nil
}

func convertParameters(schema map[string]any, path string, root bool, problems *[]string) map[string]any {
	result := map[string]any{}
	for _, key := range []string{"type", "description", "enum"} {
		if value, ok := schema[key]; ok && value != nil {
			result[key] = value
		}
	}
	if _, typed := result["type"]; !typed {
		subject := "the parameters schema"
		if path != "" {
			subject = fmt.Sprintf("parameter %q", path)
		}
		*problems = append(*problems, subject+" uses a JSON Schema feature the ElevenLabs tool dialect cannot "+
			"express (a union of types, or a recursive reference)")
	}
	if description, _ := result["description"].(string); !root && description == "" {
		*problems = append(*problems, fmt.Sprintf(
			"ElevenLabs requires a description for every tool parameter, and %q has none", path,
		))
	}
	if items, ok := schema["items"].(map[string]any); ok {
		result["items"] = convertParameters(items, path+"[]", false, problems)
	}
	if properties, ok := schema["properties"].(map[string]any); ok {
		converted := map[string]any{}
		names := make([]string, 0, len(properties))
		for name := range properties {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			if property, ok := properties[name].(map[string]any); ok {
				converted[name] = convertParameters(property, strings.TrimPrefix(path+"."+name, "."), false, problems)
			}
		}
		result["properties"] = converted
	}
	if required, ok := schema["required"]; ok && required != nil {
		result["required"] = required
	}
	return result
}

// toolConfig renders a definition as a client-tool config. An update replaces
// the whole config, so the remote tool's own response timeout is kept.
func toolConfig(tool ai.ToolDefinition, remote *agentTool) (map[string]any, error) {
	parameters, err := elevenLabsParameters(tool.Name, preparedParameters(tool.Schema))
	if err != nil {
		return nil, err
	}
	config := map[string]any{
		"type": "client", "name": tool.Name, "description": tool.Description,
		// The session settles every call and sends its result back.
		"expects_response": true, "parameters": parameters,
	}
	if remote != nil && remote.ResponseTimeoutSecs != nil {
		config["response_timeout_secs"] = *remote.ResponseTimeoutSecs
	}
	return config, nil
}

// schemasMatch compares a prepared local schema with one the server stored in
// its own dialect, which adds bookkeeping fields and drops keywords it cannot
// express, so strict equality is impossible. Compared: type, enum (a remote
// null counts as absent), the required set, property names recursively, items,
// and description only where the local schema declares one.
func schemasMatch(local, remote map[string]any) bool {
	localEnum, _ := local["enum"].([]any)
	remoteEnum, _ := remote["enum"].([]any)
	if local["type"] != remote["type"] || len(localEnum) != len(remoteEnum) ||
		len(localEnum) > 0 && !reflect.DeepEqual(localEnum, remoteEnum) {
		return false
	}
	if description, _ := local["description"].(string); description != "" && description != remote["description"] {
		return false
	}
	if !sameStrings(local["required"], remote["required"]) {
		return false
	}
	localProperties, _ := local["properties"].(map[string]any)
	remoteProperties, _ := remote["properties"].(map[string]any)
	if len(localProperties) != len(remoteProperties) {
		return false
	}
	for name, property := range localProperties {
		remoteProperty, ok := remoteProperties[name].(map[string]any)
		localProperty, _ := property.(map[string]any)
		if !ok || !schemasMatch(localProperty, remoteProperty) {
			return false
		}
	}
	localItems, localHasItems := local["items"].(map[string]any)
	remoteItems, remoteHasItems := remote["items"].(map[string]any)
	if localHasItems {
		return remoteHasItems && schemasMatch(localItems, remoteItems)
	}
	// A local array without items accepts anything; a remote one that constrains its items is a different contract.
	return !remoteHasItems
}

func sameStrings(left, right any) bool {
	collect := func(value any) []string {
		items, _ := value.([]any)
		names := make([]string, len(items))
		for index, item := range items {
			names[index] = fmt.Sprint(item)
		}
		slices.Sort(names)
		return slices.Compact(names)
	}
	return slices.Equal(collect(left), collect(right))
}

// toolMismatches describes every difference between the session's tools and
// the agent's client tools. Names must match exactly; matched tools must agree
// on description, parameters, and expecting a response.
func toolMismatches(local []ai.ToolDefinition, remote []agentTool) []string {
	remoteByName := map[string]agentTool{}
	for _, tool := range remote {
		if tool.Name != "" {
			remoteByName[tool.Name] = tool
		}
	}
	localByName := map[string]ai.ToolDefinition{}
	for _, tool := range local {
		localByName[tool.Name] = tool
	}
	var mismatches []string
	for _, name := range sortedKeys(localByName) {
		if _, ok := remoteByName[name]; !ok {
			mismatches = append(mismatches, fmt.Sprintf("tool %q is not configured on the agent", name))
		}
	}
	for _, name := range sortedKeys(remoteByName) {
		if _, ok := localByName[name]; !ok {
			mismatches = append(mismatches, fmt.Sprintf("agent client tool %q is not defined by this agent run", name))
		}
	}
	for _, name := range sortedKeys(localByName) {
		remoteTool, ok := remoteByName[name]
		if !ok {
			continue
		}
		localTool := localByName[name]
		if localTool.Description != remoteTool.Description {
			mismatches = append(mismatches, fmt.Sprintf("tool %q differs in description", name))
		}
		if !schemasMatch(preparedParameters(localTool.Schema), remoteTool.Parameters) {
			mismatches = append(mismatches, fmt.Sprintf("tool %q differs in parameters schema", name))
		}
		if remoteTool.ExpectsResponse != nil && !*remoteTool.ExpectsResponse {
			mismatches = append(mismatches, fmt.Sprintf(
				"agent client tool %q has expects_response disabled, but Pydantic AI tools always return a result", name,
			))
		}
	}
	return mismatches
}

func sortedKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}
