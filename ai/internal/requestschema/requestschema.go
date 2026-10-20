// Package requestschema applies a provider's JSON Schema transform to every
// schema a model request carries.
package requestschema

import ai "github.com/Kludex/pydantic-ai-go/ai"

// Transform returns params with transform applied to every tool, deferred-tool,
// output-tool, and native-output schema. Definitions are copied, not mutated.
func Transform(params ai.ModelRequestParams, transform func(map[string]any) map[string]any) ai.ModelRequestParams {
	params.Tools = definitions(params.Tools, transform)
	params.DeferredTools = definitions(params.DeferredTools, transform)
	if params.OutputTool != nil {
		output := definition(*params.OutputTool, transform)
		params.OutputTool = &output
	}
	if params.OutputSchema != nil {
		params.OutputSchema = transform(params.OutputSchema)
	}
	return params
}

func definitions(tools []ai.ToolDefinition, transform func(map[string]any) map[string]any) []ai.ToolDefinition {
	if tools == nil {
		return nil
	}
	transformed := make([]ai.ToolDefinition, len(tools))
	for index, tool := range tools {
		transformed[index] = definition(tool, transform)
	}
	return transformed
}

func definition(tool ai.ToolDefinition, transform func(map[string]any) map[string]any) ai.ToolDefinition {
	tool.Schema = transform(tool.Schema)
	if tool.ReturnSchema != nil {
		tool.ReturnSchema = transform(tool.ReturnSchema)
	}
	return tool
}
