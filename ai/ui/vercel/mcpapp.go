package vercel

import ai "github.com/Kludex/pydantic-ai-go/ai"

func mcpAppPayload(metadata map[string]any) (map[string]any, bool) {
	app, _ := metadata[ai.MCPAppMetadataKey].(map[string]any)
	resourceURI, uriOK := app["resource_uri"].(string)
	toolInput, inputOK := app["tool_input"].(map[string]any)
	result, resultOK := app["result"].(map[string]any)
	if !uriOK || !inputOK || !resultOK {
		return nil, false
	}
	payload := map[string]any{"resourceUri": resourceURI, "toolInput": toolInput, "result": result}
	if toolsetID, _ := app["toolset_id"].(string); toolsetID != "" {
		payload["serverId"], payload["serverHash"] = toolsetID, toolsetID
	}
	return payload, true
}
