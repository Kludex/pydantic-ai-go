package elevenlabs

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// agentTool is one tool resolved on the agent or listed in the workspace.
type agentTool struct {
	Type                string         `json:"type"`
	Name                string         `json:"name"`
	Description         string         `json:"description"`
	Parameters          map[string]any `json:"parameters"`
	ExpectsResponse     *bool          `json:"expects_response"`
	ResponseTimeoutSecs *int           `json:"response_timeout_secs"`
}

// agentPreflight is the subset of GET /v1/convai/agents/{id} the preflight reads.
type agentPreflight struct {
	ConversationConfig struct {
		Agent struct {
			Prompt struct {
				ToolIDs []string    `json:"tool_ids"`
				Tools   []agentTool `json:"tools"`
			} `json:"prompt"`
		} `json:"agent"`
		Conversation struct {
			ClientEvents []string `json:"client_events"`
		} `json:"conversation"`
	} `json:"conversation_config"`
	PlatformSettings struct {
		// Overrides mirrors conversation_config with booleans at the leaves.
		Overrides map[string]any `json:"overrides"`
	} `json:"platform_settings"`
}

func (agent agentPreflight) overridePermissions() map[string]any {
	permissions, _ := agent.PlatformSettings.Overrides["conversation_config_override"].(map[string]any)
	return permissions
}

func (agent agentPreflight) sendsContextUsage() bool {
	return slices.Contains(agent.ConversationConfig.Conversation.ClientEvents, "context_usage")
}

func (agent agentPreflight) clientTools() []agentTool {
	var tools []agentTool
	for _, tool := range agent.ConversationConfig.Agent.Prompt.Tools {
		if tool.Type == "client" {
			tools = append(tools, tool)
		}
	}
	return tools
}

func (agent agentPreflight) toolIDs() []string { return agent.ConversationConfig.Agent.Prompt.ToolIDs }

type workspaceTool struct {
	ID         string    `json:"id"`
	ToolConfig agentTool `json:"tool_config"`
}

func (model *Model) fetchAgent(ctx context.Context) (agentPreflight, error) {
	var agent agentPreflight
	if _, err := model.rest(ctx, http.MethodGet, "/v1/convai/agents/"+url.PathEscape(model.agentID), nil, nil, &agent); err != nil {
		return agent, err
	}
	// ElevenLabs deprecated the inline tools array in favor of tool_ids. While
	// it still resolves them, it is the source; otherwise each id is fetched,
	// skipping a dangling one the agent cannot call anyway.
	prompt := &agent.ConversationConfig.Agent.Prompt
	if len(prompt.Tools) > 0 || len(prompt.ToolIDs) == 0 {
		return agent, nil
	}
	for _, id := range prompt.ToolIDs {
		var tool workspaceTool
		status, err := model.rest(ctx, http.MethodGet, "/v1/convai/tools/"+url.PathEscape(id), nil, nil, &tool)
		if status == http.StatusNotFound {
			continue
		}
		if err != nil {
			return agent, err
		}
		prompt.Tools = append(prompt.Tools, tool.ToolConfig)
	}
	return agent, nil
}

// signedURL mints the short-lived conversation socket URL. It is the only
// server-side auth the socket accepts, works for public agents too, and keeps
// the API key out of the URL.
func (model *Model) signedURL(ctx context.Context) (string, error) {
	var response struct {
		SignedURL string `json:"signed_url"`
	}
	_, err := model.rest(ctx, http.MethodGet, "/v1/convai/conversation/get-signed-url",
		url.Values{"agent_id": {model.agentID}}, nil, &response)
	if err == nil && response.SignedURL == "" {
		err = errors.New("elevenlabs: the signed URL response carried no URL")
	}
	return response.SignedURL, err
}

// workspaceTools fetches every page of the workspace tool listing, so an
// attached id is never mistaken for a missing tool and duplicated.
func (model *Model) workspaceTools(ctx context.Context) (map[string]workspaceTool, error) {
	tools := map[string]workspaceTool{}
	seen := map[string]bool{}
	query := url.Values{"page_size": {"100"}}
	for {
		var page struct {
			Tools      []workspaceTool `json:"tools"`
			HasMore    bool            `json:"has_more"`
			NextCursor string          `json:"next_cursor"`
		}
		if _, err := model.rest(ctx, http.MethodGet, "/v1/convai/tools", query, nil, &page); err != nil {
			return nil, err
		}
		for _, tool := range page.Tools {
			tools[tool.ID] = tool
		}
		// A repeated cursor would refetch the same page forever.
		if !page.HasMore || page.NextCursor == "" || seen[page.NextCursor] {
			return tools, nil
		}
		seen[page.NextCursor] = true
		query.Set("cursor", page.NextCursor)
	}
}

// toolUsedElsewhere reports whether another agent references the tool, since
// changing a shared workspace tool changes it for every agent. A further page
// counts as shared, erring toward a fresh create over a shared mutation.
func (model *Model) toolUsedElsewhere(ctx context.Context, id string) (bool, error) {
	var page struct {
		Agents []struct {
			ID string `json:"id"`
		} `json:"agents"`
		HasMore bool `json:"has_more"`
	}
	if _, err := model.rest(ctx, http.MethodGet, "/v1/convai/tools/"+url.PathEscape(id)+"/dependent-agents", nil, nil, &page); err != nil {
		return false, err
	}
	return page.HasMore || slices.ContainsFunc(page.Agents, func(agent struct {
		ID string `json:"id"`
	}) bool {
		return agent.ID != model.agentID
	}), nil
}

func (model *Model) createTool(ctx context.Context, config map[string]any) (string, error) {
	var created struct {
		ID string `json:"id"`
	}
	_, err := model.rest(ctx, http.MethodPost, "/v1/convai/tools", nil, map[string]any{"tool_config": config}, &created)
	return created.ID, err
}

// syncTools makes the agent's client tools match the session's: it creates
// missing workspace tools, updates differing ones no other agent uses, and
// re-points the agent's tool_ids, keeping ElevenLabs-side server tools. It
// returns the attached ids once done.
func (model *Model) syncTools(ctx context.Context, agent agentPreflight, tools []ai.ToolDefinition) ([]string, error) {
	if len(toolMismatches(tools, agent.clientTools())) == 0 {
		return agent.toolIDs(), nil
	}
	remoteByName := map[string]agentTool{}
	for _, tool := range agent.clientTools() {
		remoteByName[tool.Name] = tool
	}
	// Every config is built before the first write, so a tool that cannot be synced costs no workspace mutation.
	configs := map[string]map[string]any{}
	var problems []error
	for _, tool := range tools {
		var remote *agentTool
		if found, ok := remoteByName[tool.Name]; ok {
			remote = &found
		}
		config, err := toolConfig(tool, remote)
		if err != nil {
			problems = append(problems, err)
		}
		configs[tool.Name] = config
	}
	if err := errors.Join(problems...); err != nil {
		return nil, err
	}
	workspace, err := model.workspaceTools(ctx)
	if err != nil {
		return nil, err
	}
	attached := agent.toolIDs()
	// Server-side tools, and ids the listing doesn't report, are ElevenLabs' and survive the re-point.
	var target []string
	clientIDs := map[string]string{}
	for _, id := range attached {
		tool, ok := workspace[id]
		if !ok || tool.ToolConfig.Type != "client" {
			target = append(target, id)
		} else if tool.ToolConfig.Name != "" {
			clientIDs[tool.ToolConfig.Name] = id
		}
	}
	// An unattached matching client tool nothing depends on was left by a failed sync, and is adopted.
	adoptable := map[string][]workspaceTool{}
	for _, id := range sortedKeys(workspace) {
		tool := workspace[id]
		if !slices.Contains(attached, id) && tool.ToolConfig.Type == "client" && tool.ToolConfig.Name != "" {
			adoptable[tool.ToolConfig.Name] = append(adoptable[tool.ToolConfig.Name], tool)
		}
	}
	for _, tool := range tools {
		config := configs[tool.Name]
		remote, configured := remoteByName[tool.Name]
		existing, attachedID := clientIDs[tool.Name]
		if !configured || !attachedID {
			id, err := model.adoptOrCreate(ctx, tool, config, adoptable[tool.Name])
			if err != nil {
				return nil, err
			}
			target = append(target, id)
			continue
		}
		if len(toolMismatches([]ai.ToolDefinition{tool}, []agentTool{remote})) > 0 {
			shared, err := model.toolUsedElsewhere(ctx, existing)
			if err != nil {
				return nil, err
			}
			if shared {
				// Updating it would change the other agents' tool too: this agent gets one of its own.
				if existing, err = model.createTool(ctx, config); err != nil {
					return nil, err
				}
			} else if _, err := model.rest(ctx, http.MethodPatch, "/v1/convai/tools/"+url.PathEscape(existing), nil,
				map[string]any{"tool_config": config}, nil); err != nil {
				return nil, err
			}
		}
		target = append(target, existing)
	}
	if !slices.Equal(target, attached) {
		body := map[string]any{"conversation_config": map[string]any{
			"agent": map[string]any{"prompt": map[string]any{"tool_ids": target}},
		}}
		if _, err := model.rest(ctx, http.MethodPatch, "/v1/convai/agents/"+url.PathEscape(model.agentID), nil, body, nil); err != nil {
			return nil, err
		}
	}
	return target, nil
}

func (model *Model) adoptOrCreate(
	ctx context.Context, tool ai.ToolDefinition, config map[string]any, candidates []workspaceTool,
) (string, error) {
	for _, candidate := range candidates {
		if len(toolMismatches([]ai.ToolDefinition{tool}, []agentTool{candidate.ToolConfig})) > 0 {
			continue
		}
		shared, err := model.toolUsedElsewhere(ctx, candidate.ID)
		if err != nil {
			return "", err
		}
		if !shared {
			return candidate.ID, nil
		}
	}
	return model.createTool(ctx, config)
}

// restrictedToolIDs leaves out the attached client tools outside allowed,
// keeping server-side tools and ids the listing doesn't report.
func (model *Model) restrictedToolIDs(ctx context.Context, attached []string, allowed map[string]bool) ([]string, error) {
	workspace, err := model.workspaceTools(ctx)
	if err != nil {
		return nil, err
	}
	restricted := []string{}
	for _, id := range attached {
		tool, ok := workspace[id]
		if !ok || tool.ToolConfig.Type != "client" || allowed[tool.ToolConfig.Name] {
			restricted = append(restricted, id)
		}
	}
	return restricted, nil
}
