package openaiprotocol

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
	"strconv"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// ToolItemPruning deletes a tool round's items from the server conversation once
// the model has answered it, so later responses are not billed for them again.
type ToolItemPruning struct {
	// Enabled turns pruning on.
	Enabled bool
	// Keep names tools whose items stay in the conversation.
	Keep []string
}

// toolItems are one tool call's items in the server conversation, deleted together.
type toolItems struct {
	call    string
	output  string
	content []string
	// answeredBy is the frame sequence at which the output joined the conversation.
	answeredBy int
}

type pruneFrame struct {
	Type     string     `json:"type"`
	Item     *pruneItem `json:"item"`
	Response *struct {
		ID     string      `json:"id"`
		Status string      `json:"status"`
		Output []pruneItem `json:"output"`
	} `json:"response"`
}

type pruneItem struct {
	ID     string `json:"id"`
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
}

// toolContentItemID names the user message carrying content a tool returned
// beside its output, so it can be deleted with the call.
func toolContentItemID(inputIndex int) string {
	return "pydantic_ai_content_" + strconv.Itoa(inputIndex)
}

// trackToolItems records tool items as they join the conversation and returns the
// calls a completed tool-free response answered, whose items can now go.
func (connection *Connection) trackToolItems(data []byte) []string {
	if !connection.config.ToolItemPruning.Enabled {
		return nil
	}
	var frame pruneFrame
	if json.Unmarshal(data, &frame) != nil {
		return nil
	}
	connection.stateMu.Lock()
	defer connection.stateMu.Unlock()
	connection.frames++
	switch frame.Type {
	case "conversation.item.added", "conversation.item.created":
		item := frame.Item
		if item == nil || item.ID == "" || item.CallID == "" {
			return nil
		}
		switch item.Type {
		case "function_call":
			if !slices.Contains(connection.config.ToolItemPruning.Keep, item.Name) {
				connection.toolItems[item.CallID] = &toolItems{call: item.ID}
			}
		case "function_call_output":
			if items := connection.toolItems[item.CallID]; items != nil {
				items.output, items.answeredBy = item.ID, connection.frames
			}
		}
	case "response.created":
		if frame.Response != nil {
			connection.responseFrames[frame.Response.ID] = connection.frames
		}
	case "response.done":
		if frame.Response == nil {
			return nil
		}
		created, known := connection.responseFrames[frame.Response.ID]
		delete(connection.responseFrames, frame.Response.ID)
		calledTool := slices.ContainsFunc(frame.Response.Output, func(item pruneItem) bool {
			return item.Type == "function_call"
		})
		if !known || frame.Response.Status != "completed" || calledTool || len(frame.Response.Output) == 0 {
			return nil
		}
		var answered []string
		for callID, items := range connection.toolItems {
			if items.output != "" && items.answeredBy < created {
				answered = append(answered, callID)
			}
		}
		slices.Sort(answered)
		return answered
	}
	return nil
}

// pruneToolItems deletes answered calls with their outputs and returned content.
// A failed delete leaves the call unpruned, so a reconnect replays it.
func (connection *Connection) pruneToolItems(ctx context.Context, callIDs []string) {
	for _, callID := range callIDs {
		connection.stateMu.Lock()
		items := connection.toolItems[callID]
		delete(connection.toolItems, callID)
		connection.stateMu.Unlock()
		for _, itemID := range append([]string{items.call, items.output}, items.content...) {
			if connection.writeJSON(ctx, map[string]any{"type": "conversation.item.delete", "item_id": itemID}) != nil {
				return
			}
		}
		connection.stateMu.Lock()
		connection.prunedCalls[callID] = true
		connection.stateMu.Unlock()
	}
}

// withoutPrunedCalls removes pruned calls from a reconnect's replay. A pruned
// result's request goes whole, with the content the tool returned beside it.
func (connection *Connection) withoutPrunedCalls(messages []ai.ModelMessage) []ai.ModelMessage {
	connection.stateMu.Lock()
	pruned := maps.Clone(connection.prunedCalls)
	connection.stateMu.Unlock()
	if len(pruned) == 0 {
		return messages
	}
	kept := make([]ai.ModelMessage, 0, len(messages))
	for _, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			if !slices.ContainsFunc(message.Parts, func(part ai.RequestPart) bool {
				switch part := part.(type) {
				case ai.ToolReturnPart:
					return pruned[part.ToolCallID]
				case ai.RetryPromptPart:
					return pruned[part.ToolCallID]
				}
				return false
			}) {
				kept = append(kept, message)
			}
		case ai.ModelResponse:
			message.Parts = slices.DeleteFunc(slices.Clone(message.Parts), func(part ai.ResponsePart) bool {
				call, ok := part.(ai.ToolCallPart)
				return ok && pruned[call.ToolCallID]
			})
			if len(message.Parts) > 0 {
				kept = append(kept, message)
			}
		}
	}
	return kept
}
