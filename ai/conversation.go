package ai

import "encoding/json"

// Conversation carries the state you need to continue or store a conversation.
// The zero value starts a new conversation; its first run generates an ID.
type Conversation struct {
	Messages             []ModelMessage        `json:"messages"`
	Usage                Usage                 `json:"usage"`
	ConversationID       string                `json:"conversation_id"`
	DeferredToolRequests *DeferredToolRequests `json:"deferred_tool_requests,omitempty"`
}

// WithConversation continues a detached snapshot of conversation. It cannot
// be combined with WithMessageHistory or WithConversationID. Answer pending
// requests separately with WithDeferredToolResults.
func WithConversation(conversation Conversation) RunOption {
	conversation.Messages = cloneModelMessages(conversation.Messages)
	conversation.Usage = conversation.Usage.Clone()
	if conversation.DeferredToolRequests != nil {
		requests := conversation.DeferredToolRequests.Clone()
		conversation.DeferredToolRequests = &requests
	}
	return func(config *runConfig) { config.conversation = &conversation }
}

// Conversation returns a detached bundle, ready to store or continue.
func (r *RunResult[Output]) Conversation() Conversation {
	return Conversation{
		Messages: cloneModelMessages(r.messages), Usage: r.Usage(),
		ConversationID: r.conversationID, DeferredToolRequests: r.Deferred(),
	}
}

// MarshalJSON encodes messages through the stable message persistence boundary.
func (c Conversation) MarshalJSON() ([]byte, error) {
	messages, err := MarshalMessages(c.Messages)
	if err != nil {
		return nil, err
	}
	return json.Marshal(conversationJSON{
		Messages: messages, Usage: c.Usage,
		ConversationID: c.ConversationID, DeferredToolRequests: c.DeferredToolRequests,
	})
}

// UnmarshalJSON restores a conversation and validates its message history.
func (c *Conversation) UnmarshalJSON(data []byte) error {
	var decoded conversationJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	if len(decoded.Messages) == 0 {
		decoded.Messages = json.RawMessage("[]")
	}
	messages, err := UnmarshalMessages(decoded.Messages)
	if err != nil {
		return err
	}
	*c = Conversation{
		Messages: messages, Usage: decoded.Usage,
		ConversationID: decoded.ConversationID, DeferredToolRequests: decoded.DeferredToolRequests,
	}
	return nil
}

type conversationJSON struct {
	Messages             json.RawMessage       `json:"messages"`
	Usage                Usage                 `json:"usage"`
	ConversationID       string                `json:"conversation_id"`
	DeferredToolRequests *DeferredToolRequests `json:"deferred_tool_requests,omitempty"`
}
