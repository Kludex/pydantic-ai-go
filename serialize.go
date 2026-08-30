package ai

import (
	"encoding/json"
	"fmt"
	"time"
)

// MarshalMessages encodes a conversation in PydanticAI's JSON message format.
func MarshalMessages(msgs []ModelMessage) ([]byte, error) {
	out := make([]json.RawMessage, len(msgs))
	for i, m := range msgs {
		b, err := marshalMessage(m)
		if err != nil {
			return nil, err
		}
		out[i] = b
	}
	return json.Marshal(out)
}

// UnmarshalMessages decodes a conversation from PydanticAI's JSON message format.
func UnmarshalMessages(data []byte) ([]ModelMessage, error) {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	msgs := make([]ModelMessage, len(raw))
	for i, r := range raw {
		m, err := unmarshalMessage(r)
		if err != nil {
			return nil, err
		}
		msgs[i] = m
	}
	return msgs, nil
}

type wireRequest struct {
	Kind  string     `json:"kind"`
	Parts []wirePart `json:"parts"`
}

type wireResponse struct {
	Kind      string     `json:"kind"`
	Parts     []wirePart `json:"parts"`
	Usage     *Usage     `json:"usage,omitempty"`
	ModelName string     `json:"model_name,omitempty"`
	Timestamp *time.Time `json:"timestamp,omitempty"`
}

type wirePart struct {
	PartKind   string          `json:"part_kind"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolName   string          `json:"tool_name,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Args       json.RawMessage `json:"args,omitempty"`
}

func marshalMessage(m ModelMessage) ([]byte, error) {
	switch msg := m.(type) {
	case ModelRequest:
		w := wireRequest{Kind: "request"}
		for _, p := range msg.Parts {
			wp, err := marshalRequestPart(p)
			if err != nil {
				return nil, err
			}
			w.Parts = append(w.Parts, wp)
		}
		return json.Marshal(w)
	case ModelResponse:
		w := wireResponse{Kind: "response", ModelName: msg.ModelName}
		if msg.Usage != (Usage{}) {
			u := msg.Usage
			w.Usage = &u
		}
		if !msg.Timestamp.IsZero() {
			t := msg.Timestamp
			w.Timestamp = &t
		}
		for _, p := range msg.Parts {
			wp, err := marshalResponsePart(p)
			if err != nil {
				return nil, err
			}
			w.Parts = append(w.Parts, wp)
		}
		return json.Marshal(w)
	default:
		return nil, fmt.Errorf("ai: unknown message type %T", m)
	}
}

func marshalRequestPart(p RequestPart) (wirePart, error) {
	switch part := p.(type) {
	case SystemPromptPart:
		return wirePart{PartKind: "system-prompt", Content: mustJSON(part.Content)}, nil
	case UserPromptPart:
		content, err := marshalUserContent(part)
		if err != nil {
			return wirePart{}, err
		}
		return wirePart{PartKind: "user-prompt", Content: content}, nil
	case ToolReturnPart:
		content, err := json.Marshal(part.Content)
		if err != nil {
			return wirePart{}, fmt.Errorf("ai: marshal tool return content: %w", err)
		}
		return wirePart{PartKind: "tool-return", Content: content, ToolName: part.ToolName, ToolCallID: part.ToolCallID}, nil
	case RetryPromptPart:
		return wirePart{PartKind: "retry-prompt", Content: mustJSON(part.Content), ToolName: part.ToolName, ToolCallID: part.ToolCallID}, nil
	default:
		return wirePart{}, fmt.Errorf("ai: unknown request part type %T", p)
	}
}

func marshalResponsePart(p ResponsePart) (wirePart, error) {
	switch part := p.(type) {
	case TextPart:
		return wirePart{PartKind: "text", Content: mustJSON(part.Content)}, nil
	case ToolCallPart:
		return wirePart{PartKind: "tool-call", ToolName: part.ToolName, Args: part.Args, ToolCallID: part.ToolCallID}, nil
	case ThinkingPart:
		return wirePart{PartKind: "thinking", Content: mustJSON(part.Content)}, nil
	default:
		return wirePart{}, fmt.Errorf("ai: unknown response part type %T", p)
	}
}

func unmarshalMessage(data []byte) (ModelMessage, error) {
	var probe struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Kind {
	case "request":
		var w wireRequest
		if err := json.Unmarshal(data, &w); err != nil {
			return nil, err
		}
		msg := ModelRequest{}
		for _, wp := range w.Parts {
			p, err := unmarshalRequestPart(wp)
			if err != nil {
				return nil, err
			}
			msg.Parts = append(msg.Parts, p)
		}
		return msg, nil
	case "response":
		var w wireResponse
		if err := json.Unmarshal(data, &w); err != nil {
			return nil, err
		}
		msg := ModelResponse{ModelName: w.ModelName}
		if w.Usage != nil {
			msg.Usage = *w.Usage
		}
		if w.Timestamp != nil {
			msg.Timestamp = *w.Timestamp
		}
		for _, wp := range w.Parts {
			p, err := unmarshalResponsePart(wp)
			if err != nil {
				return nil, err
			}
			msg.Parts = append(msg.Parts, p)
		}
		return msg, nil
	default:
		return nil, fmt.Errorf("ai: unknown message kind %q", probe.Kind)
	}
}

func unmarshalRequestPart(wp wirePart) (RequestPart, error) {
	switch wp.PartKind {
	case "system-prompt":
		return SystemPromptPart{Content: stringContent(wp.Content)}, nil
	case "user-prompt":
		return unmarshalUserContent(wp.Content)
	case "tool-return":
		var content any
		// wp.Content is raw JSON from a document that already parsed,
		// so decoding into any cannot fail.
		if len(wp.Content) > 0 {
			_ = json.Unmarshal(wp.Content, &content)
		}
		return ToolReturnPart{ToolName: wp.ToolName, Content: content, ToolCallID: wp.ToolCallID}, nil
	case "retry-prompt":
		return RetryPromptPart{Content: stringContent(wp.Content), ToolName: wp.ToolName, ToolCallID: wp.ToolCallID}, nil
	default:
		return nil, fmt.Errorf("ai: unknown request part kind %q", wp.PartKind)
	}
}

func unmarshalResponsePart(wp wirePart) (ResponsePart, error) {
	switch wp.PartKind {
	case "text":
		return TextPart{Content: stringContent(wp.Content)}, nil
	case "tool-call":
		return ToolCallPart{ToolName: wp.ToolName, Args: wp.Args, ToolCallID: wp.ToolCallID}, nil
	case "thinking":
		return ThinkingPart{Content: stringContent(wp.Content)}, nil
	default:
		return nil, fmt.Errorf("ai: unknown response part kind %q", wp.PartKind)
	}
}

func mustJSON(s string) json.RawMessage {
	b, _ := json.Marshal(s) // marshalling a string cannot fail
	return b
}

func stringContent(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return string(raw)
	}
	return s
}

type wireUserContent struct {
	Kind      string `json:"kind"`
	Content   string `json:"content,omitempty"`
	URL       string `json:"url,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

func marshalUserContent(part UserPromptPart) (json.RawMessage, error) {
	if len(part.Contents) == 0 {
		return mustJSON(part.Content), nil
	}
	items := make([]wireUserContent, 0, len(part.Contents))
	for _, c := range part.Contents {
		switch item := c.(type) {
		case TextContent:
			items = append(items, wireUserContent{Kind: "text-content", Content: item.Text})
		case ImageURL:
			items = append(items, wireUserContent{Kind: "image-url", URL: item.URL})
		case BinaryContent:
			items = append(items, wireUserContent{Kind: "binary", Data: item.Data, MediaType: item.MediaType})
		default:
			return nil, fmt.Errorf("ai: unknown user content type %T", c)
		}
	}
	b, _ := json.Marshal(items) // wireUserContent is always marshallable
	return b, nil
}

func unmarshalUserContent(raw json.RawMessage) (UserPromptPart, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return UserPromptPart{Content: s}, nil
	}
	var items []wireUserContent
	if err := json.Unmarshal(raw, &items); err != nil {
		return UserPromptPart{}, fmt.Errorf("ai: unmarshal user prompt content: %w", err)
	}
	part := UserPromptPart{}
	for _, item := range items {
		switch item.Kind {
		case "text-content":
			part.Contents = append(part.Contents, TextContent{Text: item.Content})
		case "image-url":
			part.Contents = append(part.Contents, ImageURL{URL: item.URL})
		case "binary":
			part.Contents = append(part.Contents, BinaryContent{Data: item.Data, MediaType: item.MediaType})
		default:
			return UserPromptPart{}, fmt.Errorf("ai: unknown user content kind %q", item.Kind)
		}
	}
	return part, nil
}
