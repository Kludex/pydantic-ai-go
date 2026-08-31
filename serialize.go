package ai

import (
	"encoding/json"
	"fmt"
	"slices"
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
	Kind           string         `json:"kind"`
	Parts          []wirePart     `json:"parts"`
	Timestamp      *time.Time     `json:"timestamp,omitempty"`
	Instructions   string         `json:"instructions,omitempty"`
	RunID          string         `json:"run_id,omitempty"`
	ConversationID string         `json:"conversation_id,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
	State          RequestState   `json:"state,omitempty"`
}

type wireResponse struct {
	Kind               string             `json:"kind"`
	Parts              []wirePart         `json:"parts"`
	Usage              *Usage             `json:"usage,omitempty"`
	ModelName          string             `json:"model_name,omitempty"`
	Timestamp          *time.Time         `json:"timestamp,omitempty"`
	ProviderName       string             `json:"provider_name,omitempty"`
	ProviderURL        string             `json:"provider_url,omitempty"`
	ProviderDetails    map[string]any     `json:"provider_details,omitempty"`
	VendorDetails      map[string]any     `json:"vendor_details,omitempty"`
	ProviderResponseID string             `json:"provider_response_id,omitempty"`
	VendorID           string             `json:"vendor_id,omitempty"`
	FinishReason       FinishReason       `json:"finish_reason,omitempty"`
	RunID              string             `json:"run_id,omitempty"`
	ConversationID     string             `json:"conversation_id,omitempty"`
	Metadata           map[string]any     `json:"metadata,omitempty"`
	State              ModelResponseState `json:"state,omitempty"`
}

type wirePart struct {
	PartKind        string            `json:"part_kind"`
	Content         json.RawMessage   `json:"content,omitempty"`
	ToolName        string            `json:"tool_name,omitempty"`
	ToolCallID      string            `json:"tool_call_id,omitempty"`
	ToolKind        ToolPartKind      `json:"tool_kind,omitempty"`
	Args            json.RawMessage   `json:"args,omitempty"`
	ID              string            `json:"id,omitempty"`
	Signature       string            `json:"signature,omitempty"`
	DynamicRef      string            `json:"dynamic_ref,omitempty"`
	Timestamp       *time.Time        `json:"timestamp,omitempty"`
	ProviderName    string            `json:"provider_name,omitempty"`
	ProviderDetails map[string]any    `json:"provider_details,omitempty"`
	Outcome         ToolReturnOutcome `json:"outcome,omitempty"`
	Metadata        map[string]any    `json:"metadata,omitempty"`
	ToolsAdded      []string          `json:"tools_added,omitempty"`
	Added           []string          `json:"added,omitempty"`
}

func marshalMessage(m ModelMessage) ([]byte, error) {
	switch msg := m.(type) {
	case ModelRequest:
		w := wireRequest{
			Kind: "request", Instructions: msg.Instructions, RunID: msg.RunID,
			ConversationID: msg.ConversationID, Metadata: msg.Metadata, State: msg.State,
		}
		if !msg.Timestamp.IsZero() {
			timestamp := msg.Timestamp
			w.Timestamp = &timestamp
		}
		for _, p := range msg.Parts {
			wp, err := marshalRequestPart(p)
			if err != nil {
				return nil, err
			}
			w.Parts = append(w.Parts, wp)
		}
		return json.Marshal(w)
	case ModelResponse:
		w := wireResponse{
			Kind: "response", ModelName: msg.ModelName, ProviderName: msg.ProviderName,
			ProviderURL: msg.ProviderURL, ProviderDetails: msg.ProviderDetails,
			ProviderResponseID: msg.ProviderResponseID, FinishReason: msg.FinishReason,
			RunID: msg.RunID, ConversationID: msg.ConversationID, Metadata: msg.Metadata, State: msg.State,
		}
		if !msg.Usage.IsZero() {
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
		wire := wirePart{PartKind: "system-prompt", Content: mustJSON(part.Content), DynamicRef: part.DynamicRef}
		return wirePartWithTimestamp(wire, part.Timestamp), nil
	case UserPromptPart:
		content, err := marshalUserContent(part)
		if err != nil {
			return wirePart{}, err
		}
		return wirePartWithTimestamp(wirePart{PartKind: "user-prompt", Content: content}, part.Timestamp), nil
	case ToolReturnPart:
		content, err := json.Marshal(part.Content)
		if err != nil {
			return wirePart{}, fmt.Errorf("ai: marshal tool return content: %w", err)
		}
		return wirePartWithTimestamp(wirePart{
			PartKind: "tool-return", Content: content, ToolName: part.ToolName,
			ToolCallID: part.ToolCallID, ToolKind: part.ToolKind, Outcome: part.Outcome, Metadata: part.Metadata,
		}, part.Timestamp), nil
	case ToolAvailabilityDeltaPart:
		return wirePart{
			PartKind: "tool-availability-delta", ToolsAdded: slices.Clone(part.ToolsAdded),
			ToolCallID: part.ToolCallID,
		}, nil
	case RetryPromptPart:
		content := mustJSON(part.Content)
		if part.Errors != nil {
			var err error
			content, err = json.Marshal(part.Errors)
			if err != nil {
				return wirePart{}, fmt.Errorf("ai: marshal retry validation errors: %w", err)
			}
		}
		wire := wirePart{
			PartKind: "retry-prompt", Content: content, ToolName: part.ToolName, ToolCallID: part.ToolCallID,
		}
		return wirePartWithTimestamp(wire, part.Timestamp), nil
	default:
		return wirePart{}, fmt.Errorf("ai: unknown request part type %T", p)
	}
}

func marshalResponsePart(p ResponsePart) (wirePart, error) {
	switch part := p.(type) {
	case TextPart:
		return wirePart{
			PartKind: "text", Content: mustJSON(part.Content), ID: part.ID,
			ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
		}, nil
	case FilePart:
		content, _ := json.Marshal(struct {
			Data           urlBase64Bytes `json:"data"`
			MediaType      string         `json:"media_type"`
			VendorMetadata map[string]any `json:"vendor_metadata,omitempty"`
			Kind           string         `json:"kind"`
			Identifier     string         `json:"identifier"`
		}{
			Data: urlBase64Bytes(part.Content.Data), MediaType: part.Content.MediaType,
			VendorMetadata: part.Content.VendorMetadata, Kind: "binary",
			Identifier: part.Content.ResolvedIdentifier(),
		})
		return wirePart{
			PartKind: "file", Content: content, ID: part.ID,
			ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
		}, nil
	case ToolCallPart:
		return wirePart{
			PartKind: "tool-call", ToolName: part.ToolName, Args: part.Args, ToolCallID: part.ToolCallID,
			ToolKind: part.ToolKind, ID: part.ID, ProviderName: part.ProviderName,
			ProviderDetails: part.ProviderDetails,
		}, nil
	case NativeToolCallPart:
		return wirePart{
			PartKind: "builtin-tool-call", ToolName: part.ToolName, Args: part.Args, ToolCallID: part.ToolCallID,
			ToolKind: part.ToolKind, ID: part.ID, ProviderName: part.ProviderName,
			ProviderDetails: part.ProviderDetails,
		}, nil
	case NativeToolReturnPart:
		content, err := json.Marshal(part.Content)
		if err != nil {
			return wirePart{}, fmt.Errorf("ai: marshal native tool return content: %w", err)
		}
		return wirePartWithTimestamp(wirePart{
			PartKind: "builtin-tool-return", ToolName: part.ToolName, Content: content,
			ToolCallID: part.ToolCallID, ToolKind: part.ToolKind, Metadata: part.Metadata,
			Outcome: part.Outcome, ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
		}, part.Timestamp), nil
	case ThinkingPart:
		return wirePart{
			PartKind: "thinking", Content: mustJSON(part.Content), ID: part.ID, Signature: part.Signature,
			ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
		}, nil
	case CompactionPart:
		var content json.RawMessage
		if part.Content != "" {
			content = mustJSON(part.Content)
		}
		return wirePart{
			PartKind: "compaction", Content: content, ID: part.ID,
			ProviderName: part.ProviderName, ProviderDetails: part.ProviderDetails,
		}, nil
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
		msg := ModelRequest{
			Instructions: w.Instructions, RunID: w.RunID, ConversationID: w.ConversationID,
			Metadata: w.Metadata, State: w.State,
		}
		if w.Timestamp != nil {
			msg.Timestamp = *w.Timestamp
		}
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
		providerDetails := w.ProviderDetails
		if providerDetails == nil {
			providerDetails = w.VendorDetails
		}
		providerResponseID := w.ProviderResponseID
		if providerResponseID == "" {
			providerResponseID = w.VendorID
		}
		msg := ModelResponse{
			ModelName: w.ModelName, ProviderName: w.ProviderName, ProviderURL: w.ProviderURL,
			ProviderDetails: providerDetails, ProviderResponseID: providerResponseID,
			FinishReason: w.FinishReason, RunID: w.RunID, ConversationID: w.ConversationID,
			Metadata: w.Metadata, State: w.State,
		}
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
		part := SystemPromptPart{Content: stringContent(wp.Content), DynamicRef: wp.DynamicRef}
		if wp.Timestamp != nil {
			part.Timestamp = *wp.Timestamp
		}
		return part, nil
	case "user-prompt":
		part, err := unmarshalUserContent(wp.Content)
		if err != nil {
			return nil, err
		}
		if wp.Timestamp != nil {
			part.Timestamp = *wp.Timestamp
		}
		return part, nil
	case "tool-return":
		var content any
		// wp.Content is raw JSON from a document that already parsed,
		// so decoding into any cannot fail.
		if len(wp.Content) > 0 {
			_ = json.Unmarshal(wp.Content, &content)
		}
		content, err := narrowToolReturnContent(content)
		if err != nil {
			return nil, fmt.Errorf("ai: unmarshal tool return content: %w", err)
		}
		part := ToolReturnPart{
			ToolName: wp.ToolName, Content: content, ToolCallID: wp.ToolCallID, ToolKind: wp.ToolKind,
			Outcome: wp.Outcome, Metadata: wp.Metadata,
		}
		if wp.Timestamp != nil {
			part.Timestamp = *wp.Timestamp
		}
		return part, nil
	case "tool-availability-delta":
		tools := wp.ToolsAdded
		if tools == nil {
			tools = wp.Added
		}
		return ToolAvailabilityDeltaPart{
			ToolsAdded: slices.Clone(tools), ToolCallID: wp.ToolCallID,
		}, nil
	case "retry-prompt":
		part := RetryPromptPart{ToolName: wp.ToolName, ToolCallID: wp.ToolCallID}
		if err := json.Unmarshal(wp.Content, &part.Content); err != nil {
			if err := json.Unmarshal(wp.Content, &part.Errors); err != nil {
				return nil, fmt.Errorf("ai: unmarshal retry validation errors: %w", err)
			}
		}
		if wp.Timestamp != nil {
			part.Timestamp = *wp.Timestamp
		}
		return part, nil
	default:
		return nil, fmt.Errorf("ai: unknown request part kind %q", wp.PartKind)
	}
}

func unmarshalResponsePart(wp wirePart) (ResponsePart, error) {
	switch wp.PartKind {
	case "text":
		return TextPart{
			Content: stringContent(wp.Content), ID: wp.ID,
			ProviderName: wp.ProviderName, ProviderDetails: wp.ProviderDetails,
		}, nil
	case "file":
		var content struct {
			Data           urlBase64Bytes `json:"data"`
			MediaType      string         `json:"media_type"`
			VendorMetadata map[string]any `json:"vendor_metadata"`
			Identifier     string         `json:"identifier"`
		}
		if err := json.Unmarshal(wp.Content, &content); err != nil {
			return nil, fmt.Errorf("ai: unmarshal file content: %w", err)
		}
		return FilePart{
			Content: BinaryContent{
				Data: []byte(content.Data), MediaType: content.MediaType,
				VendorMetadata: content.VendorMetadata, Identifier: content.Identifier,
			},
			ID:           wp.ID,
			ProviderName: wp.ProviderName, ProviderDetails: wp.ProviderDetails,
		}, nil
	case "tool-call":
		return ToolCallPart{
			ToolName: wp.ToolName, Args: wp.Args, ToolCallID: wp.ToolCallID, ToolKind: wp.ToolKind,
			ID: wp.ID, ProviderName: wp.ProviderName, ProviderDetails: wp.ProviderDetails,
		}, nil
	case "builtin-tool-call":
		return NativeToolCallPart{
			ToolName: wp.ToolName, Args: wp.Args, ToolCallID: wp.ToolCallID, ToolKind: wp.ToolKind,
			ID: wp.ID, ProviderName: wp.ProviderName, ProviderDetails: wp.ProviderDetails,
		}, nil
	case "builtin-tool-return":
		var content any
		if err := json.Unmarshal(wp.Content, &content); err != nil {
			return nil, fmt.Errorf("ai: unmarshal native tool return content: %w", err)
		}
		content, err := narrowToolReturnContent(content)
		if err != nil {
			return nil, fmt.Errorf("ai: unmarshal native tool return content: %w", err)
		}
		part := NativeToolReturnPart{
			ToolName: wp.ToolName, Content: content, ToolCallID: wp.ToolCallID, ToolKind: wp.ToolKind,
			Metadata: wp.Metadata, Outcome: wp.Outcome, ProviderName: wp.ProviderName,
			ProviderDetails: wp.ProviderDetails,
		}
		if wp.Timestamp != nil {
			part.Timestamp = *wp.Timestamp
		}
		return part, nil
	case "thinking":
		return ThinkingPart{
			Content: stringContent(wp.Content), ID: wp.ID, Signature: wp.Signature,
			ProviderName: wp.ProviderName, ProviderDetails: wp.ProviderDetails,
		}, nil
	case "compaction":
		return CompactionPart{
			Content: stringContent(wp.Content), ID: wp.ID,
			ProviderName: wp.ProviderName, ProviderDetails: wp.ProviderDetails,
		}, nil
	default:
		return nil, fmt.Errorf("ai: unknown response part kind %q", wp.PartKind)
	}
}

func wirePartWithTimestamp(part wirePart, timestamp time.Time) wirePart {
	if !timestamp.IsZero() {
		part.Timestamp = &timestamp
	}
	return part
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
	Kind           string         `json:"kind"`
	Content        string         `json:"content,omitempty"`
	URL            string         `json:"url,omitempty"`
	Data           urlBase64Bytes `json:"data,omitempty"`
	MediaType      string         `json:"media_type,omitempty"`
	FileID         string         `json:"file_id,omitempty"`
	ProviderName   string         `json:"provider_name,omitempty"`
	Identifier     string         `json:"identifier,omitempty"`
	VendorMetadata map[string]any `json:"vendor_metadata,omitempty"`
	Metadata       any            `json:"metadata,omitempty"`
	ForceDownload  any            `json:"force_download,omitempty"`
	TTL            CachePointTTL  `json:"ttl,omitempty"`
}

func marshalUserContent(part UserPromptPart) (json.RawMessage, error) {
	if len(part.Contents) == 0 {
		return mustJSON(part.Content), nil
	}
	items := make([]wireUserContent, 0, len(part.Contents))
	for _, c := range part.Contents {
		switch item := c.(type) {
		case TextContent:
			items = append(items, wireUserContent{Kind: "text-content", Content: item.Text, Metadata: item.Metadata})
		case ImageURL:
			wire, err := marshalFileURL(
				"image-url", item.URL, item.ResolvedMediaType, item.ResolvedIdentifier(), item.ForceDownload,
				item.VendorMetadata,
			)
			if err != nil {
				return nil, err
			}
			items = append(items, wire)
		case VideoURL:
			wire, err := marshalFileURL(
				"video-url", item.URL, item.ResolvedMediaType, item.ResolvedIdentifier(), item.ForceDownload,
				item.VendorMetadata,
			)
			if err != nil {
				return nil, err
			}
			items = append(items, wire)
		case AudioURL:
			wire, err := marshalFileURL(
				"audio-url", item.URL, item.ResolvedMediaType, item.ResolvedIdentifier(), item.ForceDownload,
				item.VendorMetadata,
			)
			if err != nil {
				return nil, err
			}
			items = append(items, wire)
		case DocumentURL:
			wire, err := marshalFileURL(
				"document-url", item.URL, item.ResolvedMediaType, item.ResolvedIdentifier(), item.ForceDownload,
				item.VendorMetadata,
			)
			if err != nil {
				return nil, err
			}
			items = append(items, wire)
		case BinaryContent:
			items = append(items, wireUserContent{
				Kind: "binary", Data: urlBase64Bytes(item.Data), MediaType: item.MediaType,
				Identifier: item.ResolvedIdentifier(), VendorMetadata: item.VendorMetadata,
			})
		case CachePoint:
			ttl, err := item.ResolvedTTL()
			if err != nil {
				return nil, err
			}
			items = append(items, wireUserContent{Kind: "cache-point", TTL: ttl})
		case UploadedFile:
			items = append(items, wireUserContent{
				Kind: "uploaded-file", FileID: item.FileID, ProviderName: item.ProviderName,
				MediaType: item.MediaType, Identifier: item.Identifier, VendorMetadata: item.VendorMetadata,
			})
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
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return UserPromptPart{}, fmt.Errorf("ai: unmarshal user prompt content: %w", err)
	}
	part := UserPromptPart{}
	for _, rawItem := range items {
		var text string
		if err := json.Unmarshal(rawItem, &text); err == nil {
			part.Contents = append(part.Contents, TextContent{Text: text})
			continue
		}
		var item wireUserContent
		if err := json.Unmarshal(rawItem, &item); err != nil {
			return UserPromptPart{}, fmt.Errorf("ai: unmarshal user content: %w", err)
		}
		content, err := unmarshalUserContentItem(item)
		if err != nil {
			return UserPromptPart{}, err
		}
		part.Contents = append(part.Contents, content)
	}
	return part, nil
}

func unmarshalUserContentItem(item wireUserContent) (UserContent, error) {
	switch item.Kind {
	case "text-content":
		return TextContent{Text: item.Content, Metadata: item.Metadata}, nil
	case "image-url":
		forceDownload, err := unmarshalFileDownloadMode(item.ForceDownload, "image")
		return ImageURL{
			URL: item.URL, MediaType: item.MediaType, Identifier: item.Identifier,
			ForceDownload: forceDownload, VendorMetadata: item.VendorMetadata,
		}, err
	case "video-url":
		forceDownload, err := unmarshalFileDownloadMode(item.ForceDownload, "video")
		return VideoURL{
			URL: item.URL, MediaType: item.MediaType, Identifier: item.Identifier,
			ForceDownload: forceDownload, VendorMetadata: item.VendorMetadata,
		}, err
	case "audio-url":
		forceDownload, err := unmarshalFileDownloadMode(item.ForceDownload, "audio")
		return AudioURL{
			URL: item.URL, MediaType: item.MediaType, Identifier: item.Identifier,
			ForceDownload: forceDownload, VendorMetadata: item.VendorMetadata,
		}, err
	case "document-url":
		forceDownload, err := unmarshalFileDownloadMode(item.ForceDownload, "document")
		return DocumentURL{
			URL: item.URL, MediaType: item.MediaType, Identifier: item.Identifier,
			ForceDownload: forceDownload, VendorMetadata: item.VendorMetadata,
		}, err
	case "binary":
		return BinaryContent{
			Data: []byte(item.Data), MediaType: item.MediaType, Identifier: item.Identifier,
			VendorMetadata: item.VendorMetadata,
		}, nil
	case "cache-point":
		point := CachePoint{TTL: item.TTL}
		ttl, err := point.ResolvedTTL()
		return CachePoint{TTL: ttl}, err
	case "uploaded-file":
		return UploadedFile{
			FileID: item.FileID, ProviderName: item.ProviderName, MediaType: item.MediaType,
			Identifier: item.Identifier, VendorMetadata: item.VendorMetadata,
		}, nil
	default:
		return nil, fmt.Errorf("ai: unknown user content kind %q", item.Kind)
	}
}

func marshalFileURL(
	kind, rawURL string,
	resolveMediaType func() (string, error),
	identifier string,
	forceDownload FileDownloadMode,
	vendorMetadata map[string]any,
) (wireUserContent, error) {
	if err := forceDownload.Validate(); err != nil {
		return wireUserContent{}, err
	}
	mediaType, err := resolveMediaType()
	if err != nil {
		return wireUserContent{}, err
	}
	wireForceDownload := any(false)
	switch forceDownload {
	case FileDownloadSafe:
		wireForceDownload = true
	case FileDownloadAllowLocal:
		wireForceDownload = string(FileDownloadAllowLocal)
	}
	return wireUserContent{
		Kind: kind, URL: rawURL, MediaType: mediaType, Identifier: identifier,
		ForceDownload: wireForceDownload, VendorMetadata: vendorMetadata,
	}, nil
}

func unmarshalFileDownloadMode(value any, kind string) (FileDownloadMode, error) {
	var mode FileDownloadMode
	switch forceDownload := value.(type) {
	case nil:
	case bool:
		if forceDownload {
			mode = FileDownloadSafe
		}
	case string:
		mode = FileDownloadMode(forceDownload)
	default:
		return "", fmt.Errorf("ai: invalid %s force_download value %T", kind, value)
	}
	if err := mode.Validate(); err != nil {
		return "", err
	}
	return mode, nil
}
