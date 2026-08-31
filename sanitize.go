package ai

import (
	"fmt"
	"net/url"
	"strings"
)

// MessageSanitizationOptions controls which values from untrusted message history are retained.
// The zero value strips system prompts, allows HTTP and HTTPS URLs, resets forced downloads,
// rejects uploaded-file references, and strips unresolved trailing tool calls.
type MessageSanitizationOptions struct {
	// AllowSystemPrompts trusts the client to own the model's highest-priority instructions.
	AllowSystemPrompts bool
	// StripCompactionParts prevents client boundaries from hiding a trusted history prefix.
	StripCompactionParts bool
	// AllowedFileURLSchemes defaults to HTTP and HTTPS when nil.
	// A non-nil empty slice rejects all explicit schemes.
	AllowedFileURLSchemes []string
	// AllowedFileDownloadModes permits selected non-default modes. FileDownloadNever is always permitted.
	AllowedFileDownloadModes []FileDownloadMode
	// AllowUploadedFiles trusts provider-hosted references supplied by the client.
	AllowUploadedFiles bool
	// ResolvedToolCallIDs retains trailing local calls that the server is resuming in this request.
	ResolvedToolCallIDs []string
}

// MessageSanitizationToolCall identifies a local tool call removed from untrusted history.
type MessageSanitizationToolCall struct {
	Name string
	ID   string
}

// MessageSanitizationReport describes security-sensitive values removed or reset by SanitizeMessages.
type MessageSanitizationReport struct {
	StrippedSystemPrompts        int
	StrippedCompactionParts      int
	DroppedFileURLSchemes        []string
	ResetFileDownloadModes       []FileDownloadMode
	DroppedUploadedFileProviders []string
	StrippedToolCalls            []MessageSanitizationToolCall
}

// Changed reports whether sanitization changed the supplied history.
func (report MessageSanitizationReport) Changed() bool {
	return report.StrippedSystemPrompts > 0 || report.StrippedCompactionParts > 0 ||
		len(report.DroppedFileURLSchemes) > 0 || len(report.ResetFileDownloadModes) > 0 ||
		len(report.DroppedUploadedFileProviders) > 0 || len(report.StrippedToolCalls) > 0
}

// SanitizeMessages removes values that are unsafe to honor from untrusted message history.
// It returns detached messages and a report suitable for application logging or audit records.
func SanitizeMessages(
	messages []ModelMessage, options MessageSanitizationOptions,
) ([]ModelMessage, MessageSanitizationReport, error) {
	state, err := newMessageSanitizer(options)
	if err != nil {
		return nil, MessageSanitizationReport{}, err
	}
	sanitized := make([]ModelMessage, 0, len(messages))
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			request, keep, sanitizeErr := state.request(message)
			if sanitizeErr != nil {
				return nil, MessageSanitizationReport{}, sanitizeErr
			}
			if keep {
				sanitized = append(sanitized, request)
			}
		case ModelResponse:
			response, keep, sanitizeErr := state.response(message)
			if sanitizeErr != nil {
				return nil, MessageSanitizationReport{}, sanitizeErr
			}
			if keep {
				sanitized = append(sanitized, response)
			}
		default:
			return nil, MessageSanitizationReport{}, fmt.Errorf("ai: sanitize unsupported message type %T", message)
		}
	}
	state.stripTrailingToolCalls(&sanitized)
	return sanitized, state.report(), nil
}

type messageSanitizer struct {
	allowSystemPrompts   bool
	stripCompactionParts bool
	allowUploadedFiles   bool
	allowedSchemes       map[string]struct{}
	allowedModes         map[FileDownloadMode]struct{}
	resolvedToolCallIDs  map[string]struct{}
	strippedSystems      int
	strippedCompactions  int
	droppedSchemes       map[string]struct{}
	resetModes           map[FileDownloadMode]struct{}
	droppedProviders     map[string]struct{}
	strippedToolCalls    []MessageSanitizationToolCall
}

func newMessageSanitizer(options MessageSanitizationOptions) (*messageSanitizer, error) {
	schemes := options.AllowedFileURLSchemes
	if schemes == nil {
		schemes = []string{"http", "https"}
	}
	state := &messageSanitizer{
		allowSystemPrompts: options.AllowSystemPrompts, stripCompactionParts: options.StripCompactionParts,
		allowUploadedFiles: options.AllowUploadedFiles, allowedSchemes: make(map[string]struct{}, len(schemes)),
		allowedModes:        make(map[FileDownloadMode]struct{}, len(options.AllowedFileDownloadModes)),
		resolvedToolCallIDs: make(map[string]struct{}, len(options.ResolvedToolCallIDs)),
		droppedSchemes:      make(map[string]struct{}), resetModes: make(map[FileDownloadMode]struct{}),
		droppedProviders: make(map[string]struct{}),
	}
	for _, scheme := range schemes {
		normalized := strings.ToLower(strings.TrimSpace(scheme))
		parsed, err := url.Parse(normalized + ":")
		if normalized == "" || err != nil || parsed.Scheme != normalized {
			return nil, fmt.Errorf("ai: invalid allowed file URL scheme %q", scheme)
		}
		state.allowedSchemes[normalized] = struct{}{}
	}
	for _, mode := range options.AllowedFileDownloadModes {
		if mode == FileDownloadNever {
			continue
		}
		if err := mode.Validate(); err != nil {
			return nil, err
		}
		state.allowedModes[mode] = struct{}{}
	}
	for _, id := range options.ResolvedToolCallIDs {
		state.resolvedToolCallIDs[id] = struct{}{}
	}
	return state, nil
}
