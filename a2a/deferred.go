package a2a

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	protocol "github.com/a2aproject/a2a-go/a2a"

	ai "github.com/Kludex/pydantic-ai-go"
)

const (
	deferredRequestsKind = "pydantic-ai-go/deferred-tool-requests"
	deferredResultsKind  = "pydantic-ai-go/deferred-tool-results"
)

// DeferredCallOutcome identifies an externally executed tool result.
type DeferredCallOutcome string

const (
	// DeferredCallSucceeded returns Value to the model.
	DeferredCallSucceeded DeferredCallOutcome = "success"
	// DeferredCallFailed returns Message as a terminal tool failure.
	DeferredCallFailed DeferredCallOutcome = "failed"
	// DeferredCallRetry returns Message as a retry prompt.
	DeferredCallRetry DeferredCallOutcome = "retry"
)

// DeferredCallResult resolves one externally executed tool call.
type DeferredCallResult struct {
	// Outcome selects successful, failed, or retry behavior.
	Outcome DeferredCallOutcome `json:"outcome"`
	// Value is the successful JSON-compatible result.
	Value any `json:"value,omitempty"`
	// Message explains a failed or retry result.
	Message string `json:"message,omitempty"`
}

// DeferredApproval resolves one local tool approval request.
type DeferredApproval struct {
	// Approved authorizes local execution.
	Approved bool `json:"approved"`
	// OverrideArgs replaces the original arguments when non-empty.
	OverrideArgs json.RawMessage `json:"override_args,omitempty"`
	// Message explains a denial to the model.
	Message string `json:"message,omitempty"`
}

// DeferredResults resolves every pending call and approval from an input-required task.
type DeferredResults struct {
	// Calls contains externally executed results by tool-call ID.
	Calls map[string]DeferredCallResult `json:"calls,omitempty"`
	// Approvals contains reviewer decisions by tool-call ID.
	Approvals map[string]DeferredApproval `json:"approvals,omitempty"`
	// Metadata contains detached application context by tool-call ID.
	Metadata map[string]map[string]any `json:"metadata,omitempty"`
}

// NewDeferredResultsPart creates the data part used to resume an input-required task.
func NewDeferredResultsPart(results DeferredResults) (protocol.DataPart, error) {
	data, err := structMap(struct {
		Type string `json:"type"`
		DeferredResults
	}{Type: deferredResultsKind, DeferredResults: results})
	if err != nil {
		return protocol.DataPart{}, fmt.Errorf("ai/a2a: encode deferred results: %w", err)
	}
	return protocol.DataPart{Data: data}, nil
}

type deferredTool struct {
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Args     json.RawMessage `json:"args"`
	Metadata map[string]any  `json:"metadata,omitempty"`
}

type deferredRequestState struct {
	Type      string         `json:"type"`
	Messages  string         `json:"messages"`
	Calls     []deferredTool `json:"calls,omitempty"`
	Approvals []deferredTool `json:"approvals,omitempty"`
}

func newDeferredRequestPart(resultMessages []ai.ModelMessage, requests ai.DeferredToolRequests) (protocol.DataPart, error) {
	// Agent-produced messages only contain serializable public part types.
	messages, _ := ai.MarshalMessages(resultMessages)
	state := deferredRequestState{Type: deferredRequestsKind, Messages: string(messages)}
	state.Calls = deferredTools(requests.Calls, requests.Metadata)
	state.Approvals = deferredTools(requests.Approvals, requests.Metadata)
	data, err := structMap(state)
	if err != nil {
		return protocol.DataPart{}, fmt.Errorf("ai/a2a: encode deferred requests: %w", err)
	}
	return protocol.DataPart{Data: data}, nil
}

func deferredTools(calls []ai.ToolCallPart, metadata map[string]map[string]any) []deferredTool {
	tools := make([]deferredTool, len(calls))
	for index, call := range calls {
		tools[index] = deferredTool{
			ID: call.ToolCallID, Name: call.ToolName, Args: slices.Clone(call.Args), Metadata: maps.Clone(metadata[call.ToolCallID]),
		}
	}
	return tools
}

func deferredState(task *protocol.Task) (*deferredRequestState, error) {
	if task == nil || task.Status.State != protocol.TaskStateInputRequired || task.Status.Message == nil {
		return nil, nil
	}
	for _, part := range task.Status.Message.Parts {
		data, ok := part.(protocol.DataPart)
		if !ok || data.Data["type"] != deferredRequestsKind {
			continue
		}
		encoded, err := json.Marshal(data.Data)
		if err != nil {
			return nil, fmt.Errorf("ai/a2a: encode stored deferred state: %w", err)
		}
		var state deferredRequestState
		if err := json.Unmarshal(encoded, &state); err != nil {
			return nil, fmt.Errorf("ai/a2a: decode stored deferred state: %w", err)
		}
		if state.Messages == "" {
			return nil, fmt.Errorf("ai/a2a: stored deferred state has no message history")
		}
		return &state, nil
	}
	return nil, nil
}

func parseDeferredResults(data map[string]any) (*DeferredResults, error) {
	encoded, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("ai/a2a: encode deferred results: %w", err)
	}
	var results DeferredResults
	if err := json.Unmarshal(encoded, &results); err != nil {
		return nil, fmt.Errorf("ai/a2a: decode deferred results: %w", err)
	}
	return &results, nil
}

func resolveDeferred(state *deferredRequestState, wire *DeferredResults) ([]ai.ModelMessage, *ai.DeferredToolResults, error) {
	if state == nil {
		if wire != nil {
			return nil, nil, fmt.Errorf("ai/a2a: deferred results require an input-required stored task")
		}
		return nil, nil, nil
	}
	if wire == nil {
		return nil, nil, fmt.Errorf("ai/a2a: input-required task needs deferred results")
	}
	messages, err := ai.UnmarshalMessages([]byte(state.Messages))
	if err != nil {
		return nil, nil, fmt.Errorf("ai/a2a: decode deferred history: %w", err)
	}
	results := &ai.DeferredToolResults{
		Calls: make(map[string]any, len(wire.Calls)), Approvals: make(map[string]ai.ToolApproval, len(wire.Approvals)),
		Metadata: wire.Metadata,
	}
	if err := resolveDeferredCalls(state.Calls, wire.Calls, results.Calls); err != nil {
		return nil, nil, err
	}
	if err := resolveDeferredApprovals(state.Approvals, wire.Approvals, results.Approvals); err != nil {
		return nil, nil, err
	}
	return messages, results, nil
}

func resolveDeferredCalls(expected []deferredTool, supplied map[string]DeferredCallResult, output map[string]any) error {
	if len(supplied) != len(expected) {
		return fmt.Errorf("ai/a2a: deferred call results are incomplete")
	}
	for _, call := range expected {
		result, exists := supplied[call.ID]
		if !exists {
			return fmt.Errorf("ai/a2a: deferred call result %q is missing", call.ID)
		}
		switch result.Outcome {
		case DeferredCallSucceeded:
			output[call.ID] = result.Value
		case DeferredCallFailed:
			output[call.ID] = ai.ToolFailedf("%s", result.Message)
		case DeferredCallRetry:
			output[call.ID] = ai.Retryf("%s", result.Message)
		default:
			return fmt.Errorf("ai/a2a: deferred call result %q has invalid outcome %q", call.ID, result.Outcome)
		}
	}
	return nil
}

func resolveDeferredApprovals(
	expected []deferredTool, supplied map[string]DeferredApproval, output map[string]ai.ToolApproval,
) error {
	if len(supplied) != len(expected) {
		return fmt.Errorf("ai/a2a: deferred approval results are incomplete")
	}
	for _, call := range expected {
		approval, exists := supplied[call.ID]
		if !exists {
			return fmt.Errorf("ai/a2a: deferred approval result %q is missing", call.ID)
		}
		if approval.Approved {
			output[call.ID] = ai.ToolApproved{OverrideArgs: slices.Clone(approval.OverrideArgs)}
		} else {
			output[call.ID] = ai.DenyTool(approval.Message)
		}
	}
	return nil
}

func structMap(value any) (map[string]any, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var data map[string]any
	// Marshal output is always valid JSON.
	_ = json.Unmarshal(encoded, &data)
	return data, nil
}
