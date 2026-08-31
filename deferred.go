package ai

import (
	"encoding/json"
	"fmt"
	"slices"
)

// DeferredToolRequests describes model tool calls that require work outside
// the current agent step. Calls require external execution. Approvals run
// locally after the caller approves them.
type DeferredToolRequests struct {
	Calls     []ToolCallPart
	Approvals []ToolCallPart
	Metadata  map[string]map[string]any
}

// Clone returns a detached copy of the pending requests.
func (r DeferredToolRequests) Clone() DeferredToolRequests {
	return DeferredToolRequests{
		Calls: cloneToolCalls(r.Calls), Approvals: cloneToolCalls(r.Approvals), Metadata: cloneDeferredMetadata(r.Metadata),
	}
}

// ToolApproval is a caller decision for a tool that requested approval.
type ToolApproval interface {
	ToolApprovalKind() string
}

// ToolApproved authorizes local execution. OverrideArgs, when non-empty,
// replaces the model-generated JSON arguments and is validated before use.
type ToolApproved struct {
	OverrideArgs json.RawMessage
}

// ToolApprovalKind identifies an approval decision.
func (ToolApproved) ToolApprovalKind() string { return "approved" }

// ToolDenied prevents execution and returns Message to the model.
type ToolDenied struct {
	Message string
}

// ToolApprovalKind identifies a denial decision.
func (ToolDenied) ToolApprovalKind() string { return "denied" }

// DeferredToolResults resolves calls from a previous deferred RunResult.
// Calls maps externally executed call IDs to plain values, ToolReturn values,
// ToolFailedf or Retryf errors, or RetryPromptPart values.
type DeferredToolResults struct {
	Calls     map[string]any
	Approvals map[string]ToolApproval
	Metadata  map[string]map[string]any
}

// ApproveTool returns a decision that executes the original arguments.
func ApproveTool() ToolApproved { return ToolApproved{} }

// ApproveToolWithArgs returns a decision with replacement arguments.
func ApproveToolWithArgs(args any) (ToolApproved, error) {
	raw, err := json.Marshal(args)
	if err != nil {
		return ToolApproved{}, fmt.Errorf("ai: marshal approved tool arguments: %w", err)
	}
	return ToolApproved{OverrideArgs: raw}, nil
}

// DenyTool returns a denial decision. An empty message uses the default.
func DenyTool(message string) ToolDenied { return ToolDenied{Message: message} }

func cloneDeferredToolResults(results DeferredToolResults) DeferredToolResults {
	cloned := DeferredToolResults{
		Calls:     make(map[string]any, len(results.Calls)),
		Approvals: make(map[string]ToolApproval, len(results.Approvals)),
		Metadata:  cloneDeferredMetadata(results.Metadata),
	}
	for id, result := range results.Calls {
		cloned.Calls[id] = cloneDeferredCallResult(result)
	}
	for id, approval := range results.Approvals {
		switch approval := approval.(type) {
		case ToolApproved:
			approval.OverrideArgs = slices.Clone(approval.OverrideArgs)
			cloned.Approvals[id] = approval
		case *ToolApproved:
			if approval != nil {
				value := *approval
				value.OverrideArgs = slices.Clone(value.OverrideArgs)
				cloned.Approvals[id] = &value
			} else {
				cloned.Approvals[id] = approval
			}
		case ToolDenied:
			cloned.Approvals[id] = approval
		case *ToolDenied:
			if approval != nil {
				value := *approval
				cloned.Approvals[id] = &value
			} else {
				cloned.Approvals[id] = approval
			}
		case nil:
			cloned.Approvals[id] = nil
		default:
			cloned.Approvals[id] = approval
		}
	}
	return cloned
}

func cloneDeferredCallResult(result any) any {
	switch result := result.(type) {
	case ToolReturn:
		return cloneToolReturn(result)
	case *ToolReturn:
		if result == nil {
			return (*ToolReturn)(nil)
		}
		cloned := cloneToolReturn(*result)
		return &cloned
	case RetryPromptPart:
		result.Errors = cloneDeferredValidationErrors(result.Errors)
		return result
	case ToolReturnPart:
		result.Metadata = cloneSchemaMap(result.Metadata)
		return result
	case json.RawMessage:
		return slices.Clone(result)
	default:
		return cloneSchemaValue(result)
	}
}

func cloneToolReturn(result ToolReturn) ToolReturn {
	result.ReturnValue = cloneSchemaValue(result.ReturnValue)
	result.Content = cloneUserContents(result.Content)
	result.Metadata = cloneSchemaMap(result.Metadata)
	result.Tools = slices.Clone(result.Tools)
	return result
}

func cloneDeferredMetadata(metadata map[string]map[string]any) map[string]map[string]any {
	if metadata == nil {
		return nil
	}
	cloned := make(map[string]map[string]any, len(metadata))
	for id, value := range metadata {
		cloned[id] = cloneSchemaMap(value)
	}
	return cloned
}

func cloneDeferredValidationErrors(errors []ValidationError) []ValidationError {
	cloned := make([]ValidationError, len(errors))
	for index, item := range errors {
		item.Location = slices.Clone(item.Location)
		item.Input = cloneSchemaValue(item.Input)
		item.Context = cloneSchemaMap(item.Context)
		cloned[index] = item
	}
	return cloned
}

func cloneToolCalls(calls []ToolCallPart) []ToolCallPart {
	cloned := make([]ToolCallPart, len(calls))
	for index, call := range calls {
		call.Args = slices.Clone(call.Args)
		call.ProviderDetails = cloneSchemaMap(call.ProviderDetails)
		cloned[index] = call
	}
	return cloned
}
