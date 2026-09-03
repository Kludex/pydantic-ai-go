# Deferred tool execution

Pause a run before a sensitive tool executes. Resume it after your application records an approval decision.

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type DeleteArgs struct {
	Path string `json:"path"`
}

func main() {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context,
		_ []ai.ModelMessage,
		_ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "delete_file",
				ToolCallID: "delete-1",
				Args: json.RawMessage(`{"path":"report.txt"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "The file was deleted."},
		}}, nil
	})

	agent := ai.NewAgent[struct{}, string](model)
	ai.AddSimpleTool(agent, "delete_file", func(_ context.Context, args DeleteArgs) (string, error) {
		return "deleted " + args.Path, nil
	}, ai.WithApprovalRequired())

	paused, err := agent.Run(context.Background(), "Delete report.txt", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	pending := paused.Deferred()
	if pending == nil || len(pending.Approvals) != 1 {
		log.Fatal("expected one approval request")
	}

	call := pending.Approvals[0]
	fmt.Printf("Approve %s with arguments %s\n", call.ToolName, call.Args)
	approval := ai.ApproveTool()
	resumed, err := agent.Run(
		context.Background(),
		"Continue after review.",
		struct{}{},
		ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{
			Approvals: map[string]ai.ToolApproval{call.ToolCallID: approval},
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resumed.Output)
}
```

`WithApprovalRequired` validates the model arguments before the run pauses. The tool does not execute until you return `ApproveTool`. A denial from `DenyTool("A reviewer rejected this deletion.")` returns feedback to the model without calling the tool.

Use `ApproveToolWithArgs` when a reviewer changes the arguments. The replacement is encoded and validated against the same tool schema before execution.

## Persist a paused run

Persist `paused.Messages()` with `MarshalMessages` before you return control to a user or worker. Persist the pending call IDs from `paused.Deferred()` in the same transaction. A later process can decode the messages and pass them through `WithMessageHistory` as shown above.

Call IDs are the join key between the paused history and `DeferredToolResults`. Reject duplicate, missing, or unknown decisions in your application before resuming. The agent also validates that every supplied decision matches a pending call and its approval or external-execution kind.

Do not treat client-submitted tool names or arguments as authoritative. Load the pending call from your own persisted history, authorize it, and key the decision by its stored `ToolCallID`.

## Execute a tool in another system

Register a typed external tool with `AddExternalTool` when a queue, browser, or separate service owns execution. The first run returns the call in `DeferredToolRequests.Calls`. Submit it to the external executor. Resume with its typed result in `DeferredToolResults.Calls`, keyed by the original call ID.

External results can also be `ToolReturn`, `ToolFailedf`, `Retryf`, `ToolReturnPart`, or `RetryPromptPart` values. Use `ToolReturn` when the worker returns metadata or trailing multimodal content.

Use `WithDynamicApproval` or `RequestToolApproval` when approval depends on validated arguments or run dependencies. Use `WithDynamicExternalExecution` or `RequestExternalToolExecution` when only some calls leave the process.

## Stream deferred calls

`RunStream` emits each tool-call event before one `DeferredToolRequestsEvent`. Persist the event's request snapshot only after you receive it. `DeferredToolResultsEvent` reports each batch resolved by an inline deferred-call handler.

A detached provider job and a deferred local tool are different states. `SuspendedRun` resumes provider-owned background generation. `DeferredToolRequests` resumes application-owned approval or execution work.
