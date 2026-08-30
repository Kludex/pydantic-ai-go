package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

const (
	streamValidationFailed = "Output validation failed during streaming, and retries are not supported."
	outputValidationFailed = "Output tool not used - output failed validation."
)

func (r *run[Deps, Output]) streamedOutput(
	ctx context.Context, response *ModelResponse,
) (*Output, int, bool, error) {
	callIndex := -1
	for _, part := range response.Parts {
		switch part := part.(type) {
		case TextPart:
			if !r.params.AllowText {
				continue
			}
			var output Output
			if r.params.OutputSchema != nil {
				if err := json.Unmarshal([]byte(response.Text()), &output); err != nil {
					return nil, -1, false, streamOutputValidationError()
				}
			} else {
				output = any(response.Text()).(Output)
			}
			if err := r.validateStreamedOutput(ctx, r.outputRunContext(""), output); err != nil {
				return nil, -1, false, err
			}
			return &output, -1, true, nil
		case ToolCallPart:
			callIndex++
			if !r.isOutputCall(part) {
				continue
			}
			var output Output
			if err := json.Unmarshal(part.Args, &output); err != nil {
				return nil, callIndex, false, streamOutputValidationError()
			}
			if err := r.validateStreamedOutput(ctx, r.outputRunContext(part.ToolCallID), output); err != nil {
				return nil, callIndex, false, err
			}
			return &output, callIndex, true, nil
		}
	}
	return nil, -1, false, nil
}

func (r *run[Deps, Output]) validatePartialOutput(
	ctx context.Context, raw, toolCallID string,
) (Output, bool, error) {
	var output Output
	if r.params.OutputSchema == nil && r.params.OutputTool == nil {
		output = any(raw).(Output)
	} else {
		schema := r.params.OutputSchema
		if r.params.OutputTool != nil {
			schema = r.params.OutputTool.Schema
		}
		var valid bool
		output, valid = decodePartialJSON[Output](raw, schema)
		if !valid {
			return output, false, nil
		}
	}
	runContext := r.outputRunContext(toolCallID)
	runContext.PartialOutput = true
	for _, validate := range r.agent.outputValidators {
		err := validate(ctx, runContext, output)
		var retry *RetryError
		switch {
		case errors.As(err, &retry):
			return output, false, nil
		case err != nil:
			return output, false, fmt.Errorf("ai: partial output validation: %w", err)
		}
	}
	return output, true, nil
}

func (r *run[Deps, Output]) validateStreamedOutput(
	ctx context.Context, runContext *RunContext[Deps], output Output,
) error {
	for _, validate := range r.agent.outputValidators {
		err := validate(ctx, runContext, output)
		var retry *RetryError
		switch {
		case errors.As(err, &retry):
			return streamOutputValidationError()
		case err != nil:
			return fmt.Errorf("ai: output validation: %w", err)
		}
	}
	return nil
}

func streamOutputValidationError() error {
	return &UnexpectedModelBehaviorError{Message: streamValidationFailed}
}

func (r *run[Deps, Output]) executeCallsWithCommittedOutput(
	ctx context.Context, calls []ToolCallPart, winningCall int,
) ([]RequestPart, error) {
	if !r.emitToolCallEvents(calls) {
		return nil, context.Canceled
	}
	outcomes := make([]callOutcome[Output], len(calls))
	if r.agent.endStrategy == EndStrategyEarly {
		for index, call := range calls {
			outcomes[index] = r.committedCallOutcome(call, index == winningCall)
		}
		if err := r.emitPendingCallResults(outcomes); err != nil {
			return nil, err
		}
		return committedParts(outcomes), nil
	}
	if r.agent.endStrategy == EndStrategyGraceful {
		batch := make([]int, 0, len(calls))
		for index, call := range calls {
			if !r.isOutputCall(call) && !r.callIsBarrier(call, true) {
				batch = append(batch, index)
				continue
			}
			if err := r.executeCommittedBatch(ctx, calls, outcomes, batch); err != nil {
				return committedParts(outcomes), err
			}
			batch = batch[:0]
			if r.isOutputCall(call) {
				outcomes[index] = r.committedCallOutcome(call, index == winningCall)
			} else {
				outcomes[index] = r.executeOneCommitted(ctx, call)
				if outcomes[index].err != nil {
					return committedParts(outcomes), outcomes[index].err
				}
			}
		}
		if err := r.executeCommittedBatch(ctx, calls, outcomes, batch); err != nil {
			return committedParts(outcomes), err
		}
		if err := r.emitPendingCallResults(outcomes); err != nil {
			return nil, err
		}
		return committedParts(outcomes), nil
	}

	indexes := make([]int, 0, len(calls))
	for index, call := range calls {
		if index == winningCall {
			outcomes[index] = r.committedCallOutcome(call, true)
			continue
		}
		indexes = append(indexes, index)
	}
	if err := r.executeCommittedSelected(ctx, calls, outcomes, indexes); err != nil {
		return committedParts(outcomes), err
	}
	if err := r.emitPendingCallResults(outcomes); err != nil {
		return nil, err
	}
	return committedParts(outcomes), nil
}

func (r *run[Deps, Output]) committedCallOutcome(call ToolCallPart, winner bool) callOutcome[Output] {
	content := outputSkipped
	if winner {
		content = finalResultProcessed
	} else if !r.isOutputCall(call) {
		content = toolSkipped
	}
	return callOutcome[Output]{
		part: ToolReturnPart{
			ToolName: call.ToolName, Content: content, ToolCallID: call.ToolCallID,
			Outcome: ToolReturnOutcomeSuccess,
		},
		outputCall:   r.isOutputCall(call),
		functionCall: !r.isOutputCall(call),
	}
}

func (r *run[Deps, Output]) executeOneCommitted(
	ctx context.Context, call ToolCallPart,
) callOutcome[Output] {
	outcome := r.executeOne(ctx, call)
	if !outcome.outputCall {
		return outcome
	}
	if outcome.err != nil {
		if errors.Is(outcome.err, context.Canceled) || errors.Is(outcome.err, context.DeadlineExceeded) {
			return outcome
		}
		return callOutcome[Output]{
			part: ToolReturnPart{
				ToolName: call.ToolName, Content: outputValidationFailed, ToolCallID: call.ToolCallID,
				Outcome: ToolReturnOutcomeSuccess,
			},
			outputCall: true,
		}
	}
	if outcome.output != nil {
		part := outcome.part.(ToolReturnPart)
		part.Content = outputNotFinal
		outcome.part = part
		outcome.output = nil
	}
	return outcome
}

func (r *run[Deps, Output]) executeCommittedSelected(
	ctx context.Context, calls []ToolCallPart, outcomes []callOutcome[Output], indexes []int,
) error {
	batch := make([]int, 0, len(indexes))
	for _, index := range indexes {
		if !r.callIsBarrier(calls[index], true) {
			batch = append(batch, index)
			continue
		}
		if err := r.executeCommittedBatch(ctx, calls, outcomes, batch); err != nil {
			return err
		}
		batch = batch[:0]
		outcomes[index] = r.executeOneCommitted(ctx, calls[index])
		if outcomes[index].err != nil {
			return outcomes[index].err
		}
	}
	return r.executeCommittedBatch(ctx, calls, outcomes, batch)
}

func (r *run[Deps, Output]) executeCommittedBatch(
	ctx context.Context, calls []ToolCallPart, outcomes []callOutcome[Output], indexes []int,
) error {
	if len(indexes) == 0 {
		return nil
	}
	if len(indexes) == 1 {
		outcomes[indexes[0]] = r.executeOneCommitted(ctx, calls[indexes[0]])
		return outcomes[indexes[0]].err
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var waitGroup sync.WaitGroup
	for _, index := range indexes {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			outcomes[index] = r.executeOneCommitted(ctx, calls[index])
			if outcomes[index].err != nil {
				cancel()
			}
		}()
	}
	waitGroup.Wait()
	for _, index := range indexes {
		if outcomes[index].err != nil {
			return outcomes[index].err
		}
	}
	return nil
}

func committedParts[Output any](outcomes []callOutcome[Output]) []RequestPart {
	parts := make([]RequestPart, 0, len(outcomes))
	for _, outcome := range outcomes {
		if outcome.part != nil {
			parts = append(parts, outcome.part)
		}
	}
	return parts
}
