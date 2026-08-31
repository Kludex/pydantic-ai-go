package ai

import (
	"context"
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
			structured := r.params.OutputSchema != nil
			output, err := r.validateAndProcessOutput(
				ctx, r.outputRunContext(""), r.outputHookContext(nil, structured, false), response.Text(),
				func(raw any) (decodedOutput, error) { return r.decodeOutput(raw, structured) },
			)
			if err := committedOutputError(err); err != nil {
				return nil, -1, false, err
			}
			return &output, -1, true, nil
		case ToolCallPart:
			callIndex++
			if !r.isOutputCall(part) {
				continue
			}
			call := part
			output, err := r.validateAndProcessOutput(
				ctx, r.outputRunContext(part.ToolCallID), r.outputHookContext(&call, true, false), part.Args,
				func(raw any) (decodedOutput, error) { return r.decodeOutput(raw, true) },
			)
			if err := committedOutputError(err); err != nil {
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
	structured := r.params.OutputSchema != nil || r.currentOutputTool != nil
	runContext := r.outputRunContext(toolCallID)
	runContext.PartialOutput = true
	var call *ToolCallPart
	if toolCallID != "" {
		call = &ToolCallPart{ToolName: outputToolName, ToolCallID: toolCallID, Args: []byte(raw)}
		if r.currentOutputTool != nil {
			call.ToolName = r.currentOutputTool.Name
		}
	}
	output, err := r.validateAndProcessOutput(
		ctx, runContext, r.outputHookContext(call, structured, true), raw,
		func(rawOutput any) (decodedOutput, error) {
			if !structured {
				return r.decodeOutput(rawOutput, false)
			}
			encoded, err := outputBytes(rawOutput)
			if err != nil {
				return decodedOutput{}, err
			}
			output, valid := decodePartialJSON(string(encoded), r.currentOutputValidator, r.decodeOutputBytes)
			if !valid {
				return output, errPartialOutputIncomplete
			}
			return output, nil
		},
	)
	if err == nil {
		return output, true, nil
	}
	var retry *RetryError
	if errors.Is(err, errPartialOutputIncomplete) || errors.As(err, &retry) {
		return output, false, nil
	}
	if validationError, ok := err.(*outputValidationHookError); ok {
		return output, false, fmt.Errorf("ai: partial output validation: %w", validationError.err)
	}
	processingError := err.(*outputProcessingHookError)
	return output, false, fmt.Errorf("ai: partial output processing: %w", processingError.err)
}

var errPartialOutputIncomplete = errors.New("partial output is incomplete")

func committedOutputError(err error) error {
	if err == nil {
		return nil
	}
	var schemaValidation *outputSchemaValidationError
	var decodeError *outputDecodeError
	var retry *RetryError
	if errors.As(err, &schemaValidation) || errors.As(err, &decodeError) || errors.As(err, &retry) {
		return streamOutputValidationError()
	}
	return err
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
		return r.committedParts(outcomes), nil
	}
	if r.agent.endStrategy == EndStrategyGraceful {
		if err := r.checkToolCallLimit(calls); err != nil {
			return nil, err
		}
		batch := make([]int, 0, len(calls))
		for index, call := range calls {
			if !r.isOutputCall(call) && !r.callIsBarrier(call, true) {
				batch = append(batch, index)
				continue
			}
			if err := r.executeCommittedBatch(ctx, calls, outcomes, batch); err != nil {
				return r.committedParts(outcomes), err
			}
			batch = batch[:0]
			if r.isOutputCall(call) {
				outcomes[index] = r.committedCallOutcome(call, index == winningCall)
			} else {
				outcomes[index] = r.executeOneCommitted(ctx, call)
				if outcomes[index].err != nil {
					return r.committedParts(outcomes), outcomes[index].err
				}
			}
		}
		if err := r.executeCommittedBatch(ctx, calls, outcomes, batch); err != nil {
			return r.committedParts(outcomes), err
		}
		if err := r.emitPendingCallResults(outcomes); err != nil {
			return nil, err
		}
		return r.committedParts(outcomes), nil
	}

	if err := r.checkToolCallLimit(calls); err != nil {
		return nil, err
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
		return r.committedParts(outcomes), err
	}
	if err := r.emitPendingCallResults(outcomes); err != nil {
		return nil, err
	}
	return r.committedParts(outcomes), nil
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
			ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeSuccess,
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
				ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeSuccess,
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

func (r *run[Deps, Output]) committedParts(outcomes []callOutcome[Output]) []RequestPart {
	parts := make([]RequestPart, 0, len(outcomes))
	for _, outcome := range outcomes {
		if outcome.part != nil {
			parts = append(parts, outcome.part)
		}
	}
	return append(parts, r.normalizeOutcomeExtraParts(outcomes)...)
}
