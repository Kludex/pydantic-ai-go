package agui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strconv"
	"sync/atomic"

	ai "github.com/Kludex/pydantic-ai-go"
)

var generatedID atomic.Uint64

// Adapter converts between AG-UI requests and one typed agent.
type Adapter[Deps, Output any] struct {
	agent  *ai.Agent[Deps, Output]
	config Config
}

// NewAdapter creates an AG-UI adapter. Client message history is sanitized on every run.
func NewAdapter[Deps, Output any](agent *ai.Agent[Deps, Output], config Config) *Adapter[Deps, Output] {
	if agent == nil {
		panic("agui: agent must not be nil")
	}
	if config.MaxRequestBytes < 0 {
		panic("agui: maximum request bytes must not be negative")
	}
	return &Adapter[Deps, Output]{agent: agent, config: config}
}

// RunStream runs one AG-UI input and emits protocol events.
func (adapter *Adapter[Deps, Output]) RunStream(
	ctx context.Context, input RunAgentInput, deps Deps, options ...ai.RunOption,
) iter.Seq2[Event, error] {
	threadID := input.ThreadID
	if threadID == "" {
		threadID = nextID("thread")
	}
	runID := input.RunID
	if runID == "" {
		runID = nextID("run")
	}
	return func(yield func(Event, error) bool) {
		prompt, history, _, err := PrepareInput(input, adapter.config.Sanitization)
		if err != nil {
			yield(Event{Type: EventRunError, Message: err.Error()}, err)
			return
		}
		runOptions := append([]ai.RunOption(nil), options...)
		runOptions = append(runOptions, ai.WithMessageHistory(history), ai.WithConversationID(threadID))
		stream := adapter.agent.RunStream(ctx, prompt.Content, deps, runOptions...)
		for event, eventErr := range TransformStream(stream.Events(), threadID, runID) {
			if !yield(event, eventErr) || eventErr != nil {
				return
			}
		}
	}
}

// TransformStream converts an existing agent event stream to AG-UI events.
func TransformStream(stream ai.EventStream, threadID string, runID string) iter.Seq2[Event, error] {
	if threadID == "" {
		threadID = nextID("thread")
	}
	if runID == "" {
		runID = nextID("run")
	}
	return func(yield func(Event, error) bool) {
		if !yield(Event{Type: EventRunStarted, ThreadID: threadID, RunID: runID}, nil) {
			return
		}
		transformer := eventTransformer{runID: runID, calls: map[string]bool{}, partCalls: map[string]string{}}
		for event, eventErr := range stream {
			if eventErr != nil {
				if !transformer.closeMessage(yield) {
					return
				}
				yield(Event{Type: EventRunError, Message: eventErr.Error()}, eventErr)
				return
			}
			if err := transformer.emit(yield, event); err != nil {
				if errors.Is(err, errConsumerStopped) {
					return
				}
				yield(Event{Type: EventRunError, Message: err.Error()}, err)
				return
			}
		}
		if !transformer.closeMessage(yield) {
			return
		}
		yield(Event{Type: EventRunFinished, ThreadID: threadID, RunID: runID}, nil)
	}
}

func nextID(kind string) string {
	return "agui-" + kind + "-" + strconv.FormatUint(generatedID.Add(1), 10)
}

func resultContent(part ai.RequestPart) (string, string, error) {
	switch value := part.(type) {
	case ai.ToolReturnPart:
		if text, ok := value.Content.(string); ok {
			return value.ToolCallID, text, nil
		}
		encoded, err := json.Marshal(value.Content)
		if err != nil {
			return "", "", fmt.Errorf("agui: encode tool result: %w", err)
		}
		return value.ToolCallID, string(encoded), nil
	case ai.RetryPromptPart:
		return value.ToolCallID, value.ModelResponse(), nil
	default:
		return "", "", fmt.Errorf("agui: unsupported tool result part %T", part)
	}
}
