package ai_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestRunStreamOutputsTextPartialsAndFinal(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "Hel"},
			ai.TextDeltaEvent{PartID: "text", Delta: "lo"},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	var validationOutputs []string
	var partialFlags []bool
	agent.AddOutputValidator(func(_ context.Context, runContext *ai.RunContext[deps], output string) error {
		validationOutputs = append(validationOutputs, output)
		partialFlags = append(partialFlags, runContext.PartialOutput)
		return nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var outputs []string
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if !slices.Equal(outputs, []string{"Hel", "Hello", "Hello"}) {
		t.Fatalf("unexpected partial/final outputs: %v", outputs)
	}
	if !slices.Equal(validationOutputs, outputs) || !slices.Equal(partialFlags, []bool{true, true, false}) {
		t.Fatalf("unexpected validator calls: outputs=%v partial=%v", validationOutputs, partialFlags)
	}
}

func TestRunStreamOutputsStructuredPartialsAndFinal(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"SF"`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `,"temp_c":1`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `8}`},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, weather](model)
	var validationOutputs []weather
	var partialFlags []bool
	agent.AddOutputValidator(func(_ context.Context, runContext *ai.RunContext[deps], output weather) error {
		validationOutputs = append(validationOutputs, output)
		partialFlags = append(partialFlags, runContext.PartialOutput)
		return nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var outputs []weather
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	want := []weather{{City: "SF", TempC: 1}, {City: "SF", TempC: 18}, {City: "SF", TempC: 18}}
	if !slices.Equal(outputs, want) {
		t.Fatalf("unexpected structured outputs: %+v", outputs)
	}
	if !slices.Equal(validationOutputs, want) || !slices.Equal(partialFlags, []bool{true, true, false}) {
		t.Fatalf("unexpected validator calls: outputs=%+v partial=%v", validationOutputs, partialFlags)
	}
}

func TestRunStreamOutputsSuppressPartialRetries(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"SF","temp_c":1`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `8}`},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, weather](model)
	agent.AddOutputValidator(func(_ context.Context, runContext *ai.RunContext[deps], output weather) error {
		if runContext.PartialOutput && output.TempC == 1 {
			return ai.Retryf("wait")
		}
		return nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var outputs []weather
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if len(outputs) != 2 || outputs[0].TempC != 18 || outputs[1].TempC != 18 {
		t.Fatalf("partial retry was not suppressed: %+v", outputs)
	}
}

func TestRunStreamOutputsPropagatePartialValidatorErrors(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "stop"}, ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddOutputValidator(func(_ context.Context, runContext *ai.RunContext[deps], _ string) error {
		if runContext.PartialOutput {
			return errors.New("partial rejected")
		}
		return nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Outputs() {
		if err != nil {
			got = err
		}
	}
	if got == nil || got.Error() != "ai: partial output validation: partial rejected" || stream.Result() != nil {
		t.Fatalf("unexpected partial validation failure: err=%v result=%+v", got, stream.Result())
	}
}

type partialNested struct {
	Value string `json:"value"`
}

type partialCollection struct {
	Items []partialNested `json:"items"`
}

func TestRunStreamOutputsCompleteNestedArraysAndEscapedStrings(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"items":[{`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `"value":"a\`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `\b"}]}`},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, partialCollection](model).RunStream(t.Context(), "go", deps{})
	var outputs []partialCollection
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if len(outputs) != 4 || len(outputs[0].Items) != 0 || outputs[1].Items[0].Value != `a\` ||
		outputs[2].Items[0].Value != `a\b` || outputs[3].Items[0].Value != `a\b` {
		t.Fatalf("unexpected escaped partial outputs: %#v", outputs)
	}
}

func TestRunStreamOutputsRejectMalformedPartialJSON(t *testing.T) {
	tests := map[string]struct {
		raw string
	}{
		"mismatched object": {raw: `}`},
		"mismatched array":  {raw: `]`},
		"wrong struct type": {raw: `[]`},
		"wrong field type":  {raw: `{"city":1,"temp_c":2}`},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				return []ai.ModelStreamEvent{
					ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
					ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: test.raw}, ai.FinishEvent{},
				}
			})
			stream := ai.NewAgent[deps, weather](model).RunStream(t.Context(), "go", deps{})
			var got error
			for _, err := range stream.Outputs() {
				if err != nil {
					got = err
				}
			}
			if got == nil {
				t.Fatal("expected malformed final output error")
			}
		})
	}

	for name, raw := range map[string]string{
		"wrong array type": `{"items":{}}`, "wrong array item": `{"items":[1]}`,
	} {
		t.Run(name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				return []ai.ModelStreamEvent{
					ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
					ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: raw}, ai.FinishEvent{},
				}
			})
			stream := ai.NewAgent[deps, partialCollection](model).RunStream(t.Context(), "go", deps{})
			var got error
			for _, err := range stream.Outputs() {
				if err != nil {
					got = err
				}
			}
			if got == nil {
				t.Fatal("expected invalid array output error")
			}
		})
	}
}

type malformedPartialEventsCapability struct {
	mode string
}

func (*malformedPartialEventsCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *malformedPartialEventsCapability) ProcessStreamEvent(
	_ context.Context, _ *ai.RunInfo, event ai.StreamEvent,
) (ai.StreamEvent, error) {
	switch value := event.(type) {
	case ai.PartStartEvent:
		switch c.mode {
		case "missing":
			return nil, nil
		case "thinking":
			value.Part = ai.ThinkingPart{Content: "hidden"}
			return value, nil
		}
	case ai.PartDeltaEvent:
		switch c.mode {
		case "missing":
			value.Index = 99
			return value, nil
		case "wrong delta":
			value.Delta = ai.ThinkingPartDelta{ContentDelta: "bad"}
			return value, nil
		}
	}
	return event, nil
}

func TestRunStreamOutputsValidateTransformedLifecycle(t *testing.T) {
	for _, mode := range []string{"missing", "thinking", "wrong delta"} {
		t.Run(mode, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				events := []ai.ModelStreamEvent{ai.TextDeltaEvent{PartID: "text", Delta: "a"}}
				if mode != "thinking" {
					events = append(events, ai.TextDeltaEvent{PartID: "text", Delta: "b"})
				}
				return append(events, ai.FinishEvent{})
			})
			agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(&malformedPartialEventsCapability{mode: mode}))
			stream := agent.RunStream(t.Context(), "go", deps{})
			var got error
			for _, err := range stream.Outputs() {
				if err != nil {
					got = err
				}
			}
			if mode == "wrong delta" && got == nil {
				t.Fatal("expected incompatible transformed delta error")
			}
			if mode != "wrong delta" && got != nil {
				t.Fatalf("unexpected transformed lifecycle error: %v", got)
			}
		})
	}
}

func TestRunStreamOutputsForwardSourceErrors(t *testing.T) {
	stream := ai.NewAgent[deps, string](&streamingErrModel{Model: newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return nil
	})}).RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Outputs() {
		if err != nil {
			got = err
		}
	}
	if got == nil || got.Error() != "mid-stream failure" {
		t.Fatalf("source error was not forwarded: %v", got)
	}
}

func TestRunStreamStructuredOutputsStopOnConsumerBreak(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"SF"`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `,"temp_c":18}`}, ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, weather](model).RunStream(t.Context(), "go", deps{})
	for range stream.Outputs() {
		break
	}
	if stream.Result() != nil {
		t.Fatal("structured partial output consumer break unexpectedly completed the run")
	}
}

func TestRunStreamOutputsStopOnConsumerBreak(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "a"},
			ai.TextDeltaEvent{PartID: "text", Delta: "b"}, ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for range stream.Outputs() {
		break
	}
	if stream.Result() != nil {
		t.Fatal("partial output consumer break unexpectedly completed the run")
	}
}
