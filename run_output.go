package ai

import (
	"context"
	"slices"
)

// RunAs executes agent with Output as this run's final output type. The agent
// must not have output validators because their input type belongs to the
// agent's declared output type.
func RunAs[Output, Deps, AgentOutput any](
	ctx context.Context,
	agent *Agent[Deps, AgentOutput],
	prompt string,
	deps Deps,
	opts ...RunOption,
) (*RunResult[Output], error) {
	specialized, err := specializeAgentOutput[Output](agent)
	if err != nil {
		return nil, err
	}
	return specialized.Run(ctx, prompt, deps, opts...)
}

// RunPartsAs is RunAs with a multimodal prompt.
func RunPartsAs[Output, Deps, AgentOutput any](
	ctx context.Context,
	agent *Agent[Deps, AgentOutput],
	contents []UserContent,
	deps Deps,
	opts ...RunOption,
) (*RunResult[Output], error) {
	specialized, err := specializeAgentOutput[Output](agent)
	if err != nil {
		return nil, err
	}
	return specialized.RunParts(ctx, contents, deps, opts...)
}

// RunStreamAs executes agent as a streaming run with Output as this run's
// final output type. Setup errors are yielded from StreamedRun.Events.
func RunStreamAs[Output, Deps, AgentOutput any](
	ctx context.Context,
	agent *Agent[Deps, AgentOutput],
	prompt string,
	deps Deps,
	opts ...RunOption,
) *StreamedRun[Output] {
	return runStreamAsPrompt[Output](ctx, agent, UserPromptPart{Content: prompt}, deps, opts)
}

// RunStreamPartsAs is RunStreamAs with a multimodal prompt.
func RunStreamPartsAs[Output, Deps, AgentOutput any](
	ctx context.Context,
	agent *Agent[Deps, AgentOutput],
	contents []UserContent,
	deps Deps,
	opts ...RunOption,
) *StreamedRun[Output] {
	return runStreamAsPrompt[Output](ctx, agent, UserPromptPart{Contents: contents}, deps, opts)
}

func runStreamAsPrompt[Output, Deps, AgentOutput any](
	ctx context.Context,
	agent *Agent[Deps, AgentOutput],
	prompt UserPromptPart,
	deps Deps,
	opts []RunOption,
) *StreamedRun[Output] {
	streamed := &StreamedRun[Output]{}
	streamed.events = func(yield func(StreamEvent, error) bool) {
		specialized, err := specializeAgentOutput[Output](agent)
		if err != nil {
			yield(nil, err)
			return
		}
		inner := specialized.runStreamPrompt(ctx, prompt, deps, opts, true)
		for event, streamErr := range inner.Events() {
			streamed.partialOutput = inner.partialOutput
			if !yield(event, streamErr) {
				return
			}
		}
		streamed.result = inner.result
	}
	return streamed
}

func specializeAgentOutput[Output, Deps, AgentOutput any](
	agent *Agent[Deps, AgentOutput],
) (*Agent[Deps, Output], error) {
	agent.started.Store(true)
	if len(agent.outputValidators) != 0 {
		return nil, ErrOutputTypeOverrideWithValidators
	}
	tools := make([]toolEntry[Deps], len(agent.tools))
	for index, tool := range agent.tools {
		tool.def = cloneToolDefinition(tool.def)
		tools[index] = tool
	}
	settingsLayers := make([]capabilitySettingsLayer, len(agent.capSettings))
	for index, layer := range agent.capSettings {
		settingsLayers[index] = capabilitySettingsLayer{
			static: cloneModelSettingsSlice(layer.static), provider: layer.provider,
		}
	}
	usageLimits := agent.usageLimits
	usageLimits.ToolCallLimit = clonePointer(usageLimits.ToolCallLimit)
	usageLimits.CostLimitUSD = clonePointer(usageLimits.CostLimitUSD)
	specialized := &Agent[Deps, Output]{
		model:              agent.model,
		instructions:       agent.instructions,
		instructionsFuncs:  slices.Clone(agent.instructionsFuncs),
		systemPrompts:      slices.Clone(agent.systemPrompts),
		systemPromptFuncs:  slices.Clone(agent.systemPromptFuncs),
		modelSettingsFuncs: slices.Clone(agent.modelSettingsFuncs),
		modelSelectors:     slices.Clone(agent.modelSelectors),
		modelIDResolvers:   slices.Clone(agent.modelIDResolvers),
		toolsPrepareFuncs:  slices.Clone(agent.toolsPrepareFuncs),
		settings:           agent.settings.Clone(),
		usageLimits:        usageLimits,
		retryLimits:        agent.retryLimits,
		outputMode:         agent.outputMode,
		outputTool:         cloneOutputToolConfig(agent.outputTool),
		promptedTemplate:   agent.promptedTemplate,
		outputToolPrepare:  slices.Clone(agent.outputToolPrepare),
		endStrategy:        agent.endStrategy,
		sequentialTools:    agent.sequentialTools,
		capabilities:       slices.Clone(agent.capabilities),
		capInstructions:    slices.Clone(agent.capInstructions),
		capSettings:        settingsLayers,
		tools:              tools,
		toolsets:           slices.Clone(agent.toolsets),
	}
	return specialized, nil
}

func cloneModelSettingsSlice(settings []ModelSettings) []ModelSettings {
	cloned := make([]ModelSettings, len(settings))
	for index, value := range settings {
		cloned[index] = value.Clone()
	}
	return cloned
}
