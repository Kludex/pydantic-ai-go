package ai

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

const outputToolName = "final_result"

const defaultPromptedOutputTemplate = `Always respond with a JSON object that's compatible with this schema:

{schema}

Don't include any text or Markdown fencing before or after.`

type toolValidateFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps], rawArgs json.RawMessage,
) (any, error)

type toolExecuteFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps], args any,
) (any, error)

// Run executes the agent loop: send the conversation to the model, execute
// any tool calls, repeat until the model produces a final output.
func (a *Agent[Deps, Output]) Run(ctx context.Context, prompt string, deps Deps, opts ...RunOption) (*RunResult[Output], error) {
	return a.runPrompt(ctx, UserPromptPart{Content: prompt}, deps, opts)
}

// RunParts is Run with a multimodal prompt: text, image URLs, and inline
// binary data. Providers reject content kinds they do not support.
func (a *Agent[Deps, Output]) RunParts(ctx context.Context, contents []UserContent, deps Deps, opts ...RunOption) (*RunResult[Output], error) {
	return a.runPrompt(ctx, UserPromptPart{Contents: contents}, deps, opts)
}

// Resume continues a provider response left suspended in history without
// adding a new user prompt. History must end with ModelResponseStateSuspended.
func (a *Agent[Deps, Output]) Resume(
	ctx context.Context, history []ModelMessage, deps Deps, opts ...RunOption,
) (*RunResult[Output], error) {
	return a.runPrompt(ctx, UserPromptPart{}, deps, suspendedRunOptions(history, opts))
}

func suspendedRunOptions(history []ModelMessage, opts []RunOption) []RunOption {
	options := slices.Clone(opts)
	options = append(options, func(cfg *runConfig) {
		cfg.history = cloneModelMessages(history)
		cfg.resumeSuspended = true
	})
	return options
}

func (a *Agent[Deps, Output]) runPrompt(ctx context.Context, prompt UserPromptPart, deps Deps, opts []RunOption) (result *RunResult[Output], err error) {
	cfg := buildRunConfig(opts)
	capabilities := append(slices.Clone(a.capabilities), cfg.capabilities...)
	if hasEventStreamCapability(capabilities) {
		stream := a.runStreamPrompt(ctx, prompt, deps, opts, false)
		for _, streamErr := range stream.Events() {
			if streamErr != nil {
				return nil, streamErr
			}
		}
		if stream.Result() == nil {
			return nil, &UnexpectedModelBehaviorError{Message: "event stream ended before the run completed"}
		}
		return stream.Result(), nil
	}
	model := a.model
	if cfg.model != nil {
		model = cfg.model
	}
	ctx, span := startRunSpan(ctx, modelName(model), !hasInstrumentationCapability(capabilities))
	defer func() {
		if result != nil {
			recordUsage(span, result.usage)
		}
		endSpan(span, err)
	}()
	a.started.Store(true)
	r, err := a.newRun(ctx, prompt, deps, cfg)
	if err != nil {
		return nil, err
	}
	r.recordSelectedModel = func(name string) { recordRunModel(span, name) }
	defer func() {
		closeErr := r.closeRunResources(context.WithoutCancel(ctx))
		r.cancellation.finish()
		if closeErr != nil {
			result = nil
			err = errors.Join(err, closeErr)
		}
	}()
	return r.wrappedLoop(r.ctx)
}

func buildRunConfig(opts []RunOption) runConfig {
	var cfg runConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func (a *Agent[Deps, Output]) newRun(
	ctx context.Context, prompt UserPromptPart, deps Deps, cfg runConfig,
) (*run[Deps, Output], error) {
	a.started.Store(true)
	runCtx, cancel := context.WithCancelCause(ctx)
	cancellation := &runCancellation{cancel: cancel, active: true}
	model := a.model
	if cfg.model != nil {
		model = cfg.model
	}
	selectionModes := 0
	if cfg.model != nil {
		selectionModes++
	}
	if cfg.modelID != "" {
		selectionModes++
	}
	if len(cfg.modelSelectors) > 0 {
		selectionModes++
	}
	if selectionModes > 1 {
		cancellation.finish()
		return nil, fmt.Errorf("ai: run model, model ID, and model selector are mutually exclusive")
	}
	availableCapabilities := append(slices.Clone(a.capabilities), cfg.capabilities...)
	runCapabilities, err := sortCapabilities(cfg.capabilities, availableCapabilities)
	if err != nil {
		cancellation.finish()
		return nil, fmt.Errorf("ai: run capability ordering: %w", err)
	}
	capabilities := append(slices.Clone(a.capabilities), runCapabilities...)
	runCapabilityInstructions := []string(nil)
	capSettings := slices.Clone(a.capSettings)
	var runCapabilityTools []capabilityTool
	for _, capability := range runCapabilities {
		registry := &CapabilityRegistry{}
		if err := capability.Setup(registry); err != nil {
			cancellation.finish()
			return nil, fmt.Errorf("ai: run capability setup: %w", err)
		}
		runCapabilityInstructions = append(runCapabilityInstructions, registry.instructions...)
		runCapabilityTools = append(runCapabilityTools, registry.tools...)
		capSettings = append(capSettings, capabilitySettingsLayer{
			static: registry.modelSettings, provider: capabilityModelSettingsProvider(capability),
		})
	}
	limits := a.usageLimits
	if cfg.usageLimits != nil {
		limits = *cfg.usageLimits
	}
	if limits != (UsageLimits{}) {
		limitCapability := usageLimitsCapability{limits: limits}
		_ = limitCapability.Setup(&CapabilityRegistry{})
		capabilities = append([]Capability{limitCapability}, capabilities...)
	}
	r := &run[Deps, Output]{
		agent: a, model: model, capabilities: capabilities, ctx: runCtx, cancellation: cancellation,
		retryLimits: a.retryLimits, toolRetries: make(map[string]int), availabilityRefused: make(map[string]struct{}),
		runSettings: cfg.settings, usageLimits: limits,
		tools: slices.Clone(a.tools), toolsets: slices.Clone(a.toolsets), capSettings: capSettings,
		runSettingsFuncs: slices.Clone(cfg.settingsFuncs), runInstructionsFuncs: slices.Clone(cfg.instructionsFuncs),
		explicitRunModel: cfg.model != nil, staticModelID: cfg.modelID,
		runModelSelectors: slices.Clone(cfg.modelSelectors), resolvedModels: make(map[string]Model),
		pendingMessages: &pendingMessageQueue{},
	}
	if cfg.deferredResults != nil {
		results := cloneDeferredToolResults(*cfg.deferredResults)
		r.deferredResults = &results
	}
	if cfg.retryLimits != nil {
		validateRetryLimits(*cfg.retryLimits)
		r.retryLimits = *cfg.retryLimits
	}
	r.outputTool = cloneOutputToolConfig(a.outputTool)
	if cfg.outputTool != nil {
		r.outputTool = cloneOutputToolConfig(*cfg.outputTool)
	}
	r.outputMaxRetries = r.retryLimits.Output
	if r.outputTool.MaxRetries != nil {
		r.outputMaxRetries = *r.outputTool.MaxRetries
	}
	for _, erased := range cfg.toolsets {
		toolset, ok := erased.(Toolset[Deps])
		if !ok {
			cancellation.finish()
			return nil, fmt.Errorf("ai: run toolset dependencies do not match agent")
		}
		r.toolsets = append(r.toolsets, toolset)
	}
	toolNames := make(map[string]struct{}, len(r.tools)+len(cfg.tools))
	for _, entry := range r.tools {
		toolNames[entry.def.Name] = struct{}{}
	}
	for _, erased := range cfg.tools {
		entry, ok := erased.entry.(toolEntry[Deps])
		if !ok {
			cancellation.finish()
			return nil, fmt.Errorf("ai: run tool dependencies do not match agent")
		}
		if _, exists := toolNames[entry.def.Name]; exists {
			cancellation.finish()
			return nil, fmt.Errorf("ai: duplicate run tool name %q", entry.def.Name)
		}
		entry.def = cloneToolDefinition(entry.def)
		r.tools = append(r.tools, entry)
		toolNames[entry.def.Name] = struct{}{}
	}
	for _, tool := range runCapabilityTools {
		if _, exists := toolNames[tool.def.Name]; exists {
			cancellation.finish()
			return nil, fmt.Errorf("ai: duplicate run capability tool name %q", tool.def.Name)
		}
		call := tool.call
		r.tools = append(r.tools, toolEntry[Deps]{
			def: cloneToolDefinition(tool.def),
			validate: func(_ context.Context, _ *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
				return slices.Clone(rawArgs), nil
			},
			execute: func(ctx context.Context, _ *RunContext[Deps], validated any) (any, error) {
				rawArgs, ok := validated.(json.RawMessage)
				if !ok {
					return nil, fmt.Errorf(
						"validated arguments for tool %q have type %T, expected json.RawMessage", tool.def.Name, validated,
					)
				}
				return call(ctx, rawArgs)
			},
		})
		toolNames[tool.def.Name] = struct{}{}
	}
	history := dropOrphanedToolResults(cfg.history)
	restoredPending, err := restorePendingMessages(history)
	if err != nil {
		cancellation.finish()
		return nil, err
	}
	for _, pending := range restoredPending {
		r.pendingMessages.add(pending)
	}
	var interruptedReturns []RequestPart
	var resumeSeed *ModelResponse
	if cfg.resumeSuspended {
		history = cloneModelMessages(history)
		if !historyEndsSuspended(history) {
			cancellation.finish()
			return nil, ErrNoSuspendedResponse
		}
		seed := history[len(history)-1].(ModelResponse)
		resumeSeed = cloneModelResponse(&seed)
	} else {
		history, interruptedReturns = repairDanglingToolCalls(history, cfg.deferredResults != nil)
	}
	runID := cfg.runID
	if runID == "" {
		runID = newRunID()
	} else if historyContainsRunID(history, runID) {
		cancellation.finish()
		return nil, fmt.Errorf("ai: run ID %q already appears in message history", runID)
	}
	conversationID := latestConversationID(history)
	if cfg.conversationID != nil {
		conversationID = *cfg.conversationID
		if conversationID == "new" {
			conversationID = newRunID()
		}
	}
	if conversationID == "" {
		conversationID = newRunID()
	}
	if resumeSeed != nil {
		history = history[:len(history)-1]
	}
	history = mergeConsecutiveMessages(history)
	r.revealedTools = revealedToolNames(history)
	r.messages = append(r.messages, history...)
	r.newMessages = len(r.messages)
	settings := mergeModelSettings(a.settings, cfg.settings)
	r.resumeSeed = resumeSeed
	r.rc = &RunContext[Deps]{
		Deps: deps, MaxRetries: r.outputMaxRetries, RunID: runID, ConversationID: conversationID,
		Model: model, ModelSettings: settings, UsageLimits: limits,
		usage: &r.usage, toolCalls: &r.toolCalls, messages: &r.messages,
		revealedTools: &r.revealedTools, pendingMessages: r.pendingMessages, cancellation: cancellation,
	}
	r.info = &RunInfo{
		RunID: runID, ConversationID: conversationID,
		usage: &r.usage, toolCalls: &r.toolCalls, messages: &r.messages, newMessages: r.newMessages,
		model: func() Model { return r.model },
	}
	r.staticInstructions = a.staticInstructions(cfg.instructions, runCapabilityInstructions)
	outputMode := a.outputMode
	if cfg.outputMode != nil {
		outputMode = *cfg.outputMode
	}
	validateOutputMode(outputMode)
	promptedTemplate := a.promptedTemplate
	if cfg.promptedTemplate != nil {
		promptedTemplate = *cfg.promptedTemplate
	}
	r.baseParams, err = a.buildParams(r.staticInstructions, settings, outputMode, r.tools)
	r.params = r.baseParams
	r.promptedTemplate = promptedTemplate
	if err != nil {
		cancellation.finish()
		return nil, err
	}
	if err := r.openToolsets(runCtx); err != nil {
		cancellation.finish()
		return nil, err
	}
	if !cfg.resumeSuspended {
		requestParts := slices.Clone(interruptedReturns)
		requestParts = append(requestParts, prompt)
		requestParts = stampRequestParts(requestParts, time.Now().UTC())
		r.messages = append(r.messages, ModelRequest{
			Parts: requestParts, RunID: runID, ConversationID: conversationID,
		})
	}
	return r, nil
}

func historyEndsSuspended(messages []ModelMessage) bool {
	if len(messages) == 0 {
		return false
	}
	response, ok := messages[len(messages)-1].(ModelResponse)
	return ok && response.State == ModelResponseStateSuspended
}

func historyContainsRunID(messages []ModelMessage, runID string) bool {
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			if message.RunID == runID {
				return true
			}
		case ModelResponse:
			if message.RunID == runID {
				return true
			}
		}
	}
	return false
}

func latestConversationID(messages []ModelMessage) string {
	for index := len(messages) - 1; index >= 0; index-- {
		switch message := messages[index].(type) {
		case ModelRequest:
			if message.ConversationID != "" {
				return message.ConversationID
			}
		case ModelResponse:
			if message.ConversationID != "" {
				return message.ConversationID
			}
		}
	}
	return ""
}

func dropOrphanedToolResults(messages []ModelMessage) []ModelMessage {
	seenCalls := map[string]struct{}{}
	repaired := make([]ModelMessage, 0, len(messages))
	for index, message := range messages {
		switch message := message.(type) {
		case ModelResponse:
			for _, call := range message.ToolCalls() {
				seenCalls[toolCallMatchKey(call.ToolName, call.ToolCallID)] = struct{}{}
			}
			repaired = append(repaired, message)
		case ModelRequest:
			parts := make([]RequestPart, 0, len(message.Parts))
			for _, part := range message.Parts {
				toolName, toolCallID, isResult := toolResultIdentity(part)
				if isResult {
					if _, seen := seenCalls[toolCallMatchKey(toolName, toolCallID)]; !seen {
						continue
					}
				}
				parts = append(parts, part)
			}
			if len(parts) > 0 || index == len(messages)-1 {
				message.Parts = parts
				repaired = append(repaired, message)
			}
		}
	}
	return repaired
}

func mergeConsecutiveMessages(messages []ModelMessage) []ModelMessage {
	merged := make([]ModelMessage, 0, len(messages))
	for _, message := range messages {
		if len(merged) == 0 {
			merged = append(merged, message)
			continue
		}
		switch message := message.(type) {
		case ModelRequest:
			previous, ok := merged[len(merged)-1].(ModelRequest)
			if !ok || previous.Instructions != "" && message.Instructions != "" &&
				previous.Instructions != message.Instructions {
				merged = append(merged, message)
				continue
			}
			combined := append(slices.Clone(previous.Parts), message.Parts...)
			parts := make([]RequestPart, 0, len(combined))
			for _, part := range combined {
				if _, _, isResult := toolResultIdentity(part); isResult {
					parts = append(parts, part)
				}
			}
			for _, part := range combined {
				if _, _, isResult := toolResultIdentity(part); !isResult {
					parts = append(parts, part)
				}
			}
			instructions := previous.Instructions
			if instructions == "" {
				instructions = message.Instructions
			}
			timestamp := message.Timestamp
			if timestamp.IsZero() {
				timestamp = previous.Timestamp
			}
			merged[len(merged)-1] = ModelRequest{
				Parts: parts, Timestamp: timestamp, Instructions: instructions,
			}
		case ModelResponse:
			previous, ok := merged[len(merged)-1].(ModelResponse)
			if !ok || previous.ModelName != "" || message.ModelName != "" ||
				previous.ProviderName != "" || message.ProviderName != "" ||
				previous.ProviderResponseID != "" || message.ProviderResponseID != "" {
				merged = append(merged, message)
				continue
			}
			previous.Parts = append(slices.Clone(previous.Parts), message.Parts...)
			merged[len(merged)-1] = previous
		}
	}
	return merged
}

type trackedToolCall struct {
	responseIndex int
	call          ToolCallPart
	active        bool
	dangling      bool
}

func repairDanglingToolCalls(messages []ModelMessage, preserveDangling bool) ([]ModelMessage, []RequestPart) {
	tracked := make([]*trackedToolCall, 0)
	open := map[string]*trackedToolCall{}
	for index, message := range messages {
		switch message := message.(type) {
		case ModelResponse:
			for _, call := range message.ToolCalls() {
				key := toolCallMatchKey(call.ToolName, call.ToolCallID)
				if shadowed := open[key]; shadowed != nil {
					shadowed.active = false
					shadowed.dangling = true
				}
				item := &trackedToolCall{responseIndex: index, call: call, active: true}
				tracked = append(tracked, item)
				open[key] = item
			}
		case ModelRequest:
			for _, part := range message.Parts {
				toolName, toolCallID, ok := toolResultIdentity(part)
				if !ok {
					continue
				}
				key := toolCallMatchKey(toolName, toolCallID)
				if item := open[key]; item != nil {
					item.active = false
					delete(open, key)
				}
			}
		}
	}
	if preserveDangling {
		return messages, nil
	}
	dangling := map[int][]RequestPart{}
	for _, item := range tracked {
		if item.active || item.dangling {
			call := item.call
			dangling[item.responseIndex] = append(dangling[item.responseIndex], ToolReturnPart{
				ToolName: call.ToolName, ToolCallID: call.ToolCallID, ToolKind: call.ToolKind,
				Content:  "The tool call was interrupted before a result was produced.",
				Outcome:  ToolReturnOutcomeInterrupted,
				Metadata: map[string]any{SynthesizedToolReturnMetadataKey: true},
			})
		}
	}
	if len(dangling) == 0 {
		return slices.Clone(messages), nil
	}
	repaired := make([]ModelMessage, 0, len(messages)+len(dangling))
	var pending []RequestPart
	for index, message := range messages {
		switch message := message.(type) {
		case ModelResponse:
			if len(pending) > 0 {
				repaired = append(repaired, ModelRequest{Parts: pending})
			}
			repaired = append(repaired, message)
			pending = dangling[index]
		case ModelRequest:
			if len(pending) > 0 {
				parts := slices.Clone(message.Parts)
				insertAt := 0
				for partIndex, part := range parts {
					if _, _, ok := toolResultIdentity(part); ok {
						insertAt = partIndex + 1
					}
				}
				parts = slices.Insert(parts, insertAt, pending...)
				message.Parts = parts
				pending = nil
			}
			repaired = append(repaired, message)
		}
	}
	return repaired, pending
}

func toolCallMatchKey(toolName, toolCallID string) string {
	if toolCallID != "" {
		return "id:" + toolCallID
	}
	return "name:" + toolName
}

func toolResultIdentity(part RequestPart) (toolName, toolCallID string, ok bool) {
	switch part := part.(type) {
	case ToolReturnPart:
		return part.ToolName, part.ToolCallID, true
	case RetryPromptPart:
		return part.ToolName, part.ToolCallID, part.ToolName != ""
	default:
		return "", "", false
	}
}

type run[Deps, Output any] struct {
	agent                      *Agent[Deps, Output]
	model                      Model
	capabilities               []Capability
	ctx                        context.Context
	cancellation               *runCancellation
	rc                         *RunContext[Deps]
	info                       *RunInfo
	params                     ModelRequestParams
	baseParams                 ModelRequestParams
	promptedTemplate           string
	messages                   []ModelMessage
	newMessages                int
	usage                      Usage
	toolCalls                  atomic.Int64
	retryLimits                RetryLimits
	outputTool                 OutputToolConfig
	outputMaxRetries           int
	toolRetries                map[string]int
	availabilityRefused        map[string]struct{}
	outputRetry                int
	retriesMu                  sync.Mutex
	currentTools               map[string]struct{}
	currentDeferredTools       map[string]struct{}
	currentDeferredDefinitions map[string]ToolDefinition
	revealedTools              map[string]struct{}
	currentToolValidators      map[string]*schema.Validator
	currentOutputTool          *ToolDefinition
	currentOutputValidator     *schema.Validator
	tools                      []toolEntry[Deps]
	toolsets                   []Toolset[Deps]
	toolsetClosers             []ToolsetCloseFunc
	currentToolEntries         map[string]toolEntry[Deps]
	staticInstructions         []InstructionPart
	systemPromptsPrepared      bool
	capSettings                []capabilitySettingsLayer
	runSettings                *ModelSettings
	usageLimits                UsageLimits
	runSettingsFuncs           []erasedModelSettingsFunc
	runInstructionsFuncs       []erasedInstructionsFunc
	explicitRunModel           bool
	staticModelID              string
	runModelSelectors          []erasedModelSelectorFunc
	resolvedModels             map[string]Model
	resumeSeed                 *ModelResponse
	detachedResponse           *ModelResponse
	enteredModels              []Model
	modelClosers               []ModelCloseFunc
	deferredResults            *DeferredToolResults
	resolvingDeferred          map[string]deferredResolution
	pendingDeferred            *DeferredToolRequests
	pendingMessages            *pendingMessageQueue
	runStep                    int
	// emit forwards stream events during streamed model execution.
	emit                 func(StreamEvent) bool
	emitMu               sync.Mutex
	commitStreamedOutput bool
	recordSelectedModel  func(string)
	observeUsage         func(Usage)
	usagePublishMu       sync.Mutex
}

func (r *run[Deps, Output]) openToolsets(ctx context.Context) error {
	resolved := make([]Toolset[Deps], len(r.toolsets))
	for index, toolset := range r.toolsets {
		var err error
		resolved[index], err = toolsetForRun(ctx, r.rc, toolset)
		if err != nil {
			return fmt.Errorf("ai: toolset for run: %w", err)
		}
	}
	opened := make([]Toolset[Deps], len(resolved))
	for index, toolset := range resolved {
		var closeFunc ToolsetCloseFunc
		var err error
		opened[index], closeFunc, err = openToolset(ctx, r.rc, toolset)
		if err != nil {
			closeErr := closeToolsetFuncs(context.WithoutCancel(ctx), r.toolsetClosers)
			r.toolsetClosers = nil
			return errors.Join(fmt.Errorf("ai: open toolset: %w", err), closeErr)
		}
		if closeFunc != nil {
			r.toolsetClosers = append(r.toolsetClosers, closeFunc)
		}
	}
	r.toolsets = opened
	return nil
}

func (r *run[Deps, Output]) prepareToolsetsForStep(ctx context.Context, rc *RunContext[Deps]) error {
	for index, toolset := range r.toolsets {
		resolved, err := toolsetForRunStep(ctx, rc, toolset)
		if err != nil {
			return fmt.Errorf("ai: toolset for run step: %w", err)
		}
		r.toolsets[index] = resolved
	}
	return nil
}

func (r *run[Deps, Output]) openSelectedModel(ctx context.Context) error {
	for _, entered := range r.enteredModels {
		if sameModelInstance(entered, r.model) {
			return nil
		}
	}
	opener, ok := r.model.(ModelOpener)
	if !ok {
		r.enteredModels = append(r.enteredModels, r.model)
		return nil
	}
	closeFunc, err := opener.OpenModel(ctx)
	if err != nil {
		return fmt.Errorf("ai: open model %q: %w", r.model.Name(), err)
	}
	r.enteredModels = append(r.enteredModels, r.model)
	if closeFunc != nil {
		r.modelClosers = append(r.modelClosers, closeFunc)
	}
	return nil
}

func (r *run[Deps, Output]) closeModels(ctx context.Context) error {
	closers := r.modelClosers
	r.modelClosers = nil
	var errs []error
	for index := len(closers) - 1; index >= 0; index-- {
		if err := closers[index](ctx); err != nil {
			errs = append(errs, err)
		}
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("ai: close model: %w", err)
	}
	return nil
}

func (r *run[Deps, Output]) closeRunResources(ctx context.Context) error {
	return errors.Join(r.closeModels(ctx), r.closeToolsets(ctx))
}

func (r *run[Deps, Output]) closeToolsets(ctx context.Context) error {
	closers := r.toolsetClosers
	r.toolsetClosers = nil
	if err := closeToolsetFuncs(ctx, closers); err != nil {
		return fmt.Errorf("ai: close toolset: %w", err)
	}
	return nil
}

func (r *run[Deps, Output]) selectModel(ctx context.Context) error {
	r.runStep++
	r.rc.RunStep = r.runStep
	if r.explicitRunModel {
		r.rc.Model = r.model
		r.rc.ModelID = ""
		return nil
	}
	if r.staticModelID != "" {
		return r.applyModelSelection(ctx, ModelSelection{ID: r.staticModelID})
	}
	if len(r.runModelSelectors) > 0 {
		for _, selector := range r.runModelSelectors {
			selection, err := selector(ctx, r.modelSelectionContext())
			if err != nil {
				return fmt.Errorf("ai: select model: %w", err)
			}
			if err := r.applyModelSelection(ctx, selection); err != nil {
				return err
			}
		}
		return nil
	}
	for _, selector := range r.agent.modelSelectors {
		selection, err := selector(ctx, r.modelSelectionContext())
		if err != nil {
			return fmt.Errorf("ai: select model: %w", err)
		}
		if err := r.applyModelSelection(ctx, selection); err != nil {
			return err
		}
	}
	for _, capability := range r.capabilities {
		provider, ok := capability.(ModelSelectionProvider)
		if !ok {
			continue
		}
		selection, err := provider.SelectModel(ctx, r.info, r.modelSelectionInfo())
		if err != nil {
			return fmt.Errorf("ai: select model: %w", err)
		}
		if err := r.applyModelSelection(ctx, selection); err != nil {
			return err
		}
	}
	return nil
}

func (r *run[Deps, Output]) modelSelectionContext() ModelSelectionContext[Deps] {
	return ModelSelectionContext[Deps]{
		Deps: r.rc.Deps, Model: r.model, ModelID: r.rc.ModelID, Step: r.runStep,
		Messages: r.modelSelectionMessages(), Usage: r.info.Usage(),
	}
}

func (r *run[Deps, Output]) modelSelectionInfo() ModelSelectionInfo {
	return ModelSelectionInfo{
		Model: r.model, ModelID: r.rc.ModelID, Step: r.runStep,
		Messages: r.modelSelectionMessages(), Usage: r.info.Usage(),
	}
}

func (r *run[Deps, Output]) modelSelectionMessages() []ModelMessage {
	messages := r.messages
	if len(messages) > 0 {
		if _, pending := messages[len(messages)-1].(ModelRequest); pending {
			messages = messages[:len(messages)-1]
		}
	}
	return cloneModelMessages(messages)
}

func cloneModelMessages(messages []ModelMessage) []ModelMessage {
	cloned := make([]ModelMessage, len(messages))
	for index, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			message.Parts = slices.Clone(message.Parts)
			message.Metadata = cloneSchemaMap(message.Metadata)
			for partIndex, part := range message.Parts {
				switch part := part.(type) {
				case UserPromptPart:
					part.Contents = cloneUserContents(part.Contents)
					message.Parts[partIndex] = part
				case ToolReturnPart:
					part.Content = cloneSchemaValue(part.Content)
					part.Metadata = cloneSchemaMap(part.Metadata)
					message.Parts[partIndex] = part
				case ToolAvailabilityDeltaPart:
					part.ToolsAdded = slices.Clone(part.ToolsAdded)
					message.Parts[partIndex] = part
				case RetryPromptPart:
					part.Errors = cloneDeferredValidationErrors(part.Errors)
					message.Parts[partIndex] = part
				}
			}
			cloned[index] = message
		case ModelResponse:
			cloned[index] = *cloneModelResponse(&message)
		}
	}
	return cloned
}

func revealedToolNames(messages []ModelMessage) map[string]struct{} {
	revealed := make(map[string]struct{})
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			for _, requestPart := range message.Parts {
				part, ok := requestPart.(ToolAvailabilityDeltaPart)
				if !ok {
					continue
				}
				for _, name := range part.ToolsAdded {
					revealed[name] = struct{}{}
				}
			}
		case ModelResponse:
			for _, responsePart := range message.Parts {
				switch part := responsePart.(type) {
				case CompactionPart:
					clear(revealed)
				case NativeToolReturnPart:
					if part.ToolKind == ToolPartKindToolSearch {
						for _, name := range toolSearchResultNames(part.Content) {
							revealed[name] = struct{}{}
						}
					}
				}
			}
		}
	}
	return revealed
}

func cloneUserContents(contents []UserContent) []UserContent {
	cloned := slices.Clone(contents)
	for index, content := range cloned {
		if binary, ok := content.(BinaryContent); ok {
			binary.Data = slices.Clone(binary.Data)
			cloned[index] = binary
		}
	}
	return cloned
}

func (r *run[Deps, Output]) applyModelSelection(ctx context.Context, selection ModelSelection) error {
	if selection.Model != nil && modelIsNil(selection.Model) {
		return fmt.Errorf("ai: selected model must not be nil")
	}
	if selection.Model != nil && selection.ID != "" {
		return fmt.Errorf("ai: model selection must contain either Model or ID, not both")
	}
	if selection.Model == nil && selection.ID == "" {
		return nil
	}
	if selection.ID != "" {
		model, err := r.resolveModelID(ctx, selection.ID)
		if err != nil {
			return err
		}
		r.model = model
		r.rc.Model = model
		r.rc.ModelID = selection.ID
		return nil
	}
	r.model = selection.Model
	r.rc.Model = selection.Model
	r.rc.ModelID = ""
	return nil
}

func (r *run[Deps, Output]) resolveModelID(ctx context.Context, modelID string) (Model, error) {
	if model, ok := r.resolvedModels[modelID]; ok {
		return model, nil
	}
	resolution := ModelResolutionContext[Deps]{Deps: r.rc.Deps}
	for _, resolver := range r.agent.modelIDResolvers {
		model, err := resolver(ctx, resolution, modelID)
		if err != nil {
			return nil, fmt.Errorf("ai: resolve model ID %q: %w", modelID, err)
		}
		if !modelIsNil(model) {
			r.resolvedModels[modelID] = model
			return model, nil
		}
	}
	for _, capability := range r.capabilities {
		resolver, ok := capability.(ModelIDResolver)
		if !ok {
			continue
		}
		model, err := resolver.ResolveModelID(ctx, r.info, modelID)
		if err != nil {
			return nil, fmt.Errorf("ai: resolve model ID %q: %w", modelID, err)
		}
		if !modelIsNil(model) {
			r.resolvedModels[modelID] = model
			return model, nil
		}
	}
	return nil, &UnknownModelIDError{ID: modelID}
}

// modelRequest is the model-request interception point: tracing plus
// capability middleware (ModelRequestWrapper), outermost first.
func (r *run[Deps, Output]) modelRequest(ctx context.Context) (*ModelResponse, error) {
	r.revealedTools = revealedToolNames(r.messages)
	if r.deferredResults == nil {
		if err := r.deliverPendingMessages(PendingMessageASAP); err != nil {
			return nil, err
		}
	}
	if err := r.selectModel(ctx); err != nil {
		return nil, err
	}
	if modelIsNil(r.model) {
		return nil, ErrNoModel
	}
	if err := r.openSelectedModel(ctx); err != nil {
		return nil, err
	}
	if r.recordSelectedModel != nil {
		r.recordSelectedModel(r.model.Name())
	}
	inner := func(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
		setLatestRequestContext(msgs, params.Instructions, r.rc.RunID, r.rc.ConversationID)
		if r.resumeSeed == nil {
			setLatestRequestContext(r.messages, params.Instructions, r.rc.RunID, r.rc.ConversationID)
		}
		r.setCurrentTools(params)
		if hasInstrumentedModel(r.model) || modelRequestSpanActive(ctx) {
			return r.doModelRequest(ctx, msgs, params)
		}
		reqCtx, reqSpan := startRequestSpan(ctx, r.model.Name())
		resp, err := r.doModelRequest(reqCtx, msgs, params)
		if err != nil {
			endSpan(reqSpan, err)
			return resp, err
		}
		recordUsage(reqSpan, resp.Usage)
		endSpan(reqSpan, nil)
		return resp, nil
	}
	params, err := r.prepareModelParams(ctx)
	if err != nil {
		return nil, err
	}
	r.params = params
	if r.resumeSeed == nil {
		setLatestRequestContext(r.messages, params.Instructions, r.rc.RunID, r.rc.ConversationID)
	}
	next := inner
	for i := len(r.capabilities) - 1; i >= 0; i-- {
		if wrapper, ok := r.capabilities[i].(ModelRequestWrapper); ok {
			innerNext := next
			next = func(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
				return wrapper.WrapModelRequest(ctx, r.info, msgs, params, innerNext)
			}
		}
	}
	r.setCurrentTools(params)
	if r.deferredResults != nil {
		parts, err := r.resolveDeferredToolResults(ctx, *r.deferredResults)
		if err != nil {
			return nil, err
		}
		r.deferredResults = nil
		if r.pendingDeferred != nil {
			if !r.emitStreamEvent(DeferredToolRequestsEvent{Requests: r.pendingDeferred.Clone()}) {
				return nil, context.Canceled
			}
			resolvedParts, remaining, err := r.handleDeferredToolCalls(ctx, *r.pendingDeferred)
			if err != nil {
				return nil, err
			}
			parts = append(parts, resolvedParts...)
			r.pendingDeferred = remaining
		}
		r.prependLatestRequestParts(parts)
		if r.pendingDeferred != nil {
			r.removeLatestUserPrompts()
			return nil, nil
		}
		if err := r.refreshRevealedDeferredTools(&params); err != nil {
			return nil, err
		}
	}
	if err := r.deliverPendingMessages(PendingMessageASAP); err != nil {
		return nil, err
	}
	requestMessages := r.messages
	if r.resumeSeed != nil {
		requestMessages = append(slices.Clone(requestMessages), *cloneModelResponse(r.resumeSeed))
	}
	request := ModelRequestContext{
		Model: r.model, ModelID: r.rc.ModelID, Messages: requestMessages, Params: params, Streaming: r.emit != nil,
	}.Clone()
	for _, capability := range r.capabilities {
		hook, ok := capability.(BeforeModelRequestHook)
		if !ok {
			continue
		}
		request, err = hook.BeforeModelRequest(ctx, r.info, request)
		if err != nil {
			return nil, err
		}
	}
	if request.ReplaceHistory {
		r.messages = cloneModelMessages(request.Messages)
		r.newMessages = 0
	}
	if !request.AdditionalUsage.IsZero() {
		r.usage.Add(request.AdditionalUsage)
		r.publishUsage(nil)
		if err := r.usageLimits.check(r.info.Usage()); err != nil {
			return nil, err
		}
	}
	if request.Params.Settings.RequestTimeout < 0 {
		return nil, fmt.Errorf(
			"ai: request timeout must be non-negative, got %s", request.Params.Settings.RequestTimeout,
		)
	}
	if err := validateModelSettings(request.Params.Settings); err != nil {
		return nil, err
	}
	if modelIsNil(request.Model) {
		if request.ModelID == "" {
			return nil, ErrNoModel
		}
		request.Model, err = r.resolveModelID(ctx, request.ModelID)
		if err != nil {
			return nil, err
		}
	}
	modelChanged := !sameModelInstance(r.model, request.Model)
	if modelChanged && request.Params.OutputMode == r.params.OutputMode && r.baseParams.OutputMode == OutputModeAuto {
		outputParams := request.Params
		outputParams.OutputMode = r.baseParams.OutputMode
		if outputParams.OutputSchema == nil && outputParams.OutputTool != nil {
			outputParams.OutputSchema = cloneSchemaMap(outputParams.OutputTool.Schema)
		}
		outputParams.OutputPrompt = ""
		outputParams.InstructionParts = removeOutputPrompt(outputParams.InstructionParts, r.params.OutputPrompt)
		instructions := make([]string, len(outputParams.InstructionParts))
		for index, part := range outputParams.InstructionParts {
			instructions[index] = part.Content
		}
		outputParams.Instructions = strings.Join(instructions, "\n\n")
		request.Params, err = resolveModelOutputParams(request.Model, outputParams, r.outputTool, r.promptedTemplate)
		if err != nil {
			return nil, err
		}
	}
	r.model = request.Model
	r.rc.Model = request.Model
	r.rc.ModelID = request.ModelID
	if modelChanged {
		if err := r.openSelectedModel(ctx); err != nil {
			return nil, err
		}
		if r.recordSelectedModel != nil {
			r.recordSelectedModel(r.model.Name())
		}
	}
	r.params = request.Params
	r.currentOutputTool = request.Params.OutputTool
	r.setCurrentTools(request.Params)
	if err := r.compileCurrentSchemas(request.Params); err != nil {
		return nil, err
	}
	response, err := next(ctx, request.Messages, request.Params)
	var retry *RetryError
	if err != nil && !errors.As(err, &retry) {
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(ModelRequestErrorHook)
			if !ok {
				continue
			}
			response, err = hook.OnModelRequestError(ctx, r.info, request, err)
			if err == nil {
				if response == nil {
					return nil, &UnexpectedModelBehaviorError{Message: "model request error hook returned no response"}
				}
				break
			}
		}
	}
	if err != nil {
		return response, err
	}
	if response == nil {
		return nil, nil
	}
	r.stampModelResponse(response)
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		hook, ok := r.capabilities[index].(AfterModelRequestHook)
		if !ok {
			continue
		}
		response, err = hook.AfterModelRequest(ctx, r.info, request, response)
		if err != nil {
			return response, err
		}
		if response == nil {
			return nil, &UnexpectedModelBehaviorError{Message: "after model request hook returned no response"}
		}
	}
	r.stampModelResponse(response)
	r.resumeSeed = nil
	return response, nil
}

func (r *run[Deps, Output]) stampModelResponse(response *ModelResponse) {
	if response.Timestamp.IsZero() {
		response.Timestamp = time.Now().UTC()
	}
	if response.RunID == "" {
		response.RunID = r.rc.RunID
	}
	if response.ConversationID == "" {
		response.ConversationID = r.rc.ConversationID
	}
	if response.ModelName == "" {
		response.ModelName = r.model.Name()
	}
}

func setLatestRequestContext(messages []ModelMessage, instructions, runID, conversationID string) {
	if len(messages) == 0 {
		return
	}
	index := len(messages) - 1
	request, ok := messages[index].(ModelRequest)
	if !ok {
		return
	}
	request.Instructions = instructions
	if request.Timestamp.IsZero() {
		request.Timestamp = time.Now().UTC()
	}
	if request.RunID == "" {
		request.RunID = runID
	}
	if request.ConversationID == "" {
		request.ConversationID = conversationID
	}
	if request.State == "" {
		request.State = RequestStateComplete
	}
	messages[index] = request
}

func (r *run[Deps, Output]) prependLatestRequestParts(parts []RequestPart) {
	if len(parts) == 0 {
		return
	}
	index := len(r.messages) - 1
	request := r.messages[index].(ModelRequest)
	request.Parts = append(slices.Clone(parts), request.Parts...)
	r.messages[index] = request
}

func (r *run[Deps, Output]) removeLatestUserPrompts() {
	index := len(r.messages) - 1
	request := r.messages[index].(ModelRequest)
	parts := make([]RequestPart, 0, len(request.Parts))
	for _, part := range request.Parts {
		if _, prompt := part.(UserPromptPart); !prompt {
			parts = append(parts, part)
		}
	}
	if len(parts) == 0 {
		r.messages = r.messages[:index]
		return
	}
	request.Parts = parts
	r.messages[index] = request
}

func (r *run[Deps, Output]) setCurrentTools(params ModelRequestParams) {
	r.currentTools = make(map[string]struct{}, len(params.Tools))
	for _, def := range params.Tools {
		r.currentTools[def.Name] = struct{}{}
	}
}

func (r *run[Deps, Output]) refreshRevealedDeferredTools(params *ModelRequestParams) error {
	visible := make(map[string]struct{}, len(params.Tools))
	for _, definition := range params.Tools {
		visible[definition.Name] = struct{}{}
	}
	for _, definition := range params.DeferredTools {
		if _, revealed := r.revealedTools[definition.Name]; !revealed {
			continue
		}
		if _, exists := visible[definition.Name]; exists {
			continue
		}
		params.Tools = append(params.Tools, cloneToolDefinition(definition))
		visible[definition.Name] = struct{}{}
	}
	r.setCurrentTools(*params)
	return r.compileCurrentSchemas(*params)
}

func (r *run[Deps, Output]) resolveDeferredToolResults(
	ctx context.Context, results DeferredToolResults,
) ([]RequestPart, error) {
	parts, _, err := r.applyDeferredToolResults(ctx, unresolvedToolCalls(r.messages), results, true)
	return parts, err
}

func (r *run[Deps, Output]) applyDeferredToolResults(
	ctx context.Context,
	pending []ToolCallPart,
	results DeferredToolResults,
	requireAll bool,
) ([]RequestPart, map[string]struct{}, error) {
	pendingByID := make(map[string]ToolCallPart, len(pending))
	for _, call := range pending {
		if call.ToolCallID == "" {
			return nil, nil, fmt.Errorf("ai: pending tool call %q has an empty tool call ID", call.ToolName)
		}
		if _, duplicate := pendingByID[call.ToolCallID]; duplicate {
			return nil, nil, fmt.Errorf("ai: pending tool calls have duplicate tool call ID %q", call.ToolCallID)
		}
		pendingByID[call.ToolCallID] = call
	}
	for id := range results.Calls {
		if _, exists := pendingByID[id]; !exists {
			return nil, nil, fmt.Errorf("ai: deferred call result %q does not match a pending tool call", id)
		}
		if _, duplicate := results.Approvals[id]; duplicate {
			return nil, nil, fmt.Errorf("ai: deferred result %q appears in calls and approvals", id)
		}
	}
	for id := range results.Approvals {
		if _, exists := pendingByID[id]; !exists {
			return nil, nil, fmt.Errorf("ai: approval result %q does not match a pending tool call", id)
		}
	}
	for id := range results.Metadata {
		if _, exists := pendingByID[id]; !exists {
			return nil, nil, fmt.Errorf("ai: deferred metadata %q does not match a pending tool call", id)
		}
	}

	r.resolvingDeferred = make(map[string]deferredResolution, len(pending))
	defer func() { r.resolvingDeferred = nil }()
	hookDeferralEnabled := false
	for _, capability := range r.capabilities {
		_, afterValidation := capability.(AfterToolValidationHook)
		_, beforeExecution := capability.(BeforeToolExecutionHook)
		_, afterExecution := capability.(AfterToolExecutionHook)
		if afterValidation || beforeExecution || afterExecution {
			hookDeferralEnabled = true
			break
		}
	}
	selected := make([]ToolCallPart, 0, len(pending))
	resolved := make(map[string]struct{}, len(pending))
	for _, call := range pending {
		entry, registered := r.findTool(call.ToolName)
		_, available := r.currentTools[call.ToolName]
		if !registered || !available {
			return nil, nil, fmt.Errorf("ai: pending tool %q is not available in the resumed run", call.ToolName)
		}
		expectedKind, hasExpectedKind := deferredToolKind(r.messages, call)
		hookDeferred := hasExpectedKind && hookDeferralEnabled
		if !entry.def.ExternalExecution && !entry.def.DynamicExternalExecution &&
			!entry.def.RequiresApproval && !entry.def.DynamicApproval && !hookDeferred {
			return nil, nil, fmt.Errorf("ai: pending tool call %q is not deferred", call.ToolCallID)
		}
		metadata := cloneSchemaMap(results.Metadata[call.ToolCallID])
		result, hasResult := results.Calls[call.ToolCallID]
		approval, hasApproval := results.Approvals[call.ToolCallID]
		switch {
		case hasResult && hasExpectedKind && expectedKind != deferredCallExternal:
			return nil, nil, fmt.Errorf("ai: approval tool call %q received an external result", call.ToolCallID)
		case hasApproval && hasExpectedKind && expectedKind != deferredCallApproval:
			return nil, nil, fmt.Errorf("ai: external tool call %q received an approval result", call.ToolCallID)
		case hasResult:
			if !hookDeferred && !entry.def.ExternalExecution && !entry.def.DynamicExternalExecution {
				return nil, nil, fmt.Errorf("ai: approval tool call %q received an external result", call.ToolCallID)
			}
			r.resolvingDeferred[call.ToolCallID] = deferredResolution{
				kind: deferredCallExternal, result: result, metadata: metadata,
			}
		case hasApproval:
			if !hookDeferred && !entry.def.RequiresApproval && !entry.def.DynamicApproval {
				return nil, nil, fmt.Errorf("ai: external tool call %q received an approval result", call.ToolCallID)
			}
			normalized, err := normalizeToolApproval(approval)
			if err != nil {
				return nil, nil, fmt.Errorf("ai: approval for tool call %q: %w", call.ToolCallID, err)
			}
			r.resolvingDeferred[call.ToolCallID] = deferredResolution{
				kind: deferredCallApproval, approval: normalized, metadata: metadata,
			}
		case requireAll:
			switch {
			case entry.def.ExternalExecution && !entry.def.RequiresApproval && !entry.def.DynamicApproval:
				return nil, nil, fmt.Errorf("ai: missing result for external tool call %q", call.ToolCallID)
			case entry.def.RequiresApproval && !entry.def.DynamicExternalExecution:
				return nil, nil, fmt.Errorf("ai: missing approval for tool call %q", call.ToolCallID)
			default:
				return nil, nil, fmt.Errorf("ai: missing deferred result for tool call %q", call.ToolCallID)
			}
		default:
			continue
		}
		selected = append(selected, call)
		resolved[call.ToolCallID] = struct{}{}
	}
	if len(selected) == 0 {
		return nil, resolved, nil
	}
	parts, _, err := r.executeCallsWithCallEvents(ctx, selected, false)
	return parts, resolved, err
}

func (r *run[Deps, Output]) handleDeferredToolCalls(
	ctx context.Context, requests DeferredToolRequests,
) ([]RequestPart, *DeferredToolRequests, error) {
	remaining := requests.Clone()
	var parts []RequestPart
	for _, capability := range r.capabilities {
		handler, ok := capability.(DeferredToolCallHandler)
		if !ok {
			continue
		}
		results, err := handler.HandleDeferredToolCalls(ctx, r.info, remaining.Clone())
		if err != nil {
			return nil, nil, fmt.Errorf("ai: handle deferred tool calls: %w", err)
		}
		if results == nil {
			continue
		}
		clonedResults := cloneDeferredToolResults(*results)
		pending := append(cloneToolCalls(remaining.Approvals), remaining.Calls...)
		r.pendingDeferred = nil
		resolvedParts, resolved, err := r.applyDeferredToolResults(ctx, pending, clonedResults, false)
		if err != nil {
			return nil, nil, err
		}
		if !r.emitStreamEvent(DeferredToolResultsEvent{Results: cloneDeferredToolResults(clonedResults)}) {
			return nil, nil, context.Canceled
		}
		parts = append(parts, resolvedParts...)
		remaining = remainingDeferredToolRequests(remaining, resolved)
		if r.pendingDeferred != nil {
			remaining = appendDeferredToolRequests(remaining, *r.pendingDeferred)
			r.pendingDeferred = nil
		}
		if len(remaining.Calls) == 0 && len(remaining.Approvals) == 0 {
			return parts, nil, nil
		}
	}
	return parts, &remaining, nil
}

func appendDeferredToolRequests(first, second DeferredToolRequests) DeferredToolRequests {
	first.Calls = append(first.Calls, cloneToolCalls(second.Calls)...)
	first.Approvals = append(first.Approvals, cloneToolCalls(second.Approvals)...)
	if len(second.Metadata) > 0 && first.Metadata == nil {
		first.Metadata = make(map[string]map[string]any, len(second.Metadata))
	}
	for id, metadata := range second.Metadata {
		first.Metadata[id] = cloneSchemaMap(metadata)
	}
	return first
}

func remainingDeferredToolRequests(
	requests DeferredToolRequests, resolved map[string]struct{},
) DeferredToolRequests {
	remaining := DeferredToolRequests{Metadata: map[string]map[string]any{}}
	for _, call := range requests.Calls {
		if _, ok := resolved[call.ToolCallID]; !ok {
			remaining.Calls = append(remaining.Calls, call)
		}
	}
	for _, call := range requests.Approvals {
		if _, ok := resolved[call.ToolCallID]; !ok {
			remaining.Approvals = append(remaining.Approvals, call)
		}
	}
	for id, metadata := range requests.Metadata {
		if _, ok := resolved[id]; !ok {
			remaining.Metadata[id] = cloneSchemaMap(metadata)
		}
	}
	if len(remaining.Metadata) == 0 {
		remaining.Metadata = nil
	}
	return remaining
}

func normalizeToolApproval(approval ToolApproval) (ToolApproval, error) {
	switch approval := approval.(type) {
	case ToolApproved:
		return approval, nil
	case *ToolApproved:
		if approval == nil {
			return nil, fmt.Errorf("approval must not be nil")
		}
		return *approval, nil
	case ToolDenied:
		return approval, nil
	case *ToolDenied:
		if approval == nil {
			return nil, fmt.Errorf("denial must not be nil")
		}
		return *approval, nil
	case nil:
		return nil, fmt.Errorf("approval must not be nil")
	default:
		return nil, fmt.Errorf("unsupported approval type %T", approval)
	}
}

func (r *run[Deps, Output]) persistDeferredToolKinds(requests DeferredToolRequests) {
	kinds := make(map[string]string, len(requests.Calls)+len(requests.Approvals))
	for _, call := range requests.Calls {
		kinds[call.ToolCallID] = "external"
	}
	for _, call := range requests.Approvals {
		kinds[call.ToolCallID] = "approval"
	}
	for index := len(r.messages) - 1; index >= 0 && len(kinds) > 0; index-- {
		response, ok := r.messages[index].(ModelResponse)
		if !ok {
			continue
		}
		stored := map[string]any{}
		if existing, ok := response.Metadata[DeferredToolKindsMetadataKey].(map[string]any); ok {
			stored = cloneSchemaMap(existing)
		}
		changed := false
		for _, call := range response.ToolCalls() {
			kind, pending := kinds[call.ToolCallID]
			if !pending {
				continue
			}
			stored[call.ToolCallID] = kind
			delete(kinds, call.ToolCallID)
			changed = true
		}
		if changed {
			response.Metadata = cloneSchemaMap(response.Metadata)
			if response.Metadata == nil {
				response.Metadata = make(map[string]any, 1)
			}
			response.Metadata[DeferredToolKindsMetadataKey] = stored
			r.messages[index] = response
		}
	}
}

func deferredToolKind(messages []ModelMessage, call ToolCallPart) (deferredCallKind, bool) {
	for index := len(messages) - 1; index >= 0; index-- {
		response, ok := messages[index].(ModelResponse)
		if !ok {
			continue
		}
		containsCall := false
		for _, candidate := range response.ToolCalls() {
			if candidate.ToolCallID == call.ToolCallID && candidate.ToolName == call.ToolName {
				containsCall = true
				break
			}
		}
		if !containsCall {
			continue
		}
		stored, _ := response.Metadata[DeferredToolKindsMetadataKey].(map[string]any)
		switch stored[call.ToolCallID] {
		case "external":
			return deferredCallExternal, true
		case "approval":
			return deferredCallApproval, true
		}
	}
	return 0, false
}

func unresolvedToolCalls(messages []ModelMessage) []ToolCallPart {
	type pendingCall struct {
		call   ToolCallPart
		active bool
	}
	tracked := make([]*pendingCall, 0)
	open := map[string][]*pendingCall{}
	for _, message := range messages {
		switch message := message.(type) {
		case ModelResponse:
			for _, call := range message.ToolCalls() {
				pending := &pendingCall{call: call, active: true}
				tracked = append(tracked, pending)
				key := toolCallMatchKey(call.ToolName, call.ToolCallID)
				open[key] = append(open[key], pending)
			}
		case ModelRequest:
			for _, part := range message.Parts {
				toolName, toolCallID, isResult := toolResultIdentity(part)
				if !isResult {
					continue
				}
				key := toolCallMatchKey(toolName, toolCallID)
				calls := open[key]
				if len(calls) == 0 {
					continue
				}
				calls[len(calls)-1].active = false
				open[key] = calls[:len(calls)-1]
			}
		}
	}
	pending := make([]ToolCallPart, 0, len(tracked))
	for _, trackedCall := range tracked {
		if trackedCall.active {
			pending = append(pending, trackedCall.call)
		}
	}
	return pending
}

func (r *run[Deps, Output]) applyResponseToolKinds(response *ModelResponse) error {
	parts := slices.Clone(response.Parts)
	for index, responsePart := range parts {
		switch part := responsePart.(type) {
		case ToolCallPart:
			if part.ToolKind != "" {
				continue
			}
			entry, ok := r.findTool(part.ToolName)
			if !ok || entry.def.ToolKind == "" {
				continue
			}
			part.ToolKind = entry.def.ToolKind
			parts[index] = part
		case NativeToolReturnPart:
			if part.ToolKind != ToolPartKindToolSearch {
				continue
			}
			for _, name := range toolSearchResultNames(part.Content) {
				if _, deferred := r.currentDeferredTools[name]; !deferred {
					continue
				}
				if err := r.activateDeferredTool(name); err != nil {
					return err
				}
				r.revealedTools[name] = struct{}{}
				r.currentTools[name] = struct{}{}
			}
		}
	}
	response.Parts = parts
	return nil
}

func (r *run[Deps, Output]) activateDeferredTool(name string) error {
	definition, ok := r.currentDeferredDefinitions[name]
	if !ok || definition.Schema == nil {
		return nil
	}
	validator, err := schema.Compile(definition.Schema)
	if err != nil {
		return fmt.Errorf("ai: tool %q schema: %w", name, err)
	}
	r.currentToolValidators[name] = validator
	return nil
}

func toolSearchResultNames(content any) []string {
	encoded, err := json.Marshal(content)
	if err != nil {
		return nil
	}
	var result ToolSearchResult
	if json.Unmarshal(encoded, &result) != nil {
		return nil
	}
	names := make([]string, len(result.DiscoveredTools))
	for index, match := range result.DiscoveredTools {
		names[index] = match.Name
	}
	return names
}

func (r *run[Deps, Output]) prepareModelParams(ctx context.Context) (ModelRequestParams, error) {
	params := r.baseParams
	rc := *r.rc
	rc.Retry = r.outputRetryCount()
	rc.MaxRetries = r.outputMaxRetries
	settings, err := r.prepareModelSettings(ctx, &rc)
	if err != nil {
		return ModelRequestParams{}, err
	}
	if err := r.prepareToolsetsForStep(ctx, &rc); err != nil {
		return ModelRequestParams{}, err
	}
	if err := r.prepareSystemPrompts(ctx, &rc); err != nil {
		return ModelRequestParams{}, err
	}
	instructionParts, err := r.prepareInstructions(ctx, &rc)
	if err != nil {
		return ModelRequestParams{}, err
	}
	params.Settings = settings
	params.InstructionParts = instructionParts
	instructions := make([]string, 0, len(instructionParts))
	for _, part := range instructionParts {
		instructions = append(instructions, part.Content)
	}
	params.Instructions = strings.Join(instructions, "\n\n")
	params, err = r.prepareOutputParams(ctx, &rc, params)
	if err != nil {
		return ModelRequestParams{}, err
	}
	r.currentOutputTool = params.OutputTool
	stepEntries := make(map[string]toolEntry[Deps], len(r.tools))
	for _, entry := range r.tools {
		stepEntries[entry.def.Name] = entry
	}
	for _, toolset := range r.toolsets {
		resolved, err := resolveToolsetTools(ctx, &rc, toolset)
		if err != nil {
			return ModelRequestParams{}, fmt.Errorf("ai: resolve toolset: %w", err)
		}
		for _, tool := range resolved {
			entry := tool.entry
			if entry.def.Name == "" {
				return ModelRequestParams{}, fmt.Errorf("ai: toolset returned an empty tool name")
			}
			if _, exists := stepEntries[entry.def.Name]; exists {
				return ModelRequestParams{}, fmt.Errorf("ai: duplicate tool name %q", entry.def.Name)
			}
			entry.def = cloneToolDefinition(entry.def)
			stepEntries[entry.def.Name] = entry
			params.Tools = append(params.Tools, entry.def)
		}
	}
	r.currentToolEntries = stepEntries
	tools := make([]ToolDefinition, 0, len(params.Tools))
	for _, def := range params.Tools {
		prepared := cloneToolDefinition(def)
		entry, _ := r.findTool(def.Name)
		if entry.prepare != nil {
			result, err := entry.prepare(ctx, &rc, prepared)
			if err != nil {
				return ModelRequestParams{}, fmt.Errorf("ai: prepare tool %q: %w", def.Name, err)
			}
			if result == nil {
				continue
			}
			prepared = *result
		}
		tools = append(tools, prepared)
	}
	for _, prepare := range r.agent.toolsPrepareFuncs {
		tools, err = prepare(ctx, &rc, tools)
		if err != nil {
			return ModelRequestParams{}, fmt.Errorf("ai: prepare tools: %w", err)
		}
	}
	known := make(map[string]struct{}, len(stepEntries))
	for name := range stepEntries {
		known[name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(tools))
	for _, def := range tools {
		if def.ExternalExecution && (def.RequiresApproval || def.DynamicApproval) {
			return ModelRequestParams{}, fmt.Errorf(
				"ai: external tool %q cannot require approval", def.Name,
			)
		}
		if def.RequiresApproval && def.DynamicApproval {
			return ModelRequestParams{}, fmt.Errorf(
				"ai: tool %q cannot use static and dynamic approval", def.Name,
			)
		}
		if def.ExternalExecution && def.DynamicExternalExecution {
			return ModelRequestParams{}, fmt.Errorf(
				"ai: tool %q cannot use static and dynamic external execution", def.Name,
			)
		}
		if _, ok := known[def.Name]; !ok {
			return ModelRequestParams{}, fmt.Errorf("ai: prepare tools returned unknown tool %q", def.Name)
		}
		if _, ok := seen[def.Name]; ok {
			return ModelRequestParams{}, fmt.Errorf("ai: prepare tools returned duplicate tool %q", def.Name)
		}
		seen[def.Name] = struct{}{}
	}
	r.currentDeferredTools = make(map[string]struct{})
	r.currentDeferredDefinitions = make(map[string]ToolDefinition)
	visibleTools := make([]ToolDefinition, 0, len(tools))
	deferredTools := make([]ToolDefinition, 0, len(tools))
	for _, definition := range tools {
		if definition.DeferLoading {
			r.currentDeferredTools[definition.Name] = struct{}{}
			r.currentDeferredDefinitions[definition.Name] = cloneToolDefinition(definition)
			deferredTools = append(deferredTools, cloneToolDefinition(definition))
			if _, revealed := r.revealedTools[definition.Name]; !revealed {
				continue
			}
		}
		visibleTools = append(visibleTools, definition)
	}
	params.Tools = visibleTools
	params.DeferredTools = deferredTools
	if err := validateToolSearchStrategies(r.model, params.Tools); err != nil {
		return ModelRequestParams{}, err
	}
	if err := r.compileCurrentSchemas(params); err != nil {
		return ModelRequestParams{}, err
	}
	return params, nil
}

func (r *run[Deps, Output]) prepareOutputParams(
	ctx context.Context, rc *RunContext[Deps], params ModelRequestParams,
) (ModelRequestParams, error) {
	params, err := resolveModelOutputParams(r.model, params, r.outputTool, r.promptedTemplate)
	if err != nil {
		return ModelRequestParams{}, err
	}
	if params.OutputTool != nil {
		prepared := cloneToolDefinition(*params.OutputTool)
		for _, prepare := range r.agent.outputToolPrepare {
			result, err := prepare(ctx, rc, prepared)
			if err != nil {
				return ModelRequestParams{}, fmt.Errorf("ai: prepare output tool: %w", err)
			}
			if result == nil {
				params.OutputTool = nil
				break
			}
			prepared = cloneToolDefinition(*result)
			params.OutputTool = &prepared
		}
		if params.OutputTool != nil && params.OutputTool.Name == "" {
			return ModelRequestParams{}, fmt.Errorf("ai: prepared output tool name must not be empty")
		}
	}
	return params, nil
}

func resolveModelOutputParams(
	model Model, params ModelRequestParams, outputTool OutputToolConfig, promptedTemplate string,
) (ModelRequestParams, error) {
	if params.OutputSchema == nil && params.OutputTool == nil {
		return params, nil
	}
	profile := modelProfile(model)
	mode := params.OutputMode
	if mode == OutputModeAuto {
		if dispatcher, ok := model.(ModelOutputProfileDispatcher); ok && dispatcher.DispatchesOutputProfile() {
			if params.OutputTool == nil {
				params.OutputTool = defaultOutputTool(params.OutputSchema, outputTool)
			}
			params.AllowText = true
			return params, nil
		}
		mode = profile.DefaultOutputMode
	}
	switch mode {
	case OutputModeTool:
		if params.OutputTool == nil {
			params.OutputTool = defaultOutputTool(params.OutputSchema, outputTool)
		}
		params.OutputSchema = nil
		params.OutputPrompt = ""
		params.AllowText = false
	case OutputModeNative, OutputModePrompted:
		params.OutputTool = nil
		params.AllowText = true
		if mode == OutputModePrompted || profile.NativeOutputRequiresPrompt {
			template := promptedTemplate
			if template == "" {
				template = profile.PromptedOutputTemplate
			}
			params.OutputPrompt = buildPromptedOutput(params.OutputSchema, template)
		}
	default:
		return ModelRequestParams{}, fmt.Errorf("ai: model profile has invalid default output mode %d", mode)
	}
	params.OutputMode = mode
	if params.OutputPrompt != "" {
		params.InstructionParts = append(params.InstructionParts, InstructionPart{Content: params.OutputPrompt})
		instructions := make([]string, len(params.InstructionParts))
		for index, part := range params.InstructionParts {
			instructions[index] = part.Content
		}
		params.Instructions = strings.Join(instructions, "\n\n")
	}
	return params, nil
}

func defaultOutputTool(schema map[string]any, config OutputToolConfig) *ToolDefinition {
	name := config.Name
	if name == "" {
		name = outputToolName
	}
	description := config.Description
	if description == "" {
		description = "The final result of the run."
	}
	return &ToolDefinition{
		Name: name, Description: description, Schema: cloneSchemaMap(schema),
		Sequential: config.Sequential, Strict: clonePointer(config.Strict),
	}
}

func removeOutputPrompt(parts []InstructionPart, prompt string) []InstructionPart {
	if prompt == "" {
		return parts
	}
	filtered := make([]InstructionPart, 0, len(parts))
	for _, part := range parts {
		if part.Content != prompt {
			filtered = append(filtered, part)
		}
	}
	return filtered
}

func buildPromptedOutput(outputSchema map[string]any, template string) string {
	encodedSchema, _ := json.Marshal(outputSchema)
	if template == "" {
		template = defaultPromptedOutputTemplate
	}
	if !strings.Contains(template, "{schema}") {
		template += "\n\n{schema}"
	}
	return strings.ReplaceAll(template, "{schema}", string(encodedSchema))
}

func validateToolSearchStrategies(model Model, tools []ToolDefinition) error {
	for _, tool := range tools {
		strategy := tool.ToolSearchStrategy
		if tool.ToolKind != ToolPartKindToolSearch ||
			(strategy != ToolSearchStrategyBM25 && strategy != ToolSearchStrategyRegex) {
			continue
		}
		support, ok := model.(ToolSearchStrategyModel)
		if !ok || !support.SupportsToolSearchStrategy(strategy) {
			return fmt.Errorf("ai: selected model does not support tool search strategy %q", strategy)
		}
	}
	return nil
}

func (r *run[Deps, Output]) compileCurrentSchemas(params ModelRequestParams) error {
	r.currentToolValidators = make(map[string]*schema.Validator, len(params.Tools))
	for _, tool := range params.Tools {
		if tool.Schema == nil {
			continue
		}
		validator, err := schema.Compile(tool.Schema)
		if err != nil {
			return fmt.Errorf("ai: tool %q schema: %w", tool.Name, err)
		}
		r.currentToolValidators[tool.Name] = validator
	}
	r.currentOutputValidator = nil
	var outputSchema map[string]any
	if params.OutputTool != nil {
		outputSchema = params.OutputTool.Schema
	} else if params.OutputSchema != nil {
		outputSchema = params.OutputSchema
	}
	if outputSchema != nil {
		validator, err := schema.Compile(outputSchema)
		if err != nil {
			return fmt.Errorf("ai: output schema: %w", err)
		}
		r.currentOutputValidator = validator
	}
	return nil
}

func cloneToolDefinition(def ToolDefinition) ToolDefinition {
	def.Schema = cloneSchemaMap(def.Schema)
	def.ReturnSchema = cloneSchemaMap(def.ReturnSchema)
	def.Metadata = cloneSchemaMap(def.Metadata)
	def.ApprovalMetadata = cloneSchemaMap(def.ApprovalMetadata)
	if def.Strict != nil {
		strict := *def.Strict
		def.Strict = &strict
	}
	return def
}

func cloneSchemaMap(schema map[string]any) map[string]any {
	if schema == nil {
		return nil
	}
	cloned := make(map[string]any, len(schema))
	for key, value := range schema {
		cloned[key] = cloneSchemaValue(value)
	}
	return cloned
}

func cloneSchemaValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneSchemaMap(value)
	case ToolSearchResult:
		value.DiscoveredTools = slices.Clone(value.DiscoveredTools)
		return value
	case []any:
		cloned := make([]any, len(value))
		for i, item := range value {
			cloned[i] = cloneSchemaValue(item)
		}
		return cloned
	default:
		return value
	}
}

// doModelRequest resolves all suspended continuation segments as one logical
// request. Usage is merged and committed once after the final segment.
func (r *run[Deps, Output]) doModelRequest(
	ctx context.Context, msgs []ModelMessage, params ModelRequestParams,
) (*ModelResponse, error) {
	baseMessages := slices.Clone(msgs)
	var response *ModelResponse
	if historyEndsSuspended(baseMessages) {
		seed := baseMessages[len(baseMessages)-1].(ModelResponse)
		response = cloneModelResponse(&seed)
		fillResponseCost(ctx, response)
		baseMessages = baseMessages[:len(baseMessages)-1]
	}
	lastMode := continuationAccumulate
	generationCount := 0
	pollCount := 0
	lastSegmentOffset := 0
	for {
		segmentMessages := baseMessages
		if response != nil {
			if response.State != ModelResponseStateSuspended {
				return response, nil
			}
			if err := continuationLimitError(response, lastMode, &generationCount, &pollCount); err != nil {
				r.cancelSuspendedResponse(response)
				return nil, err
			}
			if err := r.waitForContinuation(ctx, response); err != nil {
				r.cancelSuspendedResponse(response)
				return nil, err
			}
			segmentMessages = append(slices.Clone(baseMessages), *cloneModelResponse(response))
		}

		eventOffset := 0
		emit := r.emitStreamEvent
		if response != nil && r.emit != nil {
			if background, _ := response.ProviderDetails["background"].(bool); background {
				eventOffset = lastSegmentOffset
			} else {
				eventOffset = len(response.Parts)
			}
			emit = func(event StreamEvent) bool {
				return r.emitStreamEvent(reindexContinuationEvent(event, eventOffset))
			}
		}
		observe := func(segment *ModelResponse) {
			current := segment
			if response != nil {
				current, _ = mergeModelResponses(response, segment)
			}
			r.publishUsage(current)
		}
		segment, err := r.requestModelSegment(ctx, segmentMessages, params, emit, observe)
		if err != nil {
			prior := response
			partial := response
			if segment != nil {
				if segment.State == "" {
					segment.State = ModelResponseStateComplete
				}
				if partial == nil {
					partial = segment
				} else {
					partial, _ = mergeModelResponses(partial, segment)
				}
			}
			if errors.Is(context.Cause(r.ctx), errStreamDetached) {
				if partial != nil && partial.State == ModelResponseStateSuspended {
					r.detachedResponse = cloneModelResponse(partial)
				} else if prior != nil && prior.State == ModelResponseStateSuspended {
					r.detachedResponse = cloneModelResponse(prior)
				}
				return nil, err
			}
			if partial != nil && partial.State == ModelResponseStateSuspended {
				r.cancelSuspendedResponse(partial)
			} else {
				r.cancelSuspendedResponse(prior)
			}
			return nil, err
		}
		if segment == nil {
			err := &UnexpectedModelBehaviorError{Message: "model returned no response"}
			if response != nil {
				r.cancelSuspendedResponse(response)
			}
			return nil, err
		}
		if segment.State == "" {
			segment.State = ModelResponseStateComplete
		}
		if response == nil {
			response = segment
		} else {
			var mode continuationMergeMode
			response, mode = mergeModelResponses(response, segment)
			lastMode = mode
			switch mode {
			case continuationAccumulate:
				lastSegmentOffset = eventOffset
			case continuationReplaceSameID:
				lastSegmentOffset = eventOffset
			default:
				lastSegmentOffset = 0
			}
		}
		if response.State == ModelResponseStateSuspended {
			projected := r.usage.Clone()
			projected.Add(response.Usage)
			if err := r.rc.UsageLimits.check(projected); err != nil {
				r.cancelSuspendedResponse(response)
				return nil, err
			}
		}
	}
}

func (r *run[Deps, Output]) requestModelSegment(
	ctx context.Context, msgs []ModelMessage, params ModelRequestParams, emit func(StreamEvent) bool,
	observe func(*ModelResponse),
) (*ModelResponse, error) {
	provider := ""
	if model, ok := r.model.(NativeToolSearchHistoryModel); ok {
		provider = model.NativeToolSearchProvider()
	}
	msgs = adaptNativeToolSearchHistory(msgs, provider)
	if params.Settings.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, params.Settings.RequestTimeout)
		defer cancel()
	}
	if r.emit == nil {
		response, err := r.model.Request(ctx, msgs, params)
		fillResponseCost(ctx, response)
		if response != nil {
			observe(response)
		}
		return response, err
	}
	if sm, ok := r.model.(StreamingModel); ok {
		events, err := sm.StreamRequest(ctx, msgs, params)
		if err != nil {
			return nil, err
		}
		response, err := accumulate(events, params, emit, observe)
		fillResponseCost(ctx, response)
		return response, err
	}
	resp, err := r.model.Request(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	response, err := accumulate(replayAsEvents(resp), params, emit, observe)
	fillResponseCost(ctx, response)
	return response, err
}

func (r *run[Deps, Output]) publishUsage(response *ModelResponse) {
	if r.observeUsage == nil {
		return
	}
	r.usagePublishMu.Lock()
	defer r.usagePublishMu.Unlock()
	usage := r.usage.Clone()
	usage.ToolCalls = int(r.toolCalls.Load())
	if response != nil {
		priced := cloneModelResponse(response)
		fillResponseCost(r.ctx, priced)
		usage.Add(priced.Usage)
	}
	r.observeUsage(usage)
}

func (r *run[Deps, Output]) waitForContinuation(ctx context.Context, response *ModelResponse) error {
	delayer, ok := r.model.(ModelContinuationDelayer)
	if !ok {
		return nil
	}
	delay := delayer.ContinuationDelay(*cloneModelResponse(response))
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-timer.C:
		return nil
	}
}

func (r *run[Deps, Output]) cancelSuspendedResponse(response *ModelResponse) {
	canceler, ok := r.model.(SuspendedResponseCanceler)
	if !ok || response == nil || response.State != ModelResponseStateSuspended {
		return
	}
	_ = canceler.CancelSuspendedResponse(context.WithoutCancel(r.ctx), *cloneModelResponse(response))
}

func (r *run[Deps, Output]) loop(ctx context.Context) (*RunResult[Output], error) {
	for {
		resp, err := r.modelRequest(ctx)
		if err != nil {
			var retry *RetryError
			if !errors.As(err, &retry) {
				return nil, err
			}
			if err := r.countOutputRetry(); err != nil {
				return nil, err
			}
			if resp != nil {
				fillResponseCost(ctx, resp)
				r.usage.Add(resp.Usage)
				r.publishUsage(nil)
				if applyErr := r.applyResponseToolKinds(resp); applyErr != nil {
					return nil, applyErr
				}
				r.messages = append(r.messages, *resp)
			}
			r.recordRetry(RetryPromptPart{Content: retry.Message})
			continue
		}
		if resp == nil {
			if r.pendingDeferred != nil {
				return r.deferredResult(*r.pendingDeferred)
			}
			return nil, &UnexpectedModelBehaviorError{Message: "model request returned no response"}
		}
		fillResponseCost(ctx, resp)
		r.usage.Add(resp.Usage)
		r.publishUsage(nil)
		if err := r.applyResponseToolKinds(resp); err != nil {
			return nil, err
		}
		r.messages = append(r.messages, *resp)

		calls := resp.ToolCalls()
		if r.emit != nil && r.commitStreamedOutput {
			output, winningCall, committed, err := r.streamedOutput(ctx, resp)
			if err != nil {
				return nil, err
			}
			if committed {
				parts, err := r.executeCallsWithCommittedOutput(ctx, calls, winningCall)
				if errors.Is(context.Cause(r.ctx), ErrRunCancelled) {
					if len(parts) > 0 {
						r.appendRequest(parts, RequestStateInterrupted)
					}
					return nil, ErrRunCancelled
				}
				if err != nil {
					return nil, err
				}
				if len(parts) > 0 {
					r.appendRequest(parts, RequestStateComplete)
				}
				if redirected, err := r.redirectPendingMessages(); err != nil {
					return nil, err
				} else if redirected {
					continue
				}
				return r.result(*output), nil
			}
		}
		if len(calls) == 0 {
			result, retry, err := r.finalizeText(ctx, resp)
			if err != nil {
				return nil, err
			}
			if retry != nil {
				r.recordRetry(*retry)
				continue
			}
			if redirected, err := r.redirectPendingMessages(); err != nil {
				return nil, err
			} else if redirected {
				continue
			}
			return result, nil
		}

		if output, ok, err := r.earlyNativeOutput(ctx, resp); err != nil {
			return nil, err
		} else if ok {
			if !r.emitToolCallEvents(calls) {
				return nil, context.Canceled
			}
			parts := make([]RequestPart, 0, len(calls))
			for _, call := range calls {
				part := ToolReturnPart{
					ToolName: call.ToolName, Content: toolSkipped, ToolCallID: call.ToolCallID,
					ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeSuccess,
				}
				parts = append(parts, part)
				if !r.emitStreamEvent(FunctionToolResultEvent{Part: part}) {
					return nil, context.Canceled
				}
			}
			r.appendRequest(parts, RequestStateComplete)
			if redirected, err := r.redirectPendingMessages(); err != nil {
				return nil, err
			} else if redirected {
				continue
			}
			return r.result(*output), nil
		}

		parts, final, err := r.executeCalls(ctx, calls)
		if errors.Is(context.Cause(r.ctx), ErrRunCancelled) {
			if len(parts) > 0 {
				r.appendRequest(parts, RequestStateInterrupted)
			}
			return nil, ErrRunCancelled
		}
		if err != nil {
			return nil, err
		}
		if r.pendingDeferred != nil {
			if !r.emitStreamEvent(DeferredToolRequestsEvent{Requests: r.pendingDeferred.Clone()}) {
				return nil, context.Canceled
			}
			resolvedParts, remaining, err := r.handleDeferredToolCalls(ctx, *r.pendingDeferred)
			if err != nil {
				return nil, err
			}
			parts = append(parts, resolvedParts...)
			r.pendingDeferred = remaining
		}
		if len(parts) > 0 {
			r.appendRequest(parts, RequestStateComplete)
		}
		if final != nil {
			if redirected, err := r.redirectPendingMessages(); err != nil {
				return nil, err
			} else if redirected {
				continue
			}
			return r.result(*final), nil
		}
		if r.pendingDeferred != nil {
			return r.deferredResult(*r.pendingDeferred)
		}
	}
}

func (r *run[Deps, Output]) earlyNativeOutput(
	ctx context.Context, resp *ModelResponse,
) (*Output, bool, error) {
	if r.agent.endStrategy != EndStrategyEarly || r.params.OutputSchema == nil || resp.Text() == "" {
		return nil, false, nil
	}
	out, err := r.validateAndProcessOutput(
		ctx, r.outputRunContext(""), r.outputHookContext(nil, true, false), resp.Text(),
		func(raw any) (Output, error) { return r.decodeOutput(raw, true) },
	)
	var schemaValidation *outputSchemaValidationError
	var decodeError *outputDecodeError
	var retry *RetryError
	switch {
	case errors.As(err, &schemaValidation), errors.As(err, &decodeError), errors.As(err, &retry):
		return nil, false, nil
	case err != nil:
		return nil, false, err
	default:
		return &out, true, nil
	}
}

type deferredCallKind uint8

const (
	deferredCallExternal deferredCallKind = iota + 1
	deferredCallApproval
)

type deferredRequestPart struct {
	call     ToolCallPart
	kind     deferredCallKind
	metadata map[string]any
}

func (part deferredRequestPart) requestPartKind() string { return part.enqueueItemKind() }
func (deferredRequestPart) enqueueItemKind() string      { return "deferred" }

type deferredResolution struct {
	kind     deferredCallKind
	approval ToolApproval
	result   any
	metadata map[string]any
}

type callOutcome[Output any] struct {
	part          RequestPart
	extraParts    []RequestPart
	output        *Output
	outputCall    bool
	functionCall  bool
	resultEmitted bool
	deferred      *deferredRequestPart
	err           error
}

const (
	finalResultProcessed = "Final result processed."
	retryWins            = "Output not used as the final result - addressing tool retries from this round first."
	outputSkipped        = "Output tool not used - a final result was already processed."
	outputNotFinal       = "Output tool processed, but its value will not be the final result of the agent run."
	toolSkipped          = "Tool not executed - a final result was already processed."
)

func (r *run[Deps, Output]) emitStreamEvent(event StreamEvent) bool {
	_ = event.streamEventKind()
	if r.emit == nil {
		return true
	}
	r.emitMu.Lock()
	defer r.emitMu.Unlock()
	return r.emit(event)
}

func (r *run[Deps, Output]) emitToolCallEvents(calls []ToolCallPart) bool {
	for _, call := range calls {
		var event StreamEvent = FunctionToolCallEvent{Part: call}
		if r.isOutputCall(call) {
			event = OutputToolCallEvent{Part: call}
		}
		if !r.emitStreamEvent(event) {
			return false
		}
	}
	return true
}

// executeCalls honors the configured end strategy while preserving emission
// order in the request sent back to the model.
func (r *run[Deps, Output]) executeCalls(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	return r.executeCallsWithCallEvents(ctx, calls, true)
}

func (r *run[Deps, Output]) executeCallsWithCallEvents(
	ctx context.Context, calls []ToolCallPart, emitCalls bool,
) ([]RequestPart, *Output, error) {
	r.pendingDeferred = nil
	if emitCalls && !r.emitToolCallEvents(calls) {
		return nil, nil, context.Canceled
	}
	if r.agent.endStrategy == EndStrategyEarly {
		return r.executeCallsEarly(ctx, calls)
	}
	if r.agent.endStrategy == EndStrategyGraceful {
		return r.executeCallsGraceful(ctx, calls)
	}
	return r.executeCallsExhaustive(ctx, calls)
}

func (r *run[Deps, Output]) checkToolCallLimit(calls []ToolCallPart) error {
	if r.rc.UsageLimits.ToolCallLimit == nil {
		return nil
	}
	pending := 0
	for _, call := range calls {
		entry, registered := r.findTool(call.ToolName)
		_, available := r.currentTools[call.ToolName]
		resolution, resolving := r.resolvingDeferred[call.ToolCallID]
		if r.isOutputCall(call) || !registered || !available || entry.def.ExternalExecution ||
			resolving && resolution.kind == deferredCallExternal {
			continue
		}
		if entry.def.RequiresApproval {
			resolution, approved := r.resolvingDeferred[call.ToolCallID]
			if !approved {
				continue
			}
			if _, denied := resolution.approval.(ToolDenied); denied {
				continue
			}
		}
		pending++
	}
	projected := int(r.toolCalls.Load()) + pending
	if projected > *r.rc.UsageLimits.ToolCallLimit {
		return fmt.Errorf(
			"%w: tool call count %d exceeds limit %d",
			ErrUsageLimitExceeded, projected, *r.rc.UsageLimits.ToolCallLimit,
		)
	}
	return nil
}

func (r *run[Deps, Output]) executeCallsEarly(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	outcomes := make([]callOutcome[Output], len(calls))
	var winner *Output
	for i, call := range calls {
		if !r.isOutputCall(call) {
			continue
		}
		if winner != nil {
			outcomes[i].part = ToolReturnPart{
				ToolName: call.ToolName, Content: outputSkipped, ToolCallID: call.ToolCallID,
				ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeSuccess,
			}
			continue
		}
		outcomes[i] = r.executeOne(ctx, call)
		if outcomes[i].err != nil {
			return r.completedCallParts(outcomes), nil, outcomes[i].err
		}
		winner = outcomes[i].output
	}
	if winner != nil {
		for i, call := range calls {
			if !r.isOutputCall(call) {
				outcomes[i].part = ToolReturnPart{
					ToolName: call.ToolName, Content: toolSkipped, ToolCallID: call.ToolCallID,
					ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeSuccess,
				}
			}
		}
		return r.collectCallOutcomes(outcomes, false)
	}
	if err := r.checkToolCallLimit(calls); err != nil {
		return r.completedCallParts(outcomes), nil, err
	}
	if err := r.executeSelected(ctx, calls, outcomes, r.functionCallIndexes(calls), false); err != nil {
		return r.completedCallParts(outcomes), nil, err
	}
	return r.collectCallOutcomes(outcomes, false)
}

func (r *run[Deps, Output]) executeCallsGraceful(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	if err := r.checkToolCallLimit(calls); err != nil {
		return nil, nil, err
	}
	outcomes := make([]callOutcome[Output], len(calls))
	batch := make([]int, 0, len(calls))
	var winner *Output
	for i, call := range calls {
		if !r.isOutputCall(call) && !r.callIsBarrier(call, true) {
			batch = append(batch, i)
			continue
		}
		if err := r.executeIndexBatch(ctx, calls, outcomes, batch); err != nil {
			return r.completedCallParts(outcomes), nil, err
		}
		batch = batch[:0]
		if r.isOutputCall(call) && winner != nil {
			outcomes[i].part = ToolReturnPart{
				ToolName: call.ToolName, Content: outputSkipped, ToolCallID: call.ToolCallID,
				ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeSuccess,
			}
			continue
		}
		outcomes[i] = r.executeOne(ctx, call)
		if outcomes[i].err != nil {
			return r.completedCallParts(outcomes), nil, outcomes[i].err
		}
		if outcomes[i].output != nil {
			winner = outcomes[i].output
		}
	}
	if err := r.executeIndexBatch(ctx, calls, outcomes, batch); err != nil {
		return r.completedCallParts(outcomes), nil, err
	}
	return r.collectCallOutcomes(outcomes, true)
}

func (r *run[Deps, Output]) executeCallsExhaustive(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	if err := r.checkToolCallLimit(calls); err != nil {
		return nil, nil, err
	}
	outcomes := make([]callOutcome[Output], len(calls))
	indexes := make([]int, len(calls))
	for i := range calls {
		indexes[i] = i
	}
	if err := r.executeSelected(ctx, calls, outcomes, indexes, true); err != nil {
		return r.completedCallParts(outcomes), nil, err
	}
	return r.collectCallOutcomes(outcomes, true)
}

func (r *run[Deps, Output]) executeSelected(
	ctx context.Context,
	calls []ToolCallPart,
	outcomes []callOutcome[Output],
	indexes []int,
	outputToolsConcurrent bool,
) error {
	batch := make([]int, 0, len(indexes))
	for _, i := range indexes {
		if !r.callIsBarrier(calls[i], outputToolsConcurrent) {
			batch = append(batch, i)
			continue
		}
		if err := r.executeIndexBatch(ctx, calls, outcomes, batch); err != nil {
			return err
		}
		batch = batch[:0]
		outcomes[i] = r.executeOne(ctx, calls[i])
		if outcomes[i].err != nil {
			return outcomes[i].err
		}
	}
	return r.executeIndexBatch(ctx, calls, outcomes, batch)
}

func (r *run[Deps, Output]) executeIndexBatch(
	ctx context.Context, calls []ToolCallPart, outcomes []callOutcome[Output], indexes []int,
) error {
	if len(indexes) == 0 {
		return nil
	}
	if len(indexes) == 1 {
		outcomes[indexes[0]] = r.executeOne(ctx, calls[indexes[0]])
		return outcomes[indexes[0]].err
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	for _, i := range indexes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			outcomes[i] = r.executeOne(ctx, calls[i])
			if outcomes[i].err != nil {
				cancel()
			}
		}()
	}
	wg.Wait()
	for _, i := range indexes {
		if outcomes[i].err != nil {
			return outcomes[i].err
		}
	}
	return nil
}

func (r *run[Deps, Output]) executeOne(ctx context.Context, call ToolCallPart) callOutcome[Output] {
	part, extraParts, output, err := r.executeCall(ctx, call)
	entry, registered := r.findTool(call.ToolName)
	_, available := r.currentTools[call.ToolName]
	outcome := callOutcome[Output]{
		part: part, extraParts: extraParts, output: output, outputCall: r.isOutputCall(call),
		functionCall: registered && available, err: err,
	}
	if part != nil && part.requestPartKind() == "deferred" {
		deferred := part.(deferredRequestPart)
		outcome.part = nil
		outcome.deferred = &deferred
	}
	if err == nil && errors.Is(context.Cause(r.ctx), ErrRunCancelled) {
		outcome.part = nil
		outcome.extraParts = nil
		outcome.output = nil
		outcome.deferred = nil
	}
	resolution, resolving := r.resolvingDeferred[call.ToolCallID]
	if outcome.err == nil && outcome.part != nil && outcome.functionCall &&
		!entry.def.ExternalExecution && (!resolving || resolution.kind != deferredCallExternal) {
		if part, ok := outcome.part.(ToolReturnPart); ok && part.Outcome == ToolReturnOutcomeSuccess {
			r.toolCalls.Add(1)
			r.publishUsage(nil)
		}
	}
	if outcome.err == nil && outcome.part != nil && !outcome.outputCall {
		if !r.emitStreamEvent(FunctionToolResultEvent{Part: outcome.part}) {
			outcome.err = context.Canceled
		} else {
			outcome.resultEmitted = true
		}
	}
	return outcome
}

func (r *run[Deps, Output]) functionCallIndexes(calls []ToolCallPart) []int {
	indexes := make([]int, 0, len(calls))
	for i, call := range calls {
		if !r.isOutputCall(call) {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func (r *run[Deps, Output]) isOutputCall(call ToolCallPart) bool {
	return r.currentOutputTool != nil && call.ToolName == r.currentOutputTool.Name
}

func (r *run[Deps, Output]) callIsBarrier(call ToolCallPart, outputToolsConcurrent bool) bool {
	if r.agent.sequentialTools {
		return true
	}
	if r.isOutputCall(call) {
		return r.currentOutputTool.Sequential || !outputToolsConcurrent
	}
	entry, ok := r.findTool(call.ToolName)
	return ok && entry.def.Sequential
}

func (r *run[Deps, Output]) completedCallParts(outcomes []callOutcome[Output]) []RequestPart {
	parts := make([]RequestPart, 0, len(outcomes))
	for _, outcome := range outcomes {
		if outcome.err == nil && outcome.part != nil {
			parts = append(parts, outcome.part)
		}
	}
	return append(parts, r.normalizeOutcomeExtraParts(outcomes)...)
}

func (r *run[Deps, Output]) collectCallOutcomes(
	outcomes []callOutcome[Output], retryCanWin bool,
) ([]RequestPart, *Output, error) {
	winningIndex := -1
	functionRetry := false
	for index, outcome := range outcomes {
		if outcome.output != nil && winningIndex < 0 {
			winningIndex = index
		}
		if _, ok := outcome.part.(RetryPromptPart); ok && outcome.functionCall {
			functionRetry = true
		}
	}
	var winner *Output
	if winningIndex >= 0 {
		winner = outcomes[winningIndex].output
	}
	retryWon := retryCanWin && functionRetry && winner != nil
	if retryWon {
		winner = nil
	}
	pending := DeferredToolRequests{Metadata: map[string]map[string]any{}}
	seenDeferredIDs := map[string]struct{}{}
	parts := make([]RequestPart, 0, len(outcomes))
	for index := range outcomes {
		outcome := &outcomes[index]
		if outcome.output != nil {
			part := outcome.part.(ToolReturnPart)
			switch {
			case index != winningIndex:
				part.Content = outputNotFinal
			case retryWon:
				part.Content = retryWins
			}
			outcome.part = part
		}
		if outcome.deferred != nil {
			if winner != nil {
				outcome.part = ToolReturnPart{
					ToolName: outcome.deferred.call.ToolName, Content: toolSkipped,
					ToolCallID: outcome.deferred.call.ToolCallID, ToolKind: outcome.deferred.call.ToolKind,
					Outcome: ToolReturnOutcomeSuccess,
				}
			} else {
				id := outcome.deferred.call.ToolCallID
				if id == "" {
					return nil, nil, fmt.Errorf("ai: deferred tool call %q has an empty tool call ID", outcome.deferred.call.ToolName)
				}
				if _, duplicate := seenDeferredIDs[id]; duplicate {
					return nil, nil, fmt.Errorf("ai: deferred tool calls have duplicate tool call ID %q", id)
				}
				seenDeferredIDs[id] = struct{}{}
				if outcome.deferred.kind == deferredCallExternal {
					pending.Calls = append(pending.Calls, outcome.deferred.call)
				} else {
					pending.Approvals = append(pending.Approvals, outcome.deferred.call)
				}
				if len(outcome.deferred.metadata) > 0 {
					pending.Metadata[id] = cloneSchemaMap(outcome.deferred.metadata)
				}
			}
		}
		if outcome.part != nil {
			parts = append(parts, outcome.part)
		}
	}
	if len(pending.Metadata) == 0 {
		pending.Metadata = nil
	}
	if len(pending.Calls) > 0 || len(pending.Approvals) > 0 {
		r.persistDeferredToolKinds(pending)
		r.pendingDeferred = &pending
	} else {
		r.pendingDeferred = nil
	}
	if err := r.emitPendingCallResults(outcomes); err != nil {
		return nil, nil, err
	}
	return append(parts, r.normalizeOutcomeExtraParts(outcomes)...), winner, nil
}

func (r *run[Deps, Output]) normalizeOutcomeExtraParts(
	outcomes []callOutcome[Output],
) []RequestPart {
	var deltas []RequestPart
	var trailing []RequestPart
	for _, outcome := range outcomes {
		if outcome.err != nil || outcome.part == nil {
			continue
		}
		for _, extra := range outcome.extraParts {
			delta, ok := extra.(ToolAvailabilityDeltaPart)
			if !ok {
				trailing = append(trailing, extra)
				continue
			}
			added := make([]string, 0, len(delta.ToolsAdded))
			for _, name := range delta.ToolsAdded {
				if _, deferred := r.currentDeferredTools[name]; !deferred {
					continue
				}
				if _, revealed := r.revealedTools[name]; revealed {
					continue
				}
				r.revealedTools[name] = struct{}{}
				added = append(added, name)
			}
			if len(added) > 0 {
				delta.ToolsAdded = added
				deltas = append(deltas, delta)
			}
		}
	}
	return append(deltas, trailing...)
}

func (r *run[Deps, Output]) emitPendingCallResults(outcomes []callOutcome[Output]) error {
	for index := range outcomes {
		if outcomes[index].resultEmitted || outcomes[index].part == nil {
			continue
		}
		var event StreamEvent = FunctionToolResultEvent{Part: outcomes[index].part}
		if outcomes[index].outputCall {
			event = OutputToolResultEvent{Part: outcomes[index].part}
		}
		if !r.emitStreamEvent(event) {
			return context.Canceled
		}
		outcomes[index].resultEmitted = true
	}
	return nil
}

// executeCall runs one tool call. It returns the tool result, any trailing
// user content, and a validated value for a successful output tool.
func (r *run[Deps, Output]) executeCall(
	ctx context.Context, call ToolCallPart,
) (RequestPart, []RequestPart, *Output, error) {
	if r.isOutputCall(call) {
		part, output, err := r.finalizeOutputCall(ctx, call)
		return part, nil, output, err
	}
	entry, registered := r.findTool(call.ToolName)
	_, available := r.currentTools[call.ToolName]
	if registered && !available {
		if _, deferred := r.currentDeferredTools[call.ToolName]; deferred {
			if _, refused := r.availabilityRefused[call.ToolName]; refused {
				if err := r.countToolRetry(call.ToolName); err != nil {
					return nil, nil, nil, err
				}
			} else {
				r.availabilityRefused[call.ToolName] = struct{}{}
			}
			return RetryPromptPart{
				Content: fmt.Sprintf(
					"Tool %s is not available yet: search for it first, then call it again once you've seen its schema.",
					quoteToolName(call.ToolName),
				),
				ToolName: call.ToolName, ToolCallID: call.ToolCallID,
			}, nil, nil, nil
		}
	}
	if !registered || !available {
		if err := r.countToolRetry(call.ToolName); err != nil {
			return nil, nil, nil, err
		}
		return RetryPromptPart{
			Content: r.unknownToolMessage(call.ToolName), ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		}, nil, nil, nil
	}
	resolution, resolving := r.resolvingDeferred[call.ToolCallID]
	if resolving && resolution.kind == deferredCallExternal {
		return r.normalizeDeferredCallResult(call, resolution.result)
	}
	if resolving && resolution.kind == deferredCallApproval {
		switch approval := resolution.approval.(type) {
		case ToolDenied:
			message := approval.Message
			if message == "" {
				message = "The tool call was denied."
			}
			return ToolReturnPart{
				ToolName: call.ToolName, Content: message, ToolCallID: call.ToolCallID,
				ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeDenied,
			}, nil, nil, nil
		case ToolApproved:
			if len(approval.OverrideArgs) > 0 {
				call.Args = slices.Clone(approval.OverrideArgs)
			}
		}
	}
	toolRC := *r.rc
	toolRC.ToolName = call.ToolName
	toolRC.ToolCallID = call.ToolCallID
	toolRC.ToolCallApproved = resolving && resolution.kind == deferredCallApproval
	toolRC.ToolCallMetadata = cloneSchemaMap(resolution.metadata)
	toolRC.Retry, toolRC.MaxRetries = r.toolRetryInfo(call.ToolName)
	call, validated, err := r.validateToolCall(ctx, &toolRC, entry, call)
	var schemaValidation *toolArgsSchemaValidationError
	var validationFailed *ToolFailedError
	var validationRetry *RetryError
	switch {
	case errors.As(err, &schemaValidation):
		if retryErr := r.countToolRetry(call.ToolName); retryErr != nil {
			return nil, nil, nil, retryErr
		}
		return validationRetryPrompt(
			schemaValidation.err, call.Args, call.ToolName, call.ToolCallID, "invalid arguments",
		), nil, nil, nil
	case errors.As(err, &validationFailed):
		return ToolReturnPart{
			ToolName: call.ToolName, Content: validationFailed.Message, ToolCallID: call.ToolCallID,
			ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeFailed,
		}, nil, nil, nil
	case errors.As(err, &validationRetry):
		if retryErr := r.countToolRetry(call.ToolName); retryErr != nil {
			return nil, nil, nil, retryErr
		}
		return RetryPromptPart{
			Content: validationRetry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		}, nil, nil, nil
	case err != nil:
		return nil, nil, nil, fmt.Errorf("ai: tool %q: %w", call.ToolName, err)
	}
	if deferred, ok := validated.(toolHookDeferral); ok {
		return toolHookDeferredRequest(call, deferred), nil, nil, nil
	}
	if !resolving {
		switch {
		case entry.def.ExternalExecution:
			return deferredRequestPart{call: call, kind: deferredCallExternal}, nil, nil, nil
		case entry.def.RequiresApproval:
			return deferredRequestPart{
				call: call, kind: deferredCallApproval, metadata: cloneSchemaMap(entry.def.ApprovalMetadata),
			}, nil, nil, nil
		}
	}
	spanCtx, toolSpan := startToolSpan(
		ctx, call.ToolName, call.ToolCallID, !hasInstrumentationCapability(r.capabilities),
	)
	toolCtx := spanCtx
	var cancel context.CancelFunc = func() {}
	if entry.def.timeout > 0 {
		toolCtx, cancel = context.WithTimeout(spanCtx, entry.def.timeout)
	}
	content, err := r.callTool(toolCtx, &toolRC, entry, call, validated)
	if ctx.Err() == nil && errors.Is(toolCtx.Err(), context.DeadlineExceeded) {
		err = Retryf("Timed out after %s.", entry.def.timeout)
	}
	var approvalRequest *ToolApprovalRequest
	var externalRequest *ExternalToolRequest
	hookDeferred := false
	if err == nil {
		if deferred, ok := content.(toolHookDeferral); ok {
			hookDeferred = true
			call.Args, err = validatedToolArgsJSON(deferred.args)
			if err != nil {
				err = fmt.Errorf("marshal deferred arguments: %w", err)
			} else {
				approvalRequest = deferred.approval
				externalRequest = deferred.external
			}
		}
		switch request := content.(type) {
		case ToolApprovalRequest:
			approvalRequest = &request
		case *ToolApprovalRequest:
			approvalRequest = request
		case ExternalToolRequest:
			externalRequest = &request
		case *ExternalToolRequest:
			externalRequest = request
		}
		switch {
		case approvalRequest != nil && !hookDeferred && !entry.def.DynamicApproval:
			err = fmt.Errorf("tool returned ToolApprovalRequest without WithDynamicApproval")
		case externalRequest != nil && !hookDeferred && !entry.def.DynamicExternalExecution:
			err = fmt.Errorf("tool returned ExternalToolRequest without WithDynamicExternalExecution")
		case approvalRequest == nil && externalRequest == nil && containsNestedToolReturn(content):
			err = errors.New("return value contains nested ToolReturn; return ToolReturn directly")
		}
	}
	cancel()
	endSpan(toolSpan, err)
	var failed *ToolFailedError
	var retry *RetryError
	switch {
	case errors.As(err, &failed):
		return ToolReturnPart{
			ToolName: call.ToolName, Content: failed.Message, ToolCallID: call.ToolCallID, ToolKind: call.ToolKind,
			Outcome: ToolReturnOutcomeFailed,
		}, nil, nil, nil
	case errors.As(err, &retry):
		if err := r.countToolRetry(call.ToolName); err != nil {
			return nil, nil, nil, err
		}
		return RetryPromptPart{
			Content: retry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		}, nil, nil, nil
	case err != nil:
		return nil, nil, nil, fmt.Errorf("ai: tool %q: %w", call.ToolName, err)
	}
	if approvalRequest != nil {
		return deferredRequestPart{
			call: call, kind: deferredCallApproval, metadata: cloneSchemaMap(approvalRequest.Metadata),
		}, nil, nil, nil
	}
	if externalRequest != nil {
		return deferredRequestPart{
			call: call, kind: deferredCallExternal, metadata: cloneSchemaMap(externalRequest.Metadata),
		}, nil, nil, nil
	}

	part, extraParts := r.normalizeSuccessfulToolReturn(call, content)
	return part, extraParts, nil, nil
}

func (r *run[Deps, Output]) normalizeDeferredCallResult(
	call ToolCallPart, result any,
) (RequestPart, []RequestPart, *Output, error) {
	switch result := result.(type) {
	case ToolReturnPart:
		result.ToolName = call.ToolName
		result.ToolCallID = call.ToolCallID
		result.ToolKind = call.ToolKind
		if result.Outcome == "" {
			result.Outcome = ToolReturnOutcomeSuccess
		}
		return result, nil, nil, nil
	case RetryPromptPart:
		result.ToolName = call.ToolName
		result.ToolCallID = call.ToolCallID
		if err := r.countToolRetry(call.ToolName); err != nil {
			return nil, nil, nil, err
		}
		return result, nil, nil, nil
	}
	resultErr := resultAsError(result)
	var failed *ToolFailedError
	if errors.As(resultErr, &failed) {
		return ToolReturnPart{
			ToolName: call.ToolName, Content: failed.Message, ToolCallID: call.ToolCallID,
			ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeFailed,
		}, nil, nil, nil
	}
	var retry *RetryError
	if errors.As(resultErr, &retry) {
		if err := r.countToolRetry(call.ToolName); err != nil {
			return nil, nil, nil, err
		}
		return RetryPromptPart{
			Content: retry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		}, nil, nil, nil
	}
	if resultErr != nil {
		return nil, nil, nil, fmt.Errorf("ai: deferred tool %q: %w", call.ToolName, resultErr)
	}
	if containsNestedToolReturn(result) {
		return nil, nil, nil, fmt.Errorf("ai: deferred tool %q return value contains nested ToolReturn", call.ToolName)
	}
	part, extraParts := r.normalizeSuccessfulToolReturn(call, result)
	return part, extraParts, nil, nil
}

func resultAsError(result any) error {
	err, _ := result.(error)
	return err
}

func (r *run[Deps, Output]) normalizeSuccessfulToolReturn(
	call ToolCallPart, content any,
) (ToolReturnPart, []RequestPart) {
	returnValue := content
	var metadata map[string]any
	var extraParts []RequestPart
	var rich *ToolReturn
	switch value := content.(type) {
	case ToolReturn:
		rich = &value
	case *ToolReturn:
		rich = value
	}
	if rich != nil {
		returnValue = rich.ReturnValue
		metadata = cloneSchemaMap(rich.Metadata)
		if len(rich.Tools) > 0 {
			extraParts = append(extraParts, ToolAvailabilityDeltaPart{
				ToolsAdded: slices.Clone(rich.Tools), ToolCallID: call.ToolCallID,
			})
		}
		if len(rich.Content) > 0 {
			extraParts = append(extraParts, UserPromptPart{Contents: cloneUserContents(rich.Content)})
		}
	} else if _, ok := content.(*ToolReturn); ok {
		returnValue = nil
	}
	return ToolReturnPart{
		ToolName: call.ToolName, Content: returnValue, ToolCallID: call.ToolCallID, ToolKind: call.ToolKind,
		Outcome: ToolReturnOutcomeSuccess, Metadata: metadata,
	}, extraParts
}

func (r *run[Deps, Output]) unknownToolMessage(name string) string {
	available := make([]string, 0, len(r.currentTools)+1)
	for toolName := range r.currentTools {
		available = append(available, toolName)
	}
	if r.currentOutputTool != nil {
		available = append(available, r.currentOutputTool.Name)
	}
	slices.Sort(available)
	if len(available) == 0 {
		return fmt.Sprintf("Unknown tool name: %s. No tools available.", quoteToolName(name))
	}
	for i, toolName := range available {
		available[i] = quoteToolName(toolName)
	}
	return fmt.Sprintf("Unknown tool name: %s. Available tools: %s", quoteToolName(name), strings.Join(available, ", "))
}

func quoteToolName(name string) string {
	return "'" + strings.ReplaceAll(name, "'", `\'`) + "'"
}

func (r *run[Deps, Output]) finalizeOutputCall(ctx context.Context, call ToolCallPart) (RequestPart, *Output, error) {
	hookContext := r.outputHookContext(&call, true, false)
	out, err := r.validateAndProcessOutput(
		ctx, r.outputRunContext(call.ToolCallID), hookContext, call.Args,
		func(raw any) (Output, error) { return r.decodeOutput(raw, true) },
	)
	var schemaValidation *outputSchemaValidationError
	var retry *RetryError
	switch {
	case errors.As(err, &schemaValidation):
		if retryErr := r.countOutputRetry(); retryErr != nil {
			return nil, nil, retryErr
		}
		return validationRetryPrompt(
			schemaValidation.err, schemaValidation.raw, call.ToolName, call.ToolCallID, "invalid final result",
		), nil, nil
	case errors.As(err, &retry):
		if retryErr := r.countOutputRetry(); retryErr != nil {
			return nil, nil, retryErr
		}
		return RetryPromptPart{
			Content: retry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		}, nil, nil
	case err != nil:
		var decodeError *outputDecodeError
		if errors.As(err, &decodeError) {
			if retryErr := r.countOutputRetry(); retryErr != nil {
				return nil, nil, retryErr
			}
			return RetryPromptPart{
				Content:  fmt.Sprintf("invalid final result: %v", decodeError),
				ToolName: call.ToolName, ToolCallID: call.ToolCallID,
			}, nil, nil
		}
		return nil, nil, err
	}
	return ToolReturnPart{
		ToolName: call.ToolName, Content: finalResultProcessed, ToolCallID: call.ToolCallID,
		ToolKind: call.ToolKind, Outcome: ToolReturnOutcomeSuccess,
	}, &out, nil
}

// finalizeText handles a response with no tool calls. String outputs take
// the text as-is; native-mode structured outputs unmarshal it; tool-mode
// structured outputs must call the output tool, so text triggers a retry.
func (r *run[Deps, Output]) finalizeText(ctx context.Context, resp *ModelResponse) (*RunResult[Output], *RetryPromptPart, error) {
	if !r.params.AllowText {
		if err := r.countOutputRetry(); err != nil {
			return nil, nil, err
		}
		name := outputToolName
		if r.params.OutputTool != nil {
			name = r.params.OutputTool.Name
		}
		if r.currentOutputTool != nil {
			name = r.currentOutputTool.Name
		}
		return nil, &RetryPromptPart{Content: fmt.Sprintf(
			"Respond by calling the %s tool to provide the final result.", name,
		)}, nil
	}
	structured := r.params.OutputSchema != nil
	hookContext := r.outputHookContext(nil, structured, false)
	out, err := r.validateAndProcessOutput(
		ctx, r.outputRunContext(""), hookContext, resp.Text(),
		func(raw any) (Output, error) { return r.decodeOutput(raw, structured) },
	)
	var schemaValidation *outputSchemaValidationError
	var retry *RetryError
	switch {
	case errors.As(err, &schemaValidation):
		if retryErr := r.countOutputRetry(); retryErr != nil {
			return nil, nil, retryErr
		}
		part := validationRetryPrompt(
			schemaValidation.err, schemaValidation.raw, "", "", "invalid JSON output",
		)
		return nil, &part, nil
	case errors.As(err, &retry):
		if retryErr := r.countOutputRetry(); retryErr != nil {
			return nil, nil, retryErr
		}
		return nil, &RetryPromptPart{Content: retry.Message}, nil
	case err != nil:
		var decodeError *outputDecodeError
		if errors.As(err, &decodeError) {
			if retryErr := r.countOutputRetry(); retryErr != nil {
				return nil, nil, retryErr
			}
			return nil, &RetryPromptPart{Content: fmt.Sprintf("invalid JSON output: %v", decodeError)}, nil
		}
		return nil, nil, err
	}
	return r.result(out), nil, nil
}

func (r *run[Deps, Output]) appendRequest(parts []RequestPart, state RequestState) {
	timestamp := time.Now().UTC()
	r.messages = append(r.messages, ModelRequest{
		Parts: stampRequestParts(parts, timestamp), Timestamp: timestamp, RunID: r.rc.RunID,
		ConversationID: r.rc.ConversationID, State: state,
	})
}

func stampRequestParts(parts []RequestPart, timestamp time.Time) []RequestPart {
	stamped := slices.Clone(parts)
	for index, requestPart := range stamped {
		switch part := requestPart.(type) {
		case UserPromptPart:
			if part.Timestamp.IsZero() {
				part.Timestamp = timestamp
			}
			stamped[index] = part
		case ToolReturnPart:
			if part.Timestamp.IsZero() {
				part.Timestamp = timestamp
			}
			stamped[index] = part
		case RetryPromptPart:
			if part.Timestamp.IsZero() {
				part.Timestamp = timestamp
			}
			stamped[index] = part
		}
	}
	return stamped
}

func validationRetryPrompt(err error, raw json.RawMessage, toolName, toolCallID, prefix string) RetryPromptPart {
	issues := schema.ValidationIssues(err)
	if len(issues) == 0 {
		return RetryPromptPart{
			Content: prefix + ": " + err.Error(), ToolName: toolName, ToolCallID: toolCallID,
			Timestamp: time.Now().UTC(),
		}
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.UseNumber()
	var input any
	_ = decoder.Decode(&input)
	validationErrors := make([]ValidationError, len(issues))
	for index, issue := range issues {
		location := make([]any, len(issue.Location))
		value := input
		for partIndex, part := range issue.Location {
			location[partIndex] = part
			switch current := value.(type) {
			case map[string]any:
				value = current[part]
			case []any:
				itemIndex, parseErr := strconv.Atoi(part)
				if parseErr == nil && itemIndex >= 0 && itemIndex < len(current) {
					location[partIndex] = itemIndex
					value = current[itemIndex]
				}
			}
		}
		validationErrors[index] = ValidationError{
			Type: issue.Keyword, Location: location, Message: issue.Message, Input: value,
		}
	}
	return RetryPromptPart{
		Errors: validationErrors, ToolName: toolName, ToolCallID: toolCallID, Timestamp: time.Now().UTC(),
	}
}

func (r *run[Deps, Output]) recordRetry(part RetryPromptPart) {
	r.appendRequest([]RequestPart{part}, RequestStateComplete)
}

func (r *run[Deps, Output]) outputRunContext(toolCallID string) *RunContext[Deps] {
	rc := *r.rc
	rc.Retry = r.outputRetryCount()
	rc.MaxRetries = r.outputMaxRetries
	rc.ToolCallID = toolCallID
	return &rc
}

func (r *run[Deps, Output]) countToolRetry(name string) error {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	r.toolRetries[name]++
	_, maxRetries := r.toolRetryInfoLocked(name)
	if r.toolRetries[name] > maxRetries {
		return fmt.Errorf("%w: tool %q exceeded %d retries", ErrMaxRetriesExceeded, name, maxRetries)
	}
	return nil
}

func (r *run[Deps, Output]) toolRetryInfo(name string) (int, int) {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	return r.toolRetryInfoLocked(name)
}

func (r *run[Deps, Output]) toolRetryInfoLocked(name string) (int, int) {
	maxRetries := r.retryLimits.Tools
	if entry, ok := r.findTool(name); ok && entry.def.maxRetries != nil {
		maxRetries = *entry.def.maxRetries
	}
	return r.toolRetries[name], maxRetries
}

func (r *run[Deps, Output]) countOutputRetry() error {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	r.outputRetry++
	if r.outputRetry > r.outputMaxRetries {
		return fmt.Errorf("%w: output exceeded %d retries", ErrMaxRetriesExceeded, r.outputMaxRetries)
	}
	return nil
}

func (r *run[Deps, Output]) outputRetryCount() int {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	return r.outputRetry
}

func (r *run[Deps, Output]) result(out Output) *RunResult[Output] {
	usage := r.usage
	usage.ToolCalls = int(r.toolCalls.Load())
	return &RunResult[Output]{Output: out, usage: usage, messages: r.messages, newMessages: r.newMessages}
}

func (r *run[Deps, Output]) deferredResult(requests DeferredToolRequests) (*RunResult[Output], error) {
	if err := persistPendingMessages(r.messages, r.pendingMessages); err != nil {
		return nil, err
	}
	usage := r.usage
	usage.ToolCalls = int(r.toolCalls.Load())
	requests = requests.Clone()
	return &RunResult[Output]{
		usage: usage, messages: r.messages, newMessages: r.newMessages, deferred: &requests,
	}, nil
}

func (r *run[Deps, Output]) findTool(name string) (toolEntry[Deps], bool) {
	entry, ok := r.currentToolEntries[name]
	return entry, ok
}

func (a *Agent[Deps, Output]) staticInstructions(additional string, runCapabilityInstructions []string) []InstructionPart {
	parts := make([]InstructionPart, 0, len(a.capInstructions)+len(runCapabilityInstructions)+2)
	if a.instructions != "" {
		parts = append(parts, InstructionPart{Content: a.instructions})
	}
	for _, instructions := range a.capInstructions {
		parts = append(parts, InstructionPart{Content: instructions})
	}
	for _, instructions := range runCapabilityInstructions {
		parts = append(parts, InstructionPart{Content: instructions})
	}
	if additional != "" {
		parts = append(parts, InstructionPart{Content: additional})
	}
	return parts
}

func (r *run[Deps, Output]) prepareModelSettings(
	ctx context.Context, rc *RunContext[Deps],
) (ModelSettings, error) {
	var settings ModelSettings
	if defaults, ok := r.model.(ModelDefaultSettings); ok {
		modelSettings := defaults.DefaultModelSettings()
		settings = mergeModelSettings(settings, &modelSettings)
	}
	settings = mergeModelSettings(settings, &r.agent.settings)
	for _, fn := range r.agent.modelSettingsFuncs {
		rc.ModelSettings = settings
		resolved, err := fn(ctx, rc)
		if err != nil {
			return ModelSettings{}, fmt.Errorf("ai: model settings: %w", err)
		}
		settings = mergeModelSettings(settings, &resolved)
	}
	for _, layer := range r.capSettings {
		for index := range layer.static {
			settings = mergeModelSettings(settings, &layer.static[index])
		}
		if layer.provider != nil {
			resolved, err := layer.provider.ModelSettings(ctx, r.info, settings)
			if err != nil {
				return ModelSettings{}, fmt.Errorf("ai: model settings: %w", err)
			}
			settings = mergeModelSettings(settings, &resolved)
		}
	}
	settings = mergeModelSettings(settings, r.runSettings)
	for _, fn := range r.runSettingsFuncs {
		rc.ModelSettings = settings
		resolved, err := fn(ctx, rc)
		if err != nil {
			return ModelSettings{}, fmt.Errorf("ai: model settings: %w", err)
		}
		settings = mergeModelSettings(settings, &resolved)
	}
	if settings.RequestTimeout < 0 {
		return ModelSettings{}, fmt.Errorf("ai: request timeout must be non-negative, got %s", settings.RequestTimeout)
	}
	if err := validateModelSettings(settings); err != nil {
		return ModelSettings{}, err
	}
	rc.ModelSettings = settings
	r.rc.ModelSettings = settings
	return settings, nil
}

func (r *run[Deps, Output]) prepareSystemPrompts(ctx context.Context, rc *RunContext[Deps]) error {
	if r.systemPromptsPrepared {
		return nil
	}
	r.systemPromptsPrepared = true
	runners := make(map[string]systemPromptRunner[Deps])
	for _, runner := range r.agent.systemPromptFuncs {
		if runner.dynamic {
			runners[runner.id] = runner
		}
	}
	for messageIndex := range r.newMessages {
		request, ok := r.messages[messageIndex].(ModelRequest)
		if !ok {
			continue
		}
		parts := slices.Clone(request.Parts)
		for partIndex, requestPart := range parts {
			part, ok := requestPart.(SystemPromptPart)
			if !ok || part.DynamicRef == "" {
				continue
			}
			runner, ok := runners[part.DynamicRef]
			if !ok {
				continue
			}
			content, err := runner.fn(ctx, rc)
			if err != nil {
				return fmt.Errorf("ai: dynamic system prompt %q: %w", part.DynamicRef, err)
			}
			parts[partIndex] = SystemPromptPart{
				Content: content, Timestamp: time.Now().UTC(), DynamicRef: part.DynamicRef,
			}
		}
		request.Parts = parts
		r.messages[messageIndex] = request
	}
	if r.newMessages > 0 {
		return nil
	}
	parts := make([]RequestPart, 0, len(r.agent.systemPrompts)+len(r.agent.systemPromptFuncs))
	for _, content := range r.agent.systemPrompts {
		parts = append(parts, SystemPromptPart{Content: content, Timestamp: time.Now().UTC()})
	}
	for _, runner := range r.agent.systemPromptFuncs {
		content, err := runner.fn(ctx, rc)
		if err != nil {
			if runner.dynamic {
				return fmt.Errorf("ai: dynamic system prompt %q: %w", runner.id, err)
			}
			return fmt.Errorf("ai: system prompt: %w", err)
		}
		if content == "" && !runner.dynamic {
			continue
		}
		part := SystemPromptPart{Content: content, Timestamp: time.Now().UTC()}
		if runner.dynamic {
			part.DynamicRef = runner.id
		}
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		return nil
	}
	request := r.messages[len(r.messages)-1].(ModelRequest)
	request.Parts = append(parts, request.Parts...)
	r.messages[len(r.messages)-1] = request
	return nil
}

func (r *run[Deps, Output]) prepareInstructions(
	ctx context.Context, rc *RunContext[Deps],
) ([]InstructionPart, error) {
	parts := slices.Clone(r.staticInstructions)
	for _, toolset := range r.toolsets {
		instructions, err := resolveToolsetInstructions(ctx, rc, toolset)
		if err != nil {
			return nil, fmt.Errorf("ai: toolset instructions: %w", err)
		}
		parts = append(parts, instructions...)
	}
	for _, fn := range r.agent.instructionsFuncs {
		instructions, err := fn(ctx, rc)
		if err != nil {
			return nil, fmt.Errorf("ai: instructions: %w", err)
		}
		if instructions != "" {
			parts = append(parts, InstructionPart{Content: instructions, Dynamic: true})
		}
	}
	for _, capability := range r.capabilities {
		provider, ok := capability.(InstructionsProvider)
		if !ok {
			continue
		}
		instructions, err := provider.Instructions(ctx, r.info)
		if err != nil {
			return nil, fmt.Errorf("ai: instructions: %w", err)
		}
		if instructions != "" {
			parts = append(parts, InstructionPart{Content: instructions, Dynamic: true})
		}
	}
	for _, fn := range r.runInstructionsFuncs {
		instructions, err := fn(ctx, rc)
		if err != nil {
			return nil, fmt.Errorf("ai: instructions: %w", err)
		}
		if instructions != "" {
			parts = append(parts, InstructionPart{Content: instructions, Dynamic: true})
		}
	}
	return parts, nil
}

func (a *Agent[Deps, Output]) buildParams(
	instructionParts []InstructionPart,
	settings ModelSettings,
	outputMode OutputMode,
	tools []toolEntry[Deps],
) (ModelRequestParams, error) {
	instructions := make([]string, 0, len(instructionParts))
	for _, part := range instructionParts {
		instructions = append(instructions, part.Content)
	}
	params := ModelRequestParams{
		Instructions: strings.Join(instructions, "\n\n"), InstructionParts: instructionParts,
		Settings: settings, OutputMode: outputMode,
	}
	for _, entry := range tools {
		params.Tools = append(params.Tools, entry.def)
	}
	var out Output
	if _, isString := any(out).(string); isString {
		params.AllowText = true
		return params, nil
	}
	s := cloneSchemaMap(a.outputSchema)
	if s == nil {
		var err error
		s, err = schema.For(reflect.TypeFor[Output]())
		if err != nil {
			return params, fmt.Errorf("ai: output type: %w", err)
		}
	}
	params.OutputSchema = s
	return params, nil
}

func newRunID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// wrappedLoop is the run lifecycle interception point. Wrappers are outermost
// first, before hooks run in order, and after and error hooks run in reverse.
func (r *run[Deps, Output]) wrappedLoop(ctx context.Context) (*RunResult[Output], error) {
	next := RunFunc(func(ctx context.Context) (RunOutcome, error) {
		for _, capability := range r.capabilities {
			if hook, ok := capability.(BeforeRunHook); ok {
				if err := hook.BeforeRun(ctx, r.info); err != nil {
					return RunOutcome{}, err
				}
			}
		}
		result, err := r.loop(ctx)
		if err != nil {
			return RunOutcome{}, err
		}
		return runOutcomeFromResult(result), nil
	})
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		if wrapper, ok := r.capabilities[index].(RunWrapper); ok {
			innerNext := next
			next = func(ctx context.Context) (RunOutcome, error) {
				return wrapper.WrapRun(ctx, r.info, innerNext)
			}
		}
	}
	outcome, err := next(ctx)
	detached := errors.Is(context.Cause(r.ctx), errStreamDetached)
	if err != nil && !detached {
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(RunErrorHook)
			if !ok {
				continue
			}
			outcome, err = hook.OnRunError(ctx, r.info, err)
			if err == nil {
				break
			}
		}
	}
	if err == nil {
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(AfterRunHook)
			if !ok {
				continue
			}
			outcome, err = hook.AfterRun(ctx, r.info, outcome.Clone())
			if err != nil {
				break
			}
		}
	}
	cause := context.Cause(r.ctx)
	if errors.Is(cause, ErrRunCancelled) {
		usage := r.usage
		usage.ToolCalls = int(r.toolCalls.Load())
		cancelled := &RunCancelledError{messages: slices.Clone(r.messages), usage: usage}
		if persistErr := persistPendingMessages(r.messages, r.pendingMessages); persistErr != nil {
			return nil, errors.Join(cancelled, persistErr)
		}
		cancelled.messages = slices.Clone(r.messages)
		return nil, cancelled
	}
	if cause != nil {
		return nil, cause
	}
	if err != nil {
		return nil, err
	}
	return r.resultFromRunOutcome(outcome)
}

func runOutcomeFromResult[Output any](result *RunResult[Output]) RunOutcome {
	outcome := RunOutcome{Output: result.Output}
	if result.deferred != nil {
		deferred := result.deferred.Clone()
		outcome.Deferred = &deferred
	}
	return outcome
}

func (r *run[Deps, Output]) resultFromRunOutcome(outcome RunOutcome) (*RunResult[Output], error) {
	if outcome.Deferred != nil {
		return r.deferredResult(outcome.Deferred.Clone())
	}
	if outcome.Output == nil {
		var zero Output
		outputType := reflect.TypeFor[Output]()
		switch outputType.Kind() {
		case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
			return r.result(zero), nil
		}
	}
	output, ok := outcome.Output.(Output)
	if !ok {
		return nil, fmt.Errorf(
			"ai: run outcome has output type %T, expected %v", outcome.Output, reflect.TypeFor[Output](),
		)
	}
	return r.result(output), nil
}
