package ai

import "github.com/Kludex/pydantic-ai-go/ai/internal/contextwindow"

// ModelProfile describes model-specific output, message-preparation, and
// context-window behavior. The zero value defaults reflected output to a
// function tool, uses the standard prompted-output template, converts realtime
// speech to transcripts, and leaves the context window unknown.
type ModelProfile struct {
	// DefaultOutputMode resolves OutputModeAuto for this model.
	DefaultOutputMode OutputMode
	// PromptedOutputTemplate formats provider-specific JSON instructions.
	PromptedOutputTemplate string
	// NativeOutputRequiresPrompt adds prompted guidance beside native schema enforcement.
	NativeOutputRequiresPrompt bool
	// SupportsImageOutput allows a FilePart with an image media type as final output.
	SupportsImageOutput bool
	// SupportsAudioInput allows retained SpeechPart audio to replace its transcript
	// when realtime history is prepared for a standard model.
	SupportsAudioInput bool
	// SupportsToolAvailabilityDelta lets the provider render tool reveals directly.
	// Other models receive a provider-neutral tool-search call and result.
	SupportsToolAvailabilityDelta bool
	// ContextWindow is the maximum combined input and output token count.
	// Zero means the limit is unknown.
	ContextWindow int
}

// ModelProfiler is implemented by models that expose output, message-preparation,
// and context-window defaults.
type ModelProfiler interface {
	// ModelProfile returns model-specific output, input, and context behavior.
	ModelProfile() ModelProfile
}

// ModelOutputProfileDispatcher is implemented by composite models that resolve
// OutputModeAuto independently for each selected child model.
type ModelOutputProfileDispatcher interface {
	// DispatchesOutputProfile reports whether children resolve output profiles independently.
	DispatchesOutputProfile() bool
}

// ModelMessageProfileDispatcher is implemented by composite models that prepare
// message histories independently for each selected child model.
type ModelMessageProfileDispatcher interface {
	// DispatchesMessageProfile reports whether children prepare histories independently.
	DispatchesMessageProfile() bool
}

// ProfiledModel applies a profile to any model while preserving optional model capabilities.
type ProfiledModel struct {
	*ModelWrapper
	profile ModelProfile
}

// NewProfiledModel applies profile to model. The profile is resolved for each
// request, so OutputModeAuto also works with model selection and concurrent runs.
func NewProfiledModel(model Model, profile ModelProfile) *ProfiledModel {
	if modelIsNil(model) {
		panic("ai: profiled model must not be nil")
	}
	validateModelProfile(profile)
	return &ProfiledModel{ModelWrapper: WrapModel(model), profile: profile}
}

// ModelProfile returns the configured profile.
func (model *ProfiledModel) ModelProfile() ModelProfile { return model.profile }

// ContextWindow returns the configured context window. Zero means unknown.
func (model *ProfiledModel) ContextWindow() int { return model.profile.ContextWindow }

// DispatchesOutputProfile reports that this explicit outer profile resolves before delegation.
func (*ProfiledModel) DispatchesOutputProfile() bool { return false }

// DispatchesMessageProfile reports that this explicit outer profile prepares messages before delegation.
func (*ProfiledModel) DispatchesMessageProfile() bool { return false }

// ModelProfile delegates profile discovery to the wrapped model.
func (wrapper *ModelWrapper) ModelProfile() ModelProfile { return modelProfile(wrapper.wrapped) }

// ContextWindow delegates context-window discovery to the wrapped model.
func (wrapper *ModelWrapper) ContextWindow() int { return modelContextWindow(wrapper.wrapped) }

// SupportsToolAvailabilityDelta delegates request-specific reveal support.
func (wrapper *ModelWrapper) SupportsToolAvailabilityDelta(params ModelRequestParams) bool {
	if model, ok := wrapper.wrapped.(ToolAvailabilityDeltaModel); ok {
		return model.SupportsToolAvailabilityDelta(params)
	}
	return modelProfile(wrapper.wrapped).SupportsToolAvailabilityDelta
}

// DispatchesOutputProfile reports whether the wrapped composite resolves profiles per child model.
func (wrapper *ModelWrapper) DispatchesOutputProfile() bool {
	model, ok := wrapper.wrapped.(ModelOutputProfileDispatcher)
	return ok && model.DispatchesOutputProfile()
}

// DispatchesMessageProfile reports whether the wrapped composite prepares messages per child model.
func (wrapper *ModelWrapper) DispatchesMessageProfile() bool {
	model, ok := wrapper.wrapped.(ModelMessageProfileDispatcher)
	return ok && model.DispatchesMessageProfile()
}

func modelProfile(model Model) ModelProfile {
	profile := ModelProfile{DefaultOutputMode: OutputModeTool}
	if profiled, ok := model.(ModelProfiler); ok {
		profile = profiled.ModelProfile()
	}
	if profile.ContextWindow == 0 {
		profile.ContextWindow = modelContextWindow(model)
	}
	return profile
}

func modelContextWindow(model Model) int {
	if modelIsNil(model) {
		return 0
	}
	if windowed, ok := model.(ModelContextWindow); ok {
		return windowed.ContextWindow()
	}
	if profiled, ok := model.(ModelProfiler); ok {
		if window := profiled.ModelProfile().ContextWindow; window != 0 {
			return window
		}
	}
	if identified, ok := model.(ModelProviderIdentity); ok {
		return contextwindow.Lookup(model.Name(), identified.ProviderName(), identified.ProviderURL())
	}
	return 0
}

func validateModelProfile(profile ModelProfile) {
	switch profile.DefaultOutputMode {
	case OutputModeTool, OutputModeNative, OutputModePrompted:
	default:
		panic("ai: model profile default output mode must be tool, native, or prompted")
	}
	if profile.ContextWindow < 0 {
		panic("ai: model profile context window must not be negative")
	}
}
