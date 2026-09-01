package ai

// ModelProfile describes model-specific output and message-preparation behavior.
// The zero value defaults reflected output to a function tool, uses the standard
// prompted-output template, and converts realtime speech to transcripts.
type ModelProfile struct {
	DefaultOutputMode          OutputMode
	PromptedOutputTemplate     string
	NativeOutputRequiresPrompt bool
	// SupportsImageOutput allows a FilePart with an image media type as final output.
	SupportsImageOutput bool
	// SupportsAudioInput allows retained SpeechPart audio to replace its transcript
	// when realtime history is prepared for a standard model.
	SupportsAudioInput bool
}

// ModelProfiler is implemented by models that expose output and message-preparation defaults.
type ModelProfiler interface {
	ModelProfile() ModelProfile
}

// ModelOutputProfileDispatcher is implemented by composite models that resolve
// OutputModeAuto independently for each selected child model.
type ModelOutputProfileDispatcher interface {
	DispatchesOutputProfile() bool
}

// ModelMessageProfileDispatcher is implemented by composite models that prepare
// message histories independently for each selected child model.
type ModelMessageProfileDispatcher interface {
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

// DispatchesOutputProfile reports that this explicit outer profile resolves before delegation.
func (*ProfiledModel) DispatchesOutputProfile() bool { return false }

// DispatchesMessageProfile reports that this explicit outer profile prepares messages before delegation.
func (*ProfiledModel) DispatchesMessageProfile() bool { return false }

func (wrapper *ModelWrapper) ModelProfile() ModelProfile {
	if model, ok := wrapper.wrapped.(ModelProfiler); ok {
		return model.ModelProfile()
	}
	return ModelProfile{DefaultOutputMode: OutputModeTool}
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
	if profiled, ok := model.(ModelProfiler); ok {
		return profiled.ModelProfile()
	}
	return ModelProfile{DefaultOutputMode: OutputModeTool}
}

func validateModelProfile(profile ModelProfile) {
	switch profile.DefaultOutputMode {
	case OutputModeTool, OutputModeNative, OutputModePrompted:
	default:
		panic("ai: model profile default output mode must be tool, native, or prompted")
	}
}
