package ai

// ModelProfile describes model-specific structured-output behavior. The zero
// value defaults reflected output to a function tool and uses the standard
// prompted-output template.
type ModelProfile struct {
	DefaultOutputMode          OutputMode
	PromptedOutputTemplate     string
	NativeOutputRequiresPrompt bool
}

// ModelProfiler is implemented by models that expose structured-output defaults.
type ModelProfiler interface {
	ModelProfile() ModelProfile
}

// ModelOutputProfileDispatcher is implemented by composite models that resolve
// OutputModeAuto independently for each selected child model.
type ModelOutputProfileDispatcher interface {
	DispatchesOutputProfile() bool
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
