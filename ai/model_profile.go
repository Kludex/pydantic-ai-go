package ai

import (
	"slices"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/internal/contextwindow"
)

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
	// SupportsTextOutput reports whether the model can produce text. Nil defaults to true.
	SupportsTextOutput *bool
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
	// DefaultCacheRetention is the provider's documented prompt-cache lifetime.
	// Zero means the lifetime is unknown or depends on account configuration.
	DefaultCacheRetention time.Duration
	// SupportsCache reports request-side prompt-cache configuration support.
	SupportsCache bool
	// SupportsAutoCache reports server-managed moving cache boundaries.
	SupportsAutoCache bool
	// SupportedCacheRetentions lists honored request-side retention tiers, shortest first.
	SupportedCacheRetentions []CacheRetention
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
	profile.SupportedCacheRetentions = slices.Clone(profile.SupportedCacheRetentions)
	return &ProfiledModel{ModelWrapper: WrapModel(model), profile: profile}
}

// ModelProfile returns the configured profile.
func (model *ProfiledModel) ModelProfile() ModelProfile {
	profile := model.profile
	profile.SupportedCacheRetentions = slices.Clone(profile.SupportedCacheRetentions)
	return profile
}

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

// CacheOutlook predicts whether the next request can reuse a prompt cache.
type CacheOutlook string

const (
	// CacheOutlookWarm means the latest response is within the expected retention window.
	CacheOutlookWarm CacheOutlook = "warm"
	// CacheOutlookCold means the expected retention window has elapsed.
	CacheOutlookCold CacheOutlook = "cold"
	// CacheOutlookUnknown means the history or model has no usable retention boundary.
	CacheOutlookUnknown CacheOutlook = "unknown"
)

// PromptCacheOutlook predicts cache state from served history. An explicit
// retention replaces the profile default. A zero now uses the current UTC time.
func PromptCacheOutlook(
	messages []ModelMessage, profile *ModelProfile, retention *time.Duration, now time.Time,
) CacheOutlook {
	lastResponse := -1
	var timestamp time.Time
	for index := len(messages) - 1; index >= 0; index-- {
		if response, ok := messages[index].(ModelResponse); ok {
			lastResponse = index
			timestamp = response.Timestamp
			break
		}
	}
	if lastResponse < 0 || timestamp.IsZero() {
		return CacheOutlookUnknown
	}
	expected := expectedCacheRetention(messages[:lastResponse], profile, retention)
	if expected <= 0 {
		return CacheOutlookUnknown
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	if now.Sub(timestamp) <= expected {
		return CacheOutlookWarm
	}
	return CacheOutlookCold
}

func expectedCacheRetention(messages []ModelMessage, profile *ModelProfile, retention *time.Duration) time.Duration {
	expected := time.Duration(0)
	if retention != nil {
		expected = *retention
	} else if profile != nil {
		expected = profile.DefaultCacheRetention
	}
	if expected <= 0 {
		return 0
	}
	for _, message := range messages {
		if request, ok := message.(ModelRequest); ok {
			for _, part := range request.Parts {
				if prompt, ok := part.(UserPromptPart); ok {
					for _, content := range prompt.Contents {
						if point, ok := content.(CachePoint); ok {
							ttl, err := point.ResolvedTTL()
							if err == nil && ttl == CachePointTTL1Hour && expected < time.Hour &&
								(profile == nil || !profile.SupportsCache ||
									slices.Contains(profile.SupportedCacheRetentions, CacheRetention1Hour)) {
								expected = time.Hour
							}
						}
					}
				}
			}
		}
	}
	return expected
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
	if profile.DefaultCacheRetention < 0 {
		panic("ai: model profile cache retention must not be negative")
	}
	for _, tier := range profile.SupportedCacheRetentions {
		if tier == "" || tier == CacheRetentionDisabled || validateCacheConfig(&CacheConfig{Retention: tier}) != nil {
			panic("ai: model profile cache tiers must be 5m, 30m, or 1h")
		}
	}
}
