package realtime

// DefaultAudioSampleRate is the fallback PCM sample rate in hertz.
const DefaultAudioSampleRate = 24000

// AsyncToolCallMode describes whether generation can continue while tools run.
type AsyncToolCallMode string

const (
	// AsyncToolCallsNever makes the model wait for tool results.
	AsyncToolCallsNever AsyncToolCallMode = "never"
	// AsyncToolCallsOptional lets the session choose whether the model waits.
	AsyncToolCallsOptional AsyncToolCallMode = "optional"
	// AsyncToolCallsAlways keeps generation active regardless of the session setting.
	AsyncToolCallsAlways AsyncToolCallMode = "always"
)

// Profile describes operations supported by one realtime model.
type Profile struct {
	// SupportsImageInput allows discrete image or video frames.
	SupportsImageInput bool
	// ImageInputRequiresResponse requires images to start delegated work.
	ImageInputRequiresResponse bool
	// SynthesizesTurnBoundary reports that turn completion is inferred from silence.
	SynthesizesTurnBoundary bool
	// SupportsManualTurnControl allows commit, clear, and create-response commands.
	SupportsManualTurnControl bool
	// SupportsInterruption allows cancellation of active output.
	SupportsInterruption bool
	// SupportsOutputTruncation allows provider history to drop unheard audio.
	SupportsOutputTruncation bool
	// SupportsTextOutput allows plain text instead of speech output.
	SupportsTextOutput bool
	// SupportsSessionSeeding allows portable prior messages at connection time.
	SupportsSessionSeeding bool
	// SupportsWebRTC allows browser signaling and sideband control.
	SupportsWebRTC bool
	// SupportsSeedingImages allows prior inline images.
	SupportsSeedingImages bool
	// SupportsSeedingAudio allows retained user audio.
	SupportsSeedingAudio bool
	// SupportsThinking allows portable reasoning settings.
	SupportsThinking bool
	// SupportsAsyncToolCalls allows generation while local tools run.
	SupportsAsyncToolCalls bool
	// AsyncToolCallMode reports whether asynchronous generation is optional or required.
	AsyncToolCallMode AsyncToolCallMode
	// SupportsToolReturnSchema allows native response schemas on function declarations.
	SupportsToolReturnSchema bool
	// EmitsInputSpeechEvents reports server-side speech boundaries.
	EmitsInputSpeechEvents bool
	// SupportedNativeTools maps provider-neutral native tool kinds to support.
	SupportedNativeTools map[string]bool
	// AudioInputSampleRate is the required PCM input rate in hertz.
	AudioInputSampleRate int
	// AudioOutputSampleRate is the produced PCM output rate in hertz.
	AudioOutputSampleRate int
	// ContextWindow is the maximum combined input and output token count.
	// Zero means the limit is unknown.
	ContextWindow int
	// UsageExcludesContextWindow prevents provider usage from being treated as current context occupancy.
	UsageExcludesContextWindow bool
}

// ProfileOverride is a partial profile layer. Pointer fields distinguish omitted and zero values.
type ProfileOverride struct {
	// SupportsImageInput overrides image input support.
	SupportsImageInput *bool
	// ImageInputRequiresResponse overrides image response requirements.
	ImageInputRequiresResponse *bool
	// SynthesizesTurnBoundary overrides inferred turn completion.
	SynthesizesTurnBoundary *bool
	// SupportsManualTurnControl overrides manual turn support.
	SupportsManualTurnControl *bool
	// SupportsInterruption overrides response interruption support.
	SupportsInterruption *bool
	// SupportsOutputTruncation overrides output truncation support.
	SupportsOutputTruncation *bool
	// SupportsTextOutput overrides plain text output support.
	SupportsTextOutput *bool
	// SupportsSessionSeeding overrides history seeding support.
	SupportsSessionSeeding *bool
	// SupportsWebRTC overrides browser WebRTC support.
	SupportsWebRTC *bool
	// SupportsSeedingImages overrides history image support.
	SupportsSeedingImages *bool
	// SupportsSeedingAudio overrides history audio support.
	SupportsSeedingAudio *bool
	// SupportsThinking overrides reasoning configuration support.
	SupportsThinking *bool
	// SupportsAsyncToolCalls overrides asynchronous tool support.
	SupportsAsyncToolCalls *bool
	// AsyncToolCallMode overrides the model's asynchronous generation behavior.
	AsyncToolCallMode *AsyncToolCallMode
	// SupportsToolReturnSchema overrides function response-schema support.
	SupportsToolReturnSchema *bool
	// EmitsInputSpeechEvents overrides speech-boundary reporting.
	EmitsInputSpeechEvents *bool
	// SupportedNativeTools replaces the supported native-tool map when non-nil.
	SupportedNativeTools map[string]bool
	// AudioInputSampleRate replaces the input rate when positive.
	AudioInputSampleRate int
	// AudioOutputSampleRate replaces the output rate when positive.
	AudioOutputSampleRate int
	// ContextWindow replaces the context window. Point to zero to keep it unknown.
	ContextWindow *int
	// UsageExcludesContextWindow overrides whether usage measures context occupancy.
	UsageExcludesContextWindow *bool
}

// DefaultProfile returns conservative provider-neutral defaults.
func DefaultProfile() Profile {
	return Profile{
		SupportsTextOutput:    true,
		AsyncToolCallMode:     AsyncToolCallsNever,
		SupportedNativeTools:  map[string]bool{},
		AudioInputSampleRate:  DefaultAudioSampleRate,
		AudioOutputSampleRate: DefaultAudioSampleRate,
	}
}

// MergeProfile applies one partial override to a detached copy of base.
func MergeProfile(base Profile, override ProfileOverride) Profile {
	base = cloneProfile(base)
	mergeBool := func(target *bool, value *bool) {
		if value != nil {
			*target = *value
		}
	}
	mergeBool(&base.SupportsImageInput, override.SupportsImageInput)
	mergeBool(&base.ImageInputRequiresResponse, override.ImageInputRequiresResponse)
	mergeBool(&base.SynthesizesTurnBoundary, override.SynthesizesTurnBoundary)
	mergeBool(&base.SupportsManualTurnControl, override.SupportsManualTurnControl)
	mergeBool(&base.SupportsInterruption, override.SupportsInterruption)
	mergeBool(&base.SupportsOutputTruncation, override.SupportsOutputTruncation)
	mergeBool(&base.SupportsTextOutput, override.SupportsTextOutput)
	mergeBool(&base.SupportsSessionSeeding, override.SupportsSessionSeeding)
	mergeBool(&base.SupportsWebRTC, override.SupportsWebRTC)
	mergeBool(&base.SupportsSeedingImages, override.SupportsSeedingImages)
	mergeBool(&base.SupportsSeedingAudio, override.SupportsSeedingAudio)
	mergeBool(&base.SupportsThinking, override.SupportsThinking)
	mergeBool(&base.SupportsAsyncToolCalls, override.SupportsAsyncToolCalls)
	if override.AsyncToolCallMode != nil {
		base.AsyncToolCallMode = *override.AsyncToolCallMode
	}
	mergeBool(&base.SupportsToolReturnSchema, override.SupportsToolReturnSchema)
	mergeBool(&base.EmitsInputSpeechEvents, override.EmitsInputSpeechEvents)
	mergeBool(&base.UsageExcludesContextWindow, override.UsageExcludesContextWindow)
	if override.SupportedNativeTools != nil {
		base.SupportedNativeTools = cloneBoolMap(override.SupportedNativeTools)
	}
	if override.AudioInputSampleRate != 0 {
		base.AudioInputSampleRate = override.AudioInputSampleRate
	}
	if override.AudioOutputSampleRate != 0 {
		base.AudioOutputSampleRate = override.AudioOutputSampleRate
	}
	if override.ContextWindow != nil {
		base.ContextWindow = *override.ContextWindow
	}
	return base
}

func cloneProfile(profile Profile) Profile {
	profile.SupportedNativeTools = cloneBoolMap(profile.SupportedNativeTools)
	return profile
}

func cloneBoolMap(values map[string]bool) map[string]bool {
	if values == nil {
		return nil
	}
	cloned := make(map[string]bool, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}
