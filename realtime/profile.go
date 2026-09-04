package realtime

// DefaultAudioSampleRate is the fallback PCM sample rate in hertz.
const DefaultAudioSampleRate = 24000

// Profile describes operations supported by one realtime model.
type Profile struct {
	// SupportsImageInput allows discrete image or video frames.
	SupportsImageInput bool
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
}

// ProfileOverride is a partial profile layer. Pointer booleans can enable or disable a claim.
type ProfileOverride struct {
	// SupportsImageInput overrides image input support.
	SupportsImageInput *bool
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
}

// DefaultProfile returns conservative provider-neutral defaults.
func DefaultProfile() Profile {
	return Profile{
		SupportsTextOutput:    true,
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
	mergeBool(&base.SupportsToolReturnSchema, override.SupportsToolReturnSchema)
	mergeBool(&base.EmitsInputSpeechEvents, override.EmitsInputSpeechEvents)
	if override.SupportedNativeTools != nil {
		base.SupportedNativeTools = cloneBoolMap(override.SupportedNativeTools)
	}
	if override.AudioInputSampleRate != 0 {
		base.AudioInputSampleRate = override.AudioInputSampleRate
	}
	if override.AudioOutputSampleRate != 0 {
		base.AudioOutputSampleRate = override.AudioOutputSampleRate
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
