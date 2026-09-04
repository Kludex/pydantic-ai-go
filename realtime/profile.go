package realtime

const DefaultAudioSampleRate = 24000

// Profile describes operations supported by one realtime model.
type Profile struct {
	SupportsImageInput        bool
	SupportsManualTurnControl bool
	SupportsInterruption      bool
	SupportsOutputTruncation  bool
	SupportsTextOutput        bool
	SupportsSessionSeeding    bool
	SupportsWebRTC            bool
	SupportsSeedingImages     bool
	SupportsSeedingAudio      bool
	SupportsThinking          bool
	SupportsAsyncToolCalls    bool
	SupportsToolReturnSchema  bool
	EmitsInputSpeechEvents    bool
	SupportedNativeTools      map[string]bool
	AudioInputSampleRate      int
	AudioOutputSampleRate     int
}

// ProfileOverride is a partial profile layer. Pointer booleans can enable or disable a claim.
type ProfileOverride struct {
	SupportsImageInput        *bool
	SupportsManualTurnControl *bool
	SupportsInterruption      *bool
	SupportsOutputTruncation  *bool
	SupportsTextOutput        *bool
	SupportsSessionSeeding    *bool
	SupportsWebRTC            *bool
	SupportsSeedingImages     *bool
	SupportsSeedingAudio      *bool
	SupportsThinking          *bool
	SupportsAsyncToolCalls    *bool
	SupportsToolReturnSchema  *bool
	EmitsInputSpeechEvents    *bool
	SupportedNativeTools      map[string]bool
	AudioInputSampleRate      int
	AudioOutputSampleRate     int
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
