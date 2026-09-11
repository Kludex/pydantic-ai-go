package openaiprotocol

// InterruptsResponseOnSpeech reports whether a session configuration enables server-side cancellation on speech onset.
func InterruptsResponseOnSpeech(config map[string]any, enabledByDefault bool) bool {
	turn, found := config["turn_detection"]
	if !found {
		if audio, ok := config["audio"].(map[string]any); ok {
			if input, ok := audio["input"].(map[string]any); ok {
				turn, found = input["turn_detection"]
			}
		}
	}
	if !found {
		return enabledByDefault
	}
	settings, ok := turn.(map[string]any)
	if !ok || settings == nil {
		return false
	}
	interrupts, ok := settings["interrupt_response"].(bool)
	if !ok {
		return enabledByDefault
	}
	return interrupts
}
