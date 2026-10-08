package systemone

import (
	"fmt"
	"math"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

const settingsKey = "system_one_settings"

// Settings combines portable request settings with decision thresholds.
type Settings struct {
	Common ai.ModelSettings
	// BooleanThreshold defaults to 0.5 and applies to every boolean answer.
	BooleanThreshold *float64
	// RouteThreshold defaults to zero, accepting every selected route.
	RouteThreshold *float64
}

// Build returns detached settings accepted by agent runs.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if _, exists := common.ExtraBody[settingsKey]; exists {
		return ai.ModelSettings{}, fmt.Errorf("systemone: extra body field %q is reserved", settingsKey)
	}
	thresholds := typedSettings{BooleanThreshold: .5}
	if settings.BooleanThreshold != nil {
		thresholds.BooleanThreshold = *settings.BooleanThreshold
	}
	if settings.RouteThreshold != nil {
		thresholds.RouteThreshold = *settings.RouteThreshold
	}
	if !probability(thresholds.BooleanThreshold) || !probability(thresholds.RouteThreshold) {
		return ai.ModelSettings{}, fmt.Errorf("systemone: thresholds must be finite and between 0 and 1")
	}
	if common.ExtraBody == nil {
		common.ExtraBody = make(map[string]any)
	}
	common.ExtraBody[settingsKey] = thresholds
	return common, nil
}

type typedSettings struct {
	BooleanThreshold float64
	RouteThreshold   float64
}

func extractSettings(settings ai.ModelSettings) (ai.ModelSettings, typedSettings, error) {
	settings = settings.Clone()
	thresholds := typedSettings{BooleanThreshold: .5}
	if value, exists := settings.ExtraBody[settingsKey]; exists {
		var valid bool
		thresholds, valid = value.(typedSettings)
		if !valid {
			return settings, thresholds, fmt.Errorf("systemone: settings must be built with Settings.Build")
		}
		delete(settings.ExtraBody, settingsKey)
	}
	if settings.RequestTimeout < 0 || settings.Temperature != nil &&
		(math.IsNaN(*settings.Temperature) || math.IsInf(*settings.Temperature, 0)) {
		return settings, thresholds, fmt.Errorf("systemone: timeout must be non-negative and temperature must be finite")
	}
	return settings, thresholds, nil
}

func probability(value float64) bool {
	return !math.IsNaN(value) && value >= 0 && value <= 1
}
