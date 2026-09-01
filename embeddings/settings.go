package embeddings

import (
	"fmt"
	"maps"
)

// Settings configures an embedding request.
type Settings struct {
	Dimensions   *int
	Truncate     *bool
	ExtraHeaders map[string]string
	ExtraBody    map[string]any
}

// Validate checks portable embedding settings.
func (settings Settings) Validate() error {
	if settings.Dimensions != nil && *settings.Dimensions <= 0 {
		return fmt.Errorf("embeddings: dimensions must be greater than zero")
	}
	return nil
}

// Clone returns a detached settings value.
func (settings Settings) Clone() Settings {
	settings.Dimensions = clonePointer(settings.Dimensions)
	settings.Truncate = clonePointer(settings.Truncate)
	settings.ExtraHeaders = maps.Clone(settings.ExtraHeaders)
	settings.ExtraBody = cloneMap(settings.ExtraBody)
	return settings
}

// MergeSettings overlays non-zero override fields on base and returns a detached value.
func MergeSettings(base, override Settings) Settings {
	merged := base.Clone()
	if override.Dimensions != nil {
		merged.Dimensions = clonePointer(override.Dimensions)
	}
	if override.Truncate != nil {
		merged.Truncate = clonePointer(override.Truncate)
	}
	if override.ExtraHeaders != nil {
		merged.ExtraHeaders = maps.Clone(override.ExtraHeaders)
	}
	if override.ExtraBody != nil {
		merged.ExtraBody = cloneMap(override.ExtraBody)
	}
	return merged
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
