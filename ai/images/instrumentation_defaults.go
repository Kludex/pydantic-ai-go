package images

import (
	"sync"
	"sync/atomic"
)

type instrumentationSelection struct {
	enabled bool
	options []InstrumentationOption
}

var defaultInstrumentation atomic.Pointer[instrumentationSelection]

// WithInstrumentation enables OpenTelemetry instrumentation for one Generator.
func WithInstrumentation(options ...InstrumentationOption) Option {
	selection := &instrumentationSelection{enabled: true, options: append([]InstrumentationOption(nil), options...)}
	return func(generator *Generator) { generator.instrumentation = selection }
}

// WithoutInstrumentation disables instrumentation for one Generator.
func WithoutInstrumentation() Option {
	selection := &instrumentationSelection{}
	return func(generator *Generator) { generator.instrumentation = selection }
}

// InstrumentAll enables instrumentation for generators without an explicit instrumentation option.
// The returned function restores the previous default if no newer default replaced it.
func InstrumentAll(options ...InstrumentationOption) func() {
	return setDefaultInstrumentation(&instrumentationSelection{
		enabled: true, options: append([]InstrumentationOption(nil), options...),
	})
}

// DisableInstrumentation disables default instrumentation for generators without an explicit option.
func DisableInstrumentation() func() {
	return setDefaultInstrumentation(&instrumentationSelection{})
}

func setDefaultInstrumentation(selection *instrumentationSelection) func() {
	previous := defaultInstrumentation.Swap(selection)
	var once sync.Once
	return func() {
		once.Do(func() { defaultInstrumentation.CompareAndSwap(selection, previous) })
	}
}

func (generator *Generator) instrumentModel(model Model) Model {
	selection := generator.instrumentation
	if selection == nil {
		selection = defaultInstrumentation.Load()
	}
	if selection == nil || !selection.enabled {
		return model
	}
	return InstrumentModel(model, selection.options...)
}
