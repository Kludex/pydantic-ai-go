package images

import (
	"context"
	"fmt"
)

// Generator provides one high-level interface over a direct image model.
type Generator struct {
	model           Model
	settings        Settings
	instrumentation *instrumentationSelection
}

// Option configures a Generator.
type Option func(*Generator)

// WithSettings sets defaults overridden by per-call settings.
func WithSettings(settings Settings) Option {
	settings = settings.Clone()
	return func(generator *Generator) { generator.settings = settings.Clone() }
}

// New creates a Generator.
func New(model Model, options ...Option) *Generator {
	if modelIsNil(model) {
		panic("images: model must not be nil")
	}
	generator := &Generator{model: model}
	for _, option := range options {
		option(generator)
	}
	return generator
}

// Model returns the configured model.
func (generator *Generator) Model() Model { return generator.model }

// Generate creates images from prompt and optional references.
func (generator *Generator) Generate(
	ctx context.Context, prompt string, inputs []Input, settings ...Settings,
) (*Result, error) {
	if len(settings) > 1 {
		return nil, fmt.Errorf("images: expected at most one per-call settings value")
	}
	var override Settings
	if len(settings) == 1 {
		override = settings[0]
	}
	model := generator.modelForContext(ctx)
	merged := MergeSettings(modelDefaultSettings(model), MergeSettings(generator.settings, override))
	preparedPrompt, preparedInputs, merged, err := PrepareRequest(prompt, inputs, merged)
	if err != nil {
		return nil, err
	}
	result, err := model.Generate(ctx, preparedPrompt, preparedInputs, merged)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("images: model returned a nil result")
	}
	if len(result.Images) == 0 {
		return nil, fmt.Errorf("images: model returned no generated images")
	}
	return cloneResult(result), nil
}
