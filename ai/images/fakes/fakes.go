// Package fakes provides deterministic image generation models for tests.
package fakes

import (
	"context"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
)

var (
	tokenSeparator = regexp.MustCompile(`[\s",.:]+`)
	tinyPNG        = []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	}
)

// Model returns a deterministic image without a network request.
type Model struct {
	modelName    string
	providerName string
	settings     images.Settings
	sequence     atomic.Uint64
	mu           sync.RWMutex
	lastInputs   []images.Input
	lastSettings images.Settings
	hasCall      bool
}

// Option configures a Model.
type Option func(*Model)

// WithName sets the result model name.
func WithName(name string) Option { return func(model *Model) { model.modelName = name } }

// WithProviderName sets the result provider name.
func WithProviderName(name string) Option { return func(model *Model) { model.providerName = name } }

// WithDefaultSettings sets request defaults.
func WithDefaultSettings(settings images.Settings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.settings = settings.Clone() }
}

// NewModel creates a deterministic image model.
func NewModel(options ...Option) *Model {
	model := &Model{modelName: "test", providerName: "test"}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the model name.
func (model *Model) Name() string { return model.modelName }

// ProviderName returns the provider name.
func (model *Model) ProviderName() string { return model.providerName }

// ProviderURL reports no remote endpoint.
func (*Model) ProviderURL() string { return "" }

// DefaultSettings returns detached model defaults.
func (model *Model) DefaultSettings() images.Settings { return model.settings.Clone() }

// Generate returns one detached 1x1 PNG.
func (model *Model) Generate(
	_ context.Context, prompt string, inputs []images.Input, settings images.Settings,
) (*images.Result, error) {
	settings = images.MergeSettings(model.settings, settings)
	prompt, inputs, settings, err := images.PrepareRequest(prompt, inputs, settings)
	if err != nil {
		return nil, err
	}
	model.mu.Lock()
	model.lastInputs = append([]images.Input(nil), inputs...)
	model.lastSettings = settings.Clone()
	model.hasCall = true
	model.mu.Unlock()
	return &images.Result{
		Images: []images.GeneratedImage{{
			Content:      ai.BinaryContent{Data: append([]byte(nil), tinyPNG...), MediaType: "image/png"},
			OutputFormat: "png",
		}},
		Prompt: prompt, ModelName: model.modelName, ProviderName: model.providerName,
		Timestamp: time.Now().UTC(), Usage: ai.Usage{Requests: 1, InputTokens: estimateTokens(prompt)},
		ProviderResponseID: "test-" + strconv.FormatUint(model.sequence.Add(1), 10),
	}, nil
}

// LastCall returns detached inputs and settings from the latest completed call.
func (model *Model) LastCall() ([]images.Input, images.Settings, bool) {
	model.mu.RLock()
	defer model.mu.RUnlock()
	_, inputs, _, _ := images.PrepareRequest("snapshot", model.lastInputs, images.Settings{})
	return inputs, model.lastSettings.Clone(), model.hasCall
}

func estimateTokens(text string) int {
	return len(tokenSeparator.Split(strings.TrimSpace(text), -1))
}
