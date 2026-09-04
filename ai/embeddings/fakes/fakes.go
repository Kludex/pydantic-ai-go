// Package fakes provides deterministic embedding models for tests.
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
	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

var tokenSeparator = regexp.MustCompile(`[\s",.:]+`)

// Model returns deterministic vectors containing only 1.
type Model struct {
	modelName    string
	providerName string
	dimensions   int
	settings     embeddings.Settings
	sequence     atomic.Uint64
	mu           sync.RWMutex
	lastSettings embeddings.Settings
	hasSettings  bool
}

// Option configures a Model.
type Option func(*Model)

// WithName sets the result model name.
func WithName(name string) Option { return func(model *Model) { model.modelName = name } }

// WithProviderName sets the result provider name.
func WithProviderName(name string) Option { return func(model *Model) { model.providerName = name } }

// WithDimensions sets the default vector size.
func WithDimensions(dimensions int) Option {
	return func(model *Model) { model.dimensions = dimensions }
}

// WithDefaultSettings sets request defaults.
func WithDefaultSettings(settings embeddings.Settings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.settings = settings.Clone() }
}

// NewModel creates a deterministic embedding model.
func NewModel(options ...Option) *Model {
	model := &Model{modelName: "test", providerName: "test", dimensions: 8}
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

// Embed returns one detached constant vector per input.
func (model *Model) Embed(
	_ context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings,
) (*embeddings.Result, error) {
	settings = embeddings.MergeSettings(model.settings, settings)
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	model.mu.Lock()
	model.lastSettings = settings.Clone()
	model.hasSettings = true
	model.mu.Unlock()
	dimensions := model.dimensions
	if settings.Dimensions != nil {
		dimensions = *settings.Dimensions
	}
	vectors := make([][]float64, len(inputs))
	usage := ai.Usage{Requests: 1}
	for index, input := range inputs {
		vectors[index] = make([]float64, dimensions)
		for dimension := range vectors[index] {
			vectors[index][dimension] = 1
		}
		usage.InputTokens += estimateTokens(input)
	}
	return &embeddings.Result{
		Embeddings: vectors, Inputs: append([]string(nil), inputs...), InputType: inputType,
		ModelName: model.modelName, ProviderName: model.providerName, Timestamp: time.Now().UTC(), Usage: usage,
		ProviderResponseID: "test-" + strconv.FormatUint(model.sequence.Add(1), 10),
	}, nil
}

// LastSettings returns a detached snapshot from the latest completed call.
func (model *Model) LastSettings() (embeddings.Settings, bool) {
	model.mu.RLock()
	defer model.mu.RUnlock()
	return model.lastSettings.Clone(), model.hasSettings
}

// MaxInputTokens reports the fake model's fixed limit.
func (*Model) MaxInputTokens(context.Context) (int, bool, error) { return 1024, true, nil }

// CountTokens estimates tokens deterministically.
func (*Model) CountTokens(_ context.Context, text string) (int, error) {
	return estimateTokens(text), nil
}

func estimateTokens(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	return len(tokenSeparator.Split(text, -1))
}
