package embeddings

import (
	"context"
	"fmt"
	"reflect"
	"slices"
)

// Embedder provides query and document operations over one embedding model.
type Embedder struct {
	model    Model
	settings Settings
}

// Option configures an Embedder.
type Option func(*Embedder)

// WithSettings sets defaults overridden by per-call settings.
func WithSettings(settings Settings) Option {
	settings = settings.Clone()
	return func(embedder *Embedder) { embedder.settings = settings.Clone() }
}

// New creates an Embedder.
func New(model Model, options ...Option) *Embedder {
	if embeddingModelIsNil(model) {
		panic("embeddings: model must not be nil")
	}
	embedder := &Embedder{model: model}
	for _, option := range options {
		option(embedder)
	}
	return embedder
}

// Model returns the configured model.
func (embedder *Embedder) Model() Model { return embedder.model }

// EmbedQuery embeds one search query.
func (embedder *Embedder) EmbedQuery(ctx context.Context, query string, settings ...Settings) (*Result, error) {
	return embedder.Embed(ctx, []string{query}, InputTypeQuery, settings...)
}

// EmbedQueries embeds a batch of search queries.
func (embedder *Embedder) EmbedQueries(
	ctx context.Context, queries []string, settings ...Settings,
) (*Result, error) {
	return embedder.Embed(ctx, queries, InputTypeQuery, settings...)
}

// EmbedDocument embeds one searchable document.
func (embedder *Embedder) EmbedDocument(
	ctx context.Context, document string, settings ...Settings,
) (*Result, error) {
	return embedder.Embed(ctx, []string{document}, InputTypeDocument, settings...)
}

// EmbedDocuments embeds a batch of searchable documents.
func (embedder *Embedder) EmbedDocuments(
	ctx context.Context, documents []string, settings ...Settings,
) (*Result, error) {
	return embedder.Embed(ctx, documents, InputTypeDocument, settings...)
}

// Embed performs one embedding operation with an explicit input type.
func (embedder *Embedder) Embed(
	ctx context.Context, inputs []string, inputType InputType, settings ...Settings,
) (*Result, error) {
	if err := validateRequest(inputs, inputType); err != nil {
		return nil, err
	}
	if len(settings) > 1 {
		return nil, fmt.Errorf("embeddings: expected at most one per-call settings value")
	}
	var overrides Settings
	if len(settings) == 1 {
		overrides = settings[0]
	}
	merged := MergeSettings(embedder.settings, overrides)
	if err := merged.Validate(); err != nil {
		return nil, err
	}
	result, err := embedder.model.Embed(ctx, slices.Clone(inputs), inputType, merged)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("embeddings: model returned a nil result")
	}
	if len(result.Embeddings) != len(inputs) {
		return nil, fmt.Errorf(
			"embeddings: model returned %d vectors for %d inputs", len(result.Embeddings), len(inputs),
		)
	}
	return cloneResult(result), nil
}

// MaxInputTokens returns the selected model's known input limit.
func (embedder *Embedder) MaxInputTokens(ctx context.Context) (int, bool, error) {
	model, ok := modelCapability[MaxInputTokensModel](embedder.model)
	if !ok {
		return 0, false, nil
	}
	return model.MaxInputTokens(ctx)
}

// CountTokens counts input tokens when the selected model supports it.
func (embedder *Embedder) CountTokens(ctx context.Context, text string) (int, error) {
	model, ok := modelCapability[TokenCountingModel](embedder.model)
	if !ok {
		return 0, ErrTokenCountingUnsupported
	}
	return model.CountTokens(ctx, text)
}

func modelCapability[T any](model Model) (T, bool) {
	seen := map[modelIdentity]struct{}{}
	for range 100 {
		if embeddingModelIsNil(model) {
			break
		}
		if capability, ok := any(model).(T); ok {
			return capability, true
		}
		wrapper, ok := model.(ModelUnwrapper)
		if !ok {
			break
		}
		identity, identifiable := embeddingModelIdentity(model)
		if identifiable {
			if _, exists := seen[identity]; exists {
				break
			}
			seen[identity] = struct{}{}
		}
		model = wrapper.UnwrapModel()
	}
	var zero T
	return zero, false
}

type modelIdentity struct {
	typeOf  reflect.Type
	pointer uintptr
}

func embeddingModelIdentity(model Model) (modelIdentity, bool) {
	value := reflect.ValueOf(model)
	if value.Kind() != reflect.Pointer {
		return modelIdentity{}, false
	}
	return modelIdentity{typeOf: value.Type(), pointer: value.Pointer()}, true
}

func validateRequest(inputs []string, inputType InputType) error {
	switch inputType {
	case InputTypeQuery, InputTypeDocument:
	default:
		return fmt.Errorf("embeddings: invalid input type %q", inputType)
	}
	if len(inputs) == 0 {
		return fmt.Errorf("embeddings: inputs must not be empty")
	}
	return nil
}
