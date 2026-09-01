// Package embeddings provides provider-neutral text embeddings.
package embeddings

import (
	"context"
	"errors"
	"reflect"
)

// InputType describes how an embedding will be used.
type InputType string

const (
	// InputTypeQuery identifies text used to search a collection.
	InputTypeQuery InputType = "query"
	// InputTypeDocument identifies text stored in a searchable collection.
	InputTypeDocument InputType = "document"
)

// ErrTokenCountingUnsupported means a model cannot count tokens locally.
var ErrTokenCountingUnsupported = errors.New("embeddings: token counting is not supported")

// Model generates embeddings for one batch of text.
type Model interface {
	Embed(ctx context.Context, inputs []string, inputType InputType, settings Settings) (*Result, error)
	Name() string
	ProviderName() string
	ProviderURL() string
}

// MaxInputTokensModel reports a known input-token limit.
type MaxInputTokensModel interface {
	MaxInputTokens(ctx context.Context) (int, bool, error)
}

// TokenCountingModel counts model-specific input tokens.
type TokenCountingModel interface {
	CountTokens(ctx context.Context, text string) (int, error)
}

// Wrapper delegates every embedding operation to Model.
type Wrapper struct {
	Model
}

// ModelUnwrapper exposes the model wrapped by a decorator.
type ModelUnwrapper interface {
	UnwrapModel() Model
}

// WrapModel creates a transparent embedding model wrapper.
func WrapModel(model Model) *Wrapper {
	if embeddingModelIsNil(model) {
		panic("embeddings: cannot wrap a nil model")
	}
	return &Wrapper{Model: model}
}

// UnwrapModel returns the directly wrapped model.
func (wrapper *Wrapper) UnwrapModel() Model { return wrapper.Model }

func embeddingModelIsNil(model Model) bool {
	if model == nil {
		return true
	}
	value := reflect.ValueOf(model)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
