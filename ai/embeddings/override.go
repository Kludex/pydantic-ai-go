package embeddings

import "context"

type modelOverrideContextKey struct{}

// WithModel returns a context that temporarily selects model for Embedder operations.
// The override is scoped to the returned context and is safe for concurrent use.
func WithModel(ctx context.Context, model Model) context.Context {
	if embeddingModelIsNil(model) {
		panic("embeddings: model override must not be nil")
	}
	return context.WithValue(ctx, modelOverrideContextKey{}, model)
}

func (embedder *Embedder) modelForContext(ctx context.Context) Model {
	if model, ok := ctx.Value(modelOverrideContextKey{}).(Model); ok && !embeddingModelIsNil(model) {
		return embedder.instrumentModel(model)
	}
	return embedder.instrumentModel(embedder.model)
}
