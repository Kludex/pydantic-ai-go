package images

import "context"

type modelOverrideContextKey struct{}

// WithModel returns a context that temporarily selects model for Generator operations.
// The override is scoped to the returned context and is safe for concurrent use.
func WithModel(ctx context.Context, model Model) context.Context {
	if modelIsNil(model) {
		panic("images: model override must not be nil")
	}
	return context.WithValue(ctx, modelOverrideContextKey{}, model)
}

func (generator *Generator) modelForContext(ctx context.Context) Model {
	if model, ok := ctx.Value(modelOverrideContextKey{}).(Model); ok && !modelIsNil(model) {
		return generator.instrumentModel(model)
	}
	return generator.instrumentModel(generator.model)
}
