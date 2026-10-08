package decision

import (
	"context"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

func call(
	ctx context.Context, config Config, state any, questions map[string]Question, settings ai.ModelSettings,
) (result Response, err error) {
	tracer := trace.SpanFromContext(ctx).TracerProvider().Tracer("github.com/Kludex/pydantic-ai-go/ai")
	ctx, span := tracer.Start(ctx, "decide "+config.ModelName,
		trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(
			attribute.String("gen_ai.operation.name", "decide"),
			attribute.String("gen_ai.request.model", config.ModelName),
			attribute.String("gen_ai.provider.name", config.ProviderName),
			attribute.Int("pydantic_ai.decision.question_count", len(questions)),
		),
	)
	defer func() {
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		} else {
			span.SetAttributes(
				attribute.String("gen_ai.response.model", result.Model),
				attribute.Int("gen_ai.usage.input_tokens", result.Usage.InputTokens),
				attribute.Int("gen_ai.usage.output_tokens", result.Usage.OutputTokens),
			)
		}
		span.End()
	}()
	return config.Call(ctx, state, questions, settings)
}
