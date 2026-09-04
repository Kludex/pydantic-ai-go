package vercel

import (
	"context"
	"iter"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Adapter converts between Vercel AI requests and one typed agent.
type Adapter[Deps, Output any] struct {
	agent  *ai.Agent[Deps, Output]
	config Config
}

// NewAdapter creates a Vercel AI adapter. Client history is sanitized on every run.
func NewAdapter[Deps, Output any](agent *ai.Agent[Deps, Output], config Config) *Adapter[Deps, Output] {
	if agent == nil {
		panic("vercel: agent must not be nil")
	}
	if config.MaxRequestBytes < 0 {
		panic("vercel: maximum request bytes must not be negative")
	}
	if config.SDKVersion == 0 {
		config.SDKVersion = 5
	}
	if config.SDKVersion < 5 || config.SDKVersion > 7 {
		panic("vercel: SDK version must be 5, 6, or 7")
	}
	return &Adapter[Deps, Output]{agent: agent, config: config}
}

// RunStream runs one Vercel AI input and emits UI message chunks.
func (adapter *Adapter[Deps, Output]) RunStream(
	ctx context.Context, input RequestData, deps Deps, options ...ai.RunOption,
) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		prompt, history, deferred, err := prepareRunInput(input, adapter.config.Sanitization)
		if err != nil {
			yield(Chunk{Type: ChunkError, ErrorText: err.Error()}, err)
			return
		}
		runOptions := append([]ai.RunOption(nil), options...)
		runOptions = append(runOptions, ai.WithMessageHistory(history), ai.WithConversationID(input.ID))
		if deferred != nil {
			runOptions = append(runOptions, ai.WithDeferredToolResults(*deferred))
		}
		stream := adapter.agent.RunStream(ctx, prompt.Content, deps, runOptions...)
		for chunk, eventErr := range TransformStreamWithConfig(stream.Events(), StreamConfig{
			SDKVersion: adapter.config.SDKVersion, ServerMessageID: adapter.config.ServerMessageID,
		}) {
			if !yield(chunk, eventErr) || eventErr != nil {
				return
			}
		}
	}
}
