// Package infer resolves provider-prefixed realtime model identifiers.
package infer

import (
	"fmt"
	"strings"

	"github.com/Kludex/pydantic-ai-go/realtime"
	googlert "github.com/Kludex/pydantic-ai-go/realtime/google"
	openairt "github.com/Kludex/pydantic-ai-go/realtime/openai"
	xairt "github.com/Kludex/pydantic-ai-go/realtime/xai"
)

// Model resolves provider:model into a realtime provider adapter.
func Model(name string) (realtime.Model, error) {
	provider, model, found := strings.Cut(name, ":")
	if !found || model == "" {
		return nil, fmt.Errorf("realtime: model identifiers use provider:model, got %q", name)
	}
	switch provider {
	case "openai":
		return openairt.NewModel(model), nil
	case "xai":
		return xairt.NewModel(model), nil
	case "google":
		return googlert.NewModel(model), nil
	case "google-cloud":
		return googlert.NewModel(model, googlert.WithVertex("", "")), nil
	default:
		return nil, fmt.Errorf("realtime: unknown provider %q", provider)
	}
}
