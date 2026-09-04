// Package infer resolves provider-prefixed realtime model identifiers.
package infer

import (
	"fmt"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	azurert "github.com/Kludex/pydantic-ai-go/ai/realtime/azure"
	googlert "github.com/Kludex/pydantic-ai-go/ai/realtime/google"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	xairt "github.com/Kludex/pydantic-ai-go/ai/realtime/xai"
)

// Model resolves provider:model into a realtime provider adapter.
func Model(name string) (realtime.Model, error) {
	provider, model, found := strings.Cut(name, ":")
	if !found || model == "" {
		return nil, fmt.Errorf("realtime: model identifiers use provider:model, got %q", name)
	}
	switch provider {
	case "azure":
		return azurert.NewModel(model, azurert.Config{})
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
