package google

import (
	"fmt"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) prepareHTTPRequest(request *http.Request, settings ai.ModelSettings) error {
	request.Header.Set("Content-Type", "application/json")
	if model.apiKey != "" {
		request.Header.Set("x-goog-api-key", model.apiKey)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return fmt.Errorf("google: prepare request: %w", err)
		}
	}
	if model.transport == TransportVertexAI {
		switch settings.ServiceTier {
		case ai.ServiceTierDefault:
			request.Header.Set("X-Vertex-AI-LLM-Request-Type", "shared")
		case ai.ServiceTierFlex:
			request.Header.Set("X-Vertex-AI-LLM-Shared-Request-Type", "flex")
		case ai.ServiceTierPriority:
			request.Header.Set("X-Vertex-AI-LLM-Shared-Request-Type", "priority")
		}
	}
	setExtraHeaders(request, settings.ExtraHeaders)
	return nil
}

func setExtraHeaders(request *http.Request, headers map[string]string) {
	for name, value := range headers {
		request.Header.Set(name, value)
	}
}
