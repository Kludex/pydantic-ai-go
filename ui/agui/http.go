package agui

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go"
)

// Handler returns an HTTP handler that accepts RunAgentInput JSON and streams
// AG-UI events as server-sent events. Deps and options must be safe to reuse.
func (adapter *Adapter[Deps, Output]) Handler(deps Deps, options ...ai.RunOption) http.Handler {
	options = append([]ai.RunOption(nil), options...)
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			response.Header().Set("Allow", http.MethodPost)
			http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		maximum := adapter.config.MaxRequestBytes
		if maximum == 0 {
			maximum = 10 << 20
		}
		body, err := io.ReadAll(io.LimitReader(request.Body, maximum+1))
		if err != nil {
			http.Error(response, "read request body", http.StatusBadRequest)
			return
		}
		if int64(len(body)) > maximum {
			http.Error(response, "request body too large", http.StatusRequestEntityTooLarge)
			return
		}
		var input RunAgentInput
		if err := json.Unmarshal(body, &input); err != nil {
			http.Error(response, "invalid AG-UI request", http.StatusBadRequest)
			return
		}
		response.Header().Set("Content-Type", "text/event-stream")
		response.Header().Set("Cache-Control", "no-cache")
		response.Header().Set("X-Accel-Buffering", "no")
		flusher, _ := response.(http.Flusher)
		for event, eventErr := range adapter.RunStream(request.Context(), input, deps, options...) {
			encoded, _ := json.Marshal(event)
			if _, err := fmt.Fprintf(response, "data: %s\n\n", encoded); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			if eventErr != nil {
				return
			}
		}
	})
}
