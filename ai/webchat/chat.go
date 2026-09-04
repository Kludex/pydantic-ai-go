package webchat

import (
	"bytes"
	"encoding/json"
	"io"
	"mime"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/ui/vercel"
)

func newChatHandler[Deps, Output any](
	adapter *vercel.Adapter[Deps, Output], deps Deps, baseOptions []ai.RunOption,
	configuration *frontendConfiguration, maximum int64,
) http.Handler {
	baseOptions = append([]ai.RunOption(nil), baseOptions...)
	if maximum == 0 {
		maximum = 10 << 20
	}
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method == http.MethodOptions {
			response.WriteHeader(http.StatusOK)
			return
		}
		if request.Method != http.MethodPost {
			response.Header().Set("Allow", http.MethodPost+", "+http.MethodOptions)
			writeJSONError(response, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
		if err != nil || mediaType != "application/json" {
			writeJSONError(response, http.StatusUnsupportedMediaType, "expected Content-Type application/json")
			return
		}
		body, err := io.ReadAll(io.LimitReader(request.Body, maximum+1))
		if err != nil {
			writeJSONError(response, http.StatusBadRequest, "read request body")
			return
		}
		if int64(len(body)) > maximum {
			writeJSONError(response, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		var input vercel.RequestData
		if err := json.Unmarshal(body, &input); err != nil {
			writeJSONError(response, http.StatusBadRequest, "invalid Vercel AI request")
			return
		}
		selected, err := configuration.runOptions(input.Model, input.BuiltinTools)
		if err != nil {
			writeJSONError(response, http.StatusBadRequest, err.Error())
			return
		}
		request.Body = io.NopCloser(bytes.NewReader(body))
		options := append([]ai.RunOption(nil), baseOptions...)
		options = append(options, selected...)
		adapter.Handler(deps, options...).ServeHTTP(response, request)
	})
}
