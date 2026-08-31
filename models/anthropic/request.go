package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
)

func marshalRequest(payload any, extra map[string]any) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	if extra == nil {
		return body, nil
	}
	var fields map[string]any
	// Typed request payloads always encode as JSON objects.
	_ = json.Unmarshal(body, &fields)
	for name, value := range extra {
		if _, exists := fields[name]; exists {
			return nil, fmt.Errorf("extra body field %q conflicts with a typed request field", name)
		}
		fields[name] = value
	}
	return json.Marshal(fields)
}

func setExtraHeaders(req *http.Request, headers map[string]string) {
	for name, value := range headers {
		req.Header.Set(name, value)
	}
}
