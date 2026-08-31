package openai

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

func (m *Model) configureRequest(req *http.Request, extraHeaders map[string]string) error {
	if m.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+m.apiKey)
	}
	for name, values := range m.providerHeaders {
		req.Header.Del(name)
		for _, value := range values {
			req.Header.Add(name, value)
		}
	}
	if len(m.providerQuery) > 0 {
		query := req.URL.Query()
		for name, values := range m.providerQuery {
			query.Del(name)
			for _, value := range values {
				query.Add(name, value)
			}
		}
		req.URL.RawQuery = query.Encode()
	}
	if m.prepareRequest != nil {
		if err := m.prepareRequest(req); err != nil {
			return fmt.Errorf("openai: prepare request: %w", err)
		}
	}
	setExtraHeaders(req, extraHeaders)
	return nil
}

func (m *ResponsesModel) configureRequest(req *http.Request, extraHeaders map[string]string) error {
	return (&Model{
		apiKey: m.apiKey, providerHeaders: m.providerHeaders,
		providerQuery: m.providerQuery, prepareRequest: m.prepareRequest,
	}).configureRequest(req, extraHeaders)
}

func setExtraHeaders(req *http.Request, headers map[string]string) {
	for name, value := range headers {
		req.Header.Set(name, value)
	}
}

func cloneURLValues(values map[string][]string) map[string][]string {
	if values == nil {
		return nil
	}
	cloned := make(map[string][]string, len(values))
	for name, items := range values {
		cloned[name] = append([]string(nil), items...)
	}
	return cloned
}
