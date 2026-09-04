package google

import (
	"context"
	"encoding/json"
	"fmt"
)

// CountTokens counts text through the configured Google transport.
func (model *Model) CountTokens(ctx context.Context, text string) (int, error) {
	payload := struct {
		Contents []geminiContent `json:"contents"`
	}{Contents: []geminiContent{{Role: "user", Parts: []geminiPart{{Text: text}}}}}
	body, _ := json.Marshal(payload)
	endpoint := fmt.Sprintf("%s/models/%s:countTokens", model.baseURL, model.name)
	responseBody, err := model.doRequest(ctx, endpoint, body, nil)
	if err != nil {
		return 0, err
	}
	var response struct {
		TotalTokens *int `json:"totalTokens"`
	}
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return 0, fmt.Errorf("google embeddings: decode token count response: %w", err)
	}
	if response.TotalTokens == nil {
		return 0, fmt.Errorf("google embeddings: token count response omitted totalTokens")
	}
	return *response.TotalTokens, nil
}
