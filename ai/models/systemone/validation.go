package systemone

import (
	"encoding/json"
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
)

type wireResponse struct {
	Model   *string                        `json:"model"`
	Answers map[string]decision.WireAnswer `json:"answers"`
	Usage   *struct {
		InputTokens  json.RawMessage `json:"input_tokens"`
		OutputTokens json.RawMessage `json:"output_tokens"`
	} `json:"usage"`
	inputTokens  int
	outputTokens int
}

func (response *wireResponse) validate() error {
	if response.Model == nil || response.Answers == nil || response.Usage == nil {
		return fmt.Errorf("missing model, answers, or valid usage")
	}
	for _, field := range []struct {
		raw   json.RawMessage
		value *int
	}{
		{response.Usage.InputTokens, &response.inputTokens}, {response.Usage.OutputTokens, &response.outputTokens},
	} {
		if len(field.raw) == 0 {
			continue
		}
		if string(field.raw) == "null" {
			return fmt.Errorf("usage tokens must be non-negative integers")
		}
		if err := json.Unmarshal(field.raw, field.value); err != nil || *field.value < 0 {
			return fmt.Errorf("usage tokens must be non-negative integers")
		}
	}
	return nil
}
