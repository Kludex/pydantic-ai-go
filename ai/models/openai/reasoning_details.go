package openai

import (
	"fmt"
	"strconv"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type reasoningDetail struct {
	ID        *string `json:"id"`
	Format    *string `json:"format"`
	Index     *int    `json:"index"`
	Type      string  `json:"type"`
	Text      string  `json:"text,omitempty"`
	Summary   string  `json:"summary,omitempty"`
	Data      string  `json:"data,omitempty"`
	Signature string  `json:"signature,omitempty"`
}

func thinkingPartFromReasoningDetail(detail reasoningDetail, providerName string) (ai.ThinkingPart, error) {
	part := ai.ThinkingPart{
		ProviderName:    providerName,
		ProviderDetails: map[string]any{"type": detail.Type, "format": nil, "index": nil},
	}
	if detail.ID != nil {
		part.ID = *detail.ID
	}
	if detail.Format != nil {
		part.ProviderDetails["format"] = *detail.Format
	}
	if detail.Index != nil {
		part.ProviderDetails["index"] = *detail.Index
	}
	switch detail.Type {
	case "reasoning.text":
		part.Content = detail.Text
		part.Signature = detail.Signature
	case "reasoning.summary":
		part.Content = detail.Summary
	case "reasoning.encrypted":
		part.Signature = detail.Data
	default:
		return ai.ThinkingPart{}, fmt.Errorf("openai: unknown reasoning detail type %q", detail.Type)
	}
	return part, nil
}

func reasoningDetailFromThinkingPart(part ai.ThinkingPart) (reasoningDetail, bool) {
	typeName, ok := part.ProviderDetails["type"].(string)
	if !ok {
		return reasoningDetail{}, false
	}
	detail := reasoningDetail{Type: typeName}
	if part.ID != "" {
		detail.ID = stringPointer(part.ID)
	}
	if format, ok := part.ProviderDetails["format"].(string); ok {
		detail.Format = stringPointer(format)
	}
	if index, ok := reasoningDetailIndex(part.ProviderDetails["index"]); ok {
		detail.Index = &index
	}
	switch typeName {
	case "reasoning.text":
		detail.Text = part.Content
		detail.Signature = part.Signature
	case "reasoning.summary":
		detail.Summary = part.Content
	case "reasoning.encrypted":
		if part.Signature == "" {
			return reasoningDetail{}, false
		}
		detail.Data = part.Signature
	default:
		return reasoningDetail{}, false
	}
	return detail, true
}

func reasoningDetailIndex(value any) (int, bool) {
	switch index := value.(type) {
	case int:
		return index, true
	case float64:
		return int(index), index == float64(int(index))
	default:
		return 0, false
	}
}

func reasoningDetailPartID(detail reasoningDetail, fallbackIndex int) string {
	index := fallbackIndex
	if detail.Index != nil {
		index = *detail.Index
	}
	return "reasoning_detail_" + detail.Type + "_" + strconv.Itoa(index)
}

func stringPointer(value string) *string { return &value }
