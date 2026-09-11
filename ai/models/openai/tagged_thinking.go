package openai

import (
	"fmt"
	"iter"
	"maps"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func splitTaggedThinking(response *ai.ModelResponse) {
	parts := make([]ai.ResponsePart, 0, len(response.Parts))
	for _, part := range response.Parts {
		text, ok := part.(ai.TextPart)
		if !ok || !strings.Contains(text.Content, "<think>") {
			parts = append(parts, part)
			continue
		}
		content := text.Content
		for {
			before, after, found := strings.Cut(content, "<think>")
			if !found {
				if content != "" {
					text.Content = content
					parts = append(parts, text)
				}
				break
			}
			if before != "" {
				text.Content = before
				parts = append(parts, text)
			}
			thinking, remaining, closed := strings.Cut(after, "</think>")
			if !closed {
				if after != "" {
					text.Content = after
					parts = append(parts, text)
				}
				break
			}
			parts = append(parts, ai.ThinkingPart{
				Content: thinking, ID: text.ID, ProviderName: text.ProviderName,
				ProviderDetails: maps.Clone(text.ProviderDetails),
			})
			content = remaining
		}
	}
	response.Parts = parts
}

type taggedSegment struct {
	content  string
	thinking bool
	group    int
}

type taggedSplitter struct {
	pending  string
	thinking bool
	emitted  bool
	group    int
}

func (splitter *taggedSplitter) push(content string, final bool) []taggedSegment {
	splitter.pending += content
	var segments []taggedSegment
	for splitter.pending != "" {
		tag := "<think>"
		if splitter.thinking {
			tag = "</think>"
		}
		if index := strings.Index(splitter.pending, tag); index >= 0 {
			if index > 0 {
				segments = append(segments, taggedSegment{
					content: splitter.pending[:index], thinking: splitter.thinking, group: splitter.group,
				})
				splitter.emitted = true
			}
			splitter.pending = splitter.pending[index+len(tag):]
			splitter.thinking = !splitter.thinking
			splitter.emitted = false
			splitter.group++
			continue
		}
		if final {
			segments = append(segments, taggedSegment{
				content: splitter.pending, thinking: splitter.thinking, group: splitter.group,
			})
			splitter.emitted = true
			splitter.pending = ""
			break
		}
		keep := 0
		for size := 1; size < len(tag) && size <= len(splitter.pending); size++ {
			if strings.HasPrefix(tag, splitter.pending[len(splitter.pending)-size:]) {
				keep = size
			}
		}
		if emit := len(splitter.pending) - keep; emit > 0 {
			segments = append(segments, taggedSegment{
				content: splitter.pending[:emit], thinking: splitter.thinking, group: splitter.group,
			})
			splitter.emitted = true
			splitter.pending = splitter.pending[emit:]
		}
		break
	}
	return segments
}

func (splitter *taggedSplitter) separateTextAfterTool() []taggedSegment {
	if splitter.thinking {
		return nil
	}
	segments := splitter.push("", true)
	if splitter.emitted {
		splitter.emitted = false
		splitter.group++
	}
	return segments
}

func splitTaggedThinkingEvents(
	stream iter.Seq2[ai.ModelStreamEvent, error],
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		splitters := map[string]*taggedSplitter{}
		providers := map[string]string{}
		var order []string
		emit := func(partID, provider string, segments []taggedSegment) bool {
			for _, segment := range segments {
				segmentID := partID
				if segment.group > 0 {
					segmentID = fmt.Sprintf("%s:%d", partID, segment.group)
				}
				if segment.thinking {
					if !yield(ai.ThinkingDeltaEvent{
						PartID: segmentID, Delta: segment.content, ProviderName: provider,
					}, nil) {
						return false
					}
				} else if !yield(ai.TextDeltaEvent{
					PartID: segmentID, Delta: segment.content, ProviderName: provider,
				}, nil) {
					return false
				}
			}
			return true
		}
		for event, err := range stream {
			if err != nil {
				for _, partID := range order {
					if !emit(partID, providers[partID], splitters[partID].push("", true)) {
						return
					}
				}
				yield(nil, err)
				return
			}
			if text, ok := event.(ai.TextDeltaEvent); ok {
				splitter := splitters[text.PartID]
				if splitter == nil {
					splitter = &taggedSplitter{}
					splitters[text.PartID] = splitter
					providers[text.PartID] = text.ProviderName
					order = append(order, text.PartID)
				}
				if !emit(text.PartID, text.ProviderName, splitter.push(text.Delta, false)) {
					return
				}
				continue
			}
			if _, ok := event.(ai.ToolCallStartEvent); ok {
				for _, partID := range order {
					if !emit(partID, providers[partID], splitters[partID].separateTextAfterTool()) {
						return
					}
				}
			}
			if _, ok := event.(ai.FinishEvent); ok {
				for _, partID := range order {
					if !emit(partID, providers[partID], splitters[partID].push("", true)) {
						return
					}
				}
			}
			if !yield(event, nil) {
				return
			}
		}
	}
}
