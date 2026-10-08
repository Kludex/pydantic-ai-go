package ai

import (
	"container/list"
	"sync"
	"time"
)

type cacheKey struct{ provider, endpoint, model string }

type cacheMark struct {
	established       int
	lastSeen          time.Time
	compactions       int
	alerted           bool
	notEnabledAlerted bool
}

type conversationCacheMarks struct {
	id      string
	marks   map[cacheKey]cacheMark
	updated time.Time
	element *list.Element
}

var promptCacheMarks = struct {
	sync.Mutex
	conversations map[string]*conversationCacheMarks
	order         list.List
}{conversations: make(map[string]*conversationCacheMarks)}

func bindCacheMarks(id string) *conversationCacheMarks {
	promptCacheMarks.Lock()
	defer promptCacheMarks.Unlock()
	pruneCacheMarks(time.Now())
	if marks := promptCacheMarks.conversations[id]; marks != nil {
		return marks
	}
	return &conversationCacheMarks{id: id, marks: make(map[cacheKey]cacheMark)}
}

func pruneCacheMarks(now time.Time) {
	for element := promptCacheMarks.order.Front(); element != nil; element = promptCacheMarks.order.Front() {
		marks := element.Value.(*conversationCacheMarks)
		if len(promptCacheMarks.conversations) <= 4096 && now.Sub(marks.updated) <= 24*time.Hour {
			break
		}
		delete(promptCacheMarks.conversations, marks.id)
		promptCacheMarks.order.Remove(element)
		marks.element = nil
	}
}

func observeCacheHealth(
	marks *conversationCacheMarks, messages []ModelMessage, response, measured *ModelResponse, retention time.Duration,
	notEnabled bool,
) *cacheHealth {
	// ponytail: one process lock; shard by conversation only if telemetry contention matters.
	promptCacheMarks.Lock()
	defer promptCacheMarks.Unlock()
	if stored := promptCacheMarks.conversations[marks.id]; stored != nil && stored != marks {
		for key, mark := range marks.marks {
			if _, exists := stored.marks[key]; !exists {
				stored.marks[key] = mark
			}
		}
		marks.marks = stored.marks
		marks = stored
	}
	key := cacheKey{response.ProviderName, response.ProviderURL, response.ModelName}
	mark := marks.marks[key]
	usage := measured.Usage
	read, write := usage.CacheReadTokens, usage.CacheWriteTokens
	unreported := read == 0 && write == 0
	if unreported && mark.established == 0 {
		if notEnabled && !mark.notEnabledAlerted {
			now := time.Now()
			mark.notEnabledAlerted = true
			marks.marks[key] = mark
			storeCacheMarks(marks, now)
			return &cacheHealth{notEnabled: true}
		}
		return nil
	}
	now := time.Now()
	compactions := 0
	for _, message := range append(messages[:len(messages):len(messages)], *response) {
		if response, ok := message.(ModelResponse); ok {
			for _, part := range response.Parts {
				if _, ok := part.(CompactionPart); ok {
					compactions++
				}
			}
		}
	}
	health := &cacheHealth{established: mark.established, previous: mark.established, missed: mark.established - read}
	if usage.InputTokens != 0 {
		health.ratio = float64(read) / float64(usage.InputTokens)
	}
	if health.missed >= 2000 && float64(health.missed) > float64(mark.established)*0.05 {
		switch {
		case unreported:
			health.reason = "unreported"
		case compactions > mark.compactions:
			health.reason = "compacted"
		case retention <= 0:
			health.reason = "unknown"
		case now.Sub(mark.lastSeen) > retention:
			health.reason = "ttl_expired"
		default:
			health.reason = "unexpected"
		}
		health.alert = health.reason == "unexpected" && !mark.alerted
	}
	if unreported {
		return health
	}
	passes, counted := usage.Details["message_iterations"]
	summed := counted && passes+usage.Details["compaction_iterations"] > 1
	if _, separate := usage.Details["tool_use_prompt_tokens"]; !counted && !separate {
		for _, part := range measured.Parts {
			if _, native := part.(NativeToolCallPart); native {
				summed = true
			}
		}
	}
	if summed {
		if mark.lastSeen.IsZero() {
			return health
		}
		mark.alerted = mark.alerted || health.alert
	} else {
		if health.reason == "" {
			mark.established = max(mark.established, read+write)
			mark.alerted = false
		} else {
			mark.established = read + write
			mark.alerted = mark.alerted || health.alert
		}
		mark.compactions = compactions
	}
	mark.lastSeen = now
	marks.marks[key] = mark
	health.established = mark.established
	storeCacheMarks(marks, now)
	return health
}

func storeCacheMarks(marks *conversationCacheMarks, now time.Time) {
	if marks.id != "" {
		marks.updated = now
		promptCacheMarks.conversations[marks.id] = marks
		if marks.element == nil {
			marks.element = promptCacheMarks.order.PushBack(marks)
		} else {
			promptCacheMarks.order.MoveToBack(marks.element)
		}
		pruneCacheMarks(now)
	}
}
