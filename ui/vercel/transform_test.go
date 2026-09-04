package vercel_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/ui/vercel"
)

func TestTransformStreamVariants(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		events := []ai.StreamEvent{
			ai.PartStartEvent{PartID: "text", Part: ai.TextPart{
				Content: "a", ID: "text-id", ProviderName: "openai", ProviderDetails: map[string]any{"phase": "final"},
			}},
			ai.PartDeltaEvent{PartID: "text", Delta: ai.TextPartDelta{
				ContentDelta: "b", ProviderName: "openai", ProviderDetails: map[string]any{"index": 1},
			}},
			ai.PartEndEvent{PartID: "text", Part: ai.TextPart{ID: "text-id", ProviderName: "openai"}},
			ai.PartStartEvent{PartID: "thinking", Part: ai.ThinkingPart{
				Content: "c", ID: "thinking-id", Signature: "signature", ProviderName: "anthropic",
			}},
			ai.PartDeltaEvent{PartID: "thinking", Delta: ai.ThinkingPartDelta{
				ContentDelta: "d", SignatureDelta: "next-signature", ProviderName: "anthropic",
				ProviderDetails: map[string]any{"redacted": false},
			}},
			ai.PartEndEvent{PartID: "thinking", Part: ai.ThinkingPart{
				ID: "thinking-id", Signature: "signature", ProviderName: "anthropic",
			}},
			ai.PartStartEvent{PartID: "call", Part: ai.ToolCallPart{
				ToolName: "tool", ToolCallID: "call-1", Args: []byte(`{"x":`), ID: "call-id",
				ProviderName: "openai", ToolKind: ai.ToolPartKindWebSearch,
			}},
			ai.PartDeltaEvent{PartID: "call", Delta: ai.ToolCallPartDelta{ArgsDelta: "1}"}},
			ai.PartDeltaEvent{PartID: "call", Delta: ai.ToolCallPartDelta{ToolCallID: "call-1"}},
			ai.FunctionToolCallEvent{Part: ai.ToolCallPart{ToolName: "tool", ToolCallID: "call-1", Args: []byte(`{"x":1}`)}},
			ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call-1", Content: map[string]any{"ok": true}}},
			ai.OutputToolCallEvent{Part: ai.ToolCallPart{ToolName: "final", ToolCallID: "call-2"}},
			ai.OutputToolResultEvent{Part: ai.RetryPromptPart{ToolCallID: "call-2", Content: "retry"}},
			ai.PartStartEvent{PartID: "native", Part: ai.NativeToolCallPart{ToolName: "search", ToolCallID: "native-1"}},
			ai.PartDeltaEvent{PartID: "native", Delta: ai.NativeToolCallPartDelta(ai.ToolCallPartDelta{ArgsDelta: `{}`})},
			ai.PartStartEvent{PartID: "native-return", Part: ai.NativeToolReturnPart{
				ToolCallID: "native-1", Content: "found", ProviderName: "openai",
				ProviderDetails: map[string]any{"status": "complete"}, ToolKind: ai.ToolPartKindWebSearch,
			}},
			ai.PartStartEvent{PartID: "file", Part: ai.FilePart{
				Content: ai.BinaryContent{
					Data: []byte("first"), MediaType: "image/png", VendorMetadata: map[string]any{"quality": "high"},
				},
				ID: "file-id", ProviderName: "openai", ProviderDetails: map[string]any{"kind": "image"},
			}},
			ai.PartDeltaEvent{PartID: "file", Delta: ai.FilePartDelta{Part: ai.FilePart{Content: ai.BinaryContent{
				Data: []byte("second"), MediaType: "image/png",
			}}}},
			ai.PartEndEvent{PartID: "file", Part: ai.FilePart{}},
			ai.PartStartEvent{PartID: "compaction", Part: ai.CompactionPart{
				Content: "summary", ID: "compact-1", ProviderName: "openai",
				ProviderDetails: map[string]any{"encrypted": "value"},
			}},
			ai.ToolAvailabilityDeltaEvent{Part: ai.ToolAvailabilityDeltaPart{
				ToolsAdded: []string{"search"}, ToolCallID: "reveal-1",
			}},
			ai.FinishEvent{
				FinishReason: ai.FinishReasonToolCall, Timestamp: time.Date(2026, 9, 4, 10, 0, 0, 0, time.UTC),
				Metadata: map[string]any{"request": "metadata"},
			},
		}
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	})
	var chunks []vercel.Chunk
	for chunk, err := range vercel.TransformStream(stream, "") {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if chunks[0].Type != vercel.ChunkStart || chunks[0].MessageID == "" ||
		chunks[len(chunks)-1].Type != vercel.ChunkDone {
		t.Fatalf("unexpected lifecycle: %#v", chunks)
	}
	for _, kind := range []vercel.ChunkType{
		vercel.ChunkTextStart, vercel.ChunkTextDelta, vercel.ChunkTextEnd,
		vercel.ChunkReasoningStart, vercel.ChunkReasoningDelta, vercel.ChunkReasoningEnd,
		vercel.ChunkToolInputStart, vercel.ChunkToolInputDelta, vercel.ChunkToolInputAvailable,
		vercel.ChunkToolOutputAvailable, vercel.ChunkToolOutputError, vercel.ChunkFile,
		vercel.ChunkDataCompaction, vercel.ChunkDataToolAvailability,
	} {
		if chunkIndex(chunks, kind) < 0 {
			t.Fatalf("missing %s: %#v", kind, chunks)
		}
	}
	finish := chunks[chunkIndex(chunks, vercel.ChunkFinish)]
	if finish.FinishReason != "tool-calls" {
		t.Fatalf("unexpected finish: %#v", finish)
	}
	var files []vercel.Chunk
	for _, chunk := range chunks {
		if chunk.Type == vercel.ChunkFile {
			files = append(files, chunk)
		}
	}
	if len(files) != 2 || files[0].URL != "data:image/png;base64,Zmlyc3Q=" ||
		files[1].URL != "data:image/png;base64,c2Vjb25k" || files[1].MediaType != "image/png" {
		t.Fatalf("unexpected files: %#v", files)
	}
	compaction := chunks[chunkIndex(chunks, vercel.ChunkDataCompaction)].Data
	if compaction["content"] != "summary" || compaction["id"] != "compact-1" ||
		compaction["provider_name"] != "openai" ||
		compaction["provider_details"].(map[string]any)["encrypted"] != "value" {
		t.Fatalf("unexpected compaction data: %#v", compaction)
	}
	availability := chunks[chunkIndex(chunks, vercel.ChunkDataToolAvailability)].Data
	if availability["tool_call_id"] != "reveal-1" || availability["added"].([]string)[0] != "search" {
		t.Fatalf("unexpected tool availability data: %#v", availability)
	}
	textMetadata := chunks[chunkIndex(chunks, vercel.ChunkTextStart)].ProviderMetadata["pydantic_ai"].(map[string]any)
	if textMetadata["id"] != "text-id" || textMetadata["provider_name"] != "openai" ||
		textMetadata["provider_details"].(map[string]any)["phase"] != "final" {
		t.Fatalf("unexpected text metadata: %#v", textMetadata)
	}
	finishMetadata := chunks[chunkIndex(chunks, vercel.ChunkFinish)].MessageMetadata
	if finishMetadata["request"] != "metadata" ||
		finishMetadata["pydantic_ai"].(map[string]any)["timestamp"] != "2026-09-04T10:00:00Z" {
		t.Fatalf("unexpected message metadata: %#v", finishMetadata)
	}
}

func TestTransformExternalToolMetadata(t *testing.T) {
	calls := []ai.ToolCallPart{{ToolCallID: "first"}, {ToolCallID: "second"}}
	metadata := map[string]any{"nested": map[string]any{"value": "original"}}
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.DeferredToolRequestsEvent{Requests: ai.DeferredToolRequests{Calls: calls}}, nil)
		yield(ai.DeferredToolRequestsEvent{Requests: ai.DeferredToolRequests{Calls: calls[:1]}}, nil)
		yield(ai.DeferredToolResultsEvent{Results: ai.DeferredToolResults{Calls: map[string]any{
			"first": "resolved",
		}}}, nil)
		yield(ai.FinishEvent{FinishReason: ai.FinishReasonToolCall, Metadata: metadata}, nil)
	})
	var finish map[string]any
	for chunk, err := range vercel.TransformStream(stream, "") {
		if err != nil {
			t.Fatal(err)
		}
		if chunk.Type == vercel.ChunkFinish {
			finish = chunk.MessageMetadata
		}
	}
	calls[0].ToolCallID = "changed"
	metadata["nested"].(map[string]any)["value"] = "changed"
	wrapped := finish["pydantic_ai"].(map[string]any)
	ids := wrapped["external_tool_call_ids"].([]string)
	if len(ids) != 1 || ids[0] != "second" || finish["nested"].(map[string]any)["value"] != "original" {
		t.Fatalf("unexpected external metadata: %#v", finish)
	}
	finish["nested"].(map[string]any)["value"] = "client"
	if metadata["nested"].(map[string]any)["value"] != "changed" {
		t.Fatal("finish metadata shares application metadata")
	}
}

func TestTransformApprovalVersions(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.DeferredToolRequestsEvent{Requests: ai.DeferredToolRequests{Approvals: []ai.ToolCallPart{{
			ToolCallID: "call",
		}}}}, nil)
	})
	for _, version := range []int{5, 6, 7} {
		seen := false
		for chunk, err := range vercel.TransformStreamWithConfig(stream, vercel.StreamConfig{SDKVersion: version}) {
			if err != nil {
				t.Fatal(err)
			}
			if chunk.Type == vercel.ChunkToolApprovalRequest {
				seen = true
			}
		}
		if seen != (version >= 6) {
			t.Fatalf("version=%d approval=%v", version, seen)
		}
	}
	vercel.TransformStreamWithConfig(stream, vercel.StreamConfig{SDKVersion: 6, ServerMessageID: "message"})(
		func(chunk vercel.Chunk, _ error) bool { return chunk.Type != vercel.ChunkToolApprovalRequest },
	)
	for _, version := range []int{4, 8} {
		var got error
		for _, err := range vercel.TransformStreamWithConfig(nil, vercel.StreamConfig{SDKVersion: version}) {
			got = err
		}
		if got == nil {
			t.Fatalf("version=%d: expected error", version)
		}
	}
	empty := ai.EventStream(func(func(ai.StreamEvent, error) bool) {})
	for _, err := range vercel.TransformStreamWithConfig(empty, vercel.StreamConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestTransformFinishReasons(t *testing.T) {
	tests := []struct {
		reason ai.FinishReason
		want   string
	}{
		{ai.FinishReasonStop, "stop"},
		{ai.FinishReasonLength, "length"},
		{ai.FinishReasonContentFilter, "content-filter"},
		{ai.FinishReasonToolCall, "tool-calls"},
		{ai.FinishReasonError, "error"},
		{"future", "other"},
	}
	for _, test := range tests {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			yield(ai.FinishEvent{FinishReason: test.reason}, nil)
		})
		var finish vercel.Chunk
		for chunk, err := range vercel.TransformStream(stream, "message") {
			if err != nil {
				t.Fatal(err)
			}
			if chunk.Type == vercel.ChunkFinish {
				finish = chunk
			}
		}
		if finish.FinishReason != test.want {
			t.Fatalf("reason=%q: got %q", test.reason, finish.FinishReason)
		}
	}
}

func TestTransformErrors(t *testing.T) {
	tests := []struct {
		name   string
		events []ai.StreamEvent
		err    error
		match  string
	}{
		{name: "stream", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{}}}, err: errors.New("stream failed"), match: "stream failed"},
		{name: "tool input", events: []ai.StreamEvent{ai.FunctionToolCallEvent{Part: ai.ToolCallPart{Args: []byte("{")}}}, match: "unexpected end"},
		{name: "result", events: []ai.StreamEvent{ai.FunctionToolResultEvent{Part: ai.UserPromptPart{}}}, match: "unsupported tool result"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
				for _, event := range test.events {
					yield(event, nil)
				}
				if test.err != nil {
					yield(nil, test.err)
				}
			})
			var got error
			for chunk, err := range vercel.TransformStream(stream, "message") {
				if err != nil {
					got = err
					if chunk.Type != vercel.ChunkError {
						t.Fatalf("unexpected error chunk: %#v", chunk)
					}
				}
			}
			if got == nil || !strings.Contains(got.Error(), test.match) {
				t.Fatalf("unexpected error: %v", got)
			}
		})
	}

	cancelled := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil)
		yield(nil, fmt.Errorf("cancelled: %w", ai.ErrRunCancelled))
	})
	var chunks []vercel.Chunk
	for chunk, err := range vercel.TransformStream(cancelled, "message") {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if chunks[len(chunks)-2].Type != vercel.ChunkAbort ||
		chunks[len(chunks)-2].Reason != "The agent run was cancelled." || chunks[len(chunks)-1].Type != vercel.ChunkDone {
		t.Fatalf("unexpected cancellation chunks: %#v", chunks)
	}
}

func TestTransformConsumerStops(t *testing.T) {
	streams := []ai.EventStream{
		func(yield func(ai.StreamEvent, error) bool) { yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil) },
		func(yield func(ai.StreamEvent, error) bool) { yield(ai.PartStartEvent{Part: ai.ThinkingPart{}}, nil) },
		func(yield func(ai.StreamEvent, error) bool) {
			yield(ai.PartStartEvent{Part: ai.ToolCallPart{Args: []byte(`{}`)}}, nil)
		},
		func(yield func(ai.StreamEvent, error) bool) { yield(ai.FinishEvent{}, nil) },
	}
	for _, stream := range streams {
		calls := 0
		vercel.TransformStream(stream, "message")(func(vercel.Chunk, error) bool {
			calls++
			return calls < 2
		})
		if calls != 2 {
			t.Fatalf("unexpected callback count: %d", calls)
		}
	}

	stopCases := []struct {
		events []ai.StreamEvent
		stopAt vercel.ChunkType
	}{
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{}}}, stopAt: vercel.ChunkTextStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{Content: "x"}}}, stopAt: vercel.ChunkTextDelta},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ThinkingPart{}}}, stopAt: vercel.ChunkReasoningStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ThinkingPart{Content: "x"}}}, stopAt: vercel.ChunkReasoningDelta},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ToolCallPart{}}}, stopAt: vercel.ChunkToolInputStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.NativeToolCallPart{}}}, stopAt: vercel.ChunkToolInputStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ToolCallPart{Args: []byte(`{}`)}}}, stopAt: vercel.ChunkToolInputDelta},
	}
	for _, test := range stopCases {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			for _, event := range test.events {
				yield(event, nil)
			}
		})
		vercel.TransformStream(stream, "message")(func(chunk vercel.Chunk, _ error) bool {
			return chunk.Type != test.stopAt
		})
	}

	empty := ai.EventStream(func(func(ai.StreamEvent, error) bool) {})
	vercel.TransformStream(empty, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkFinish
	})
	vercel.TransformStream(empty, "message")(func(vercel.Chunk, error) bool { return false })

	open := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil)
	})
	vercel.TransformStream(open, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkFinishStep
	})
	failed := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil)
		yield(nil, errors.New("failed"))
	})
	vercel.TransformStream(failed, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkFinishStep
	})
}
