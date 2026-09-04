package vercel_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ui/vercel"
)

func TestTransformStreamVariants(t *testing.T) {
	valid := true
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
			ai.FunctionToolCallEvent{
				Part:      ai.ToolCallPart{ToolName: "tool", ToolCallID: "call-1", Args: []byte(`{"x":1}`)},
				ArgsValid: &valid,
			},
			ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call-1", Content: map[string]any{"ok": true}}},
			ai.OutputToolCallEvent{Part: ai.ToolCallPart{ToolName: "final", ToolCallID: "call-2"}},
			ai.OutputToolResultEvent{Part: ai.RetryPromptPart{ToolCallID: "call-2", Content: "retry"}},
			ai.PartStartEvent{PartID: "native", Part: ai.NativeToolCallPart{ToolName: "search", ToolCallID: "native-1"}},
			ai.PartDeltaEvent{PartID: "native", Delta: ai.NativeToolCallPartDelta(ai.ToolCallPartDelta{ArgsDelta: `{}`})},
			ai.PartEndEvent{PartID: "native", Part: ai.NativeToolCallPart{
				ToolName: "search", ToolCallID: "native-1", Args: []byte(`{}`),
			}},
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
	compaction := chunks[chunkIndex(chunks, vercel.ChunkDataCompaction)].Data.(map[string]any)
	if compaction["content"] != "summary" || compaction["id"] != "compact-1" ||
		compaction["provider_name"] != "openai" ||
		compaction["provider_details"].(map[string]any)["encrypted"] != "value" {
		t.Fatalf("unexpected compaction data: %#v", compaction)
	}
	availability := chunks[chunkIndex(chunks, vercel.ChunkDataToolAvailability)].Data.(map[string]any)
	if availability["tool_call_id"] != "reveal-1" || availability["added"].([]string)[0] != "search" {
		t.Fatalf("unexpected tool availability data: %#v", availability)
	}
	textMetadata := chunks[chunkIndex(chunks, vercel.ChunkTextStart)].ProviderMetadata["pydantic_ai"].(map[string]any)
	if textMetadata["id"] != "text-id" || textMetadata["provider_name"] != "openai" ||
		textMetadata["provider_details"].(map[string]any)["phase"] != "final" {
		t.Fatalf("unexpected text metadata: %#v", textMetadata)
	}
	finishMetadata := chunks[chunkIndex(chunks, vercel.ChunkMessageMetadata)].MessageMetadata
	if finishMetadata["request"] != "metadata" ||
		finishMetadata["pydantic_ai"].(map[string]any)["timestamp"] != "2026-09-04T10:00:00Z" {
		t.Fatalf("unexpected message metadata: %#v", finishMetadata)
	}
}

func TestTransformToolResultChunks(t *testing.T) {
	data := map[string]any{"nested": map[string]any{"value": "original"}}
	metadata, err := vercel.ToolResultMetadata(
		vercel.Chunk{Type: "data-custom", ID: "data-1", Data: data},
		vercel.Chunk{Type: "data-null", Transient: true},
		vercel.Chunk{
			Type: vercel.ChunkSourceURL, SourceID: "source-1", URL: "https://example.com", Title: "Example",
			ProviderMetadata: map[string]any{"provider": map[string]any{"id": "citation"}},
		},
		vercel.Chunk{
			Type: vercel.ChunkSourceDocument, SourceID: "source-2", MediaType: "application/pdf",
			Title: "Document", Filename: "document.pdf",
		},
		vercel.Chunk{Type: vercel.ChunkFile, URL: "data:text/plain;base64,aGk=", MediaType: "text/plain"},
	)
	if err != nil {
		t.Fatal(err)
	}
	data["nested"].(map[string]any)["value"] = "changed"
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{
			ToolCallID: "call", Content: "done", Metadata: metadata,
		}}, nil)
	})
	var chunks []vercel.Chunk
	for chunk, err := range vercel.TransformStream(stream, "") {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	custom := chunks[chunkIndex(chunks, "data-custom")]
	if custom.ID != "data-1" || custom.Data.(map[string]any)["nested"].(map[string]any)["value"] != "original" {
		t.Fatalf("unexpected custom data chunk: %#v", custom)
	}
	nullChunk := chunks[chunkIndex(chunks, "data-null")]
	encoded, err := json.Marshal(nullChunk)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"data":null`) || !strings.Contains(string(encoded), `"transient":true`) {
		t.Fatalf("unexpected null data chunk: %s", encoded)
	}
	source := chunks[chunkIndex(chunks, vercel.ChunkSourceURL)]
	document := chunks[chunkIndex(chunks, vercel.ChunkSourceDocument)]
	file := chunks[chunkIndex(chunks, vercel.ChunkFile)]
	if source.SourceID != "source-1" || source.Title != "Example" ||
		source.ProviderMetadata["provider"].(map[string]any)["id"] != "citation" ||
		document.SourceID != "source-2" || document.Filename != "document.pdf" ||
		file.MediaType != "text/plain" {
		t.Fatalf("unexpected data-carrying chunks: source=%#v document=%#v file=%#v", source, document, file)
	}
	custom.Data.(map[string]any)["nested"].(map[string]any)["value"] = "client"
	stored := metadata["pydantic_ai_go_vercel_chunks"].([]any)[0].(map[string]any)
	if stored["data"].(map[string]any)["nested"].(map[string]any)["value"] != "original" {
		t.Fatal("emitted custom data shares tool metadata")
	}
}

func TestToolResultMetadataValidation(t *testing.T) {
	tests := []vercel.Chunk{
		{Type: vercel.ChunkStart},
		{Type: vercel.ChunkSourceURL},
		{Type: vercel.ChunkSourceDocument},
		{Type: vercel.ChunkFile},
		{Type: "data-bad", Data: func() {}},
	}
	for _, chunk := range tests {
		if _, err := vercel.ToolResultMetadata(chunk); err == nil {
			t.Fatalf("expected validation error for %#v", chunk)
		}
	}
	for _, value := range []any{
		"invalid",
		[]any{func() {}},
		[]any{map[string]any{"type": 1}},
		[]any{map[string]any{"type": "start"}},
	} {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			yield(ai.OutputToolResultEvent{Part: ai.ToolReturnPart{Metadata: map[string]any{
				"pydantic_ai_go_vercel_chunks": value,
			}}}, nil)
		})
		var got error
		for _, err := range vercel.TransformStream(stream, "") {
			got = err
		}
		if got == nil {
			t.Fatalf("expected malformed tool result metadata error for %#v", value)
		}
	}
}

func TestTransformToolResultConsumerStops(t *testing.T) {
	metadata, err := vercel.ToolResultMetadata(vercel.Chunk{Type: "data-custom", Data: "value"})
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range []ai.StreamEvent{
		ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call", Content: "done", Metadata: metadata}},
		ai.OutputToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call", Content: "done", Metadata: metadata}},
	} {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) { yield(event, nil) })
		vercel.TransformStream(stream, "")(func(chunk vercel.Chunk, _ error) bool {
			return chunk.Type != vercel.ChunkToolOutputAvailable
		})
	}
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{
			ToolCallID: "call", Content: "done", Metadata: metadata,
		}}, nil)
	})
	vercel.TransformStream(stream, "")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != "data-custom"
	})
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
		if chunk.Type == vercel.ChunkMessageMetadata {
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

func TestTransformToolValidationVersions(t *testing.T) {
	invalid := false
	for _, version := range []int{5, 6} {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			call := ai.ToolCallPart{
				ToolName: "weather", ToolCallID: "call", Args: []byte(`{"city":1}`), ID: "provider-call",
				ProviderName: "openai",
			}
			yield(ai.PartStartEvent{PartID: "part", Part: call}, nil)
			yield(ai.FunctionToolCallEvent{Part: call, ArgsValid: &invalid}, nil)
			yield(ai.FunctionToolResultEvent{Part: ai.RetryPromptPart{
				ToolName: "weather", ToolCallID: "call", Content: "city must be a string",
			}}, nil)
		})
		var chunks []vercel.Chunk
		for chunk, err := range vercel.TransformStreamWithConfig(stream, vercel.StreamConfig{SDKVersion: version}) {
			if err != nil {
				t.Fatal(err)
			}
			chunks = append(chunks, chunk)
		}
		start := chunks[chunkIndex(chunks, vercel.ChunkToolInputStart)]
		if (start.ProviderMetadata != nil) != (version >= 6) {
			t.Fatalf("version=%d start metadata=%#v", version, start.ProviderMetadata)
		}
		if version == 5 {
			if chunkIndex(chunks, vercel.ChunkToolInputAvailable) < 0 ||
				chunkIndex(chunks, vercel.ChunkToolOutputError) < 0 {
				t.Fatalf("v5 validation lifecycle: %#v", chunks)
			}
		} else {
			index := chunkIndex(chunks, vercel.ChunkToolInputError)
			if index < 0 || chunks[index].ToolName != "weather" || chunks[index].ErrorText == "" ||
				chunkIndex(chunks, vercel.ChunkToolInputAvailable) >= 0 ||
				chunkIndex(chunks, vercel.ChunkToolOutputError) >= 0 {
				t.Fatalf("v6 validation lifecycle: %#v", chunks)
			}
		}
	}
}

func TestTransformInvalidToolResultEdges(t *testing.T) {
	invalid := false
	for _, test := range []struct {
		args   string
		result ai.RequestPart
	}{
		{args: `{`, result: ai.RetryPromptPart{ToolCallID: "call", Content: "invalid"}},
		{args: `{}`, result: ai.ToolReturnPart{ToolCallID: "call", Content: "invalid"}},
	} {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			call := ai.ToolCallPart{ToolName: "tool", ToolCallID: "call", Args: []byte(test.args)}
			yield(ai.PartStartEvent{PartID: "call", Part: call}, nil)
			yield(ai.FunctionToolCallEvent{Part: call, ArgsValid: &invalid}, nil)
			yield(ai.FunctionToolResultEvent{Part: test.result}, nil)
		})
		var inputError vercel.Chunk
		for chunk, err := range vercel.TransformStreamWithConfig(stream, vercel.StreamConfig{SDKVersion: 6}) {
			if err != nil {
				t.Fatal(err)
			}
			if chunk.Type == vercel.ChunkToolInputError {
				inputError = chunk
			}
		}
		if !strings.Contains(inputError.ErrorText, "invalid") {
			t.Fatalf("unexpected input error: %#v", inputError)
		}
		if test.args == `{` && inputError.Input.(map[string]any)["INVALID_JSON"] != `{` {
			t.Fatalf("malformed input was not preserved: %#v", inputError.Input)
		}
	}
}

func TestTransformToolLifecycleEdges(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		pending := ai.ToolCallPart{ToolName: "pending", ToolCallID: "pending", Args: []byte(`{}`)}
		yield(ai.PartStartEvent{PartID: "pending", Part: pending}, nil)
		yield(ai.FunctionToolCallEvent{Part: pending}, nil)
		backfill := ai.ToolCallPart{ToolName: "backfill", ToolCallID: "backfill", Args: []byte(`{"x":`)}
		yield(ai.PartStartEvent{PartID: "backfill", Part: backfill}, nil)
		backfill.Args = []byte(`{"x":1}`)
		yield(ai.PartEndEvent{PartID: "backfill", Part: backfill}, nil)
		yield(ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{
			ToolName: "backfill", ToolCallID: "backfill", Content: "done",
		}}, nil)
		yield(ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{
			ToolName: "denied", ToolCallID: "denied", Content: "not allowed", Outcome: ai.ToolReturnOutcomeDenied,
		}}, nil)
		yield(ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{
			ToolName: "interrupted", ToolCallID: "interrupted", Content: "stopped",
			Outcome: ai.ToolReturnOutcomeInterrupted,
		}}, nil)
		yield(ai.PartStartEvent{PartID: "native", Part: ai.NativeToolReturnPart{
			ToolName: "native", ToolCallID: "native", Content: "denied", Outcome: ai.ToolReturnOutcomeDenied,
		}}, nil)
	})
	var chunks []vercel.Chunk
	for chunk, err := range vercel.TransformStreamWithConfig(stream, vercel.StreamConfig{SDKVersion: 6}) {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	available := 0
	denied := 0
	for _, chunk := range chunks {
		switch chunk.Type {
		case vercel.ChunkToolInputAvailable:
			available++
		case vercel.ChunkToolOutputDenied:
			denied++
		}
	}
	if available != 2 || denied != 2 {
		t.Fatalf("unexpected edge lifecycle: %#v", chunks)
	}
	interrupted := false
	backfilled := false
	for _, chunk := range chunks {
		if chunk.Type == vercel.ChunkToolOutputAvailable && chunk.ToolCallID == "interrupted" {
			interrupted = true
		}
		if chunk.Type == vercel.ChunkToolInputAvailable && chunk.ToolCallID == "backfill" &&
			chunk.Input.(map[string]any)["x"] == float64(1) {
			backfilled = true
		}
	}
	if !interrupted || !backfilled {
		t.Fatalf("tool lifecycle was not preserved: %#v", chunks)
	}
	v5 := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{
			ToolCallID: "denied", Content: "not allowed", Outcome: ai.ToolReturnOutcomeDenied,
		}}, nil)
	})
	for chunk, err := range vercel.TransformStreamWithConfig(v5, vercel.StreamConfig{SDKVersion: 5}) {
		if err != nil {
			t.Fatal(err)
		}
		if chunk.ToolCallID == "denied" && chunk.Type != vercel.ChunkToolOutputAvailable {
			t.Fatalf("v5 denied result was not neutral: %#v", chunk)
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

	openTools := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		ordinary := ai.ToolCallPart{ToolName: "ordinary", ToolCallID: "ordinary", Args: []byte(`{}`)}
		yield(ai.PartStartEvent{PartID: "ordinary", Part: ordinary}, nil)
		yield(ai.FunctionToolCallEvent{Part: ordinary}, nil)
		yield(ai.PartStartEvent{PartID: "native", Part: ai.NativeToolCallPart{
			ToolName: "native", ToolCallID: "native", Args: []byte(`{}`),
		}}, nil)
		yield(nil, errors.New("stream failed"))
	})
	flushedNative := false
	for chunk := range vercel.TransformStream(openTools, "message") {
		if chunk.Type == vercel.ChunkToolInputAvailable && chunk.ToolCallID == "native" &&
			chunk.ProviderExecuted != nil && *chunk.ProviderExecuted {
			flushedNative = true
		}
	}
	if !flushedNative {
		t.Fatal("open native tool input was not flushed")
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
		{events: []ai.StreamEvent{
			ai.PartStartEvent{Part: ai.ToolCallPart{ToolCallID: "call", Args: []byte(`{}`)}},
			ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call"}},
		}, stopAt: vercel.ChunkToolInputAvailable},
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
	failedTool := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.ToolCallPart{ToolCallID: "call", Args: []byte(`{}`)}}, nil)
		yield(nil, errors.New("failed"))
	})
	vercel.TransformStream(failedTool, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkToolInputAvailable
	})
	metadata := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.FinishEvent{Metadata: map[string]any{"key": "value"}}, nil)
	})
	vercel.TransformStream(metadata, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkMessageMetadata
	})
}
