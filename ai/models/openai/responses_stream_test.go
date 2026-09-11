package openai_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestResponsesStreamCodeExecution(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.created","response":{"id":"response","model":"gpt-5","created_at":100,"status":"in_progress"}}`,
		`{"type":"response.output_item.added","item":{"id":"code-1","type":"code_interpreter_call","container_id":"container-1","code":"","status":"in_progress"}}`,
		`{"type":"response.code_interpreter_call.in_progress","item_id":"code-1"}`,
		`{"type":"response.code_interpreter_call.interpreting","item_id":"code-1"}`,
		`{"type":"response.code_interpreter_call_code.delta","item_id":"code-1","delta":"print(\"hi\")"}`,
		`{"type":"response.code_interpreter_call_code.done","item_id":"code-1"}`,
		`{"type":"response.code_interpreter_call.completed","item_id":"code-1"}`,
		`{"type":"response.output_item.done","item":{"id":"code-1","type":"code_interpreter_call","container_id":"container-1","code":"print(\"hi\")","status":"completed","outputs":[{"type":"logs","logs":"hi\n"},{"type":"image","url":"data:image/png;base64,aW1hZ2U="}]}}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[{"id":"code-1","type":"code_interpreter_call","container_id":"container-1","code":"print(\"hi\")","status":"completed","outputs":[{"type":"logs","logs":"hi\n"},{"type":"image","url":"data:image/png;base64,aW1hZ2U="}]}]}}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ModelStreamEvent
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	var start ai.ToolCallStartEvent
	var deltas string
	var file ai.FilePart
	var returned ai.NativeToolReturnPart
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			deltas += event.ArgsDelta
		case ai.FileEvent:
			file = event.Part
		case ai.NativeToolReturnEvent:
			returned = event.Part
		case ai.FinishEvent:
			finish = event
		}
	}
	if start.ToolKind != ai.ToolPartKindCodeExecution || !start.Native ||
		deltas != `{"container_id":"container-1","code":"print(\"hi\")"}` ||
		file.Content.MediaType != "image/png" || string(file.Content.Data) != "image" ||
		returned.Content.(map[string]any)["logs"].([]string)[0] != "hi\n" || len(finish.Parts) != 3 {
		t.Fatalf("unexpected code execution stream: events=%#v finish=%+v", events, finish)
	}
}

func TestResponsesStreamCodeExecutionEdges(t *testing.T) {
	events := []string{
		`{"type":"response.output_item.added","item":{"id":"code","type":"code_interpreter_call","container_id":"container","code":""}}`,
		`{"type":"response.code_interpreter_call_code.delta","item_id":"code","delta":"pass"}`,
		`{"type":"response.code_interpreter_call_code.done","item_id":"code"}`,
		`{"type":"response.output_item.done","item":{"id":"code","type":"code_interpreter_call","status":"completed","outputs":[{"type":"image","url":"data:image/png;base64,aQ=="}]}}`,
	}
	for breakAfter := 1; breakAfter <= 6; breakAfter++ {
		t.Run(fmt.Sprintf("consumer break %d", breakAfter), func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, events))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			seen := 0
			for _, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				seen++
				if seen == breakAfter {
					break
				}
			}
			if seen != breakAfter {
				t.Fatalf("stream ended after %d events", seen)
			}
		})
	}
	t.Run("invalid output", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.output_item.done","item":{"id":"code","type":"code_interpreter_call","outputs":[{"type":"unknown"}]}}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for _, err := range stream {
			if err == nil || !strings.Contains(err.Error(), "unknown code interpreter output type") {
				t.Fatalf("unexpected streamed code output error: %v", err)
			}
			return
		}
		t.Fatal("expected streamed code output error")
	})
}

func TestResponsesStaticCodeExecutionFileStream(t *testing.T) {
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodGet {
			t.Fatalf("unexpected method %s", request.Method)
		}
		_, _ = response.Write([]byte(`{"id":"background","status":"completed","output":[
			{"type":"code_interpreter_call","id":"code","container_id":"container","code":"pass","status":"completed","outputs":[{"type":"image","url":"data:image/png;base64,aQ=="}]}
		]}`))
	})
	history := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "background", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true},
	}}
	stream, err := model.StreamRequest(t.Context(), history, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		seen++
		if seen == 3 {
			break
		}
	}
	if seen != 3 {
		t.Fatalf("unexpected static code stream length: %d", seen)
	}
}

func TestResponsesStreamMCPServer(t *testing.T) {
	chunks := []string{
		`{"type":"response.created","response":{"id":"response","model":"gpt-5","created_at":100,"status":"in_progress","output":[],"usage":{}}}`,
		`{"type":"response.output_item.added","item":{"type":"mcp_list_tools","id":"list","server_label":"docs","tools":[]}}`,
		`{"type":"response.mcp_list_tools.in_progress","item_id":"list"}`,
		`{"type":"response.mcp_list_tools.completed","item_id":"list"}`,
		`{"type":"response.mcp_list_tools.failed","item_id":"ignored"}`,
		`{"type":"response.output_item.done","item":{"type":"mcp_list_tools","id":"list","server_label":"docs","tools":[{"name":"search","input_schema":{"type":"object"}}],"error":null}}`,
		`{"type":"response.output_item.added","item":{"type":"mcp_call","id":"call","server_label":"docs","name":"search","arguments":""}}`,
		`{"type":"response.mcp_call.in_progress","item_id":"call"}`,
		`{"type":"response.mcp_call_arguments.delta","item_id":"call","delta":"{\"query\":"}`,
		`{"type":"response.mcp_call_arguments.delta","item_id":"call","delta":"\"Go\"}"}`,
		`{"type":"response.mcp_call_arguments.done","item_id":"call"}`,
		`{"type":"response.mcp_call.completed","item_id":"call"}`,
		`{"type":"response.mcp_call.failed","item_id":"ignored"}`,
		`{"type":"response.output_item.done","item":{"type":"mcp_call","id":"call","server_label":"docs","name":"search","arguments":"{\"query\":\"Go\"}","output":"result","error":null}}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[{"type":"mcp_list_tools","id":"list","server_label":"docs","tools":[{"name":"search","input_schema":{"type":"object"}}],"error":null},{"type":"mcp_call","id":"call","server_label":"docs","name":"search","arguments":"{\"query\":\"Go\"}","output":"result","error":null}],"usage":{}}}`,
	}
	model := newResponsesServer(t, sseHandler(t, chunks))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.MCPServerTool{ID: "docs", URL: "https://example.com/mcp"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ModelStreamEvent
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	var starts []ai.ToolCallStartEvent
	var deltas []ai.ToolCallDeltaEvent
	var returns []ai.NativeToolReturnEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			starts = append(starts, event)
		case ai.ToolCallDeltaEvent:
			deltas = append(deltas, event)
		case ai.NativeToolReturnEvent:
			returns = append(returns, event)
		}
	}
	if len(starts) != 2 || len(deltas) != 5 || len(returns) != 2 ||
		starts[0].ToolName != "mcp_server:docs" || starts[0].ToolKind != ai.ToolPartKindMCPServer ||
		!starts[0].Native || deltas[0].ArgsDelta != `{"action":"list_tools"}` ||
		deltas[1].ArgsDelta != `{"action":"call_tool","tool_name":"search","tool_args":` ||
		deltas[4].ArgsDelta != `}` || returns[0].Part.ToolCallID != "list" ||
		returns[1].Part.Content.(map[string]any)["output"] != "result" {
		t.Fatalf("unexpected MCP stream events: %#v", events)
	}
}

func TestResponsesStreamMCPListToolsBackfill(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"type":"mcp_list_tools","id":"list","server_label":"docs","tools":[]}}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[{"type":"mcp_list_tools","id":"list","server_label":"docs","tools":[],"error":"unavailable"}],"usage":{}}}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var returned *ai.NativeToolReturnEvent
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.NativeToolReturnEvent); ok {
			returned = &event
			break
		}
	}
	if returned == nil || returned.Part.Content.(map[string]any)["error"] != "unavailable" ||
		returned.Part.Timestamp.Unix() != 100 {
		t.Fatalf("missing MCP list-tools backfill: %#v", returned)
	}
}

func TestResponsesStreamMCPServerCanStop(t *testing.T) {
	chunks := []string{
		`{"type":"response.output_item.added","item":{"type":"mcp_call","id":"call","server_label":"docs","name":"search"}}`,
		`{"type":"response.mcp_call_arguments.delta","item_id":"call","delta":"{}"}`,
		`{"type":"response.mcp_call_arguments.done","item_id":"call"}`,
		`{"type":"response.output_item.done","item":{"type":"mcp_call","id":"call","server_label":"docs","name":"search","arguments":"{}","output":"ok"}}`,
	}
	for stopAfter := 1; stopAfter <= 5; stopAfter++ {
		t.Run(fmt.Sprint(stopAfter), func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, chunks))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for _, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				count++
				if count == stopAfter {
					break
				}
			}
			if count != stopAfter {
				t.Fatalf("stream ended after %d events", count)
			}
		})
	}
}

func TestResponsesStreamMCPApprovalErrors(t *testing.T) {
	for _, eventType := range []string{"response.output_item.added", "response.output_item.done"} {
		t.Run(eventType, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, []string{
				fmt.Sprintf(`{"type":%q,"item":{"type":"mcp_approval_request"}}`, eventType),
			}))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			var got error
			for _, err := range stream {
				if err != nil {
					got = err
				}
			}
			if got == nil || !strings.Contains(got.Error(), "MCP approval requests are not supported") {
				t.Fatalf("unexpected MCP approval stream error: %v", got)
			}
		})
	}
}

func TestResponsesStreamMCPServerMalformedArguments(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"type":"mcp_call","id":"call","server_label":"docs","name":"search"}}`,
		`{"type":"response.output_item.done","item":{"type":"mcp_call","id":"call","server_label":"docs","name":"search","arguments":"{"}}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var got error
	for _, err := range stream {
		if err != nil {
			got = err
		}
	}
	if got == nil || !strings.Contains(got.Error(), "parse MCP tool arguments") {
		t.Fatalf("unexpected malformed MCP stream error: %v", got)
	}
}

func TestResponsesStreamFileSearch(t *testing.T) {
	model := newResponsesServerWithOptions(t, sseHandler(t, []string{
		`{"type":"response.created","response":{"id":"response","model":"gpt-5","created_at":100,"status":"in_progress"}}`,
		`{"type":"response.output_item.added","item":{"id":"search-1","type":"file_search_call","status":"in_progress","queries":[]}}`,
		`{"type":"response.file_search_call.in_progress","item_id":"search-1"}`,
		`{"type":"response.file_search_call.searching","item_id":"search-1"}`,
		`{"type":"response.file_search_call.completed","item_id":"search-1"}`,
		`{"type":"response.output_item.done","item":{"id":"search-1","type":"file_search_call","status":"completed","queries":["revenue"],"results":[{"file_id":"file-1","text":"grew"}]}}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[{"id":"search-1","type":"file_search_call","status":"completed","queries":["revenue"],"results":[{"file_id":"file-1","text":"grew"}]}],"usage":{}}}`,
	}), openai.WithResponsesFileSearchResults(true))
	events, err := collect(t, model, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.FileSearchTool{FileStoreIDs: []string{"vs"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var start ai.ToolCallStartEvent
	var delta ai.ToolCallDeltaEvent
	var returned ai.NativeToolReturnEvent
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			delta = event
		case ai.NativeToolReturnEvent:
			returned = event
		case ai.FinishEvent:
			finish = event
		}
	}
	results := returned.Part.Content.(map[string]any)["results"].([]map[string]any)
	if !start.Native || start.ToolKind != ai.ToolPartKindFileSearch || start.ToolCallID != "search-1" ||
		delta.ArgsDelta != `{"queries":["revenue"]}` || delta.ToolCallID != "search-1" ||
		returned.Part.ToolKind != ai.ToolPartKindFileSearch || results[0]["text"] != "grew" ||
		len(finish.Parts) != 2 {
		t.Fatalf("unexpected file search stream: %#v", events)
	}
}

func TestResponsesStreamFileSearchCanStop(t *testing.T) {
	events := []string{
		`{"type":"response.output_item.added","item":{"id":"search","type":"file_search_call"}}`,
		`{"type":"response.output_item.done","item":{"id":"search","type":"file_search_call","queries":[]}}`,
	}
	for breakAfter := 1; breakAfter <= 3; breakAfter++ {
		t.Run(fmt.Sprintf("event %d", breakAfter), func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, events))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			seen := 0
			for _, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				seen++
				if seen == breakAfter {
					break
				}
			}
			if seen != breakAfter {
				t.Fatalf("stream ended after %d events", seen)
			}
		})
	}
}

func TestResponsesStreamImageGeneration(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.created","response":{"id":"response","model":"gpt-5","created_at":100,"status":"in_progress"}}`,
		`{"type":"response.output_item.added","item":{"id":"image-1","type":"image_generation_call","status":"in_progress"}}`,
		`{"type":"response.image_generation_call.generating","item_id":"image-1"}`,
		`{"type":"response.image_generation_call.in_progress","item_id":"image-1"}`,
		`{"type":"response.image_generation_call.partial_image","item_id":"image-1","partial_image_index":0,"partial_image_b64":"aQ=="}`,
		`{"type":"response.image_generation_call.partial_image","item_id":"image-1","partial_image_index":1,"partial_image_b64":"aW0=","output_format":"webp"}`,
		`{"type":"response.image_generation_call.completed","item_id":"image-1"}`,
		`{"type":"response.output_item.done","item":{"id":"image-1","type":"image_generation_call","status":"generating","output_format":"jpeg","result":"aW1hZ2U="}}`,
		`{"type":"response.output_text.delta","output_index":1,"content_index":0,"item_id":"message","delta":"done"}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","created_at":100,"status":"completed","output":[{"id":"image-1","type":"image_generation_call","status":"generating","output_format":"jpeg","result":"aW1hZ2U="},{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{}}}`,
	}))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "draw", struct{}{})
	fileStarts := 0
	fileDeltas := 0
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			if file, ok := event.Part.(ai.FilePart); ok {
				fileStarts++
				if string(file.Content.Data) != "i" || file.Content.MediaType != "image/png" {
					t.Fatalf("unexpected first partial image: %+v", file)
				}
			}
		case ai.PartDeltaEvent:
			if delta, ok := event.Delta.(ai.FilePartDelta); ok {
				fileDeltas++
				if fileDeltas == 2 && (string(delta.Part.Content.Data) != "image" ||
					delta.Part.Content.MediaType != "image/jpeg") {
					t.Fatalf("unexpected final image delta: %+v", delta.Part)
				}
			}
		}
	}
	result := stream.Result()
	if result == nil {
		t.Fatal("image generation stream did not complete")
	}
	response := result.Messages()[1].(ai.ModelResponse)
	file := response.Parts[1].(ai.FilePart)
	if result.Output != "done" || fileStarts != 1 || fileDeltas != 2 ||
		string(file.Content.Data) != "image" || file.Content.MediaType != "image/jpeg" {
		t.Fatalf("unexpected image generation stream: result=%+v starts=%d deltas=%d", result, fileStarts, fileDeltas)
	}
}

func TestResponsesStreamImageGenerationErrorsAndStopping(t *testing.T) {
	for name, event := range map[string]string{
		"partial": `{"type":"response.image_generation_call.partial_image","item_id":"image","partial_image_b64":"!"}`,
		"done":    `{"type":"response.output_item.done","item":{"id":"image","type":"image_generation_call","result":"!"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, []string{event}))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			for _, err := range stream {
				if err == nil || !strings.Contains(err.Error(), "decode generated image") {
					t.Fatalf("unexpected generated image error: %v", err)
				}
				return
			}
			t.Fatal("expected generated image error")
		})
	}

	events := []string{
		`{"type":"response.output_item.added","item":{"id":"image","type":"image_generation_call"}}`,
		`{"type":"response.image_generation_call.partial_image","item_id":"image","partial_image_b64":"aQ=="}`,
		`{"type":"response.output_item.done","item":{"id":"image","type":"image_generation_call","result":"aQ=="}}`,
	}
	for breakAfter := 1; breakAfter <= 4; breakAfter++ {
		t.Run(fmt.Sprintf("consumer break %d", breakAfter), func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, events))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			seen := 0
			for _, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				seen++
				if seen == breakAfter {
					break
				}
			}
			if seen != breakAfter {
				t.Fatalf("stream ended after %d events", seen)
			}
		})
	}
}

func TestResponsesStreamRefusal(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.refusal.delta","delta":"I cannot "}`,
		`{"type":"response.refusal.delta","delta":"help."}`,
		`{"type":"response.refusal.done","refusal":"I cannot help with that."}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","output":[{"id":"message","type":"message","content":[{"type":"refusal","refusal":"I cannot help with that."}]}]}}`,
	}))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "blocked", struct{}{})
	var streamErr error
	for _, err := range stream.Events() {
		if err != nil {
			streamErr = err
		}
	}
	var filtered *ai.ContentFilterError
	if !errors.As(streamErr, &filtered) || filtered.Response().FinishReason != ai.FinishReasonContentFilter ||
		filtered.Response().ProviderDetails["refusal"] != "I cannot help with that." ||
		filtered.Response().ProviderDetails["finish_reason"] != nil {
		t.Fatalf("unexpected streamed Responses refusal: %v response=%+v", streamErr, filtered)
	}
}

func TestResponsesStreamRefusalFromSnapshot(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","status":"completed","output":[{"id":"message","type":"message","content":[{"type":"refusal","refusal":"Blocked by snapshot."}]}]}}`,
	}))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "blocked", struct{}{})
	var streamErr error
	for _, err := range stream.Events() {
		if err != nil {
			streamErr = err
		}
	}
	var filtered *ai.ContentFilterError
	if !errors.As(streamErr, &filtered) || filtered.Response().ProviderDetails["refusal"] != "Blocked by snapshot." {
		t.Fatalf("unexpected snapshot refusal: %v response=%+v", streamErr, filtered)
	}
}

func TestResponsesStreamEvents(t *testing.T) {
	var streamed bool
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		streamed = body.Stream
		sseHandler(t, []string{
			`{"type":"response.created","response":{"model":"gpt-5"}}`,
			`{"type":"response.output_item.added","item":{"id":"cmp","type":"compaction","encrypted_content":"opaque"}}`,
			`{"type":"response.output_item.added","item":{"id":"msg","type":"message","phase":"final_answer"}}`,
			`{"type":"response.output_text.delta","item_id":"msg","delta":"Hi"}`,
			`{"type":"response.output_text.annotation.added","item_id":"msg","annotation":{"type":"url_citation","url":"https://example.com"}}`,
			`{"type":"response.output_text.done","item_id":"msg","logprobs":[{"token":"Hi","logprob":-0.1}]}`,
			`{"type":"response.output_item.added","item":{"id":"reason","type":"reasoning","encrypted_content":"signature"}}`,
			`{"type":"response.reasoning_summary_part.added","item_id":"reason","part":{"text":"A"}}`,
			`{"type":"response.reasoning_summary_text.delta","item_id":"reason","delta":"B"}`,
			`{"type":"response.reasoning_text.delta","item_id":"reason","delta":"C"}`,
			`{"type":"response.output_item.added","item":{"id":"fc","type":"function_call","call_id":"c1","name":"work","namespace":"tools","arguments":""}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc","delta":"{\"x\":"}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc","delta":"1}"}`,
			`{"type":"response.output_text.done"}`,
			`{"type":"response.completed","response":{"id":"response-stream","model":"gpt-5","created_at":1735689600.25,"status":"completed","output":[{"id":"cmp","type":"compaction","encrypted_content":"opaque"}],"usage":{"input_tokens":5,"output_tokens":3}}}`,
			`[DONE]`,
		})(w, r)
	})
	events, err := collect(t, model, ai.ModelRequestParams{Settings: rawAnnotationSettings(t)})
	if err != nil {
		t.Fatal(err)
	}
	if !streamed {
		t.Fatal("Responses request did not enable streaming")
	}
	var text, thinking, args string
	var textPartID, textID, textPhase, thinkingPartID, thinkingID, thinkingSignature, argsPartID string
	var textAnnotations, textLogprobs int
	var start ai.ToolCallStartEvent
	var compaction ai.CompactionEvent
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			text += event.Delta
			textPartID = event.PartID
			textID = event.ID
			if phase, ok := event.ProviderDetails["phase"].(string); ok {
				textPhase = phase
			}
			if annotations, ok := event.ProviderDetails["annotations"].([]map[string]any); ok {
				textAnnotations = len(annotations)
			}
			if logprobs, ok := event.ProviderDetails["logprobs"].([]map[string]any); ok {
				textLogprobs = len(logprobs)
			}
		case ai.ThinkingDeltaEvent:
			thinking += event.Delta
			thinkingPartID = event.PartID
			if event.ID != "" {
				thinkingID = event.ID
			}
			if event.SignatureDelta != "" {
				thinkingSignature = event.SignatureDelta
			}
		case ai.CompactionEvent:
			compaction = event
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			args += event.ArgsDelta
			argsPartID = event.PartID
		case ai.FinishEvent:
			finish = event
		}
	}
	if text != "Hi" || thinking != "ABC" || start.ToolName != "work" || start.ToolCallID != "c1" ||
		args != `{"x":1}` || compaction.PartID != "item:cmp" || compaction.ID != "cmp" ||
		compaction.ProviderDetails["encrypted_content"] != "opaque" {
		t.Fatalf(
			"unexpected events text=%q thinking=%q compaction=%+v start=%+v args=%q",
			text, thinking, compaction, start, args,
		)
	}
	if textPartID != "output:0:content:0:text" || textID != "msg" || textPhase != "final_answer" ||
		textAnnotations != 1 || textLogprobs != 1 || thinkingPartID != "item:reason:thinking:0" || thinkingID != "reason" || thinkingSignature != "signature" ||
		start.PartID != "item:fc" || start.ID != "fc" || start.ProviderDetails["namespace"] != "tools" ||
		argsPartID != start.PartID {
		t.Fatalf("unstable Responses part IDs: text=%q thinking=%q start=%q args=%q", textPartID, thinkingPartID, start.PartID, argsPartID)
	}
	if finish.ModelName != "gpt-5" || finish.Usage.Requests != 1 || finish.Usage.InputTokens != 5 ||
		finish.Usage.OutputTokens != 3 || finish.ProviderName != "openai" || finish.ProviderURL == "" ||
		finish.ProviderDetails["compaction"] != true ||
		finish.ProviderResponseID != "response-stream" || finish.FinishReason != ai.FinishReasonStop ||
		finish.State != ai.ModelResponseStateComplete || finish.Timestamp.IsZero() ||
		finish.ProviderDetails["finish_reason"] != "completed" || finish.ProviderDetails["timestamp"] == nil {
		t.Fatalf("unexpected finish %+v", finish)
	}
}

func TestResponsesStreamMetadataReachesNormalizedEventsAndHistory(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"id":"message","type":"message","phase":"commentary"}}`,
		`{"type":"response.output_text.delta","item_id":"message","delta":"Hi"}`,
		`{"type":"response.output_text.annotation.added","item_id":"message","annotation":{"type":"url_citation","url":"https://example.com"}}`,
		`{"type":"response.output_text.done","item_id":"message","logprobs":[{"token":"Hi","logprob":-0.1}]}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","status":"completed"}}`,
	}))
	stream := ai.NewAgent[struct{}, string](
		model, ai.WithModelSettings(rawAnnotationSettings(t)),
	).RunStream(t.Context(), "go", struct{}{})
	var started ai.TextPart
	var metadataDelta ai.TextPartDelta
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			if text, ok := event.Part.(ai.TextPart); ok {
				started = text
			}
		case ai.PartDeltaEvent:
			if delta, ok := event.Delta.(ai.TextPartDelta); ok && len(delta.ProviderDetails) > 0 {
				metadataDelta = delta
			}
		}
	}
	if started.ProviderDetails["phase"] != "commentary" ||
		len(metadataDelta.ProviderDetails["annotations"].([]map[string]any)) != 1 ||
		len(metadataDelta.ProviderDetails["logprobs"].([]map[string]any)) != 1 {
		t.Fatalf("stream metadata was not normalized: start=%+v delta=%+v", started, metadataDelta)
	}
	metadataDelta.ProviderDetails["annotations"].([]map[string]any)[0]["url"] = "changed"
	result := stream.Result()
	response := result.Messages()[len(result.Messages())-1].(ai.ModelResponse)
	text := response.Parts[0].(ai.TextPart)
	if text.ProviderDetails["phase"] != "commentary" ||
		len(text.ProviderDetails["annotations"].([]map[string]any)) != 1 ||
		text.ProviderDetails["annotations"].([]map[string]any)[0]["url"] != "https://example.com" ||
		len(text.ProviderDetails["logprobs"].([]map[string]any)) != 1 {
		t.Fatalf("stream metadata was not retained in history: %+v", text)
	}
}

func TestResponsesStreamRawAnnotationsDisabledByDefault(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_text.delta","item_id":"message","delta":"Hi"}`,
		`{"type":"response.output_text.annotation.added","item_id":"message","annotation":{"type":"url_citation"}}`,
		`{"type":"response.output_text.done","item_id":"message"}`,
		`{"type":"response.completed","response":{"status":"completed","output":[` +
			`{"id":"message","type":"message","content":[` +
			`{"type":"output_text","text":"Hi","annotations":[{"type":"url_citation"}]}` +
			`]}]}}`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range events {
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			if _, exists := event.ProviderDetails["annotations"]; exists {
				t.Fatalf("stream retained raw annotations by default: %+v", event)
			}
		case ai.FinishEvent:
			text := event.Parts[0].(ai.TextPart)
			if _, exists := text.ProviderDetails["annotations"]; exists {
				t.Fatalf("snapshot retained raw annotations by default: %+v", text)
			}
		}
	}
}

func TestResponsesStreamMetadataConsumerStop(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_text.done","item_id":"message","logprobs":[{"token":"Hi"}]}`,
		`{"type":"response.completed","response":{"status":"completed"}}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
}

func TestResponsesStreamPhaseWithoutTextDelta(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"id":"message","type":"message","phase":"final_answer"}}`,
		`{"type":"response.output_text.done","item_id":"message"}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","status":"completed"}}`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var text ai.TextDeltaEvent
	for _, event := range events {
		if delta, ok := event.(ai.TextDeltaEvent); ok {
			text = delta
		}
	}
	if text.ID != "message" || text.ProviderDetails["phase"] != "final_answer" {
		t.Fatalf("phase-only text event was lost: %+v", text)
	}
}

func TestResponsesStreamUsesAuthoritativeCompletedSnapshot(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_text.delta","item_id":"message","delta":"suffix"}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","status":"completed","output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"full output"}]}],"usage":{}}}`,
	}))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "go", struct{}{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if result := stream.Result(); result == nil || result.Output != "full output" {
		t.Fatalf("completed snapshot did not replace partial deltas: %+v", result)
	}
}

func TestResponsesCompletedSnapshotErrorsAndConsumerStop(t *testing.T) {
	t.Run("sequence-only metadata", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","sequence_number":1,"response":{"id":"job"}}`,
			`{"type":"response.completed","response":{"id":"job","status":"completed"}}`,
		}))
		events, err := collect(t, model, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		metadata, ok := events[0].(ai.ResponseMetadataEvent)
		if !ok || metadata.ProviderDetails["sequence_number"] != 1 {
			t.Fatalf("unexpected sequence metadata: %+v", events)
		}
	})

	t.Run("metadata consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","response":{"id":"job","status":"queued","background":true}}`,
			`{"type":"mystery"}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for range stream {
			break
		}
	})

	t.Run("invalid snapshot", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","name":"work","arguments":"bad"}]}}`,
		}))
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected completed snapshot error")
		}
	})

	t.Run("consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.completed","response":{"status":"completed","output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}]}}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for range stream {
			break
		}
	})

	t.Run("invalid pending snapshot", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","response":{"id":"job","status":"queued","background":true,"output":[{"type":"function_call","name":"work","arguments":"bad"}]}}`,
		}))
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected pending snapshot error")
		}
	})

	t.Run("pending consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","response":{"id":"job","status":"queued","background":true,"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"partial"}]}]}}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for event := range stream {
			if _, ok := event.(ai.TextDeltaEvent); ok {
				break
			}
		}
	})
}

func TestResponsesStreamDetachPreservesBackgroundJob(t *testing.T) {
	var paths []string
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		sseHandler(t, []string{
			`{"type":"response.created","sequence_number":4,"response":{"id":"job","model":"gpt-5","status":"queued","background":true,"usage":{"input_tokens":2}}}`,
			`{"type":"response.output_text.delta","sequence_number":5,"item_id":"message","delta":"partial"}`,
		})(w, r)
	}, openai.WithBackgroundMode(true), openai.WithBackgroundPollInterval(0))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "go", struct{}{})
	for range stream.Events() {
		break
	}
	suspended := stream.Suspended()
	if suspended == nil || suspended.Response().ProviderResponseID != "job" ||
		suspended.Response().Text() != "partial" || suspended.Response().ProviderDetails["sequence_number"] != 5 {
		t.Fatalf("unexpected OpenAI detached snapshot: %+v", suspended)
	}
	if len(paths) != 1 || paths[0] != "/responses" {
		t.Fatalf("detach polled or canceled the background job: %v", paths)
	}

	resumeModel := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/responses/job" {
			t.Fatalf("unexpected resume request %s %s", r.Method, r.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"id":"job","model":"gpt-5","status":"completed","background":true,
			"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],
			"usage":{"input_tokens":2,"output_tokens":1}
		}`))
	}, openai.WithBackgroundPollInterval(0))
	result, err := ai.NewAgent[struct{}, string](resumeModel).Resume(t.Context(), suspended.Messages(), struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("OpenAI detached history did not resume: result=%+v err=%v", result, err)
	}
}

func TestResponsesResumedStreamDetachesWithoutCreatedEvent(t *testing.T) {
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Query().Get("starting_after") != "1" {
			t.Fatalf("unexpected resumed stream request: %s %s", r.Method, r.URL.String())
		}
		sseHandler(t, []string{
			`{"type":"response.output_text.delta","sequence_number":2,"item_id":"message","delta":"partial"}`,
		})(w, r)
	}, openai.WithBackgroundPollInterval(0))
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
		ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", ModelName: "gpt-5",
			ProviderDetails: map[string]any{"background": true, "sequence_number": 1},
			State:           ai.ModelResponseStateSuspended,
		},
	}
	stream := ai.NewAgent[struct{}, string](model).ResumeStream(t.Context(), history, struct{}{})
	for range stream.Events() {
		break
	}
	suspended := stream.Suspended()
	if suspended == nil || suspended.Response().ProviderResponseID != "job" ||
		suspended.Response().ProviderDetails["sequence_number"] != 2 || suspended.Response().Text() != "partial" {
		t.Fatalf("unexpected resumed detached snapshot: %+v", suspended)
	}
}

func TestResponsesStreamContinuesBackgroundJob(t *testing.T) {
	var methods []string
	var queries []string
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		methods = append(methods, r.Method)
		queries = append(queries, r.URL.RawQuery)
		if r.Method == http.MethodPost {
			sseHandler(t, []string{
				`{"type":"response.created","sequence_number":4,"response":{"id":"job","status":"queued","background":true,"output":[{"id":"partial","type":"message","content":[{"type":"output_text","text":"partial"}]}],"usage":{"input_tokens":1}}}`,
			})(w, r)
			return
		}
		sseHandler(t, []string{
			`{"type":"response.output_text.delta","sequence_number":5,"item_id":"message","delta":"done"}`,
			`{"type":"response.output_text.annotation.added","item_id":"message","annotation":{"type":"url_citation"}}`,
			`{"type":"response.output_text.done","item_id":"message"}`,
			`{"type":"response.completed","sequence_number":8,"response":{"id":"job","model":"gpt-5","status":"completed","background":true,"usage":{"input_tokens":1,"output_tokens":1}}}`,
		})(w, r)
	}, openai.WithBackgroundPollInterval(0))
	stream := ai.NewAgent[struct{}, string](
		model, ai.WithModelSettings(rawAnnotationSettings(t)),
	).RunStream(t.Context(), "go", struct{}{})
	var finishes []ai.FinishEvent
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.FinishEvent); ok {
			finishes = append(finishes, event)
		}
	}
	result := stream.Result()
	if result == nil || result.Output != "done" || result.Usage().Requests != 1 || len(finishes) != 2 {
		t.Fatalf("unexpected streamed background result=%+v finishes=%+v", result, finishes)
	}
	response := result.NewMessages()[1].(ai.ModelResponse)
	text := response.Parts[0].(ai.TextPart)
	if len(text.ProviderDetails["annotations"].([]map[string]any)) != 1 {
		t.Fatalf("resumed stream annotations were lost: %+v", text)
	}
	if len(methods) != 2 || methods[0] != http.MethodPost || methods[1] != http.MethodGet ||
		!strings.Contains(queries[1], "starting_after=4") || !strings.Contains(queries[1], "stream=true") {
		t.Fatalf("unexpected stream retrieval methods=%v queries=%v", methods, queries)
	}
	if finishes[0].State != ai.ModelResponseStateSuspended || finishes[0].ProviderDetails["sequence_number"] != 4 {
		t.Fatalf("pending stream metadata was not retained: %+v", finishes[0])
	}
}

func TestResponsesStreamRetrievesWithoutSequenceAsStaticEvents(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/responses/job" {
			t.Fatalf("unexpected static retrieval %s %s", r.Method, r.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"id":"job","model":"gpt-5","status":"completed","background":true,
			"output":[
				{"id":"reason","type":"reasoning","encrypted_content":"signature","summary":[{"text":"thinking"}]},
				{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]},
				{"id":"call","type":"function_call","call_id":"call","name":"work","arguments":{"x":1}}
			],
			"usage":{"input_tokens":1,"output_tokens":2}
		}`))
	})
	messages := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true},
	}}
	events, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var got []ai.ModelStreamEvent
	for event, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, event)
	}
	if len(got) != 5 {
		t.Fatalf("unexpected static event count: %d (%+v)", len(got), got)
	}
	if _, ok := got[0].(ai.ThinkingDeltaEvent); !ok {
		t.Fatalf("missing static thinking event: %+v", got)
	}
	if _, ok := got[1].(ai.TextDeltaEvent); !ok {
		t.Fatalf("missing static text event: %+v", got)
	}
	if _, ok := got[2].(ai.ToolCallStartEvent); !ok {
		t.Fatalf("missing static tool start: %+v", got)
	}
	if finish, ok := got[4].(ai.FinishEvent); !ok || finish.State != ai.ModelResponseStateComplete {
		t.Fatalf("missing static finish: %+v", got)
	}
}

func TestResponsesStaticStreamSupportsEarlyConsumerStops(t *testing.T) {
	for stopAt := 1; stopAt <= 4; stopAt++ {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"id":"job","model":"gpt-5","status":"completed",
				"output":[
					{"id":"reason","type":"reasoning","summary":[{"text":"thinking"}]},
					{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]},
					{"id":"call","type":"function_call","call_id":"call","name":"work","arguments":{}}
				]
			}`))
		})
		messages := []ai.ModelMessage{ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
			ProviderDetails: map[string]any{"background": true},
		}}
		stream, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for range stream {
			count++
			if count == stopAt {
				break
			}
		}
	}
}

func TestResponsesStreamAcceptsSerializedSequenceNumber(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("starting_after") != "7" {
			t.Fatalf("unexpected sequence query: %s", r.URL.RawQuery)
		}
		sseHandler(t, []string{
			`{"type":"response.completed","sequence_number":8,"response":{"id":"job","model":"gpt-5","background":false,"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{}}}`,
		})(w, r)
	})
	messages := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true, "sequence_number": float64(7)},
	}}
	stream, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.TextDeltaEvent); ok {
			text += event.Delta
		}
	}
	if text != "done" {
		t.Fatalf("completed retrieval output was not replayed: %q", text)
	}
}

func TestResponsesBackgroundStreamRequestFailures(t *testing.T) {
	messages := func(sequence bool) []ai.ModelMessage {
		details := map[string]any{"background": true}
		if sequence {
			details["sequence_number"] = 1
		}
		return []ai.ModelMessage{ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
			ProviderDetails: details,
		}}
	}
	t.Run("seed consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{`{"type":"mystery"}`}))
		stream, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for range stream {
			break
		}
	})

	t.Run("static retrieve", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "missing", http.StatusNotFound)
		})
		if _, err := model.StreamRequest(t.Context(), messages(false), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected static retrieve error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve URL error")
		}
	})
	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve transport error")
		}
	})
	t.Run("HTTP", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "bad", http.StatusBadRequest)
		})
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve API error")
		}
	})
	t.Run("truncated HTTP", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("short"))
		})
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve read error")
		}
	})
}

func TestResponsesStreamWebSearchNativeTool(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.created","response":{"id":"response","created_at":100,"status":"in_progress"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"web-1","status":"in_progress"}}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"web-1","status":"completed","action":{"type":"search","query":"Go news"}}}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","created_at":100,"status":"completed","usage":{}}}`,
		`[DONE]`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}})
	if err != nil {
		t.Fatal(err)
	}
	var start ai.ToolCallStartEvent
	var delta ai.ToolCallDeltaEvent
	var returned ai.NativeToolReturnEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			delta = event
		case ai.NativeToolReturnEvent:
			returned = event
		}
	}
	if !start.Native || start.ToolKind != ai.ToolPartKindWebSearch || start.ToolCallID != "web-1" ||
		delta.ArgsDelta != `{"type":"search","query":"Go news"}` || returned.Part.ToolCallID != "web-1" ||
		returned.Part.ToolKind != ai.ToolPartKindWebSearch || returned.Part.Content.(map[string]any)["status"] != "completed" {
		t.Fatalf("unexpected streamed web search events: start=%+v delta=%+v return=%+v", start, delta, returned)
	}
}

func TestResponsesStreamWebSearchConsumerBreaks(t *testing.T) {
	for _, breakAfter := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("event %d", breakAfter), func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, []string{
				`{"type":"response.output_item.added","item":{"type":"web_search_call","id":"web","status":"in_progress"}}`,
				`{"type":"response.output_item.done","item":{"type":"web_search_call","id":"web","status":"completed"}}`,
				`[DONE]`,
			}))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			seen := 0
			for _, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				seen++
				if seen == breakAfter {
					break
				}
			}
			if seen != breakAfter {
				t.Fatalf("stream yielded %d events before break, want %d", seen, breakAfter)
			}
		})
	}
}

func TestResponsesStreamUsesNativeDeferredToolSearch(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		sseHandler(t, []string{
			`{"type":"response.completed","response":{"model":"gpt-5","status":"completed","usage":{}}}`,
			`[DONE]`,
		})(w, r)
	})
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	_, err := collect(t, model, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
			ToolSearchStrategy: ai.ToolSearchStrategyCustom,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := gotBody["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["type"] != "function" ||
		tools[0].(map[string]any)["name"] != "hidden" || tools[0].(map[string]any)["defer_loading"] != true ||
		tools[1].(map[string]any)["type"] != "tool_search" ||
		tools[1].(map[string]any)["execution"] != "client" {
		t.Fatalf("stream did not use native deferred search: %+v", tools)
	}
}

func TestResponsesStreamClientToolSearchCall(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"search-item","type":"tool_search_call","call_id":"provisional","execution":"client","arguments":null}}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"search-item","type":"tool_search_call","call_id":"search-final","execution":"client","arguments":{"query":"weather"},"status":"completed"}}`,
		`{"type":"response.completed","response":{"model":"gpt-5","status":"completed","usage":{}}}`,
		`[DONE]`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	start, ok := events[0].(ai.ToolCallStartEvent)
	if !ok || start.ToolName != ai.ToolSearchName || start.ToolKind != ai.ToolPartKindToolSearch ||
		start.ToolCallID != "" || start.PartID != "item:search-item" ||
		start.ProviderDetails["execution"] != "client" {
		t.Fatalf("unexpected client search start: %+v", events[0])
	}
	delta, ok := events[1].(ai.ToolCallDeltaEvent)
	if !ok || delta.PartID != start.PartID || delta.ToolCallID != "search-final" ||
		delta.ArgsDelta != `{"query":"weather"}` {
		t.Fatalf("unexpected client search completion: %+v", events[1])
	}
}

func TestResponsesStreamToolSearchEdgeCases(t *testing.T) {
	for name, events := range map[string][]string{
		"invalid function arguments": {
			`{"type":"response.output_item.added","item":{"id":"call","type":"function_call","call_id":"call","name":"work","arguments":"bad"}}`,
		},
		"invalid search arguments": {
			`{"type":"response.output_item.added","item":{"id":"search","type":"tool_search_call","execution":"client"}}`,
			`{"type":"response.output_item.done","item":{"id":"search","type":"tool_search_call","execution":"client","arguments":"bad"}}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, events))
			if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected tool-search stream error")
			}
		})
	}

	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"id":"search","type":"tool_search_call","execution":"client"}}`,
		`{"type":"response.output_item.done","item":{"id":"search","type":"tool_search_call","execution":"client","arguments":{"query":"x"}}}`,
		`{"type":"response.completed","response":{"status":"completed","usage":{}}}`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if delta := events[1].(ai.ToolCallDeltaEvent); delta.ToolCallID != "search" {
		t.Fatalf("item ID was not used as call ID: %+v", delta)
	}
}

func TestResponsesStreamConsumerBreakOnToolSearch(t *testing.T) {
	for name, stopAt := range map[string]string{"start": "start", "delta": "delta"} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, []string{
				`{"type":"response.output_item.added","item":{"id":"search","type":"tool_search_call","execution":"client"}}`,
				`{"type":"response.output_item.done","item":{"id":"search","type":"tool_search_call","call_id":"search","execution":"client","arguments":{"query":"x"}}}`,
				`{"type":"mystery"}`,
			}))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			for event, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				if stopAt == "start" {
					if _, ok := event.(ai.ToolCallStartEvent); ok {
						break
					}
				} else if _, ok := event.(ai.ToolCallDeltaEvent); ok {
					break
				}
			}
		})
	}
}

func TestResponsesStreamConsumerBreakOnEncryptedReasoning(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"id":"reason","type":"reasoning","encrypted_content":"signature"}}`,
		`{"type":"mystery"}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		if event.(ai.ThinkingDeltaEvent).SignatureDelta != "signature" {
			t.Fatalf("unexpected reasoning event: %+v", event)
		}
		break
	}
}

func TestResponsesStreamEndToEnd(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_text.delta","delta":"hello"}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":1}}}`,
	}))
	agent := ai.NewAgent[struct{}, string](model)
	stream := agent.RunStream(t.Context(), "go", struct{}{})
	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += normalizedText(event)
	}
	if text != "hello" || stream.Result().Output != "hello" {
		t.Fatalf("unexpected text %q and result %+v", text, stream.Result())
	}
}

func TestResponsesStreamProtocolErrors(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   string
	}{
		{name: "malformed", chunks: []string{`not json`}, want: "parse Responses stream"},
		{name: "error", chunks: []string{`{"type":"error","error":{"code":"busy","message":"later"}}`}, want: "busy"},
		{name: "unknown", chunks: []string{`{"type":"mystery"}`}, want: "unknown Responses stream"},
		{name: "invalid terminal output", chunks: []string{`{"type":"response.failed","response":{"status":"failed","output":[{"type":"mcp_approval_request"}]}}`}, want: "MCP approval requests"},
		{name: "missing completed", chunks: []string{`{"type":"response.in_progress"}`, `[DONE]`}, want: "without response.completed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, test.chunks))
			_, err := collect(t, model, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q error, got %v", test.want, err)
			}
		})
	}
}

func TestResponsesStreamTerminalFinishReasons(t *testing.T) {
	for name, test := range map[string]struct {
		chunk string
		want  ai.FinishReason
		raw   any
	}{
		"failed": {
			chunk: `{"type":"response.failed","response":{"id":"id","model":"served","status":"failed","error":{"code":"bad","message":"request"},"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"partial"}]}]}}`,
			want:  ai.FinishReasonError, raw: "failed",
		},
		"incomplete": {
			chunk: `{"type":"response.incomplete","response":{"id":"id","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"id":"message","type":"message","content":[{"type":"refusal","refusal":"blocked"}]}]}}`,
			want:  ai.FinishReasonContentFilter, raw: nil,
		},
		"refusal without status": {
			chunk: `{"type":"response.incomplete","response":{"id":"id","output":[{"id":"message","type":"message","content":[{"type":"refusal","refusal":"blocked"}]}]}}`,
			want:  ai.FinishReasonContentFilter, raw: nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, []string{test.chunk}))
			events, err := collect(t, model, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			finish := events[len(events)-1].(ai.FinishEvent)
			if finish.FinishReason != test.want || finish.ProviderDetails["finish_reason"] != test.raw {
				t.Fatalf("unexpected finish event: %+v", finish)
			}
		})
	}
}

func TestResponsesTerminalSnapshotConsumerBreak(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.failed","response":{"id":"id","status":"failed","output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"partial"}]}]}}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
}

func TestResponsesStreamRequestErrors(t *testing.T) {
	t.Run("bad payload", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, nil))
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{
			"type": "object", "properties": map[string]any{}, "required": []string{}, "bad": make(chan int),
		}}}}
		if _, err := collect(t, model, params); err == nil {
			t.Fatal("expected marshal error")
		}
	})
	t.Run("bad message", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, nil))
		if _, err := model.StreamRequest(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected message error")
		}
	})
	t.Run("native output", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.output_text.delta","item_id":"message","delta":"{}"}`,
			`{"type":"response.completed","response":{"model":"gpt-5","status":"completed","usage":{}}}`,
			`[DONE]`,
		}))
		if _, err := collect(t, model, ai.ModelRequestParams{
			OutputSchema: map[string]any{"type": "object"}, OutputMode: ai.OutputModeNative,
		}); err != nil {
			t.Fatalf("native output should stream: %v", err)
		}
		prompted := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.completed","response":{"model":"gpt-5","status":"completed","usage":{}}}`,
			`[DONE]`,
		}))
		if _, err := collect(t, prompted, ai.ModelRequestParams{
			OutputSchema: map[string]any{"type": "object"}, OutputMode: ai.OutputModePrompted,
		}); err != nil {
			t.Fatalf("prompted output should not request native mode: %v", err)
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("http error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad"))
		})
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("truncated error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("short"))
		})
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestResponsesStreamScannerError(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(make([]byte, 2*1024*1024))
	})
	_, err := collect(t, model, ai.ModelRequestParams{})
	var transportError *ai.ModelTransportError
	if !errors.As(err, &transportError) || transportError.Operation != "read Responses stream" {
		t.Fatalf("unexpected scanner error: %v", err)
	}
}

func TestResponsesStreamEarlyBreak(t *testing.T) {
	chunks := []string{
		`{"type":"response.reasoning_summary_part.added","part":{"text":""}}`,
		`{"type":"response.reasoning_summary_part.added","part":{"text":"p"}}`,
		`{"type":"response.output_text.delta","delta":"a"}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"b"}`,
		`{"type":"response.output_item.added","item":{"type":"function_call","call_id":"c","name":"work","arguments":"{}"}}`,
		`{"type":"response.function_call_arguments.delta","delta":"{}"}`,
		`{"type":"response.completed","response":{"model":"gpt-5","usage":{}}}`,
	}
	for breakAt := 1; breakAt <= 7; breakAt++ {
		model := newResponsesServer(t, sseHandler(t, chunks))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for range stream {
			count++
			if count == breakAt {
				break
			}
		}
	}
}

func TestStoppingResponsesCompactionStreams(t *testing.T) {
	t.Run("SSE", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.output_item.added","item":{"id":"cmp","type":"compaction","encrypted_content":"opaque"}}`,
			`{"type":"response.completed","response":{"id":"response","model":"gpt-5","status":"completed","usage":{}}}`,
		}))
		events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		seen := false
		for event, eventErr := range events {
			if eventErr != nil {
				t.Fatal(eventErr)
			}
			if _, ok := event.(ai.CompactionEvent); ok {
				seen = true
				break
			}
		}
		if !seen {
			t.Fatal("SSE compaction was not emitted")
		}
	})

	t.Run("static retrieval", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"id":"job", "model":"gpt-5", "status":"completed",
				"output":[
					{"id":"cmp","type":"compaction","encrypted_content":"opaque"},
					{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}
				], "usage":{}
			}`))
		})
		history := []ai.ModelMessage{ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
			ProviderDetails: map[string]any{"background": true},
		}}
		events, err := model.StreamRequest(t.Context(), history, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		seen := false
		for event, eventErr := range events {
			if eventErr != nil {
				t.Fatal(eventErr)
			}
			if _, ok := event.(ai.CompactionEvent); ok {
				seen = true
				break
			}
		}
		if !seen {
			t.Fatal("static compaction was not emitted")
		}
	})
}

func TestResponsesStreamServerManagedToolSearch(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.created","sequence_number":0,"response":{"id":"response-1","model":"gpt-5.4","created_at":1735689600,"status":"in_progress","usage":{}}}`,
		`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"ts-1","type":"tool_search_call","call_id":null,"execution":"server","status":"in_progress","arguments":{}}}`,
		`{"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"id":"ts-1","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["weather"]}}}`,
		`{"type":"response.output_item.added","sequence_number":3,"output_index":1,"item":{"id":"tso-1","type":"tool_search_output","call_id":null,"execution":"server","status":"in_progress","tools":[]}}`,
		`{"type":"response.output_item.done","sequence_number":4,"output_index":1,"item":{"id":"tso-1","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]}}`,
		`{"type":"response.completed","sequence_number":5,"response":{"id":"response-1","model":"gpt-5.4","created_at":1735689600,"status":"completed","output":[{"id":"ts-1","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["weather"]}},{"id":"tso-1","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]}],"usage":{}}}`,
		`[DONE]`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var start *ai.ToolCallStartEvent
	var delta *ai.ToolCallDeltaEvent
	var returned *ai.NativeToolReturnEvent
	var finish *ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			start = &event
		case ai.ToolCallDeltaEvent:
			delta = &event
		case ai.NativeToolReturnEvent:
			returned = &event
		case ai.FinishEvent:
			finish = &event
		}
	}
	if start == nil || !start.Native || start.ToolCallID != "ts-1" ||
		start.ToolKind != ai.ToolPartKindToolSearch || start.ProviderDetails["call_id"] != nil {
		t.Fatalf("unexpected server search start: %+v", start)
	}
	if delta == nil || delta.ToolCallID != "ts-1" || delta.ArgsDelta != `{"queries":["weather"]}` {
		t.Fatalf("unexpected server search delta: %+v", delta)
	}
	if returned == nil || returned.Part.ToolCallID != "ts-1" ||
		returned.Part.Content.(ai.ToolSearchResult).DiscoveredTools[0].Name != "weather" ||
		returned.Part.Timestamp.IsZero() {
		t.Fatalf("unexpected server search return: %+v", returned)
	}
	if finish == nil || len(finish.Parts) != 2 {
		t.Fatalf("missing authoritative server search snapshot: %+v", finish)
	}
	if _, ok := finish.Parts[0].(ai.NativeToolCallPart); !ok {
		t.Fatalf("unexpected final call part %T", finish.Parts[0])
	}
	if _, ok := finish.Parts[1].(ai.NativeToolReturnPart); !ok {
		t.Fatalf("unexpected final return part %T", finish.Parts[1])
	}
}

func TestResponsesStaticServerSearchStreamLifecycle(t *testing.T) {
	completed := `{"type":"response.completed","response":{"id":"response-1","model":"gpt-5.4","status":"completed","output":[{"id":"ts-1","type":"tool_search_call","call_id":"call-1","execution":"server","status":"completed","arguments":{"paths":["weather"]}},{"id":"tso-1","type":"tool_search_output","call_id":"call-1","execution":"server","status":"completed","tools":[{"type":"function","name":"weather"}]}],"usage":{}}}`
	model := newResponsesServer(t, sseHandler(t, []string{completed, `[DONE]`}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var starts, deltas, returns int
	for _, event := range events {
		switch event.(type) {
		case ai.ToolCallStartEvent:
			starts++
		case ai.ToolCallDeltaEvent:
			deltas++
		case ai.NativeToolReturnEvent:
			returns++
		}
	}
	if starts != 1 || deltas != 1 || returns != 1 {
		t.Fatalf("unexpected static native events: %+v", events)
	}

	for name, stop := range map[string]func(ai.ModelStreamEvent) bool{
		"call":   func(event ai.ModelStreamEvent) bool { _, ok := event.(ai.ToolCallStartEvent); return ok },
		"delta":  func(event ai.ModelStreamEvent) bool { _, ok := event.(ai.ToolCallDeltaEvent); return ok },
		"return": func(event ai.ModelStreamEvent) bool { _, ok := event.(ai.NativeToolReturnEvent); return ok },
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, []string{completed, `[DONE]`}))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			for event, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				if stop(event) {
					break
				}
			}
		})
	}
}

func TestResponsesLiveServerSearchStreamCanStop(t *testing.T) {
	events := []string{
		`{"type":"response.output_item.added","item":{"id":"ts-1","type":"tool_search_call","execution":"server","status":"in_progress"}}`,
		`{"type":"response.output_item.done","item":{"id":"ts-1","type":"tool_search_call","execution":"server","status":"completed","arguments":{"paths":[]}}}`,
		`{"type":"response.output_item.done","item":{"id":"tso-1","type":"tool_search_output","execution":"server","status":"completed","tools":[]}}`,
	}
	for name, stop := range map[string]func(ai.ModelStreamEvent) bool{
		"call":   func(event ai.ModelStreamEvent) bool { _, ok := event.(ai.ToolCallStartEvent); return ok },
		"return": func(event ai.ModelStreamEvent) bool { _, ok := event.(ai.NativeToolReturnEvent); return ok },
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, events))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			for event, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				if stop(event) {
					break
				}
			}
		})
	}
}
