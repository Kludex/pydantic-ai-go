package xai_test

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/xai"
)

func TestModelStream(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"response","model":"grok-4.3","created_at":100,"status":"in_progress"}}`,
		`{"type":"response.output_item.added","item":{"id":"x-1","type":"x_search_call","status":"in_progress"}}`,
		`{"type":"response.output_item.done","item":{"id":"x-1","type":"x_search_call","status":"completed","action":{"query":"Go"},"output":{"citations":["https://x.com/post"]}}}`,
		`{"type":"response.output_item.added","item":{"id":"c-1","type":"collections_search_call","status":"in_progress"}}`,
		`{"type":"response.output_item.done","item":{"id":"c-1","type":"collections_search_call","status":"completed","queries":["docs"],"results":[{"text":"answer"}]}}`,
		`{"type":"response.output_item.added","item":{"id":"message","type":"message"}}`,
		`{"type":"response.output_text.delta","item_id":"message","content_index":0,"delta":"done"}`,
		`{"type":"response.completed","response":{"id":"response","model":"grok-4.3","created_at":100,"status":"completed","usage":{"input_tokens":2,"output_tokens":3},"output":[{"id":"x-1","type":"x_search_call","status":"completed","action":{"query":"Go"},"output":{"citations":["https://x.com/post"]}},{"id":"c-1","type":"collections_search_call","status":"completed","queries":["docs"],"results":[{"text":"answer"}]},{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}]}}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("missing stream header")
		}
		response.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(response, "data: %s\n\n", event)
		}
		_, _ = io.WriteString(response, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := xai.NewModel(
		"grok-4.3", xai.WithBaseURL(server.URL), xai.WithAPIKey("key"), xai.WithHTTPClient(server.Client()),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.XSearchTool{}, ai.FileSearchTool{FileStoreIDs: []string{"collection"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var starts []ai.ToolCallStartEvent
	var returns []ai.NativeToolReturnEvent
	var finish ai.FinishEvent
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			starts = append(starts, event)
		case ai.NativeToolReturnEvent:
			returns = append(returns, event)
		case ai.FinishEvent:
			finish = event
		}
	}
	if len(starts) != 2 || starts[0].ToolKind != ai.ToolPartKindXSearch ||
		starts[1].ToolKind != ai.ToolPartKindFileSearch || len(returns) != 2 ||
		returns[0].Part.ProviderName != "xai" || len(finish.Parts) != 5 || finish.Usage.TotalTokens() != 5 {
		t.Fatalf("unexpected stream: starts=%#v returns=%#v finish=%#v", starts, returns, finish)
	}
}
