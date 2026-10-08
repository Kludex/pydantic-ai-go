package openai_test

import (
	"errors"
	"net/http"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestOpenAIResponseErrorsAreFallbackEligible(t *testing.T) {
	for _, body := range []string{
		string([]byte{0xff}), `{"error":{"code":"invalid","message":"bad"}}`,
		`{"status":"failed"}`, `{"error":{"message":"bad"}}`,
	} {
		for _, responses := range []bool{false, true} {
			handler := func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(body)) }
			var model ai.Model = newServer(t, handler)
			if responses {
				model = newResponsesServer(t, handler)
			}
			if body == `{"status":"failed"}` && !responses {
				continue
			}
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
			var apiError ai.ModelAPIError
			if !errors.As(err, &apiError) || !apiError.IsModelAPIError() {
				t.Fatalf("responses=%t body=%q error=%v", responses, body, err)
			}
		}
	}
	_, err := collect(t, newServer(t, sseHandler(t, []string{string([]byte{0xff})})), ai.ModelRequestParams{})
	var apiError ai.ModelAPIError
	if !errors.As(err, &apiError) {
		t.Fatalf("invalid UTF-8 stream was not normalized: %v", err)
	}
}

func TestResponsesWebSearchUsage(t *testing.T) {
	for _, test := range []struct {
		body string
		want int
	}{
		{body: `{"status":"completed","output":[{"type":"web_search_call","id":"search","action":{"type":"search","query":"Go"}},{"type":"web_search_call","id":"page","action":{"type":"open_page","url":"https://go.dev"}}]}`, want: 1},
		{body: `{"status":"completed","tool_usage":{"web_search":{"num_requests":3}},"output":[]}`, want: 3},
		{body: `{"status":"in_progress","tool_usage":{"web_search":{"num_requests":3}}}`},
		{body: `{"status":"completed","tool_usage":{"web_search":{"num_requests":0}},"output":[{"type":"message","content":[{"type":"output_text","text":""}]}]}`},
	} {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(test.body)) })
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil || response.Usage.Details["web_search_requests"] != test.want {
			t.Fatalf("body=%s response=%+v err=%v", test.body, response, err)
		}
	}
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.completed","response":{"status":"completed","tool_usage":{"web_search":{"num_requests":2}},"output":[]}}`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil || events[len(events)-1].(ai.FinishEvent).Usage.Details["web_search_requests"] != 2 {
		t.Fatalf("stream usage=%+v err=%v", events, err)
	}
}

func TestOpenAICacheRetentionUsesGuaranteedWindow(t *testing.T) {
	settings, err := (openai.Settings{PromptCacheRetention: openai.PromptCacheRetention24Hours}).Build()
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range []ai.Model{openai.NewModel("gpt-5.6"), openai.NewResponsesModel("gpt-6.1-sol")} {
		profile := model.(ai.ModelProfiler).ModelProfile()
		if profile.DefaultCacheRetention != 30*time.Minute {
			t.Fatalf("missing cache profile: %+v", profile)
		}
		if duration, known := ai.ResolveCacheRetention(model, &settings); known || duration != 0 {
			t.Fatalf("maximum retention widened expected window: %s %t", duration, known)
		}
	}
	for _, model := range []ai.Model{openai.NewModel("gpt-5"), openai.NewResponsesModel("gpt-5")} {
		if duration, known := ai.ResolveCacheRetention(model, &settings); !known || duration != 24*time.Hour {
			t.Fatalf("legacy retention=%s known=%t", duration, known)
		}
		if duration, known := ai.ResolveCacheRetention(model, nil); known || duration != 0 {
			t.Fatalf("unspecified retention=%s known=%t", duration, known)
		}
	}
}

func TestResponsesStreamTerminalAndTextEdges(t *testing.T) {
	chunks := []string{
		`{"type":"response.refusal.done","refusal":""}`,
		`{"type":"response.output_text.delta","delta":""}`,
		`{"type":"response.output_text.delta","item_id":"message","delta":"answer"}`,
		`{"type":"response.output_item.added","item":{"type":"message","id":"message","phase":"commentary"}}`,
		`{"type":"response.output_text.done","item_id":"message","logprobs":[{"token":"answer"}]}`,
		`{"type":"response.output_item.done","item":{"type":"tool_search_output","execution":"server","id":"search","tools":[]}}`,
		`{"type":"response.incomplete","response":{"status":"incomplete","output":[{"type":"message","content":[{"type":"output_text","text":"partial"}]}]}}`,
	}
	if _, err := collect(t, newResponsesServer(t, sseHandler(t, chunks)), ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	for _, stop := range []int{1, 2} {
		model := newResponsesServer(t, sseHandler(t, chunks))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		seen := 0
		for event, err := range stream {
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := event.(ai.TextDeltaEvent); ok {
				seen++
				if seen == stop {
					break
				}
			}
		}
	}
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.incomplete","response":{"output":[{"type":"mcp_approval_request"}]}}`,
	}))
	if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
		t.Fatal("incomplete response dropped invalid output")
	}
	model = newResponsesServer(t, sseHandler(t, []string{chunks[len(chunks)-1]}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
}
