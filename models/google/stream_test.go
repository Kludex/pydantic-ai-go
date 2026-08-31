package google_test

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func googleSSE(t *testing.T, chunks []string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
		}
	}
}

func collectGoogleStream(
	t *testing.T, model ai.StreamingModel, params ai.ModelRequestParams,
) ([]ai.ModelStreamEvent, error) {
	t.Helper()
	stream, err := model.StreamRequest(t.Context(), nil, params)
	if err != nil {
		return nil, err
	}
	var events []ai.ModelStreamEvent
	for event, err := range stream {
		if err != nil {
			return events, err
		}
		events = append(events, event)
	}
	return events, nil
}

func TestGoogleStreamWebSearchGroundingMetadata(t *testing.T) {
	grounding := `"groundingMetadata":{"webSearchQueries":["Go news"],"groundingChunks":[{"web":{"uri":"https://go.dev","title":"Go"}}]}`
	model := newServer(t, googleSSE(t, []string{
		`{"responseId":"response","candidates":[{"content":{"parts":[{"text":"first"}]},` + grounding + `}]}`,
		`{"responseId":"response","candidates":[{"content":{"parts":[{"text":"second"}]},` + grounding + `}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}`,
	}))
	events, err := collectGoogleStream(t, model, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var starts, deltas, returns int
	var returned ai.NativeToolReturnEvent
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			if event.Native && event.ToolKind == ai.ToolPartKindWebSearch {
				starts++
			}
		case ai.ToolCallDeltaEvent:
			if strings.Contains(event.ArgsDelta, "Go news") {
				deltas++
			}
		case ai.NativeToolReturnEvent:
			returns++
			returned = event
		case ai.FinishEvent:
			finish = event
		}
	}
	if starts != 1 || deltas != 1 || returns != 1 || returned.Part.Timestamp.IsZero() ||
		returned.Part.Content.([]map[string]any)[0]["title"] != "Go" ||
		finish.ProviderDetails["grounding_metadata"] == nil {
		t.Fatalf("unexpected streamed grounding events: %#v", events)
	}
}

func TestGoogleStreamWebFetchURLContext(t *testing.T) {
	metadata := `"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://go.dev","urlRetrievalStatus":"URL_RETRIEVAL_STATUS_SUCCESS"}]}`
	model := newServer(t, googleSSE(t, []string{
		`{"responseId":"response","candidates":[{"content":{"parts":[{"text":"first"}]},` + metadata + `}]}`,
		`{"responseId":"response","candidates":[{"content":{"parts":[{"text":"second"}]},` + metadata + `}]}`,
	}))
	events, err := collectGoogleStream(t, model, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebFetchTool{}}})
	if err != nil {
		t.Fatal(err)
	}
	var starts, returns int
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			if event.ToolKind == ai.ToolPartKindWebFetch {
				starts++
			}
		case ai.NativeToolReturnEvent:
			if event.Part.ToolKind == ai.ToolPartKindWebFetch {
				returns++
			}
		case ai.FinishEvent:
			finish = event
		}
	}
	if starts != 1 || returns != 1 || finish.ProviderDetails["url_context_metadata"] == nil {
		t.Fatalf("unexpected streamed URL context events: %#v", events)
	}
}

func TestGoogleStreamCodeExecution(t *testing.T) {
	model := newServer(t, googleSSE(t, []string{
		`{"responseId":"response","candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(1)"}}]}}]}`,
		`{"responseId":"response","candidates":[{"content":{"parts":[{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1\n"}}]}}]}`,
	}))
	events, err := collectGoogleStream(t, model, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}}})
	if err != nil {
		t.Fatal(err)
	}
	start := events[0].(ai.ToolCallStartEvent)
	delta := events[1].(ai.ToolCallDeltaEvent)
	returned := events[2].(ai.NativeToolReturnEvent)
	if !start.Native || start.ToolKind != ai.ToolPartKindCodeExecution ||
		start.ToolCallID != "response:code_execution:0" || !strings.Contains(delta.ArgsDelta, "print(1)") ||
		returned.Part.ToolCallID != start.ToolCallID || returned.Part.Content.(map[string]any)["output"] != "1\n" {
		t.Fatalf("unexpected streamed code execution events: %#v", events)
	}
}

func TestGoogleStreamCodeExecutionEdges(t *testing.T) {
	t.Run("orphan result", func(t *testing.T) {
		model := newServer(t, googleSSE(t, []string{
			`{"candidates":[{"content":{"parts":[{"codeExecutionResult":{"outcome":"OUTCOME_FAILED","output":"bad"}}]}}]}`,
		}))
		events, err := collectGoogleStream(t, model, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if returned := events[0].(ai.NativeToolReturnEvent); returned.Part.ToolCallID != "code_execution:0" {
			t.Fatalf("unexpected orphan code result: %+v", returned)
		}
	})
	for _, breakAfter := range []int{1, 2} {
		t.Run(fmt.Sprintf("consumer break %d", breakAfter), func(t *testing.T) {
			model := newServer(t, googleSSE(t, []string{
				`{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"pass"}}]}}]}`,
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
		})
	}
	t.Run("consumer break on result", func(t *testing.T) {
		model := newServer(t, googleSSE(t, []string{
			`{"candidates":[{"content":{"parts":[{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"done"}}]}}]}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for _, err := range stream {
			if err != nil {
				t.Fatal(err)
			}
			break
		}
	})
}

func TestGoogleStreamWebFetchConsumerBreak(t *testing.T) {
	model := newServer(t, googleSSE(t, []string{
		`{"responseId":"response","candidates":[{"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://go.dev"}]}}]}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		break
	}
}

func TestGoogleStreamWebSearchConsumerBreak(t *testing.T) {
	model := newServer(t, googleSSE(t, []string{
		`{"responseId":"response","candidates":[{"groundingMetadata":{"webSearchQueries":["query"]}}]}`,
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
		break
	}
	if seen != 1 {
		t.Fatalf("stream yielded %d events before break", seen)
	}
}

func normalizedGoogleText(event ai.StreamEvent) string {
	switch event := event.(type) {
	case ai.PartStartEvent:
		if text, ok := event.Part.(ai.TextPart); ok {
			return text.Content
		}
	case ai.PartDeltaEvent:
		if text, ok := event.Delta.(ai.TextPartDelta); ok {
			return text.ContentDelta
		}
	}
	return ""
}

func TestGoogleStreamPromptFeedbackBlock(t *testing.T) {
	model := newServer(t, googleSSE(t, []string{
		`{"responseId":"empty"}`,
		`{"responseId":"blocked","promptFeedback":{"blockReason":"PROHIBITED_CONTENT","blockReasonMessage":"The prompt was blocked.","safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","blocked":true}]}}`,
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
		filtered.Response().ProviderDetails["block_reason"] != "PROHIBITED_CONTENT" ||
		filtered.Response().ProviderDetails["block_reason_message"] != "The prompt was blocked." {
		t.Fatalf("unexpected streamed prompt block: %v response=%+v", streamErr, filtered)
	}
	if ratings, ok := filtered.Response().ProviderDetails["safety_ratings"].([]map[string]any); !ok || len(ratings) != 1 {
		t.Fatalf("unexpected streamed safety ratings: %#v", filtered.Response().ProviderDetails)
	}
}

func TestStreamEvents(t *testing.T) {
	var path, query, accept, custom string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		path, query, accept = r.URL.Path, r.URL.RawQuery, r.Header.Get("Accept")
		custom = r.Header.Get("x-custom")
		w.Header().Set("x-gemini-service-tier", "FLEX")
		googleSSE(t, []string{
			`{"responseId":"response-stream","modelVersion":"gemini-stream","candidates":[{"content":{"parts":[{"text":"Hel","thoughtSignature":"text-signature"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"thought":true,"text":"plan","thoughtSignature":"thinking-signature"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"functionCall":{"id":"c1","name":"work","args":{"x":1}},"thoughtSignature":"tool-signature"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP","safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"NEGLIGIBLE"}],"avgLogprobs":-0.25,"logprobsResult":{"chosenCandidates":[{"token":"lo"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3}}`,
		})(w, r)
	})
	events, err := collectGoogleStream(t, model, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraHeaders: map[string]string{"x-custom": "stream"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(path, ":streamGenerateContent") || query != "alt=sse" ||
		accept != "text/event-stream" || custom != "stream" {
		t.Fatalf("unexpected request path=%q query=%q accept=%q custom=%q", path, query, accept, custom)
	}
	var text, thinking, args string
	var textPartID, textSignature, thinkingPartID, thinkingSignature, argsPartID string
	var start ai.ToolCallStartEvent
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			text += event.Delta
			textPartID = event.PartID
			if signature, ok := event.ProviderDetails["thought_signature"].(string); ok {
				textSignature = signature
			}
		case ai.ThinkingDeltaEvent:
			thinking += event.Delta
			thinkingPartID = event.PartID
			if signature, ok := event.ProviderDetails["thought_signature"].(string); ok {
				thinkingSignature = signature
			}
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			args += event.ArgsDelta
			argsPartID = event.PartID
		case ai.FinishEvent:
			finish = event
		}
	}
	if text != "Hello" || thinking != "plan" || start.ToolName != "work" || start.ToolCallID != "c1" || args != `{"x":1}` {
		t.Fatalf("unexpected events text=%q thinking=%q start=%+v args=%q", text, thinking, start, args)
	}
	if textPartID != "text:0" || textSignature != "text-signature" ||
		thinkingPartID != "thinking:0" || thinkingSignature != "thinking-signature" ||
		start.PartID != "tool:0" || start.ProviderDetails["thought_signature"] != "tool-signature" ||
		argsPartID != "tool:0" {
		t.Fatalf("unstable Gemini part IDs: text=%q thinking=%q start=%q args=%q", textPartID, thinkingPartID, start.PartID, argsPartID)
	}
	if finish.ModelName != "gemini-stream" || finish.Usage.Requests != 1 || finish.Usage.InputTokens != 5 ||
		finish.Usage.OutputTokens != 3 || finish.ProviderName != "google" || finish.ProviderURL == "" ||
		finish.ProviderResponseID != "response-stream" || finish.FinishReason != ai.FinishReasonStop ||
		finish.ProviderDetails["finish_reason"] != "STOP" || finish.ProviderDetails["service_tier"] != "flex" ||
		finish.ProviderDetails["avg_logprobs"] != -0.25 || finish.ProviderDetails["logprobs"] == nil ||
		len(finish.ProviderDetails["safety_ratings"].([]map[string]any)) != 1 {
		t.Fatalf("unexpected finish %+v", finish)
	}
}

func TestStreamEndToEnd(t *testing.T) {
	model := newServer(t, googleSSE(t, []string{
		`{"modelVersion":"gemini","candidates":[{"content":{"parts":[{"text":"hello"}]}}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1}}`,
	}))
	agent := ai.NewAgent[struct{}, string](model)
	stream := agent.RunStream(t.Context(), "go", struct{}{})
	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += normalizedGoogleText(event)
	}
	if text != "hello" || stream.Result().Output != "hello" {
		t.Fatalf("unexpected stream text %q and result %+v", text, stream.Result())
	}
}

func TestStreamProtocolErrors(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   string
	}{
		{name: "malformed", chunks: []string{`not json`}, want: "parse stream chunk"},
		{name: "binary", chunks: []string{`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"aGk="}}]}}]}`}, want: "binary output"},
		{name: "empty", want: "without a response"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newServer(t, googleSSE(t, test.chunks))
			_, err := collectGoogleStream(t, model, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q error, got %v", test.want, err)
			}
		})
	}
}

func TestStreamRequestErrors(t *testing.T) {
	t.Run("bad payload", func(t *testing.T) {
		model := newServer(t, googleSSE(t, nil))
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{"x": make(chan int)}}}}
		if _, err := collectGoogleStream(t, model, params); err == nil {
			t.Fatal("expected marshal error")
		}
	})
	t.Run("bad message", func(t *testing.T) {
		model := newServer(t, googleSSE(t, nil))
		if _, err := model.StreamRequest(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected message error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := google.NewModel("gemini", google.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := google.NewModel("gemini", google.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("http error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad"))
		})
		if _, err := collectGoogleStream(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("truncated error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("short"))
		})
		if _, err := collectGoogleStream(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestStreamScannerError(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(make([]byte, 2*1024*1024))
	})
	if _, err := collectGoogleStream(t, model, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected scanner error")
	}
}

func TestStreamEarlyBreak(t *testing.T) {
	chunks := []string{
		`{"candidates":[]}`,
		`{"candidates":[{"content":{"parts":[{}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"thought":true,"text":"a"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"id":"c","name":"work","args":{"x":1}}}]}}]}`,
	}
	for breakAt := 1; breakAt <= 4; breakAt++ {
		model := newServer(t, googleSSE(t, chunks))
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
