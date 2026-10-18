package google_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"os"
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// requestLog records the JSON body of every request sent through it.
type requestLog struct {
	next   http.RoundTripper
	bodies []map[string]any
}

func (l *requestLog) RoundTrip(request *http.Request) (*http.Response, error) {
	data, err := io.ReadAll(request.Body)
	if err != nil {
		return nil, err
	}
	request.Body = io.NopCloser(bytes.NewReader(data))
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		return nil, err
	}
	l.bodies = append(l.bodies, body)
	return l.next.RoundTrip(request)
}

// vcrModel replays committed cassettes; with a real GOOGLE_API_KEY and a
// missing cassette it records new interactions, filtering credentials.
func vcrModel(t *testing.T, name, modelName string) (*google.Model, *requestLog) {
	t.Helper()
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("GOOGLE_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	r, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(i *cassette.Interaction) error {
			delete(i.Request.Headers, "X-Goog-Api-Key")
			delete(i.Response.Headers, "Set-Cookie")
			return nil
		}, recorder.AfterCaptureHook),
		recorder.WithMatcher(cassette.MatcherFunc(func(r *http.Request, i cassette.Request) bool {
			return r.Method == i.Method && r.URL.String() == i.URL
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})
	log := &requestLog{next: r}
	model := google.NewModel(modelName, google.WithHTTPClient(&http.Client{Transport: log}))
	return model, log
}

func TestRecordedGemini25Search(t *testing.T) {
	model, log := vcrModel(t, "gemini_2_5_search", "gemini-2.5-flash")
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithNativeTools(ai.WebSearchTool{}),
		ai.WithInstructions("Search the web, then answer in under ten words."),
		ai.WithModelSettings(ai.ModelSettings{
			MaxTokens: 100, Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
		}),
	)
	first, err := agent.Run(t.Context(), "Who won the 2022 FIFA World Cup?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	call, returned := lastSearch(t, first.NewMessages())
	sources, _ := returned.Content.([]map[string]any)
	if call.ToolCallID == "" || returned.ToolCallID != call.ToolCallID || len(sources) == 0 || sources[0]["uri"] == nil {
		t.Fatalf("unexpected reconstructed search: call=%+v return=%+v", call, returned)
	}

	if _, err := agent.Run(t.Context(), "And in 2018?", struct{}{},
		ai.WithMessageHistory(first.Messages())); err != nil {
		t.Fatal(err)
	}
	for _, part := range modelParts(t, log.bodies[1]) {
		if part["toolCall"] != nil || part["toolResponse"] != nil {
			t.Fatalf("reconstructed search part was replayed: %#v", part)
		}
	}
}

func TestRecordedGemini3Search(t *testing.T) {
	model, log := vcrModel(t, "gemini_3_search", "gemini-3.8-flash")
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithNativeTools(ai.WebSearchTool{}),
		ai.WithInstructions("Search the web, then answer in under ten words."),
		ai.WithModelSettings(ai.ModelSettings{
			MaxTokens: 1024, Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelMinimal},
		}),
	)
	first, err := agent.Run(t.Context(), "Who won the 2022 FIFA World Cup?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	call, returned := lastSearch(t, first.NewMessages())
	content, _ := returned.Content.(map[string]any)
	sources, _ := content["sources"].([]map[string]any)
	if call.ToolCallID == "" || returned.ToolCallID != call.ToolCallID || len(sources) == 0 || sources[0]["uri"] == nil {
		t.Fatalf("unexpected explicit search: call=%+v return=%+v", call, returned)
	}

	if _, err := agent.Run(t.Context(), "And in 2018?", struct{}{},
		ai.WithMessageHistory(first.Messages())); err != nil {
		t.Fatal(err)
	}
	want := maps.Clone(content)
	delete(want, "sources")
	var replayed map[string]any
	for _, part := range modelParts(t, log.bodies[1]) {
		if response, ok := part["toolResponse"].(map[string]any); ok && response["id"] == returned.ToolCallID {
			replayed = part
		}
	}
	if replayed == nil || replayed["thoughtSignature"] == nil ||
		!reflect.DeepEqual(replayed["toolResponse"].(map[string]any)["response"], want) {
		t.Fatalf("search return not replayed as the provider sent it: %#v", replayed)
	}
}

func TestRecordedGemini3StreamSearch(t *testing.T) {
	model, _ := vcrModel(t, "gemini_3_stream_search", "gemini-3.8-flash")
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithNativeTools(ai.WebSearchTool{}),
		ai.WithInstructions("Search the web, then answer in under ten words."),
		ai.WithModelSettings(ai.ModelSettings{
			MaxTokens: 1024, Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelMinimal},
		}),
	)
	stream := agent.RunStream(t.Context(), "Who won the 2022 FIFA World Cup?", struct{}{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	result := stream.Result()
	if result == nil || result.Output == "" {
		t.Fatalf("expected streamed output: %#v", result)
	}
	call, returned := lastSearch(t, result.NewMessages())
	content, _ := returned.Content.(map[string]any)
	sources, _ := content["sources"].([]map[string]any)
	if call.ToolCallID == "" || returned.ToolCallID != call.ToolCallID || len(sources) == 0 {
		t.Fatalf("unexpected streamed search: call=%+v return=%+v", call, returned)
	}
}

func TestRecordedGemini3SearchWithFunction(t *testing.T) {
	model, log := vcrModel(t, "gemini_3_search_function", "gemini-3.8-flash")
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithNativeTools(ai.WebSearchTool{}),
		ai.WithInstructions("Search the web for the answer, pass it to record_winner, then reply with one word."),
		ai.WithModelSettings(ai.ModelSettings{
			MaxTokens: 1024, Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelMinimal},
		}),
	)
	var winner string
	ai.AddSimpleTool(agent, "record_winner", func(_ context.Context, args struct {
		Country string `json:"country"`
	}) (string, error) {
		winner = args.Country
		return "recorded", nil
	}, ai.WithDescription("Record the winning country"))

	result, err := agent.Run(t.Context(), "Who won the 2022 FIFA World Cup?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if winner == "" || result.Output == "" {
		t.Fatalf("expected the function tool and output: winner=%q output=%q", winner, result.Output)
	}
	call, returned := lastSearch(t, result.NewMessages())
	if call.ToolCallID == "" || returned.ToolCallID != call.ToolCallID {
		t.Fatalf("unexpected search: call=%+v return=%+v", call, returned)
	}
	var calls, returns []string
	for _, message := range result.NewMessages() {
		switch message := message.(type) {
		case ai.ModelResponse:
			for _, part := range message.Parts {
				if part, ok := part.(ai.ToolCallPart); ok {
					calls = append(calls, part.ToolCallID)
				}
			}
		case ai.ModelRequest:
			for _, part := range message.Parts {
				if part, ok := part.(ai.ToolReturnPart); ok {
					returns = append(returns, part.ToolCallID)
				}
			}
		}
	}
	if len(calls) != 1 || !reflect.DeepEqual(calls, returns) {
		t.Fatalf("function call IDs %v do not pair with returns %v", calls, returns)
	}
	var replayed bool
	for _, part := range modelParts(t, log.bodies[1]) {
		if response, ok := part["toolResponse"].(map[string]any); ok && response["id"] == returned.ToolCallID {
			replayed = response["response"] != nil && response["response"].(map[string]any)["sources"] == nil
		}
	}
	if !replayed {
		t.Fatalf("search return missing from the function result request: %#v", log.bodies[1]["contents"])
	}
}

func lastSearch(t *testing.T, messages []ai.ModelMessage) (ai.NativeToolCallPart, ai.NativeToolReturnPart) {
	t.Helper()
	var call ai.NativeToolCallPart
	var returned ai.NativeToolReturnPart
	for _, message := range messages {
		response, ok := message.(ai.ModelResponse)
		if !ok {
			continue
		}
		for _, part := range response.Parts {
			switch part := part.(type) {
			case ai.NativeToolCallPart:
				if part.ToolKind == ai.ToolPartKindWebSearch {
					call = part
				}
			case ai.NativeToolReturnPart:
				if part.ToolKind == ai.ToolPartKindWebSearch {
					returned = part
				}
			}
		}
	}
	return call, returned
}

func modelParts(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	var parts []map[string]any
	contents, _ := body["contents"].([]any)
	for _, content := range contents {
		content, _ := content.(map[string]any)
		if content["role"] != "model" {
			continue
		}
		entries, _ := content["parts"].([]any)
		for _, part := range entries {
			part, _ := part.(map[string]any)
			parts = append(parts, part)
		}
	}
	return parts
}
