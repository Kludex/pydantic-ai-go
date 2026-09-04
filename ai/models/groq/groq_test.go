package groq_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/groq"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestModelAndSettings(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("unexpected request: %s %v", request.URL.Path, request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{
			"id":"completion-1","model":"openai/gpt-oss-20b",
			"choices":[{"finish_reason":"stop","message":{"reasoning":"thinking","content":"hello"}}],
			"usage":{"prompt_tokens":3,"completion_tokens":2}
		}`))
	}))
	defer server.Close()

	common := ai.ModelSettings{ExtraBody: map[string]any{"custom": true}}
	settings, err := (groq.Settings{
		Common: common, ReasoningFormat: groq.ReasoningFormatParsed, ReasoningEffort: groq.ReasoningEffortHigh,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	common.ExtraBody["custom"] = false
	model := groq.NewModel(
		"openai/gpt-oss-20b", groq.WithBaseURL(server.URL), groq.WithAPIKey("secret"),
		groq.WithHTTPClient(server.Client()), groq.WithDefaultSettings(settings),
	)
	response, err := model.Request(context.Background(), nil, ai.ModelRequestParams{Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "openai/gpt-oss-20b" || model.ProviderName() != "groq" || model.ProviderURL() != server.URL ||
		response.Text() != "hello" || len(response.Parts) != 2 {
		t.Fatalf("unexpected model or response: model=%q provider=%q url=%q response=%#v",
			model.Name(), model.ProviderName(), model.ProviderURL(), response)
	}
	if body["reasoning_format"] != "parsed" || body["reasoning_effort"] != "high" || body["custom"] != true {
		t.Fatalf("unexpected Groq request body: %#v", body)
	}
	defaults := model.DefaultModelSettings()
	defaults.ExtraBody["custom"] = false
	if model.DefaultModelSettings().ExtraBody["custom"] != true {
		t.Fatal("default settings were not detached")
	}
}

func TestTaggedThinking(t *testing.T) {
	var staticRequests int
	var streamRequests int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "application/json")
		if request.Header.Get("Accept") == "text/event-stream" {
			streamRequests++
			switch streamRequests {
			case 1:
				_, _ = response.Write([]byte("data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"pre<th\"}}]}\n\n" +
					"data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"ink>rea\"}}]}\n\n" +
					"data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"son</think>post\"}}]}\n\n" +
					"data: [DONE]\n\n"))
			case 2:
				_, _ = response.Write([]byte("data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"<think>x\"}}]}\n\n" +
					"data: [DONE]\n\n"))
			case 3:
				_, _ = response.Write([]byte("data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"tail<\"}}]}\n\n" +
					"data: invalid\n\n"))
			case 4:
				_, _ = response.Write([]byte("data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"tool_calls\":[{" +
					"\"index\":0,\"id\":\"call\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]}}]}\n\n" +
					"data: [DONE]\n\n"))
			case 5:
				_, _ = response.Write([]byte("data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"tail<\"}}]}\n\n" +
					"data: invalid\n\n"))
			case 6:
				_, _ = response.Write([]byte("data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"tail<\"}}]}\n\n" +
					"data: [DONE]\n\n"))
			default:
				_, _ = response.Write([]byte("data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"content\":\"text\"}}]}\n\n" +
					"data: [DONE]\n\n"))
			}
			return
		}
		staticRequests++
		content := `<think>a</think>b<think>c</think>d`
		if staticRequests == 2 {
			content = `before<think>unclosed`
		}
		_, _ = response.Write([]byte(`{"model":"model","choices":[{"finish_reason":"stop","message":{"content":` +
			strconv.Quote(content) + `}}]}`))
	}))
	defer server.Close()
	model := groq.NewModel("deepseek-r1", groq.WithBaseURL(server.URL), groq.WithAPIKey("key"),
		groq.WithHTTPClient(server.Client()))

	response, err := model.Request(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil || len(response.Parts) != 4 {
		t.Fatalf("unexpected tagged response: %#v %v", response, err)
	}
	if response.Parts[0].(ai.ThinkingPart).Content != "a" || response.Parts[1].(ai.TextPart).Content != "b" ||
		response.Parts[2].(ai.ThinkingPart).Content != "c" || response.Parts[3].(ai.TextPart).Content != "d" {
		t.Fatalf("unexpected tagged response parts: %#v", response.Parts)
	}
	response, err = model.Request(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil || len(response.Parts) != 2 || response.Parts[0].(ai.TextPart).Content != "before" ||
		response.Parts[1].(ai.TextPart).Content != "unclosed" {
		t.Fatalf("unexpected unclosed tagged response: %#v %v", response, err)
	}

	stream, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ModelStreamEvent
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		events = append(events, event)
	}
	if len(events) != 5 || events[0].(ai.TextDeltaEvent).Delta != "pre" ||
		events[1].(ai.ThinkingDeltaEvent).Delta != "rea" ||
		events[2].(ai.ThinkingDeltaEvent).Delta != "son" ||
		events[3].(ai.TextDeltaEvent).Delta != "post" {
		t.Fatalf("unexpected tagged stream: %#v", events)
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events = nil
	for event, eventErr := range stream {
		if eventErr != nil {
			break
		}
		events = append(events, event)
	}
	if len(events) != 2 || events[0].(ai.TextDeltaEvent).Delta != "tail" ||
		events[1].(ai.TextDeltaEvent).Delta != "<" {
		t.Fatalf("pending tagged text was lost on error: %#v", events)
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for range stream {
		count++
		if count == 2 {
			break
		}
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	count = 0
	for range stream {
		count++
		if count == 2 {
			break
		}
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
}

func TestToolUseFailedResponses(t *testing.T) {
	bodies := []string{
		`{"error":{"code":"tool_use_failed","failed_generation":"{\"name\":\"lookup\",\"arguments\":{\"city\":\"Paris\"}}"}}`,
		`{"code":"tool_use_failed","failed_generation":"plain fallback"}`,
		`{"error":{"code":"tool_use_failed","failed_generation":""}}`,
		`not json`,
		`{"error":{"code":"tool_use_failed","failed_generation":"{\"name\":\"lookup\",\"arguments\":{}}"}}`,
		`{"code":"tool_use_failed","failed_generation":"plain fallback"}`,
		`{"error":{"code":"tool_use_failed","failed_generation":""}}`,
		`{"error":{"code":"other","failed_generation":"ignored"}}`,
		`{"error":{"code":"tool_use_failed","failed_generation":"{\"name\":\"lookup\",\"arguments\":{}}"}}`,
		`{"code":"tool_use_failed","failed_generation":"plain fallback"}`,
	}
	request := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		body := bodies[request]
		request++
		response.Header().Set("Content-Type", "application/json")
		response.WriteHeader(http.StatusBadRequest)
		_, _ = response.Write([]byte(body))
	}))
	defer server.Close()
	model := groq.NewModel("model", groq.WithBaseURL(server.URL), groq.WithAPIKey("key"),
		groq.WithHTTPClient(server.Client()))

	result, err := model.Request(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil || len(result.Parts) != 1 || result.FinishReason != ai.FinishReasonError {
		t.Fatalf("unexpected failed-tool response: %#v %v", result, err)
	}
	call := result.Parts[0].(ai.ToolCallPart)
	if call.ToolName != "lookup" || string(call.Args) != `{"city":"Paris"}` ||
		!strings.HasPrefix(call.ToolCallID, "groq-failed:") {
		t.Fatalf("unexpected recovered tool call: %#v", call)
	}
	result, err = model.Request(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil || result.Text() != "plain fallback" {
		t.Fatalf("unexpected fallback text: %#v %v", result, err)
	}
	result, err = model.Request(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil || len(result.Parts) != 0 {
		t.Fatalf("unexpected empty failed generation: %#v %v", result, err)
	}
	if _, err = model.Request(context.Background(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("unrelated API error was normalized")
	}

	stream, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ModelStreamEvent
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		events = append(events, event)
	}
	if len(events) != 3 {
		t.Fatalf("unexpected failed-tool stream: %#v", events)
	}
	start := events[0].(ai.ToolCallStartEvent)
	if start.ToolName != "lookup" || events[1].(ai.ToolCallDeltaEvent).ArgsDelta != `{}` ||
		events[2].(ai.FinishEvent).FinishReason != ai.FinishReasonError {
		t.Fatalf("unexpected recovered tool stream: %#v", events)
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events = nil
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		events = append(events, event)
	}
	if len(events) != 2 || events[0].(ai.TextDeltaEvent).Delta != "plain fallback" {
		t.Fatalf("unexpected fallback text stream: %#v", events)
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events = nil
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		events = append(events, event)
	}
	if len(events) != 1 {
		t.Fatalf("unexpected empty failed stream: %#v", events)
	}
	if stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{}); err == nil || stream != nil {
		t.Fatalf("unrelated streamed API error was normalized: %v", err)
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
	stream, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
}

func TestGroqUserContent(t *testing.T) {
	var requests int
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{
			"id":"completion","model":"model",
			"choices":[{"finish_reason":"stop","message":{"content":"done"}}]
		}`))
	}))
	defer server.Close()
	model := groq.NewModel("model", groq.WithBaseURL(server.URL), groq.WithAPIKey("key"),
		groq.WithHTTPClient(server.Client()))
	messages := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "history"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "system"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "describe"},
				ai.ImageURL{URL: "https://example.com/image.png"},
				ai.BinaryContent{Data: []byte("image"), MediaType: "IMAGE/PNG"},
				ai.CachePoint{},
			}},
		}},
	}
	result, err := model.Request(context.Background(), messages, ai.ModelRequestParams{})
	if err != nil || result.Text() != "done" {
		t.Fatalf("unexpected multimodal response: %#v %v", result, err)
	}
	wireMessages := body["messages"].([]any)
	content := wireMessages[2].(map[string]any)["content"].([]any)
	if len(content) != 3 || content[0].(map[string]any)["text"] != "describe" ||
		content[1].(map[string]any)["image_url"].(map[string]any)["url"] != "https://example.com/image.png" ||
		!strings.HasPrefix(content[2].(map[string]any)["image_url"].(map[string]any)["url"].(string),
			"data:IMAGE/PNG;base64,") {
		t.Fatalf("unexpected Groq user content: %#v", body)
	}

	invalid := []ai.UserContent{
		ai.BinaryContent{MediaType: "audio/wav"},
		ai.DocumentURL{URL: "https://example.com/file.pdf"},
		ai.AudioURL{URL: "https://example.com/audio.wav"},
		ai.VideoURL{URL: "https://example.com/video.mp4"},
		ai.UploadedFile{ProviderName: "groq", FileID: "file"},
	}
	for _, content := range invalid {
		request := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Contents: []ai.UserContent{content}},
		}}}
		if _, err := model.Request(context.Background(), request, ai.ModelRequestParams{}); err == nil ||
			!strings.Contains(err.Error(), "groq:") {
			t.Fatalf("unexpected unsupported content error for %T: %v", content, err)
		}
		stream, err := model.StreamRequest(context.Background(), request, ai.ModelRequestParams{})
		if err == nil || stream != nil || !strings.Contains(err.Error(), "groq:") {
			t.Fatalf("unexpected streamed content error for %T: %v", content, err)
		}
	}
	if requests != 1 {
		t.Fatalf("unsupported content reached transport: %d requests", requests)
	}
}

func TestModelFamilyThinking(t *testing.T) {
	var bodies []map[string]any
	var warnings []groq.ReasoningWarning
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		response.Header().Set("Content-Type", "application/json")
		if body["stream"] == true {
			_, _ = response.Write([]byte("data: {\"id\":\"completion\",\"model\":\"model\"," +
				"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n"))
			return
		}
		_, _ = response.Write([]byte(`{
			"id":"completion","model":"model",
			"choices":[{"finish_reason":"stop","message":{"content":"done"}}]
		}`))
	}))
	defer server.Close()

	tests := []struct {
		name           string
		model          string
		level          ai.ThinkingLevel
		extra          map[string]any
		expectedFormat any
		expectedEffort any
	}{
		{name: "unsupported", model: "llama-3.3-70b-versatile", level: ai.ThinkingLevelHigh},
		{name: "qwen disabled", model: "qwen/qwen3-32b", level: ai.ThinkingLevelDisabled,
			extra: map[string]any{"reasoning_effort": "high"}, expectedEffort: "none"},
		{name: "qwen enabled", model: "qwen/qwen3-32b", level: ai.ThinkingLevelEnabled,
			expectedFormat: "parsed"},
		{name: "gpt enabled", model: "openai/gpt-oss-20b", level: ai.ThinkingLevelEnabled,
			expectedFormat: "parsed", expectedEffort: "medium"},
		{name: "gpt minimal", model: "openai/gpt-oss-20b", level: ai.ThinkingLevelMinimal,
			expectedFormat: "parsed", expectedEffort: "low"},
		{name: "gpt low", model: "openai/gpt-oss-20b", level: ai.ThinkingLevelLow,
			expectedFormat: "parsed", expectedEffort: "low"},
		{name: "gpt medium", model: "openai/gpt-oss-20b", level: ai.ThinkingLevelMedium,
			expectedFormat: "parsed", expectedEffort: "medium"},
		{name: "gpt high", model: "openai/gpt-oss-20b", level: ai.ThinkingLevelHigh,
			expectedFormat: "parsed", expectedEffort: "high"},
		{name: "gpt xhigh", model: "openai/gpt-oss-20b", level: ai.ThinkingLevelXHigh,
			expectedFormat: "parsed", expectedEffort: "high"},
		{name: "gpt disabled", model: "openai/gpt-oss-20b", level: ai.ThinkingLevelDisabled,
			expectedFormat: "hidden"},
		{name: "legacy enabled", model: "deepseek-r1-distill-llama-70b", level: ai.ThinkingLevelEnabled,
			expectedFormat: "parsed"},
		{name: "legacy disabled", model: "qwen-qwq-32b", level: ai.ThinkingLevelDisabled,
			expectedFormat: "hidden"},
		{name: "explicit", model: "openai/gpt-oss-120b", level: ai.ThinkingLevelHigh,
			extra:          map[string]any{"reasoning_format": "raw", "reasoning_effort": "low"},
			expectedFormat: "raw", expectedEffort: "low"},
		{name: "empty", model: "llama-4-maverick-17b", level: ""},
	}
	for index, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := groq.NewModel(test.model, groq.WithBaseURL(server.URL), groq.WithAPIKey("key"),
				groq.WithHTTPClient(server.Client()), groq.WithReasoningWarningHandler(func(warning groq.ReasoningWarning) {
					warnings = append(warnings, warning)
				}))
			settings := ai.ModelSettings{
				Thinking: &ai.ThinkingSettings{Level: test.level}, ExtraBody: test.extra,
			}
			if _, err := model.Request(context.Background(), nil, ai.ModelRequestParams{Settings: settings}); err != nil {
				t.Fatal(err)
			}
			body := bodies[index]
			if body["reasoning_format"] != test.expectedFormat || body["reasoning_effort"] != test.expectedEffort {
				t.Fatalf("unexpected reasoning request: %#v", body)
			}
			if settings.Thinking == nil || settings.Thinking.Level != test.level {
				t.Fatalf("caller settings were mutated: %#v", settings)
			}
		})
	}
	if len(warnings) != 1 || warnings[0].ModelName != "qwen/qwen3-32b" || warnings[0].Message == "" {
		t.Fatalf("unexpected reasoning warnings: %#v", warnings)
	}
	model := groq.NewModel("openai/gpt-oss-20b", groq.WithBaseURL(server.URL), groq.WithAPIKey("key"),
		groq.WithHTTPClient(server.Client()))
	stream, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelMinimal},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
	}
	body := bodies[len(bodies)-1]
	if body["reasoning_format"] != "parsed" || body["reasoning_effort"] != "low" {
		t.Fatalf("unexpected streamed reasoning request: %#v", body)
	}
}

func TestCompoundWebSearch(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		response.Header().Set("Content-Type", "application/json")
		if body["stream"] == true {
			_, _ = response.Write([]byte("data: {\"id\":\"completion\",\"model\":\"groq/compound\"," +
				"\"choices\":[{\"index\":0,\"delta\":{\"executed_tools\":[{\"index\":0,\"type\":\"search\"," +
				"\"arguments\":\"{\\\"query\\\":\\\"Go\\\"}\",\"search_results\":{\"results\":[]}}]}}]}\n\n" +
				"data: {\"id\":\"completion\",\"model\":\"groq/compound\"," +
				"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"found\",\"executed_tools\":[{" +
				"\"index\":0,\"type\":\"search\",\"arguments\":\"{\\\"query\\\":\\\"Go\\\"}\"," +
				"\"output\":\"stream result\"}]},\"finish_reason\":\"stop\"}]," +
				"\"x_groq\":{\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}}\n\n" +
				"data: [DONE]\n\n"))
			return
		}
		_, _ = response.Write([]byte(`{
			"id":"completion","model":"groq/compound",
			"choices":[{"finish_reason":"stop","message":{
				"content":"found","executed_tools":[{
					"index":0,"type":"search","arguments":"{\"query\":\"Go\"}",
					"output":"fallback","search_results":{"results":[{"title":"Go"}]}
				}]
			}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1}
		}`))
	}))
	defer server.Close()

	model := groq.NewModel("groq/compound", groq.WithBaseURL(server.URL), groq.WithAPIKey("key"),
		groq.WithHTTPClient(server.Client()))
	search := ai.WebSearchTool{
		AllowedDomains: []string{"allowed.example"}, BlockedDomains: []string{"blocked.example"},
	}
	response, err := model.Request(context.Background(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{search},
	})
	if err != nil || response.Text() != "found" {
		t.Fatalf("unexpected compound response: %#v %v", response, err)
	}
	if len(response.Parts) != 3 {
		t.Fatalf("unexpected executed search parts: %#v", response.Parts)
	}
	call, callOK := response.Parts[0].(ai.NativeToolCallPart)
	result, resultOK := response.Parts[1].(ai.NativeToolReturnPart)
	results, resultsOK := result.Content.(map[string]any)
	if !callOK || !resultOK || !resultsOK || call.ToolKind != ai.ToolPartKindWebSearch ||
		call.ToolCallID != result.ToolCallID || results["results"].([]any)[0].(map[string]any)["title"] != "Go" {
		t.Fatalf("unexpected executed search normalization: %#v", response.Parts)
	}
	stream, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&search},
	})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ModelStreamEvent
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		events = append(events, event)
	}
	if len(events) != 5 {
		t.Fatalf("unexpected executed search events: %#v", events)
	}
	start, startOK := events[0].(ai.ToolCallStartEvent)
	delta, deltaOK := events[1].(ai.ToolCallDeltaEvent)
	streamResult, resultOK := events[2].(ai.NativeToolReturnEvent)
	text, textOK := events[3].(ai.TextDeltaEvent)
	finish, finishOK := events[4].(ai.FinishEvent)
	if !startOK || !deltaOK || !resultOK || !textOK || !finishOK || !start.Native ||
		start.ToolKind != ai.ToolPartKindWebSearch || delta.ArgsDelta != `{"query":"Go"}` ||
		streamResult.Part.Content != "stream result" || text.Delta != "found" ||
		finish.Usage.InputTokens != 5 || finish.Usage.OutputTokens != 2 {
		t.Fatalf("unexpected executed search lifecycle: %#v", events)
	}
	if len(bodies) != 2 {
		t.Fatalf("unexpected request count: %d", len(bodies))
	}
	for _, body := range bodies {
		settings := body["search_settings"].(map[string]any)
		if settings["include_domains"].([]any)[0] != "allowed.example" ||
			settings["exclude_domains"].([]any)[0] != "blocked.example" {
			t.Fatalf("unexpected search settings: %#v", body)
		}
		if _, exists := body["tools"]; exists {
			t.Fatalf("implicit search emitted a tool declaration: %#v", body)
		}
	}
	if !model.SupportsNativeTool(search) || !model.SupportsNativeTool(&search) ||
		model.SupportsNativeTool(ai.CodeExecutionTool{}) {
		t.Fatal("unexpected compound native-tool support")
	}
	var nilSearch *ai.WebSearchTool
	if model.SupportsNativeTool(nilSearch) || groq.NewModel("model").SupportsNativeTool(search) {
		t.Fatal("unsupported search was accepted")
	}
}

func TestCompoundWebSearchValidation(t *testing.T) {
	compound := groq.NewModel("compound-beta")
	var nilSearch *ai.WebSearchTool
	tests := []struct {
		name   string
		model  *groq.Model
		params ai.ModelRequestParams
		match  string
	}{
		{name: "model", model: groq.NewModel("model"), params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
		}, match: "requires a compound model"},
		{name: "nil", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nil},
		}, match: "must not be nil"},
		{name: "typed nil", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nilSearch},
		}, match: "must not be nil"},
		{name: "tool", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}},
		}, match: "is not supported"},
		{name: "constraint", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{MaxUses: 1}},
		}, match: "only supports domain filters"},
		{name: "conflict", model: compound, params: ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
			Settings:    ai.ModelSettings{ExtraBody: map[string]any{"search_settings": map[string]any{}}},
		}, match: "conflicts"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.model.Request(context.Background(), nil, test.params)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("expected %q, got %v", test.match, err)
			}
			stream, streamErr := test.model.StreamRequest(context.Background(), nil, test.params)
			if streamErr == nil || stream != nil || !strings.Contains(streamErr.Error(), test.match) {
				t.Fatalf("expected streaming %q, got %v", test.match, streamErr)
			}
		})
	}
}

func TestProviderConfiguration(t *testing.T) {
	t.Setenv("GROQ_API_KEY", "environment-key")
	t.Setenv("GROQ_BASE_URL", "https://groq.example/v1")
	provider := groq.NewProviderConfig()
	if provider.Name != "groq" || provider.APIKey != "environment-key" || provider.BaseURL != "https://groq.example/v1" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
	t.Setenv("GROQ_BASE_URL", "")
	if provider := groq.NewProviderConfig(); provider.BaseURL != "https://api.groq.com/openai/v1" {
		t.Fatalf("unexpected default URL: %q", provider.BaseURL)
	}

	model := groq.NewModel("model", groq.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: "https://gateway.example/v1", APIKey: "key",
	}))
	if model.ProviderName() != "gateway" || model.ProviderURL() != "https://gateway.example/v1" {
		t.Fatalf("unexpected gateway identity: %q %q", model.ProviderName(), model.ProviderURL())
	}
}

func TestSettingsValidation(t *testing.T) {
	tests := []struct {
		name     string
		settings groq.Settings
		match    string
	}{
		{name: "format", settings: groq.Settings{ReasoningFormat: "invalid"}, match: "invalid reasoning format"},
		{name: "effort", settings: groq.Settings{ReasoningEffort: "invalid"}, match: "invalid reasoning effort"},
		{name: "format conflict", settings: groq.Settings{
			Common:          ai.ModelSettings{ExtraBody: map[string]any{"reasoning_format": "raw"}},
			ReasoningFormat: groq.ReasoningFormatParsed,
		}, match: "reasoning_format"},
		{name: "effort conflict", settings: groq.Settings{
			Common:          ai.ModelSettings{ExtraBody: map[string]any{"reasoning_effort": "low"}},
			ReasoningEffort: groq.ReasoningEffortHigh,
		}, match: "reasoning_effort"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	empty, err := (groq.Settings{}).Build()
	if err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v err=%v", empty, err)
	}
	for _, format := range []groq.ReasoningFormat{
		groq.ReasoningFormatHidden, groq.ReasoningFormatRaw, groq.ReasoningFormatParsed,
	} {
		if _, err := (groq.Settings{ReasoningFormat: format}).Build(); err != nil {
			t.Fatal(err)
		}
	}
	for _, effort := range []groq.ReasoningEffort{
		groq.ReasoningEffortNone, groq.ReasoningEffortDefault, groq.ReasoningEffortLow,
		groq.ReasoningEffortMedium, groq.ReasoningEffortHigh,
	} {
		if _, err := (groq.Settings{ReasoningEffort: effort}).Build(); err != nil {
			t.Fatal(err)
		}
	}
}
