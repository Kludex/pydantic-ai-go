package openaidecisions_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openaidecisions"
)

func TestQuestionMapping(t *testing.T) {
	severity := make([]string, 11)
	for level := range severity {
		severity[level] = fmt.Sprintf(`{"const":%d,"description":"%d of 10"}`, level, level)
	}
	params := output(`{"type":"object","properties":{
		"refund":{"type":"boolean","description":"Overdue > 1 day & unpaid?","anyOf":[
			{"const":true,"description":"They ask for their money back."},{"const":false,"description":"They do not."}]},
		"frustration":{"type":"integer","description":"How frustrated?","anyOf":[
			{"const":0,"description":"Calm"},{"const":1,"description":"Frustrated"},{"const":2,"description":"Very angry"}]},
		"team":{"type":"string","enum":["bug","billing"],"description":"Which team?"},
		"severity":{"type":"integer","description":"How severe?","anyOf":[` + strings.Join(severity, ",") + `]},
		"customer":{"type":"object","description":"Who is writing in.",
			"properties":{"vip":{"type":"boolean","description":"Enterprise plan?"}},"required":["vip"]}
	},"required":["refund","frustration","team","severity","customer"]}`)
	severityProbabilities := make([]string, 11)
	for level := range severityProbabilities {
		severityProbabilities[level] = fmt.Sprintf(`{"value":"%d","probability":0}`, level)
	}
	severityProbabilities[10] = `{"value":"10","probability":1}`
	fake := newFake(t, `{"model":"gpt-6-luna","answers":[
		{"type":"predicate","name":"customer.vip","probability":0.8},
		{"type":"score","name":"frustration","score":1.68,"confidence":0.52,"probabilities":[
			{"value":0,"label":"0","probability":0},{"value":1,"label":"1","probability":0.32},
			{"value":2,"label":"2","probability":0.68}]},
		{"type":"predicate","name":"refund","probability":0.1},
		{"type":"choice","name":"severity","choice":"10","confidence":1,"probabilities":[`+
		strings.Join(severityProbabilities, ",")+`]},
		{"type":"choice","name":"team","choice":"billing","confidence":0.9,"probabilities":[
			{"value":"billing","probability":0.9},{"value":"bug","probability":0.1}]}
	],"usage":{"input_tokens":347}}`)
	response, err := fake.model().Request(t.Context(), prompt("Third time asking. Fix it NOW."), params)
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, json.RawMessage(toolCall(t, response).Args),
		`{"customer":{"vip":true},"frustration":2,"refund":false,"severity":10,"team":"billing"}`)
	if scores := response.ProviderDetails["scores"].(map[string]float64); scores["frustration"] != 1.68 {
		t.Fatalf("details=%v", response.ProviderDetails)
	}
	questions := fake.only().body["questions"].([]any)
	assertJSON(t, questions[:3], `[
		{"type":"predicate","name":"customer.vip",
			"instructions":"{\"context\":[\"customer: Who is writing in.\"],\"field\":\"customer.vip\",\"question\":\"Enterprise plan?\"}"},
		{"type":"score","name":"frustration",
			"instructions":"{\"field\":\"frustration\",\"question\":\"How frustrated?\"}",
			"levels":[{"label":"0","description":"Calm"},{"label":"1","description":"Frustrated"},{"label":"2","description":"Very angry"}]},
		{"type":"predicate","name":"refund",
			"instructions":"{\"field\":\"refund\",\"no\":\"They do not.\",\"question\":\"Overdue > 1 day & unpaid?\",\"yes\":\"They ask for their money back.\"}"}
	]`)
	severityQuestion := questions[3].(map[string]any)
	choices := severityQuestion["choices"].([]any)
	if severityQuestion["type"] != "choice" || len(choices) != 11 {
		t.Fatalf("severity=%v", severityQuestion)
	}
	assertJSON(t, choices[2], `{"value":"10","description":"10 of 10"}`)
	assertJSON(t, questions[4], `{"type":"choice","name":"team",
		"instructions":"{\"field\":\"team\",\"question\":\"Which team?\"}",
		"choices":[{"value":"billing"},{"value":"bug"}]}`)
}

func TestScalarQuestionInstructions(t *testing.T) {
	meanings := `"anyOf":[{"const":true,"description":"Reply now."},{"const":false,"description":"Reply later."}]`
	for _, test := range []struct {
		name, schema, answer, question string
	}{
		{
			name:     "predicate",
			schema:   `{"type":"boolean","description":"Is this urgent?"}`,
			answer:   `{"type":"predicate","name":"response","probability":0.9}`,
			question: `{"type":"predicate","name":"response","instructions":"Is this urgent?"}`,
		},
		{
			name:   "predicate with meanings",
			schema: `{"type":"boolean","description":"Is this urgent?",` + meanings + `}`,
			answer: `{"type":"predicate","name":"response","probability":0.9}`,
			question: `{"type":"predicate","name":"response",
				"instructions":"{\"no\":\"Reply later.\",\"question\":\"Is this urgent?\",\"yes\":\"Reply now.\"}"}`,
		},
		{
			name:     "meanings without instructions",
			schema:   `{"type":"boolean","anyOf":[{"const":true,"description":"Reply now."},{"const":false}]}`,
			answer:   `{"type":"predicate","name":"response","probability":0.9}`,
			question: `{"type":"predicate","name":"response","instructions":"{\"yes\":\"Reply now.\"}"}`,
		},
		{
			name:   "choice without instructions",
			schema: `{"type":"string","enum":["a","b"]}`,
			answer: `{"type":"choice","name":"response","choice":"a","confidence":1,
				"probabilities":[{"value":"a","probability":1},{"value":"b","probability":0}]}`,
			question: `{"type":"choice","name":"response","instructions":"","choices":[{"value":"a"},{"value":"b"}]}`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			fake := newFake(t, `{"model":"gpt-6-luna","answers":[`+test.answer+`],"usage":{}}`)
			if _, err := fake.model().Request(t.Context(), prompt("Down?"), output(test.schema)); err != nil {
				t.Fatal(err)
			}
			assertJSON(t, fake.only().body["questions"], "["+test.question+"]")
		})
	}
}

func routeParams(tools ...ai.ToolDefinition) ai.ModelRequestParams {
	params := output(`{"type":"object","title":"Ticket","properties":{
		"urgent":{"type":"boolean","description":"Is this urgent?"}},"required":["urgent"]}`)
	params.Tools = append([]ai.ToolDefinition{{
		Name: "escalate", Description: "Hand the ticket to a human.", Schema: map[string]any{"type": "object"},
	}}, tools...)
	return params
}

func routeReply(choice string, ticket float64) string {
	return fmt.Sprintf(`{"model":"gpt-6-luna","answers":[
		{"type":"predicate","name":"Ticket.urgent","probability":0.9},
		{"type":"choice","name":"route","choice":%q,"confidence":0.9,"probabilities":[
			{"value":"Ticket","probability":%v},{"value":"escalate","probability":%v}]}
	],"usage":{"input_tokens":10}}`, choice, ticket, 1-ticket)
}

func TestRoute(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "I was charged twice."}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "We are looking into it."}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "Still no refund!"}}},
	}
	for _, test := range []struct {
		choice, tool, args string
		probability        float64
	}{
		{choice: "Ticket", tool: "final_result", args: `{"urgent":true}`, probability: .89},
		{choice: "escalate", tool: "escalate", args: `{}`, probability: .2},
	} {
		t.Run(test.choice, func(t *testing.T) {
			fake := newFake(t, routeReply(test.choice, test.probability))
			response, err := fake.model().Request(t.Context(), history, routeParams())
			if err != nil {
				t.Fatal(err)
			}
			call := toolCall(t, response)
			if call.ToolName != test.tool || string(call.Args) != test.args {
				t.Fatalf("call=%+v", call)
			}
			route := response.ProviderDetails["route"].(map[string]any)
			if route["choice"] != test.choice {
				t.Fatalf("route=%v", route)
			}
			body := fake.only().body
			var state map[string]any
			if err := json.Unmarshal([]byte(body["input"].(string)), &state); err != nil || state["text"] != "Still no refund!" {
				t.Fatalf("input=%v error=%v", body["input"], err)
			}
			questions := body["questions"].([]any)
			routeQuestion := questions[1].(map[string]any)
			if len(questions) != 2 || questions[0].(map[string]any)["name"] != "Ticket.urgent" ||
				routeQuestion["name"] != "route" || routeQuestion["type"] != "choice" {
				t.Fatalf("questions=%v", questions)
			}
			assertJSON(t, routeQuestion["choices"].([]any)[1], `{"value":"escalate","description":"Hand the ticket to a human."}`)
		})
	}
}

func TestRouteHandOff(t *testing.T) {
	settings, err := openaidecisions.Settings{RouteThreshold: floatPointer(.95)}.Build()
	if err != nil {
		t.Fatal(err)
	}
	params := routeParams()
	params.Settings = settings
	fake := newFake(t, routeReply("Ticket", .89))
	_, err = fake.model().Request(t.Context(), prompt("Down?"), params)
	var unsure *openaidecisions.UnsureRoute
	if !errors.As(err, &unsure) || unsure.ModelName != "gpt-6-luna" || unsure.Threshold != .95 ||
		unsure.Probabilities["Ticket"] != .89 || !strings.Contains(err.Error(), `"Ticket" with probability 0.89`) {
		t.Fatalf("error=%#v", err)
	}
	assertHandOff(t, err, "Ticket", .89)

	note := ai.ToolDefinition{Name: "note", Description: "Write a note.", Schema: map[string]any{
		"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}},
	}}
	fake = newFake(t, `{"model":"gpt-6-luna","answers":[
		{"type":"predicate","name":"Ticket.urgent","probability":0.9},
		{"type":"choice","name":"route","choice":"note","confidence":0.7,"probabilities":[
			{"value":"Ticket","probability":0.2},{"value":"escalate","probability":0.1},{"value":"note","probability":0.7}]}
	],"usage":{}}`)
	_, err = fake.model().Request(t.Context(), prompt("Down?"), routeParams(note))
	var unfillable *openaidecisions.UnfillableRoute
	if !errors.As(err, &unfillable) || unfillable.ToolName != "note" || unfillable.ModelName != "gpt-6-luna" ||
		!strings.Contains(err.Error(), `"note" (probability 0.70) but cannot fill it`) {
		t.Fatalf("error=%#v", err)
	}
	assertHandOff(t, err, "note", .7)
}

func assertHandOff(t *testing.T, err error, route string, probability float64) {
	t.Helper()
	var handOff openaidecisions.DecisionHandOff
	if !errors.As(err, &handOff) || !handOff.IsModelAPIError() {
		t.Fatalf("error=%#v", err)
	}
	if gotRoute, gotProbability := handOff.DecisionRoute(); gotRoute != route || gotProbability != probability {
		t.Fatalf("route=%q probability=%v", gotRoute, gotProbability)
	}
}

func tagTool(count int) ai.ToolDefinition {
	labels := make([]any, count)
	for index := range labels {
		labels[index] = fmt.Sprintf("t%03d", index)
	}
	return ai.ToolDefinition{Name: "tag", Description: "Tag the ticket.", Schema: map[string]any{
		"type": "object", "properties": map[string]any{"tags": map[string]any{
			"type": "array", "description": "Which tags apply?", "items": map[string]any{"enum": labels},
		}},
	}}
}

func tagAnswers(count int, refuse bool) string {
	answers := make([]string, count)
	for index := range answers {
		answers[index] = fmt.Sprintf(`{"type":"predicate","name":"tags.t%03d","probability":0.1}`, index)
	}
	answers[1] = `{"type":"predicate","name":"tags.t001","probability":0.9}`
	if refuse {
		answers[2] = `{"type":"refusal","name":"tags.t002"}`
	}
	return `{"model":"gpt-6-luna-fill","answers":[` + strings.Join(answers, ",") +
		`],"usage":{"input_tokens":20,"output_tokens":1}}`
}

func TestRouteThenFill(t *testing.T) {
	routeOnly := `{"model":"gpt-6-luna","answers":[
		{"type":"predicate","name":"Ticket.urgent","probability":0.9},
		{"type":"choice","name":"route","choice":"tag","confidence":0.8,"probabilities":[
			{"value":"Ticket","probability":0.1},{"value":"escalate","probability":0.1},{"value":"tag","probability":0.8}]}
	],"usage":{"input_tokens":10}}`
	fake := newFake(t, routeOnly, tagAnswers(200, false))
	response, err := fake.model().Request(t.Context(), prompt(strings.Repeat("context ", 4000)), routeParams(tagTool(200)))
	if err != nil {
		t.Fatal(err)
	}
	if len(fake.exchanges) != 2 || len(fake.exchanges[0].body["questions"].([]any)) != 2 ||
		len(fake.exchanges[1].body["questions"].([]any)) != 200 {
		t.Fatalf("requests=%d", len(fake.exchanges))
	}
	call := toolCall(t, response)
	if call.ToolName != "tag" || string(call.Args) != `{"tags":["t001"]}` {
		t.Fatalf("call=%+v", call)
	}
	if response.Usage.Requests != 2 || response.Usage.InputTokens != 30 || response.Usage.OutputTokens != 1 ||
		response.ProviderResponseID != "req_2" || response.ModelName != "gpt-6-luna-fill" ||
		response.ProviderDetails["requests"] != 2 {
		t.Fatalf("response=%+v", response)
	}

	fake = newFake(t, routeOnly, tagAnswers(200, true))
	_, err = fake.model().Request(t.Context(), prompt("Down?"), routeParams(tagTool(200)))
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) || !strings.Contains(err.Error(), `selected "tag" but failed to fill`) {
		t.Fatalf("error=%v", err)
	}

	fake = newFake(t, routeOnly)
	_, err = fake.model().Request(t.Context(), prompt("Down?"), routeParams(tagTool(201)))
	var unfillable *openaidecisions.UnfillableRoute
	if !errors.As(err, &unfillable) || unfillable.ToolName != "tag" || len(fake.exchanges) != 1 {
		t.Fatalf("error=%v requests=%d", err, len(fake.exchanges))
	}
}

func TestQuestionCapDropsSpeculation(t *testing.T) {
	fake := newFake(t, `{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"Ticket.urgent","probability":0.9},
		{"type":"choice","name":"route","choice":"tag","confidence":0.9,"probabilities":[
		{"value":"Ticket","probability":0.05},{"value":"escalate","probability":0.05},{"value":"tag","probability":0.9}]}],
		"usage":{}}`, tagAnswers(199, false))
	if _, err := fake.model().Request(t.Context(), prompt(strings.Repeat("context ", 8000)), routeParams(tagTool(199))); err != nil {
		t.Fatal(err)
	}
	if asked := fake.exchanges[0].body["questions"].([]any); len(asked) != 2 {
		t.Fatalf("speculative questions=%d, want the output type's beside the route", len(asked))
	}

	params := routeParams()
	params.OutputTool = &ai.ToolDefinition{Name: "final_result", Schema: tagTool(200).Schema}
	fake = newFake(t, `{"model":"gpt-6-luna","answers":[{"type":"choice","name":"route","choice":"output",
		"confidence":0.9,"probabilities":[{"value":"escalate","probability":0.1},{"value":"output","probability":0.9}]}],
		"usage":{}}`, tagAnswers(200, false))
	response, err := fake.model().Request(t.Context(), prompt("Down?"), params)
	if err != nil {
		t.Fatal(err)
	}
	if asked := fake.exchanges[0].body["questions"].([]any); len(asked) != 1 || toolCall(t, response).ToolName != "final_result" {
		t.Fatalf("speculative questions=%d", len(asked))
	}
}

func TestLimitsBeforeTransport(t *testing.T) {
	image := ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}
	images := func(count int) []ai.ModelMessage {
		contents := []ai.UserContent{ai.TextContent{Text: "Which is broken?"}}
		for range count {
			contents = append(contents, image)
		}
		return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: contents}}}}
	}
	tools := make([]ai.ToolDefinition, 255)
	for index := range tools {
		tools[index] = ai.ToolDefinition{Name: fmt.Sprintf("tool_%d", index), Schema: map[string]any{"type": "object"}}
	}
	tooManyRoutes := urgentParams()
	tooManyRoutes.Tools = tools
	tagOutput := ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: tagTool(201).Schema}}
	for _, test := range []struct {
		name     string
		messages []ai.ModelMessage
		params   ai.ModelRequestParams
		match    string
		apiError bool
	}{
		{"questions", prompt("Down?"), tagOutput, `accepts at most 200 questions; got 201`, true},
		{"images", images(129), urgentParams(), `accepts at most 128 images; got 129`, true},
		{"routes", prompt("Down?"), tooManyRoutes, `too many options (maximum 255)`, false},
		{"text output", prompt("Down?"), ai.ModelRequestParams{AllowText: true}, `text output is not supported`, false},
		{"non-image file", []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
			Contents: []ai.UserContent{ai.BinaryContent{Data: []byte("%PDF"), MediaType: "application/pdf"}},
		}}}}, urgentParams(), `only image files are supported`, false},
		{"video", []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
			Contents: []ai.UserContent{ai.VideoURL{URL: "https://example.com/video.mp4"}},
		}}}}, urgentParams(), `only text and image input is supported`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			fake := newFake(t)
			_, err := fake.model().Request(t.Context(), test.messages, test.params)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("error=%v", err)
			}
			var apiError interface{ IsModelAPIError() bool }
			if errors.As(err, &apiError) != test.apiError || test.apiError && !apiError.IsModelAPIError() {
				t.Fatalf("model API error=%v for %#v", !test.apiError, err)
			}
		})
	}

	fake := newFake(t, urgentReply(.9))
	if _, err := fake.model().Request(t.Context(), images(128), urgentParams()); err != nil {
		t.Fatal(err)
	}
	if content := fake.only().body["input"].([]any)[0].(map[string]any)["content"].([]any); len(content) != 1+2*128 {
		t.Fatalf("content parts=%d", len(content))
	}
}

func TestImages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/notes.txt" {
			writer.Header().Set("Content-Type", "text/plain")
			_, _ = writer.Write([]byte("not an image"))
			return
		}
		writer.Header().Set("Content-Type", "image/gif")
		_, _ = writer.Write([]byte("gif"))
	}))
	defer server.Close()
	metadata := map[string]any{"detail": "low"}
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "Here is the dashboard."},
			ai.BinaryContent{Data: []byte("png"), MediaType: "image/png", VendorMetadata: map[string]any{"detail": "high"}},
		}}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{Data: []byte("jpeg"), MediaType: "IMAGE/JPEG"}}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "Is it broken?"},
			ai.ImageURL{URL: server.URL + "/cat", MediaType: "image/gif", ForceDownload: ai.FileDownloadAllowLocal, VendorMetadata: metadata},
		}}}},
	}
	fake := newFake(t, urgentReply(.9))
	if _, err := fake.model().Request(t.Context(), messages, urgentParams()); err != nil {
		t.Fatal(err)
	}
	metadata["detail"] = "mutated"
	input := fake.only().body["input"].([]any)
	content := input[0].(map[string]any)["content"].([]any)
	state := content[0].(map[string]any)["text"].(string)
	if !strings.Contains(state, `"assistant":"<image 2>"`) || !strings.Contains(state, `"text":"Is it broken?\n\n<image 3>"`) {
		t.Fatalf("state=%s", state)
	}
	data := func(mediaType, body string) string {
		return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString([]byte(body))
	}
	assertJSON(t, input, `[{"role":"user","content":[
		{"type":"input_text","text":`+string(must(json.Marshal(state)))+`},
		{"type":"input_text","text":"<image 1>:"},
		{"type":"input_image","image_url":"`+data("image/png", "png")+`","detail":"high"},
		{"type":"input_text","text":"<image 2>:"},
		{"type":"input_image","image_url":"`+data("IMAGE/JPEG", "jpeg")+`"},
		{"type":"input_text","text":"<image 3>:"},
		{"type":"input_image","image_url":"`+data("image/gif", "gif")+`","detail":"low"}
	]}]`)

	for name, image := range map[string]ai.ImageURL{
		"invalid mode": {URL: server.URL + "/cat", ForceDownload: "invalid"},
		"download":     {URL: "://bad", ForceDownload: ai.FileDownloadSafe},
	} {
		t.Run(name, func(t *testing.T) {
			fake := newFake(t)
			_, err := fake.model().Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{image}},
			}}}, urgentParams())
			if err == nil || len(fake.exchanges) != 0 {
				t.Fatalf("image validation = %v requests=%d", err, len(fake.exchanges))
			}
		})
	}

	fake = newFake(t)
	notImage := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.ImageURL{URL: server.URL + "/notes.txt", ForceDownload: ai.FileDownloadAllowLocal},
	}}}}}
	if _, err := fake.model().Request(t.Context(), notImage, urgentParams()); err == nil ||
		!strings.Contains(err.Error(), "non-image media type") || len(fake.exchanges) != 0 {
		t.Fatalf("error=%v requests=%d", err, len(fake.exchanges))
	}
}

func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}
