package typesafe_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/typesafe"
)

type typesafeRoundTripFunc func(*http.Request) (*http.Response, error)

func (function typesafeRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type typesafeFailingBody struct{}

func (typesafeFailingBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (typesafeFailingBody) Close() error             { return nil }

func requestTypeSafe(
	t *testing.T, schema map[string]any, responseBody string,
) (map[string]any, map[string]any, error) {
	return requestTypeSafeWithSettings(t, schema, responseBody, ai.ModelSettings{})
}

func requestTypeSafeWithSettings(
	t *testing.T, schema map[string]any, responseBody string, settings ai.ModelSettings,
) (map[string]any, map[string]any, error) {
	t.Helper()
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(responseBody))
	}))
	defer server.Close()
	response, err := typesafe.NewModel(
		"jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()),
	).Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "judge"},
	}}}, ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: schema}, Settings: settings})
	if err != nil {
		return requestBody, nil, err
	}
	var arguments map[string]any
	if err := json.Unmarshal(response.Parts[0].(ai.ToolCallPart).Args, &arguments); err != nil {
		t.Fatal(err)
	}
	return requestBody, arguments, nil
}

func TestTypeSafeResolvedAndFannedSchemas(t *testing.T) {
	resolved := map[string]any{
		"type": "object",
		"$defs": map[string]any{
			"outer": map[string]any{"type": "object", "properties": map[string]any{
				"inner": map[string]any{"$ref": "#/$defs/inner"},
			}},
			"inner": map[string]any{"type": "object", "properties": map[string]any{
				"flag": map[string]any{"type": "boolean", "description": "Flag?"},
			}},
		},
		"properties": map[string]any{"outer": map[string]any{"$ref": "#/$defs/outer"}},
	}
	_, arguments, err := requestTypeSafe(t, resolved,
		`{"answers":{"outer.inner.flag":{"type":"noul","noul":0.9}},"model":"jev","usage":{}}`)
	if err != nil || !reflect.DeepEqual(arguments, map[string]any{"outer": map[string]any{"inner": map[string]any{"flag": true}}}) {
		t.Fatalf("unexpected resolved result: %#v err=%v", arguments, err)
	}

	request, arguments, err := requestTypeSafe(t, objectSchema("labels", map[string]any{
		"type": "array", "items": map[string]any{"anyOf": []any{
			map[string]any{"const": "a", "description": "First"}, map[string]any{"const": "b"},
		}},
	}), `{"answers":{"labels.a":{"type":"noul","noul":0.9},"labels.b":{"type":"noul","noul":0.1}},"model":"jev","usage":{}}`)
	if err != nil || !reflect.DeepEqual(arguments["labels"], []any{"a"}) {
		t.Fatalf("unexpected fanned result: %#v err=%v", arguments, err)
	}
	option := request["questions"].(map[string]any)["labels.a"].(map[string]any)["instructions"].(map[string]any)["option"]
	if option != "a: First" {
		t.Fatalf("unexpected described option: %#v", option)
	}
}

func TestTypeSafeNewSchemaEdges(t *testing.T) {
	for _, test := range []struct {
		name   string
		schema map[string]any
	}{
		{name: "invalid property", schema: objectSchema("bad", "not-a-schema")},
		{name: "malformed anyOf member", schema: objectSchema("bad", map[string]any{"anyOf": []any{"bad", map[string]any{"type": "null"}}})},
		{name: "missing const", schema: objectSchema("bad", map[string]any{"anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "string"}}})},
		{name: "optional boolean", schema: objectSchema("bad", map[string]any{"anyOf": []any{map[string]any{"type": "boolean"}, map[string]any{"type": "null"}}})},
		{name: "two nulls", schema: objectSchema("bad", map[string]any{"anyOf": []any{map[string]any{"type": "null"}, map[string]any{"type": "null"}}})},
		{name: "integer list", schema: objectSchema("bad", map[string]any{"type": "array", "items": map[string]any{"enum": []any{1, 2}}})},
		{name: "one option", schema: objectSchema("bad", map[string]any{"enum": []any{"one"}})},
		{name: "mixed boolean choice", schema: objectSchema("bad", map[string]any{"anyOf": []any{map[string]any{"const": true, "description": "yes"}, map[string]any{"const": "no"}}})},
		{name: "optional rubric", schema: objectSchema("bad", map[string]any{"anyOf": []any{
			map[string]any{"anyOf": []any{map[string]any{"const": 0, "description": "low"}, map[string]any{"const": 1, "description": "high"}}},
			map[string]any{"type": "null"},
		}})},
		{name: "bare boolean", schema: map[string]any{"type": "boolean"}},
		{name: "invalid number", schema: objectSchema("bad", map[string]any{"type": "number", "minimum": 1, "maximum": 2, "description": "number"})},
		{name: "unsigned overflow", schema: objectSchema("bad", map[string]any{"enum": []any{uint64(math.MaxUint64), 1}, "description": "value"})},
		{name: "fractional option", schema: objectSchema("bad", map[string]any{"enum": []any{1.5, 2}, "description": "value"})},
		{name: "invalid anyOf", schema: objectSchema("bad", map[string]any{"anyOf": []any{map[string]any{"const": "a"}, "bad"}, "description": "value"})},
		{name: "anyOf without const", schema: objectSchema("bad", map[string]any{"anyOf": []any{map[string]any{"const": "a"}, map[string]any{"type": "string"}}, "description": "value"})},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := requestTypeSafe(t, test.schema, "")
			if err == nil {
				t.Fatal("unsupported schema was accepted")
			}
		})
	}
}

func TestTypeSafeOptionalNameAndNestedDefaults(t *testing.T) {
	optional := map[string]any{
		"description": "Which team?", "default": "shipping", "anyOf": []any{
			map[string]any{"enum": []any{"none", "none_", "billing"}},
			map[string]any{"type": "null"},
		},
	}
	request, arguments, err := requestTypeSafe(t, objectSchema("team", optional),
		`{"answers":{"team":{"type":"choice","choice":"none__","confidence":1}},"model":"jev","usage":{}}`)
	if err != nil {
		t.Fatal(err)
	}
	criteria := request["questions"].(map[string]any)["team"].(map[string]any)["criteria"].(map[string]any)
	if criteria["none__"] != "None of these." || len(arguments) != 0 {
		t.Fatalf("unexpected optional result: criteria=%#v args=%#v", criteria, arguments)
	}

	nested := map[string]any{
		"type": "object", "default": map[string]any{"team": "shipping"},
		"properties": map[string]any{
			"team":   optional,
			"urgent": map[string]any{"type": "boolean", "description": "Urgent?"},
		},
	}
	_, arguments, err = requestTypeSafe(t, objectSchema("nested", nested),
		`{"answers":{"nested.team":{"type":"choice","choice":"none__","confidence":1},"nested.urgent":{"type":"noul","noul":0.9}},"model":"jev","usage":{}}`)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(arguments, map[string]any{"nested": map[string]any{"urgent": true}}) {
		t.Fatalf("answered nested object was removed: %#v", arguments)
	}
}

func TestTypeSafeIntegerRepresentationsAndLargeChoice(t *testing.T) {
	values := []any{int(11), int8(12), int16(13), int32(14), int64(15), uint(16), uint8(17), uint16(18), uint32(19), uint64(20), float32(21), float64(22), json.Number("23")}
	_, arguments, err := requestTypeSafe(t, objectSchema("value", map[string]any{
		"enum": values, "description": "Which value?",
	}), `{"answers":{"value":{"type":"choice","choice":"23","confidence":1}},"model":"jev","usage":{}}`)
	if err != nil {
		t.Fatal(err)
	}
	if arguments["value"] != float64(23) {
		t.Fatalf("unexpected integer result: %#v", arguments)
	}

	options := make([]any, 11)
	for index := range options {
		options[index] = map[string]any{"const": index, "description": "level"}
	}
	request, _, err := requestTypeSafe(t, objectSchema("value", map[string]any{
		"anyOf": options, "description": "Which value?",
	}), `{"answers":{"value":{"type":"choice","choice":"10","confidence":1}},"model":"jev","usage":{}}`)
	if err != nil {
		t.Fatal(err)
	}
	if request["questions"].(map[string]any)["value"].(map[string]any)["type"] != "choice" {
		t.Fatalf("large integer set was not a choice: %#v", request)
	}

	_, _, err = requestTypeSafe(t, objectSchema("value", map[string]any{
		"anyOf":       []any{map[string]any{"const": 0, "description": "zero"}, map[string]any{"const": 0, "description": "again"}},
		"description": "Which value?",
	}), `{"answers":{"value":{"type":"choice","choice":"0","confidence":1}},"model":"jev","usage":{}}`)
	if err != nil {
		t.Fatal(err)
	}
}

func TestTypeSafeUnknownChoiceAndThresholdEdges(t *testing.T) {
	_, arguments, err := requestTypeSafe(t, objectSchema("value", map[string]any{
		"enum": []any{"a", "b"}, "description": "Value?",
	}), `{"answers":{"value":{"type":"choice","choice":"other","confidence":1}},"model":"jev","usage":{}}`)
	if err != nil || arguments["value"] != "other" {
		t.Fatalf("unexpected unknown choice: %#v err=%v", arguments, err)
	}
	for _, test := range []struct {
		name        string
		probability float64
		threshold   float64
		want        bool
	}{
		{name: "certainty threshold", probability: 1, threshold: 1, want: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			settings, err := (typesafe.Settings{BooleanThreshold: &test.threshold}).Build()
			if err != nil {
				t.Fatal(err)
			}
			_, arguments, err := requestTypeSafeWithSettings(t,
				objectSchema("value", map[string]any{"type": "boolean", "description": "Value?"}),
				fmt.Sprintf(`{"answers":{"value":{"type":"noul","noul":%g}},"model":"jev","usage":{}}`, test.probability),
				settings,
			)
			if err != nil || arguments["value"] != test.want {
				t.Fatalf("unexpected threshold result: %#v err=%v", arguments, err)
			}
		})
	}
}

func TestTypeSafeMalformedAnswers(t *testing.T) {
	tests := []struct {
		name, response string
		schema         map[string]any
	}{
		{name: "missing boolean", schema: objectSchema("value", map[string]any{"type": "boolean", "description": "Value?"}), response: `{"answers":{"value":{"type":"noul"}},"model":"jev","usage":{}}`},
		{name: "missing choice", schema: objectSchema("value", map[string]any{"enum": []any{"a", "b"}, "description": "Value?"}), response: `{"answers":{"value":{"type":"choice"}},"model":"jev","usage":{}}`},
		{name: "missing score", schema: objectSchema("value", map[string]any{"anyOf": []any{map[string]any{"const": 0, "description": "low"}, map[string]any{"const": 1, "description": "high"}}}), response: `{"answers":{"value":{"type":"score"}},"model":"jev","usage":{}}`},
		{name: "missing fanned answer", schema: objectSchema("value", map[string]any{"type": "array", "items": map[string]any{"enum": []any{"a", "b"}}}), response: `{"answers":{},"model":"jev","usage":{}}`},
		{name: "missing mapped answer", schema: objectSchema("value", map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "boolean"}, "propertyNames": map[string]any{"enum": []any{"a", "b"}}}), response: `{"answers":{},"model":"jev","usage":{}}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := requestTypeSafe(t, test.schema, test.response)
			if err == nil {
				t.Fatal("malformed answer was accepted")
			}
		})
	}
	_, arguments, err := requestTypeSafe(t,
		objectSchema("value", map[string]any{"type": "boolean", "description": "Value?"}),
		`{"answers":{"value":{"type":"noul","noul":2}},"model":"jev","usage":{}}`,
	)
	if err != nil || arguments["value"] != false {
		t.Fatalf("unexpected out-of-range probability result: %#v err=%v", arguments, err)
	}
}

func TestTypeSafeRoutingEdges(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "judge"}}}}
	boolSchema := objectSchema("tool", map[string]any{"type": "boolean", "description": "Done?"})
	tool := ai.ToolDefinition{Name: "act", Description: "Act.", Schema: map[string]any{"type": "object"}}

	for _, responseBody := range []string{
		`{"answers":{"tool_":{"type":"choice","choice":"missing","probabilities":{"missing":1}}},"model":"jev","usage":{}}`,
		`{"answers":{"tool_":{"type":"choice","choice":"act","probabilities":{"act":2}}},"model":"jev","usage":{}}`,
	} {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
			_, _ = response.Write([]byte(responseBody))
		}))
		_, err := typesafe.NewModel(
			"jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()),
		).Request(t.Context(), messages, ai.ModelRequestParams{
			Instructions: "decide", OutputTool: &ai.ToolDefinition{Name: "out", Schema: boolSchema},
			Tools: []ai.ToolDefinition{tool},
		})
		server.Close()
		if err == nil {
			t.Fatal("invalid route answer was accepted")
		}
	}

	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"answers":{"tool":{"type":"choice","choice":"act","confidence":1,"probabilities":{"out":0,"act":1}}},"model":"jev","usage":{}}`))
	}))
	model := typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	toolThreshold := 0.5
	routeSettings, err := (typesafe.Settings{ToolCallThreshold: &toolThreshold}).Build()
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "out", Description: "The final result of the run.", Schema: map[string]any{"type": "object"}},
		Tools:      []ai.ToolDefinition{tool}, Settings: routeSettings,
	})
	server.Close()
	if err != nil || string(response.Parts[0].(ai.ToolCallPart).Args) != `{}` ||
		response.ProviderDetails["tool"] == nil {
		t.Fatalf("unexpected empty tool route: %#v err=%v", response, err)
	}

	server = httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"answers":{"value":{"type":"noul","noul":0.9},"tool":{"type":"choice","choice":"act","confidence":1,"probabilities":{"out":0,"act":1}}},"model":"jev","usage":{}}`))
	}))
	model = typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	_, err = model.Request(t.Context(), messages, ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "out", Schema: objectSchema("value", map[string]any{"type": "boolean", "description": "Value?"})},
		Tools:      []ai.ToolDefinition{{Name: "act", Schema: objectSchema("bad", "invalid")}},
	})
	server.Close()
	var proposed *typesafe.ToolCallProposed
	if !errors.As(err, &proposed) || proposed.ToolName != "act" {
		t.Fatalf("unexpected proposed tool error: %v", err)
	}

	calls := 0
	server = httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			_, _ = response.Write([]byte(`{"answers":{"value":{"type":"noul","noul":0.9},"tool":{"type":"choice","choice":"act","confidence":1,"probabilities":{"out":0,"act":1}}},"model":"jev","usage":{}}`))
			return
		}
		response.WriteHeader(http.StatusBadGateway)
	}))
	model = typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	_, err = model.Request(t.Context(), messages, ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "out", Schema: objectSchema("value", map[string]any{"type": "boolean", "description": "Value?"})},
		Tools:      []ai.ToolDefinition{{Name: "act", Schema: objectSchema("argument", map[string]any{"type": "boolean", "description": "Argument?"})}},
	})
	server.Close()
	if err == nil || !strings.Contains(err.Error(), "failed to fill arguments") {
		t.Fatalf("unexpected second request error: %v", err)
	}
}

func TestTypeSafeRequestFailures(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "judge"}}}}
	valid := ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: objectSchema("value", map[string]any{
		"type": "boolean", "description": "Value?",
	})}}

	model := typesafe.NewModel("jev")
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "no typed question") {
		t.Fatalf("unexpected empty request error: %v", err)
	}

	threshold := 0.5
	settings, err := (typesafe.Settings{BooleanThreshold: &threshold}).Build()
	if err != nil {
		t.Fatal(err)
	}
	threshold = 2
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{OutputTool: valid.OutputTool, Settings: settings}); err == nil ||
		!strings.Contains(err.Error(), "thresholds") {
		t.Fatalf("unexpected mutated threshold error: %v", err)
	}

	prepareError := errors.New("prepare failed")
	model = typesafe.NewModel("jev", typesafe.WithProvider(typesafe.ProviderConfig{
		Name: "typesafe", BaseURL: "https://example.com", PrepareRequest: func(*http.Request) error { return prepareError },
	}))
	if _, err := model.Request(t.Context(), messages, valid); !errors.Is(err, prepareError) {
		t.Fatalf("unexpected preparation error: %v", err)
	}

	model = typesafe.NewModel("jev", typesafe.WithBaseURL(":"))
	if _, err := model.Request(t.Context(), messages, valid); err == nil {
		t.Fatal("invalid URL was accepted")
	}

	transportError := errors.New("offline")
	model = typesafe.NewModel("jev", typesafe.WithHTTPClient(&http.Client{Transport: typesafeRoundTripFunc(
		func(*http.Request) (*http.Response, error) { return nil, transportError },
	)}))
	if _, err := model.Request(t.Context(), messages, valid); !errors.Is(err, transportError) {
		t.Fatalf("unexpected transport error: %v", err)
	}

	for _, test := range []struct {
		name, body, match string
		status            int
	}{
		{name: "HTTP", status: http.StatusBadRequest, body: "secret", match: "status 400"},
		{name: "invalid JSON", status: http.StatusOK, body: "not-json", match: "decode response"},
		{name: "missing model", status: http.StatusOK, body: `{}`, match: "omitted model"},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := typesafe.NewModel("jev", typesafe.WithHTTPClient(&http.Client{Transport: typesafeRoundTripFunc(
				func(*http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: test.status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(test.body))}, nil
				},
			)}))
			if _, err := model.Request(t.Context(), messages, valid); err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	model = typesafe.NewModel("jev", typesafe.WithHTTPClient(&http.Client{Transport: typesafeRoundTripFunc(
		func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: typesafeFailingBody{}}, nil
		},
	)}))
	if _, err := model.Request(t.Context(), messages, valid); err == nil || !strings.Contains(err.Error(), "read failed") {
		t.Fatalf("unexpected read error: %v", err)
	}

	model = typesafe.NewModel("jev", typesafe.WithHTTPClient(&http.Client{Transport: typesafeRoundTripFunc(
		func(*http.Request) (*http.Response, error) {
			t.Fatal("invalid body reached transport")
			return nil, nil
		},
	)}))
	invalidBody := ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: objectSchema("value", map[string]any{
		"description": "Value?", "anyOf": []any{
			map[string]any{"const": "a", "description": make(chan int)}, map[string]any{"const": "b"},
		},
	})}}
	if _, err := model.Request(t.Context(), messages, invalidBody); err == nil {
		t.Fatal("unencodable question was accepted")
	}
}

func TestTypeSafeHistoryAndRouteFallbacks(t *testing.T) {
	model := typesafe.NewModel("jev", typesafe.WithBaseURL(":"))
	valid := ai.ModelRequestParams{OutputTool: &ai.ToolDefinition{Name: "out", Schema: objectSchema("value", map[string]any{
		"type": "boolean", "description": "Value?",
	})}}
	if _, err := model.Request(t.Context(), nil, valid); err == nil || !strings.Contains(err.Error(), "needs text") {
		t.Fatalf("unexpected empty history error: %v", err)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.FilePart{Content: ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"}},
	}}}, valid); err == nil || !strings.Contains(err.Error(), "files") {
		t.Fatalf("unexpected file history error: %v", err)
	}

	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"answers":{"value":{"type":"noul","noul":0.9},"tool":{"type":"choice","choice":"out","confidence":1,"probabilities":{"out":1,"act":0}}},"model":"jev","usage":{}}`))
	}))
	defer server.Close()
	model = typesafe.NewModel("jev", typesafe.WithBaseURL(server.URL), typesafe.WithHTTPClient(server.Client()))
	_, err := model.Request(t.Context(), []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "old", Args: []byte("not-json")}}},
	}, ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "out", Schema: objectSchema("value", map[string]any{"type": "boolean", "description": "Value?"})},
		Tools:      []ai.ToolDefinition{{Name: "act", Description: "Act.", Schema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	criteria := requestBody["questions"].(map[string]any)["tool"].(map[string]any)["criteria"].(map[string]any)
	if criteria["out"] != "Structured output" {
		t.Fatalf("unexpected default output purpose: %#v", criteria)
	}
}

func objectSchema(name string, property any) map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{name: property}}
}
