package openai_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestOpenAIStrictSchemaInference(t *testing.T) {
	tests := []struct {
		name       string
		schema     map[string]any
		strict     bool
		wantStrict bool
	}{
		{
			name: "compatible",
			schema: map[string]any{
				"title": "Input", "type": "object", "additionalProperties": false,
				"properties": map[string]any{"value": map[string]any{"type": "string", "format": "email"}},
				"required":   []any{"value"},
			},
			wantStrict: true,
		},
		{
			name: "optional property",
			schema: map[string]any{
				"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}},
			},
		},
		{
			name: "additional properties",
			schema: map[string]any{
				"type": "object", "properties": map[string]any{}, "required": []string{},
				"additionalProperties": map[string]any{"type": "string"},
			},
		},
		{
			name: "unsupported constraint",
			schema: map[string]any{
				"type": "object", "properties": map[string]any{
					"value": map[string]any{"type": "string", "format": "uri", "pattern": `\(?=literal`, "default": "x"},
				},
				"required": []string{"value"},
			},
		},
		{
			name: "one of",
			schema: map[string]any{
				"type": "object", "properties": map[string]any{
					"value": map[string]any{"oneOf": []any{map[string]any{"type": "string"}}},
				},
				"required": []string{"value"},
			},
		},
		{
			name: "untyped array",
			schema: map[string]any{
				"type": "object", "properties": map[string]any{"values": map[string]any{"type": "array", "items": map[string]any{}}},
				"required": []string{"values"},
			},
		},
		{
			name: "typed array",
			schema: map[string]any{
				"type": "object", "properties": map[string]any{"values": map[string]any{
					"type": "array", "items": map[string]any{"enum": []any{"x"}},
				}},
				"required": []string{"values"},
			},
			wantStrict: true,
		},
		{
			name: "tuple array",
			schema: map[string]any{
				"type": "object", "properties": map[string]any{"values": map[string]any{
					"type": "array", "prefixItems": []any{map[string]any{"type": "string"}},
				}},
				"required": []string{"values"},
			},
			wantStrict: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var gotBody map[string]any
			model := newServer(t, openAIRequestRecorder(t, &gotBody))
			params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{Name: "work", Schema: test.schema}}}
			if _, err := model.Request(t.Context(), nil, params); err != nil {
				t.Fatal(err)
			}
			function := gotBody["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
			_, strict := function["strict"]
			if strict != test.wantStrict {
				t.Fatalf("expected strict=%v, got %v", test.wantStrict, function)
			}
			parameters := function["parameters"].(map[string]any)
			if parameters["title"] != nil {
				t.Fatalf("title was not removed: %v", parameters)
			}
			if test.name == "optional property" && parameters["additionalProperties"] != false {
				t.Fatalf("default additionalProperties not added: %v", parameters)
			}
		})
	}
}

func TestOpenAIExplicitStrictSchemaRewrite(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, openAIRequestRecorder(t, &gotBody))
	strict := true
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
		Name: "work",
		Schema: map[string]any{
			"$schema": "draft", "type": "object", "discriminator": map[string]any{},
			"properties": map[string]any{
				"value": map[string]any{
					"type": "string", "default": "x", "minLength": 2, "format": "uri",
					"pattern": "(?=x)", "description": "Value",
				},
				"choice": map[string]any{"oneOf": []any{map[string]any{"type": "string"}}},
				"empty":  map[string]any{"type": "object"},
				"other":  map[string]any{"type": "string", "minLength": 1},
			},
			"additionalProperties": map[string]any{"type": "string"},
		},
		Strict: &strict,
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	function := gotBody["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if function["strict"] != true {
		t.Fatalf("strict flag missing: %v", function)
	}
	schema := function["parameters"].(map[string]any)
	if schema["additionalProperties"] != false || schema["$schema"] != nil || schema["discriminator"] != nil {
		t.Fatalf("root was not rewritten: %v", schema)
	}
	required := schema["required"].([]any)
	if strings.Join(
		[]string{required[0].(string), required[1].(string), required[2].(string), required[3].(string)}, ",",
	) != "choice,empty,other,value" {
		t.Fatalf("required properties are not deterministic: %v", required)
	}
	properties := schema["properties"].(map[string]any)
	value := properties["value"].(map[string]any)
	if value["default"] != nil || value["minLength"] != nil || value["format"] != nil || value["pattern"] != nil {
		t.Fatalf("strict-incompatible constraints remain: %v", value)
	}
	if !strings.Contains(value["description"].(string), "minLength=2") || !strings.Contains(value["description"].(string), "pattern=(?=x)") {
		t.Fatalf("removed constraints were not preserved in description: %v", value)
	}
	choice := properties["choice"].(map[string]any)
	if choice["oneOf"] != nil || choice["anyOf"] == nil {
		t.Fatalf("oneOf was not rewritten: %v", choice)
	}
	empty := properties["empty"].(map[string]any)
	if empty["additionalProperties"] != false || empty["properties"] == nil {
		t.Fatalf("nested object was not completed: %v", empty)
	}
	if properties["other"].(map[string]any)["description"] != "minLength=1" {
		t.Fatalf("constraint description was not created: %v", properties["other"])
	}
}

func TestOpenAIRecursiveRootSchema(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, openAIRequestRecorder(t, &gotBody))
	params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
		Name: "tree",
		Schema: map[string]any{
			"$ref": "#/$defs/Node",
			"$defs": map[string]any{
				"Node": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"next": map[string]any{"$ref": "#/$defs/Node", "description": "Next node"},
					},
					"required": []string{"name", "next"},
				},
			},
		},
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	function := gotBody["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if function["strict"] != true {
		t.Fatalf("recursive schema should infer strict mode: %v", function)
	}
	schema := function["parameters"].(map[string]any)
	if schema["$ref"] != nil || schema["type"] != "object" || schema["additionalProperties"] != false {
		t.Fatalf("root definition was not expanded: %v", schema)
	}
	next := schema["properties"].(map[string]any)["next"].(map[string]any)
	refs := next["anyOf"].([]any)
	if refs[0].(map[string]any)["$ref"] != "#" || next["description"] != "Next node" {
		t.Fatalf("recursive reference was not normalized: %v", next)
	}
	definition := schema["$defs"].(map[string]any)["Node"].(map[string]any)
	if definition["additionalProperties"] != false {
		t.Fatalf("recursive definition was not transformed: %v", definition)
	}
}

func TestOpenAIStrictSchemaErrors(t *testing.T) {
	strict := true
	tests := []struct {
		name   string
		schema map[string]any
		want   string
	}{
		{name: "root", schema: map[string]any{"type": "string"}, want: "root must have type object"},
		{name: "root defs", schema: map[string]any{"$ref": "#/$defs/Missing"}, want: "has no $defs"},
		{name: "root definition", schema: map[string]any{
			"$ref": "#/$defs/Missing", "$defs": map[string]any{"Other": map[string]any{"type": "object"}},
		}, want: "no matching definition"},
		{name: "array", schema: map[string]any{
			"type": "object", "properties": map[string]any{"values": map[string]any{"type": "array", "items": true}},
		}, want: "array items"},
		{name: "empty prefix", schema: map[string]any{
			"type": "object", "properties": map[string]any{"values": map[string]any{"type": "array", "prefixItems": []any{}}},
		}, want: "array items"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newServer(t, openAIRequestRecorder(t, nil))
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
				Name: "bad", Schema: test.schema, Strict: &strict,
			}}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q error, got %v", test.want, err)
			}
		})
	}
}

func TestOpenAISchemaErrorsAcrossOutputModes(t *testing.T) {
	strict := true
	bad := ai.ToolDefinition{Name: "bad", Schema: map[string]any{
		"type": "object", "properties": map[string]any{"values": map[string]any{"type": "array"}},
	}, Strict: &strict}

	chat := newServer(t, openAIRequestRecorder(t, nil))
	if _, err := chat.Request(t.Context(), nil, ai.ModelRequestParams{OutputTool: &bad}); err == nil {
		t.Fatal("expected chat output-tool schema error")
	}
	if _, err := chat.Request(t.Context(), nil, ai.ModelRequestParams{OutputSchema: bad.Schema}); err == nil {
		t.Fatal("expected native output schema error")
	}
	chatStream, err := chat.StreamRequest(t.Context(), nil, ai.ModelRequestParams{Tools: []ai.ToolDefinition{bad}})
	if err == nil || chatStream != nil {
		t.Fatal("expected chat stream schema error")
	}

	responses := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[],"usage":{}}`))
	})
	if _, err := responses.Request(t.Context(), nil, ai.ModelRequestParams{OutputTool: &bad}); err == nil {
		t.Fatal("expected Responses output-tool schema error")
	}
	responsesStream, err := responses.StreamRequest(t.Context(), nil, ai.ModelRequestParams{Tools: []ai.ToolDefinition{bad}})
	if err == nil || responsesStream != nil {
		t.Fatal("expected Responses stream schema error")
	}
}

func TestOpenAIStrictOptOutAndProfileOverride(t *testing.T) {
	server := func(t *testing.T, opts ...openai.Option) (*openai.Model, *map[string]any) {
		t.Helper()
		var body map[string]any
		model := newOpenAIServer(t, openAIRequestRecorder(t, &body), opts...)
		return model, &body
	}
	strict := false
	model, body := server(t)
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
		Name: "work", Schema: map[string]any{"type": "object", "additionalProperties": true}, Strict: &strict,
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	function := (*body)["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if _, ok := function["strict"]; ok {
		t.Fatalf("strict false should be omitted: %v", function)
	}
	if function["parameters"].(map[string]any)["additionalProperties"] != true {
		t.Fatalf("opt-out schema was changed: %v", function)
	}

	model, body = server(t, openai.WithStrictToolSupport(false))
	params.Tools[0].Strict = nil
	params.Tools[0].Schema = map[string]any{
		"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}},
		"required": []string{"value"}, "additionalProperties": false,
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	function = (*body)["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if _, ok := function["strict"]; ok {
		t.Fatalf("profile override should omit strict: %v", function)
	}

	model, body = server(t, openai.WithStrictToolSupport(false))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{OutputSchema: params.Tools[0].Schema}); err != nil {
		t.Fatal(err)
	}
	jsonSchema := (*body)["response_format"].(map[string]any)["json_schema"].(map[string]any)
	if _, ok := jsonSchema["strict"]; ok {
		t.Fatalf("native strict flag should respect profile override: %v", jsonSchema)
	}

	var responsesBody map[string]any
	httpServer := newHTTPServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&responsesBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[],"usage":{}}`))
	})
	responses := openai.NewResponsesModel(
		"gpt-5", openai.WithBaseURL(httpServer.URL), openai.WithHTTPClient(httpServer.Client()),
		openai.WithStrictToolSupport(false),
	)
	if _, err := responses.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if _, ok := responsesBody["tools"].([]any)[0].(map[string]any)["strict"]; ok {
		t.Fatalf("Responses strict flag should respect profile override: %v", responsesBody)
	}
	if _, err := responses.Request(t.Context(), nil, ai.ModelRequestParams{
		OutputSchema: params.Tools[0].Schema, OutputMode: ai.OutputModeNative,
	}); err != nil {
		t.Fatal(err)
	}
	format := responsesBody["text"].(map[string]any)["format"].(map[string]any)
	if _, ok := format["strict"]; ok {
		t.Fatalf("Responses native output should respect strict profile override: %v", format)
	}
}

func TestResponsesStrictSchemaInference(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[],"usage":{}}`))
	})
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
		Name: "work", Schema: map[string]any{
			"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}},
			"required": []string{"value"}, "additionalProperties": false,
		},
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	tool := gotBody["tools"].([]any)[0].(map[string]any)
	if tool["strict"] != true || tool["parameters"].(map[string]any)["additionalProperties"] != false {
		t.Fatalf("Responses tool was not prepared: %v", tool)
	}
}

func openAIRequestRecorder(t *testing.T, body *map[string]any) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if body != nil {
			if err := json.NewDecoder(r.Body).Decode(body); err != nil {
				t.Error(err)
			}
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"content":"ok"}}],"usage":{}}`))
	}
}

func newOpenAIServer(t *testing.T, handler http.HandlerFunc, extra ...openai.Option) *openai.Model {
	t.Helper()
	server := newHTTPServer(t, handler)
	opts := []openai.Option{
		openai.WithAPIKey("test-key"), openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()),
	}
	return openai.NewModel("gpt-5", append(opts, extra...)...)
}

func newHTTPServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}
