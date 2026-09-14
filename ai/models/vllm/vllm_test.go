package vllm_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/infer"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/models/vllm"
)

func TestVLLMRequest(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("unexpected request: %s authorization=%q", request.URL.Path, request.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_, _ = io.WriteString(response, `{"model":"Qwen/Qwen3-32B","choices":[{"message":{"reasoning":"think","reasoning_content":"think","content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	model, err := vllm.NewModel("Qwen/Qwen3-32B", vllm.WithBaseURL(server.URL+"/v1"),
		vllm.WithAPIKey("secret"), vllm.WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "history"}, ai.UserPromptPart{Content: "hello"},
	}}}, ai.ModelRequestParams{
		Instructions: "current", Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}},
	})
	if err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	if len(messages) != 2 || messages[0].(map[string]any)["content"] != "current\n\nhistory" ||
		body["reasoning_effort"] != "high" {
		t.Fatalf("unexpected request body: %#v", body)
	}
	if len(response.Parts) != 2 || response.Parts[0].(ai.ThinkingPart).Content != "think" ||
		response.Parts[0].(ai.ThinkingPart).ID != "reasoning" {
		t.Fatalf("unexpected response: %#v", response)
	}
}

func TestVLLMConfigurationAndInference(t *testing.T) {
	t.Setenv("VLLM_BASE_URL", "")
	if _, err := vllm.NewProviderConfig(); err == nil {
		t.Fatal("expected missing provider base URL error")
	}
	if _, err := vllm.NewModel("model"); err == nil {
		t.Fatal("expected missing base URL error")
	}
	t.Setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
	t.Setenv("VLLM_API_KEY", "key")
	model, err := infer.Model("vllm:model")
	if err != nil {
		t.Fatal(err)
	}
	identity := model.(ai.ModelProviderIdentity)
	if identity.ProviderName() != "vllm" || identity.ProviderURL() != "http://localhost:8000/v1" {
		t.Fatalf("unexpected inferred model: %q %q", identity.ProviderName(), identity.ProviderURL())
	}
	profile := model.(ai.ModelProfiler).ModelProfile()
	if !profile.NativeOutputRequiresPrompt {
		t.Fatal("vLLM native output must include schema instructions")
	}
}

func TestVLLMOptionsAndStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Test") != "value" {
			t.Errorf("provider header missing: %v", request.Header)
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\",\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	model, err := vllm.NewModel("deepseek-ai/DeepSeek-V4-Pro",
		vllm.WithAPIKey("ignored"), vllm.WithBaseURL("https://ignored.example"),
		vllm.WithHTTPClient(http.DefaultClient), vllm.WithProvider(openai.ProviderConfig{
			Name: "gateway", BaseURL: server.URL, HTTPClient: server.Client(), Headers: http.Header{"X-Test": {"value"}},
		}), vllm.WithDefaultSettings(ai.ModelSettings{MaxTokens: 10}))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := model.StreamRequest(t.Context(), []ai.ModelMessage{ai.ModelResponse{}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var thinking bool
	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if delta, ok := event.(ai.ThinkingDeltaEvent); ok {
			thinking = delta.ID == "reasoning_content" && delta.Delta == "think"
		}
	}
	if !thinking || model.DefaultModelSettings().MaxTokens != 10 {
		t.Fatal("vLLM options or streamed reasoning were not applied")
	}
}

func TestVLLMProviderConfig(t *testing.T) {
	t.Setenv("VLLM_BASE_URL", "https://vllm.example/v1")
	t.Setenv("VLLM_API_KEY", "key")
	provider, err := vllm.NewProviderConfig()
	if err != nil || provider.BaseURL != "https://vllm.example/v1" || provider.APIKey != "key" {
		t.Fatalf("unexpected provider config: %#v %v", provider, err)
	}
}

func TestVLLMForcedToolChoiceByModel(t *testing.T) {
	for name, test := range map[string]struct {
		model      string
		thinking   *ai.ThinkingSettings
		toolChoice string
	}{
		"Qwen coder":                {model: "Qwen/Qwen3-Coder-480B-A35B-Instruct", toolChoice: "required"},
		"GPT OSS":                   {model: "openai/gpt-oss-20b", toolChoice: "auto"},
		"DeepSeek default thinking": {model: "deepseek-ai/DeepSeek-V4-Pro", toolChoice: "required"},
		"DeepSeek thinking disabled": {
			model: "deepseek-ai/DeepSeek-V4-Pro", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
			toolChoice: "required",
		},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				_ = json.NewDecoder(request.Body).Decode(&body)
				_, _ = io.WriteString(response, `{"choices":[{"message":{"tool_calls":[{"id":"call","type":"function","function":{"name":"final","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
			}))
			defer server.Close()
			model, err := vllm.NewModel(test.model, vllm.WithBaseURL(server.URL), vllm.WithHTTPClient(server.Client()))
			if err != nil {
				t.Fatal(err)
			}
			_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
				OutputTool: &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}, Strict: boolPointer(true)},
				Settings:   ai.ModelSettings{Thinking: test.thinking},
			})
			if err != nil {
				t.Fatal(err)
			}
			if body["tool_choice"] != test.toolChoice {
				t.Fatalf("unexpected tool choice: %#v", body)
			}
		})
	}
}

func boolPointer(value bool) *bool { return &value }

func TestVLLMFamilyProfiles(t *testing.T) {
	for name, test := range map[string]struct {
		model         string
		thinking      *ai.ThinkingSettings
		wantReasoning string
		wantInline    bool
	}{
		"Meta schema": {model: "meta-llama/Llama-4", wantInline: true},
		"Google schema": {
			model: "google/gemma-4-pro", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
			wantInline: true,
		},
		"Qwen always-on": {
			model: "Qwen/Qwen3-235B-A22B-Thinking-2507", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
			wantInline: true,
		},
		"DeepSeek R1 always-on": {
			model: "deepseek-ai/DeepSeek-R1", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
		},
		"Magistral always-on": {
			model: "mistralai/Magistral-Small-2509", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
		},
		"Cohere always-on": {
			model: "CohereLabs/command-a-reasoning-08-2025", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
		},
		"GPT OSS always-on": {
			model: "openai/gpt-oss-20b", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
		},
		"GLM always-on": {
			model: "zai/GLM-5.3", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
		},
		"GLM effort": {
			model: "zai/GLM-5.2", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh},
			wantReasoning: "high",
		},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				_ = json.NewDecoder(request.Body).Decode(&body)
				_, _ = io.WriteString(response, `{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`)
			}))
			defer server.Close()
			model, err := vllm.NewModel(test.model, vllm.WithBaseURL(server.URL), vllm.WithHTTPClient(server.Client()))
			if err != nil {
				t.Fatal(err)
			}
			schema := map[string]any{
				"$defs": map[string]any{"Value": map[string]any{
					"type": "object", "title": "Value", "properties": map[string]any{"name": map[string]any{"type": "string"}},
					"required": []any{"name"}, "additionalProperties": false,
				}},
				"type": "object", "properties": map[string]any{"value": map[string]any{"$ref": "#/$defs/Value"}},
				"required": []any{"value"}, "additionalProperties": false,
			}
			_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
				Tools:    []ai.ToolDefinition{{Name: "inspect", Schema: schema, Strict: boolPointer(false)}},
				Settings: ai.ModelSettings{Thinking: test.thinking},
			})
			if err != nil {
				t.Fatal(err)
			}
			if body["reasoning_effort"] != valueOrNil(test.wantReasoning) {
				t.Fatalf("unexpected thinking payload: %#v", body)
			}
			wireSchema := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)["parameters"].(map[string]any)
			value := wireSchema["properties"].(map[string]any)["value"].(map[string]any)
			inlined := wireSchema["$defs"] == nil && value["$ref"] == nil
			if inlined != test.wantInline {
				t.Fatalf("unexpected transformed schema: %#v", wireSchema)
			}
		})
	}
}

func TestVLLMSchemaTransformsAllDefinitions(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, body)
		_, _ = io.WriteString(response, `{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	strict := false
	complex := map[string]any{
		"$schema": "draft", "title": "Root", "type": "object", "discriminator": map[string]any{},
		"examples": []any{map[string]any{}}, "exclusiveMinimum": 1, "exclusiveMaximum": 9,
		"$defs": map[string]any{
			"Value/Name": map[string]any{"type": "string", "format": "email", "description": "Address"},
			"Alias":      map[string]any{"$ref": "#/$defs/Value~1Name"},
			"Missing":    map[string]any{"$ref": "#/$defs/Absent"},
		},
		"properties": map[string]any{
			"reference": map[string]any{"$ref": "#/$defs/Value~1Name", "description": "Override"},
			"constant":  map[string]any{"const": true},
			"enum":      map[string]any{"enum": []any{nil, true, false, 2}},
			"nullable": map[string]any{"anyOf": []any{
				map[string]any{"type": "null"}, map[string]any{"type": "string", "format": "uuid"},
			}},
			"invalid_union": map[string]any{"anyOf": []any{"bad", map[string]any{"type": "null"}}},
			"choice": map[string]any{"oneOf": []any{
				map[string]any{"type": "string"}, map[string]any{"type": "integer"}, map[string]any{"type": "boolean"},
			}},
			"formatted": map[string]any{"type": "string", "format": "date"},
			"missing":   map[string]any{"$ref": "#/$defs/Missing"},
		},
		"required": []any{"reference"}, "additionalProperties": false,
	}
	plain := map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}
	model, err := vllm.NewModel("google/gemma-4", vllm.WithBaseURL(server.URL), vllm.WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	included := true
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: "inspect", Schema: complex, ReturnSchema: complex, IncludeReturnSchema: &included, Strict: &strict,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "later", Schema: plain, ReturnSchema: plain, Strict: &strict}},
		OutputTool:    &ai.ToolDefinition{Name: "final", Schema: complex, ReturnSchema: complex, Strict: &strict},
		OutputSchema:  complex,
		OutputMode:    ai.OutputModeNative,
	})
	if err != nil {
		t.Fatal(err)
	}
	if complex["title"] != "Root" || complex["$defs"] == nil {
		t.Fatal("schema transform mutated caller data")
	}
	wireSchema := bodies[0]["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)["parameters"].(map[string]any)
	properties := wireSchema["properties"].(map[string]any)
	if wireSchema["$defs"] != nil || wireSchema["title"] != nil ||
		properties["constant"].(map[string]any)["enum"].([]any)[0] != "True" ||
		properties["formatted"].(map[string]any)["description"] != "Format: date" ||
		properties["nullable"].(map[string]any)["nullable"] != true {
		t.Fatalf("Gemma schema was not transformed: %#v", wireSchema)
	}

	recursive := map[string]any{
		"$defs": map[string]any{"Node": map[string]any{
			"type": "object", "properties": map[string]any{"children": map[string]any{
				"anyOf": []any{map[string]any{"$ref": "#/$defs/Node"}},
			}},
		}},
		"type": "object", "properties": map[string]any{"node": map[string]any{"$ref": "#/$defs/Node"}},
	}
	qwen, err := vllm.NewModel("Qwen/Qwen3-32B", vllm.WithBaseURL(server.URL), vllm.WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = qwen.Request(t.Context(), nil, ai.ModelRequestParams{Tools: []ai.ToolDefinition{{
		Name: "recursive", Schema: recursive, Strict: &strict,
	}}}); err != nil {
		t.Fatal(err)
	}
	recursiveWire := bodies[1]["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)["parameters"].(map[string]any)
	if recursiveWire["$defs"] == nil {
		t.Fatalf("recursive schema definitions were removed: %#v", recursiveWire)
	}
}

func valueOrNil(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func TestVLLMUnsupportedThinkingIsOmitted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(request.Body).Decode(&body)
		if _, exists := body["reasoning_effort"]; exists {
			t.Errorf("unsupported reasoning effort was sent: %#v", body)
		}
		_, _ = io.WriteString(response, `{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	model, err := vllm.NewModel("Qwen/Qwen3.8-27B", vllm.WithBaseURL(server.URL), vllm.WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}},
	})
	if err != nil {
		t.Fatal(err)
	}
}
