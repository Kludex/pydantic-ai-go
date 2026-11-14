package bedrock_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/bedrock"
)

func TestTypedSettingsAndNativeOutput(t *testing.T) {
	common := ai.ModelSettings{ExtraBody: map[string]any{"top_k": 40}}
	settings, err := (bedrock.Settings{
		Common: common, CacheInstructions: bedrock.CacheTTL1Hour,
		CacheMessages: bedrock.CacheTTL5Minutes, CacheToolDefinitions: bedrock.CacheTTL1Hour,
		InferenceProfile: "profile-arn",
		Guardrail: &bedrock.GuardrailConfig{
			Identifier: "guardrail", Version: "1", Trace: types.GuardrailTraceEnabledFull,
		},
		PerformanceLatency:                types.PerformanceConfigLatencyOptimized,
		RequestMetadata:                   map[string]string{"tenant": "one"},
		AdditionalModelResponseFieldPaths: []string{"/stop_sequence"},
		PromptVariables:                   map[string]string{"name": "Ada"},
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	common.ExtraBody["top_k"] = 99
	client := &fakeClient{converse: func(
		input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		if *input.ModelId != "profile-arn" || input.GuardrailConfig == nil ||
			*input.GuardrailConfig.GuardrailIdentifier != "guardrail" ||
			input.PerformanceConfig == nil || input.PerformanceConfig.Latency != types.PerformanceConfigLatencyOptimized ||
			input.RequestMetadata["tenant"] != "one" || len(input.AdditionalModelResponseFieldPaths) != 1 ||
			input.PromptVariables["name"].(*types.PromptVariableValuesMemberText).Value != "Ada" {
			t.Fatalf("Bedrock request settings missing: %#v", input)
		}
		if len(input.System) != 2 {
			t.Fatalf("instruction cache point missing: %#v", input.System)
		}
		if _, ok := input.System[1].(*types.SystemContentBlockMemberCachePoint); !ok {
			t.Fatalf("unexpected instruction cache block: %T", input.System[1])
		}
		if input.ToolConfig == nil || len(input.ToolConfig.Tools) != 2 {
			t.Fatalf("tool-definition cache point missing: %#v", input.ToolConfig)
		}
		if _, ok := input.ToolConfig.Tools[1].(*types.ToolMemberCachePoint); !ok {
			t.Fatalf("unexpected tool cache block: %T", input.ToolConfig.Tools[1])
		}
		points := 0
		for _, message := range input.Messages {
			for _, block := range message.Content {
				if _, ok := block.(*types.ContentBlockMemberCachePoint); ok {
					points++
				}
			}
		}
		if points != 2 {
			t.Fatalf("expected two newest message cache points, got %d", points)
		}
		if input.OutputConfig == nil || input.OutputConfig.TextFormat == nil ||
			input.OutputConfig.TextFormat.Type != types.OutputFormatTypeJsonSchema {
			t.Fatalf("native output config missing: %#v", input.OutputConfig)
		}
		structured := input.OutputConfig.TextFormat.Structure.(*types.OutputFormatStructureMemberJsonSchema)
		if *structured.Value.Name != "answer" || *structured.Value.Description != "Return an answer." ||
			!strings.Contains(*structured.Value.Schema, `"type":"object"`) {
			t.Fatalf("unexpected native output schema: %#v", structured.Value)
		}
		encoded, err := input.AdditionalModelRequestFields.MarshalSmithyDocument()
		if err != nil || string(encoded) != `{"top_k":40}` {
			t.Fatalf("unexpected additional fields: %s err=%v", encoded, err)
		}
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	model := bedrock.NewModel("model", bedrock.WithClient(client))
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "old"}, ai.CachePoint{}, ai.TextContent{Text: "middle"},
			ai.CachePoint{}, ai.TextContent{Text: "new"}, ai.CachePoint{},
		}}}},
	}
	_, err = model.Request(context.Background(), messages, ai.ModelRequestParams{
		Instructions: "system", Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}},
		OutputSchema: map[string]any{"type": "object"}, OutputMode: ai.OutputModeNative,
		OutputTool: &ai.ToolDefinition{Name: "answer", Description: "Return an answer."}, Settings: settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	if retention, ok := ai.ResolveCacheRetention(model, &settings); !ok || retention != time.Hour {
		t.Fatalf("unexpected cache retention: %s %v", retention, ok)
	}

	fiveMinutes, err := (bedrock.Settings{CacheMessages: bedrock.CacheTTL5Minutes}).Build()
	if err != nil {
		t.Fatal(err)
	}
	if retention, ok := model.PromptCacheRetention(fiveMinutes); !ok || retention != 5*time.Minute {
		t.Fatalf("unexpected five-minute retention: %s %v", retention, ok)
	}
	if retention, ok := model.PromptCacheRetention(ai.ModelSettings{}); ok || retention != 0 {
		t.Fatalf("unexpected empty retention: %s %v", retention, ok)
	}
}

func TestBedrockRaisesEarlierCacheTTL(t *testing.T) {
	settings, err := (bedrock.Settings{CacheInstructions: bedrock.CacheTTL5Minutes}).Build()
	if err != nil {
		t.Fatal(err)
	}
	client := &fakeClient{converse: func(
		input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		instruction := input.System[1].(*types.SystemContentBlockMemberCachePoint)
		message := input.Messages[0].Content[1].(*types.ContentBlockMemberCachePoint)
		if instruction.Value.Ttl != types.CacheTTLOneHour || message.Value.Ttl != types.CacheTTLOneHour {
			t.Fatalf("cache TTL order = instruction %q message %q", instruction.Value.Ttl, message.Value.Ttl)
		}
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	model := bedrock.NewModel("anthropic.claude-sonnet-4-5", bedrock.WithClient(client))
	_, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "context"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour}}},
	}}}, ai.ModelRequestParams{Instructions: "system", Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
}

func TestOpenAIConverseDropsUnsupportedSampling(t *testing.T) {
	for _, name := range []string{
		"us.openai.gpt-5.6-sol", "us.openai.gpt-5.6-luna", "us.openai.gpt-5.6-terra",
		"global.openai.gpt-6-sol", "global.openai.gpt-6-luna", "global.openai.gpt-6-astra", "global.openai.gpt-6.1-sol",
	} {
		t.Run(name, func(t *testing.T) {
			client := &fakeClient{converse: func(
				input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
			) (*bedrockruntime.ConverseOutput, error) {
				if input.InferenceConfig == nil || input.InferenceConfig.MaxTokens == nil || *input.InferenceConfig.MaxTokens != 64 ||
					input.InferenceConfig.Temperature != nil || input.InferenceConfig.TopP != nil {
					t.Fatalf("unexpected inference settings: %#v", input.InferenceConfig)
				}
				return completeOutput(types.StopReasonEndTurn), nil
			}}
			temperature, topP := float64(.2), float64(.3)
			_, err := bedrock.NewModel(name, bedrock.WithClient(client)).Request(t.Context(), nil, ai.ModelRequestParams{
				Settings: ai.ModelSettings{MaxTokens: 64, Temperature: &temperature, TopP: &topP},
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}

	client := &fakeClient{converse: func(
		input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		if input.InferenceConfig == nil || input.InferenceConfig.Temperature == nil || input.InferenceConfig.TopP == nil {
			t.Fatalf("GPT-OSS sampling settings were dropped: %#v", input.InferenceConfig)
		}
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	temperature, topP := float64(.2), float64(.3)
	_, err := bedrock.NewModel("openai.gpt-oss-120b-1:0", bedrock.WithClient(client)).Request(
		t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{Temperature: &temperature, TopP: &topP}},
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBedrockSettingsValidation(t *testing.T) {
	empty, err := (bedrock.Settings{}).Build()
	if err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v err=%v", empty, err)
	}

	tests := []struct {
		name     string
		settings bedrock.Settings
		match    string
	}{
		{name: "invalid TTL", settings: bedrock.Settings{CacheMessages: "forever"}, match: "invalid cache TTL"},
		{name: "reserved field", settings: bedrock.Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{
			"bedrock_cache_messages": true,
		}}}, match: "is reserved"},
		{name: "reserved request settings", settings: bedrock.Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{
			"bedrock_request_settings": true,
		}}}, match: "is reserved"},
		{name: "incomplete guardrail", settings: bedrock.Settings{
			Guardrail: &bedrock.GuardrailConfig{Identifier: "guardrail"},
		}, match: "identifier and version are required"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	client := &fakeClient{converse: func(
		*bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		t.Fatal("client should not be called")
		return nil, nil
	}}
	model := bedrock.NewModel("model", bedrock.WithClient(client))
	requestTests := []struct {
		name     string
		messages []ai.ModelMessage
		params   ai.ModelRequestParams
		match    string
	}{
		{name: "wrong cache type", params: ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{
			"bedrock_cache_messages": "5m",
		}}}, match: "must use CacheTTL"},
		{name: "invalid extracted TTL", params: ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{
			"bedrock_cache_messages": bedrock.CacheTTL("forever"),
		}}}, match: "invalid cache TTL"},
		{name: "unbuilt request settings", params: ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{
			"bedrock_request_settings": true,
		}}}, match: "must be built with Settings.Build"},
		{name: "instructions required", params: ai.ModelRequestParams{Settings: mustBedrockSettings(t, bedrock.Settings{
			CacheInstructions: bedrock.CacheTTL5Minutes,
		})}, match: "requires instructions"},
		{name: "message required", params: ai.ModelRequestParams{Settings: mustBedrockSettings(t, bedrock.Settings{
			CacheMessages: bedrock.CacheTTL5Minutes,
		})}, match: "message caching"},
		{name: "tools required", params: ai.ModelRequestParams{
			Instructions: "system", Settings: mustBedrockSettings(t, bedrock.Settings{
				CacheToolDefinitions: bedrock.CacheTTL5Minutes,
			}),
		}, match: "requires tools"},
		{name: "invalid output schema", params: ai.ModelRequestParams{
			OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"invalid": make(chan int)},
		}, match: "marshal output schema"},
	}
	for _, test := range requestTests {
		t.Run(test.name, func(t *testing.T) {
			_, err := model.Request(context.Background(), test.messages, test.params)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	invalid := ai.ModelSettings{ExtraBody: map[string]any{"bedrock_cache_messages": true}}
	if retention, ok := model.PromptCacheRetention(invalid); ok || retention != 0 {
		t.Fatalf("invalid settings reported retention: %s %v", retention, ok)
	}
}

func mustBedrockSettings(t *testing.T, settings bedrock.Settings) ai.ModelSettings {
	t.Helper()
	built, err := settings.Build()
	if err != nil {
		t.Fatal(err)
	}
	return built
}

func TestPromptedOutputOmitsNativeConfiguration(t *testing.T) {
	client := &fakeClient{converse: func(
		input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		if input.OutputConfig != nil {
			t.Fatalf("prompted output sent native configuration: %#v", input.OutputConfig)
		}
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	_, err := bedrock.NewModel("model", bedrock.WithClient(client)).Request(
		context.Background(), nil, ai.ModelRequestParams{
			OutputMode: ai.OutputModePrompted, OutputSchema: map[string]any{"type": "object"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestNativeOutputSchemaIsJSON(t *testing.T) {
	client := &fakeClient{converse: func(
		input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error) {
		structure := input.OutputConfig.TextFormat.Structure.(*types.OutputFormatStructureMemberJsonSchema)
		var schema map[string]any
		if err := json.Unmarshal([]byte(*structure.Value.Schema), &schema); err != nil {
			t.Fatal(err)
		}
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	_, err := bedrock.NewModel("model", bedrock.WithClient(client)).Request(
		context.Background(), nil, ai.ModelRequestParams{
			OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "string"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGLM53InlinesSchemaDefinitions(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"item": map[string]any{"$ref": "#/$defs/Item"}},
		"$defs":      map[string]any{"Item": map[string]any{"type": "object"}},
	}
	for model, inlined := range map[string]bool{"us.zai.glm-5.3": true, "zai.glm-5": false} {
		t.Run(model, func(t *testing.T) {
			client := &fakeClient{converse: func(
				input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options),
			) (*bedrockruntime.ConverseOutput, error) {
				spec := input.ToolConfig.Tools[0].(*types.ToolMemberToolSpec).Value
				tool, _ := spec.InputSchema.(*types.ToolInputSchemaMemberJson).Value.MarshalSmithyDocument()
				output := *input.OutputConfig.TextFormat.Structure.(*types.OutputFormatStructureMemberJsonSchema).Value.Schema
				for _, encoded := range []string{string(tool), output} {
					if strings.Contains(encoded, "$defs") == inlined {
						t.Fatalf("unexpected schema: %s", encoded)
					}
				}
				return completeOutput(types.StopReasonEndTurn), nil
			}}
			_, err := bedrock.NewModel(model, bedrock.WithClient(client)).Request(t.Context(), nil, ai.ModelRequestParams{
				Tools:      []ai.ToolDefinition{{Name: "lookup", Schema: schema}},
				OutputMode: ai.OutputModeNative, OutputSchema: schema,
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestThinkingSettings(t *testing.T) {
	tests := []struct {
		name, model, effort string
		level               ai.ThinkingLevel
	}{
		{name: "xhigh", model: "us.anthropic.claude-opus-5", level: ai.ThinkingLevelXHigh, effort: "xhigh"},
		{name: "max fallback", model: "us.anthropic.claude-sonnet-4-6", level: ai.ThinkingLevelXHigh, effort: "max"},
		{name: "minimal", model: "us.anthropic.claude-sonnet-4-6", level: ai.ThinkingLevelMinimal, effort: "low"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{converse: func(input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
				encoded, err := input.AdditionalModelRequestFields.MarshalSmithyDocument()
				if err != nil || !strings.Contains(string(encoded), `"type":"adaptive"`) || !strings.Contains(string(encoded), `"effort":"`+test.effort+`"`) {
					t.Fatalf("unexpected thinking fields: %s err=%v", encoded, err)
				}
				return completeOutput(types.StopReasonEndTurn), nil
			}}
			model := bedrock.NewModel(test.model, bedrock.WithClient(client))
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: test.level}}})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, test := range []struct {
		name, model, contains string
		thinking              ai.ThinkingSettings
	}{
		{name: "disabled", model: "anthropic.claude-3", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}, contains: `"type":"disabled"`},
		{name: "budget", model: "anthropic.claude-3", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}, contains: `"budget_tokens":16384`},
		{name: "openai", model: "openai.gpt-5.6-luna", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}, contains: `"reasoning_effort":"high"`},
		{name: "glm on", model: "zai.glm-4.7-flash", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelLow}, contains: `"reasoning_effort":"high"`},
		{name: "glm off", model: "zai.glm-5", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}, contains: `"reasoning_effort":"none"`},
		{name: "glm 5.3 enabled", model: "us.zai.glm-5.3", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelEnabled}, contains: `"reasoning_effort":"max"`},
		{name: "glm 5.3 off", model: "us.zai.glm-5.3", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}, contains: `"reasoning_effort":"low"`},
		{name: "glm 5.3 medium", model: "us.zai.glm-5.3", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelMedium}, contains: `"reasoning_effort":"medium"`},
		{name: "deepseek on", model: "deepseek.v3.2", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelXHigh}, contains: `"reasoning_effort":"high"`},
		{name: "deepseek off", model: "deepseek.v3.2", thinking: ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}, contains: `"reasoning_effort":"none"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{converse: func(input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
				encoded, _ := input.AdditionalModelRequestFields.MarshalSmithyDocument()
				if !strings.Contains(string(encoded), test.contains) {
					t.Fatalf("unexpected fields: %s", encoded)
				}
				return completeOutput(types.StopReasonEndTurn), nil
			}}
			_, err := bedrock.NewModel(test.model, bedrock.WithClient(client)).Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{Thinking: &test.thinking}})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
	budget := 123
	client := &fakeClient{converse: func(input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
		encoded, _ := input.AdditionalModelRequestFields.MarshalSmithyDocument()
		if !strings.Contains(string(encoded), `"budget_tokens":123`) {
			t.Fatalf("unexpected budget: %s", encoded)
		}
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	_, err := bedrock.NewModel("anthropic.claude-3", bedrock.WithClient(client)).Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelLow, TokenBudget: &budget}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, params := range []ai.ModelRequestParams{
		{Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: "invalid"}}},
		{OutputTool: &ai.ToolDefinition{Name: "answer"}, Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}}},
	} {
		if _, err := bedrock.NewModel("anthropic.claude-3", bedrock.WithClient(client)).Request(t.Context(), nil, params); err == nil {
			t.Fatal("expected incompatible thinking error")
		}
	}
	for _, model := range []string{"zai.glm-5", "deepseek.v3.2"} {
		if _, err := bedrock.NewModel(model, bedrock.WithClient(client)).Request(t.Context(), nil, ai.ModelRequestParams{
			Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: "invalid"}},
		}); err == nil {
			t.Fatalf("%s: expected invalid thinking level error", model)
		}
	}
}

func TestReasoningEffortThinkingWithOutputTool(t *testing.T) {
	output := &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}}
	params := ai.ModelRequestParams{
		OutputTool: output, OutputMode: ai.OutputModeTool,
		Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelEnabled}},
	}
	var choice types.ToolChoice
	client := &fakeClient{converse: func(input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
		choice = input.ToolConfig.ToolChoice
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	if _, err := bedrock.NewModel("zai.glm-5", bedrock.WithClient(client)).Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if _, ok := choice.(*types.ToolChoiceMemberAuto); !ok {
		t.Fatalf("GLM output tool was forced while thinking: %T", choice)
	}
	if _, err := bedrock.NewModel("deepseek.v3.2", bedrock.WithClient(client)).Request(t.Context(), nil, params); err == nil ||
		!strings.Contains(err.Error(), "does not support thinking and output tools") {
		t.Fatalf("DeepSeek forced an output tool while thinking: %v", err)
	}
	params.Settings.Thinking = &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}
	if _, err := bedrock.NewModel("deepseek.v3.2", bedrock.WithClient(client)).Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if _, ok := choice.(*types.ToolChoiceMemberTool); !ok {
		t.Fatalf("output tool was not forced without thinking: %T", choice)
	}
}

func TestClaudeMaxTokensCappedAtMaximumOutput(t *testing.T) {
	for _, test := range []struct {
		model           string
		requested, sent int32
	}{
		{model: "us.anthropic.claude-opus-5-5-v1:0", requested: 200_000, sent: 128_000},
		{model: "anthropic.claude-sonnet-4-5-20250929-v1:0", requested: 100_000, sent: 64_000},
		{model: "anthropic.claude-sonnet-4-5-20250929-v1:0", requested: 1_000, sent: 1_000},
		{model: "anthropic.claude-3-5-sonnet", requested: 200_000, sent: 200_000},
		{model: "zai.glm-5", requested: 200_000, sent: 200_000},
	} {
		var sent int32
		client := &fakeClient{converse: func(input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
			sent = *input.InferenceConfig.MaxTokens
			return completeOutput(types.StopReasonEndTurn), nil
		}}
		if _, err := bedrock.NewModel(test.model, bedrock.WithClient(client)).Request(t.Context(), nil, ai.ModelRequestParams{
			Settings: ai.ModelSettings{MaxTokens: int(test.requested)},
		}); err != nil {
			t.Fatal(err)
		}
		if sent != test.sent {
			t.Fatalf("%s: max tokens %d, want %d", test.model, sent, test.sent)
		}
	}
}
