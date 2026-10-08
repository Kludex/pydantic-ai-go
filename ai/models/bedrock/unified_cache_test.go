package bedrock_test

import (
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/bedrock"
)

func TestUnifiedBedrockCaching(t *testing.T) {
	var input *bedrockruntime.ConverseInput
	client := &fakeClient{converse: func(request *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
		input = request
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	noMessages := false
	for _, test := range []struct {
		name, model string
		cache       *ai.CacheConfig
		local       bool
		want        bool
		ttl         types.CacheTTL
		tools       bool
		retention   time.Duration
	}{
		{name: "default", model: "anthropic.claude-sonnet-4-5", cache: &ai.CacheConfig{}, want: true, tools: true, retention: 5 * time.Minute},
		{name: "hour", model: "anthropic.claude-sonnet-4-5", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, want: true, ttl: types.CacheTTLOneHour, tools: true, retention: time.Hour},
		{name: "old Claude", model: "us.anthropic.claude-sonnet-4", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, want: true, ttl: types.CacheTTLFiveMinutes, tools: true, retention: 5 * time.Minute},
		{name: "Nova", model: "amazon.nova-pro", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, want: true, ttl: types.CacheTTLFiveMinutes, retention: 5 * time.Minute},
		{name: "stable prefix", model: "amazon.nova-pro", cache: &ai.CacheConfig{Messages: &noMessages}, want: true, retention: 5 * time.Minute},
		{name: "unsupported", model: "meta.llama", cache: &ai.CacheConfig{}},
		{name: "disabled", model: "anthropic.claude-sonnet-4-5", cache: &ai.CacheConfig{Retention: ai.CacheRetentionDisabled}},
		{name: "local disabled", model: "anthropic.claude-sonnet-4-5", cache: &ai.CacheConfig{}, local: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			settings := ai.ModelSettings{Cache: test.cache}
			if test.local {
				settings.ExtraBody = map[string]any{"bedrock_cache_messages": false}
			}
			model := bedrock.NewModel(test.model, bedrock.WithClient(client))
			params := ai.ModelRequestParams{Instructions: "stable", Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}}, Settings: settings}
			messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}
			if _, err := model.Request(t.Context(), messages, params); err != nil {
				t.Fatal(err)
			}
			if (len(input.System) == 2) != test.want {
				t.Fatalf("system=%+v", input.System)
			}
			if test.want && input.System[1].(*types.SystemContentBlockMemberCachePoint).Value.Ttl != test.ttl {
				t.Fatalf("TTL=%+v", input.System[1])
			}
			if (len(input.ToolConfig.Tools) == 2) != test.tools {
				t.Fatalf("tool caching=%+v", input.ToolConfig)
			}
			wantMessages := test.want && (test.cache.Messages == nil || *test.cache.Messages)
			if (len(input.Messages[0].Content) == 2) != wantMessages {
				t.Fatalf("messages=%+v", input.Messages)
			}
			if got, known := ai.ResolveCacheRetention(model, &settings); known != test.want || known && got != test.retention {
				t.Fatalf("retention=%s %v", got, known)
			}
			if model.CachingNotEnabled(settings) {
				t.Fatal("configured cache reported missing")
			}
		})
	}
	model := bedrock.NewModel("anthropic.claude-sonnet-4-5", bedrock.WithClient(client))
	if !model.CachingNotEnabled(ai.ModelSettings{}) || bedrock.NewModel("meta.llama").CachingNotEnabled(ai.ModelSettings{}) {
		t.Fatal("incorrect configuration diagnostics")
	}
	bad := ai.ModelSettings{Cache: &ai.CacheConfig{Retention: "week"}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: bad}); err == nil {
		t.Fatal("invalid cache accepted")
	}
	if _, known := model.PromptCacheRetention(bad); known {
		t.Fatal("invalid retention claimed")
	}
	for _, parts := range [][]ai.InstructionPart{nil, {{Content: "static"}, {Content: "dynamic", Dynamic: true}}, {{Content: "dynamic", Dynamic: true}}} {
		params := ai.ModelRequestParams{InstructionParts: parts, Settings: ai.ModelSettings{Cache: &ai.CacheConfig{Messages: &noMessages}}}
		if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}, params); err != nil {
			t.Fatal(err)
		}
		if len(parts) == 2 {
			if _, ok := input.System[1].(*types.SystemContentBlockMemberCachePoint); !ok {
				t.Fatal("stable instruction boundary was not cached")
			}
		}
	}
}

func TestBedrockCachePreviousTailAfterWideTurn(t *testing.T) {
	var input *bedrockruntime.ConverseInput
	client := &fakeClient{converse: func(request *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
		input = request
		return completeOutput(types.StopReasonEndTurn), nil
	}}
	model := bedrock.NewModel("anthropic.claude-sonnet-4-5", bedrock.WithClient(client))
	calls := make([]ai.ResponsePart, 12)
	returns := make([]ai.RequestPart, 12)
	for index := range calls {
		calls[index] = ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`)}
		returns[index] = ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call", Content: "result"}
	}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}}, ai.ModelResponse{Parts: calls}, ai.ModelRequest{Parts: returns}}
	settings := ai.ModelSettings{Cache: &ai.CacheConfig{}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{Settings: settings}); err != nil {
		t.Fatal(err)
	}
	if len(input.Messages[0].Content) != 2 || len(input.Messages[2].Content) != 13 {
		t.Fatalf("previous and new tails not cached: %+v", input.Messages)
	}
	if _, err := model.Request(t.Context(), messages[1:], ai.ModelRequestParams{Settings: settings}); err != nil {
		t.Fatal(err)
	}
	messages[0] = ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "old"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
		ai.BinaryContent{Data: []byte("document"), MediaType: "application/pdf"},
	}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{Settings: settings}); err != nil {
		t.Fatal(err)
	}
	if got := input.Messages[0].Content[1].(*types.ContentBlockMemberCachePoint).Value.Ttl; got != types.CacheTTLOneHour {
		t.Fatalf("explicit cache TTL replaced: %s", got)
	}
}
