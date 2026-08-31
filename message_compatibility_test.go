package ai_test

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestUnmarshalUpstreamBasicMessageFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_basic.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(messages))
	}
	request := messages[0].(ai.ModelRequest)
	systemPrompt := request.Parts[0].(ai.SystemPromptPart)
	userPrompt := request.Parts[1].(ai.UserPromptPart)
	if systemPrompt.Content != "be nice" || systemPrompt.Timestamp.IsZero() || userPrompt.Timestamp.IsZero() {
		t.Fatalf("unexpected prompt metadata: system=%+v user=%+v", systemPrompt, userPrompt)
	}
	response := messages[1].(ai.ModelResponse)
	if response.Usage.InputTokens != 10 || response.Usage.OutputTokens != 5 {
		t.Fatalf("unexpected usage %+v", response.Usage)
	}
	call := response.ToolCalls()[0]
	if call.ToolName != "search" || call.ToolCallID != "c1" || string(call.Args) != `{"q": "go"}` {
		t.Fatalf("unexpected tool call %+v", call)
	}
	toolReturn := messages[2].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if toolReturn.Outcome != ai.ToolReturnOutcomeSuccess || toolReturn.Timestamp.IsZero() {
		t.Fatalf("unexpected tool return %+v", toolReturn)
	}
	if messages[3].(ai.ModelResponse).Text() != "done" {
		t.Fatalf("unexpected final response %+v", messages[3])
	}
}

func TestUnmarshalUpstreamMultimodalMessageFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_multimodal.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	prompt := messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
	if len(prompt.Contents) != 3 {
		t.Fatalf("expected 3 content items, got %+v", prompt.Contents)
	}
	if prompt.Contents[0].(ai.TextContent).Text != "look" {
		t.Fatalf("unexpected text content %+v", prompt.Contents[0])
	}
	if prompt.Contents[1].(ai.ImageURL).URL != "https://example.com/a.png" {
		t.Fatalf("unexpected image URL %+v", prompt.Contents[1])
	}
	binary := prompt.Contents[2].(ai.BinaryContent)
	if string(binary.Data) != "hi" || binary.MediaType != "image/png" {
		t.Fatalf("unexpected binary content %+v", binary)
	}
}

func TestUnmarshalUpstreamToolSearchFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_tool_search.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	call := messages[0].(ai.ModelResponse).Parts[0].(ai.ToolCallPart)
	parts := messages[1].(ai.ModelRequest).Parts
	returned := parts[0].(ai.ToolReturnPart)
	delta := parts[1].(ai.ToolAvailabilityDeltaPart)
	content := returned.Content.(map[string]any)
	discovered := content["discovered_tools"].([]any)[0].(map[string]any)
	if call.ToolKind != ai.ToolPartKindToolSearch || returned.ToolKind != ai.ToolPartKindToolSearch ||
		discovered["name"] != "github_get_me" || !slices.Equal(delta.ToolsAdded, []string{"github_get_me"}) {
		t.Fatalf("unexpected tool search fixture: %+v", messages)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"tool_kind":"tool-search"`) {
		t.Fatalf("typed tool search identity was not serialized: %s", encoded)
	}
}

func TestUnmarshalUpstreamNativeToolFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_native_tools.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	response := messages[0].(ai.ModelResponse)
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	content := returned.Content.(map[string]any)
	if call.ToolKind != ai.ToolPartKindToolSearch || call.ToolCallID != "search-1" ||
		returned.ToolKind != ai.ToolPartKindToolSearch || returned.Outcome != ai.ToolReturnOutcomeSuccess ||
		content["discovered_tools"].([]any)[0].(map[string]any)["name"] != "weather" {
		t.Fatalf("unexpected native tool fixture: %+v", response)
	}
	cloned := (ai.ModelRequestContext{Messages: messages}).Clone()
	clonedCall := cloned.Messages[0].(ai.ModelResponse).Parts[0].(ai.NativeToolCallPart)
	clonedCall.Args[0] = '['
	clonedCall.ProviderDetails["execution"] = "changed"
	clonedReturn := cloned.Messages[0].(ai.ModelResponse).Parts[1].(ai.NativeToolReturnPart)
	clonedReturn.Content.(map[string]any)["discovered_tools"] = nil
	clonedReturn.Metadata["local"] = false
	clonedReturn.ProviderDetails["execution"] = "changed"
	if string(call.Args) == string(clonedCall.Args) || call.ProviderDetails["execution"] != "server" ||
		content["discovered_tools"] == nil || returned.Metadata["local"] != true ||
		returned.ProviderDetails["execution"] != "server" {
		t.Fatalf("native tool clone aliases source: original=%+v clone=%+v", response, cloned.Messages[0])
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"part_kind":"builtin-tool-call"`) ||
		!strings.Contains(string(encoded), `"part_kind":"builtin-tool-return"`) {
		t.Fatalf("native tool identity was not serialized: %s", encoded)
	}

	typed := ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "weather"}}}
	typedMessages := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.NativeToolReturnPart{
		ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, Content: typed,
	}}}}
	typedClone := (ai.ModelRequestContext{Messages: typedMessages}).Clone()
	clonedTyped := typedClone.Messages[0].(ai.ModelResponse).Parts[0].(ai.NativeToolReturnPart).Content.(ai.ToolSearchResult)
	clonedTyped.DiscoveredTools[0].Name = "changed"
	originalTyped := typedMessages[0].(ai.ModelResponse).Parts[0].(ai.NativeToolReturnPart).Content.(ai.ToolSearchResult)
	if originalTyped.DiscoveredTools[0].Name != "weather" {
		t.Fatalf("typed native search result clone aliases source: %+v", originalTyped)
	}
}

func TestUnmarshalUpstreamToolAvailabilityFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_tool_availability.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	part := messages[0].(ai.ModelRequest).Parts[0].(ai.ToolAvailabilityDeltaPart)
	if !slices.Equal(part.ToolsAdded, []string{"secret", "archive"}) || part.ToolCallID != "reveal-1" {
		t.Fatalf("unexpected tool availability fixture: %+v", part)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"tools_added":["secret","archive"]`) {
		t.Fatalf("tool availability was not serialized: %s", encoded)
	}

	aliases, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"tool-availability-delta","added":["legacy"]}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	aliased := aliases[0].(ai.ModelRequest).Parts[0].(ai.ToolAvailabilityDeltaPart)
	if !slices.Equal(aliased.ToolsAdded, []string{"legacy"}) {
		t.Fatalf("legacy added alias was not decoded: %+v", aliased)
	}
}

func TestUnmarshalUpstreamInterruptedMessageFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_interrupted.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	request := messages[0].(ai.ModelRequest)
	part := request.Parts[0].(ai.ToolReturnPart)
	if request.State != ai.RequestStateInterrupted || part.Outcome != ai.ToolReturnOutcomeInterrupted {
		t.Fatalf("unexpected interrupted history: %+v %+v", request, part)
	}
}

func TestUnmarshalUpstreamSynthesizedReturnFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_synthesized.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	part := messages[0].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if part.Metadata[ai.SynthesizedToolReturnMetadataKey] != true ||
		part.Outcome != ai.ToolReturnOutcomeInterrupted {
		t.Fatalf("unexpected synthesized return fixture: %+v", part)
	}
}

func TestUnmarshalUpstreamRetryErrorsFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_retry_errors.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	part := messages[0].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
	if part.Content != "" || len(part.Errors) != 2 || part.Errors[0].Type != "string_type" ||
		part.Errors[0].Location[1] != float64(0) || part.Errors[1].Context["gt"] != float64(0) ||
		part.Timestamp.IsZero() || !strings.Contains(part.ModelResponse(), "2 validation errors") {
		t.Fatalf("unexpected structured retry: %+v", part)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"type":"greater_than"`) ||
		!strings.Contains(string(encoded), `"timestamp":"2026-01-03T04:05:06Z"`) {
		t.Fatalf("structured retry was not serialized: %s", encoded)
	}
}

func TestUnmarshalUpstreamDynamicSystemPromptFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_dynamic_system_prompt.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	part := messages[0].(ai.ModelRequest).Parts[0].(ai.SystemPromptPart)
	if part.Content != "Policy tenant-a" || part.DynamicRef != "tenant-policy" || part.Timestamp.IsZero() {
		t.Fatalf("unexpected dynamic system prompt: %+v", part)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"dynamic_ref":"tenant-policy"`) ||
		!strings.Contains(string(encoded), `"timestamp":"2026-01-02T03:04:05Z"`) {
		t.Fatalf("dynamic system prompt metadata was not serialized: %s", encoded)
	}
}

func TestUnmarshalUpstreamPartMetadataFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_part_metadata.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	parts := messages[0].(ai.ModelResponse).Parts
	text := parts[0].(ai.TextPart)
	thinking := parts[1].(ai.ThinkingPart)
	call := parts[2].(ai.ToolCallPart)
	if text.ID != "text-1" || text.ProviderName != "fixture" || text.ProviderDetails["phase"] != "final" ||
		thinking.ID != "thinking-1" || thinking.Signature != "signature" ||
		thinking.ProviderDetails["encrypted"] != true || call.ID != "tool-1" ||
		call.ProviderName != "fixture" || call.ProviderDetails["namespace"] != "ns" {
		t.Fatalf("unexpected part metadata fixture: %+v", parts)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"signature":"signature"`) ||
		!strings.Contains(string(encoded), `"provider_details":{"namespace":"ns"}`) {
		t.Fatalf("part metadata was not serialized: %s", encoded)
	}
}

func TestUnmarshalUpstreamResponseMetadataFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_response_metadata.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	request := messages[0].(ai.ModelRequest)
	response := messages[1].(ai.ModelResponse)
	if request.RunID != "run-1" || request.ConversationID != "conversation-1" ||
		request.Metadata["request"] != true || response.ProviderName != "fixture-provider" ||
		response.ProviderURL != "https://provider.example" || response.ProviderDetails["tier"] != "fast" ||
		response.ProviderResponseID != "response-1" || response.FinishReason != ai.FinishReasonStop ||
		response.RunID != "run-1" || response.ConversationID != "conversation-1" ||
		response.Metadata["response"] != true || response.State != ai.ModelResponseStateComplete {
		t.Fatalf("unexpected response metadata fixture: %+v", messages)
	}
}

func TestUnmarshalUpstreamInstructionsFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_instructions.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	request := messages[0].(ai.ModelRequest)
	if request.Instructions != "Be concise." {
		t.Fatalf("unexpected persisted instructions: %+v", request)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"instructions":"Be concise."`) {
		t.Fatalf("instructions were not serialized: %s", encoded)
	}
}

func TestUnmarshalUpstreamUsageDetailsFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_usage_details.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	response := messages[0].(ai.ModelResponse)
	if response.Usage.InputTokens != 10 || response.Usage.OutputTokens != 4 ||
		response.Usage.Details["provider_units"] != 7 {
		t.Fatalf("unexpected detailed usage fixture: %+v", response.Usage)
	}
}

func TestMarshalMultimodalMessageUsesUpstreamDiscriminators(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "look"},
			ai.ImageURL{URL: "https://example.com/a.png"},
			ai.BinaryContent{Data: []byte("hi"), MediaType: "image/png"},
		}},
	}}}
	data, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	var wire []struct {
		Parts []struct {
			Content []map[string]any `json:"content"`
		} `json:"parts"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	content := wire[0].Parts[0].Content
	if content[0]["kind"] != "text-content" || content[0]["content"] != "look" {
		t.Fatalf("unexpected text wire format %v", content[0])
	}
	if content[1]["kind"] != "image-url" || content[1]["url"] != "https://example.com/a.png" {
		t.Fatalf("unexpected image wire format %v", content[1])
	}
	if content[2]["kind"] != "binary" || content[2]["data"] != "aGk=" || content[2]["media_type"] != "image/png" {
		t.Fatalf("unexpected binary wire format %v", content[2])
	}
}
