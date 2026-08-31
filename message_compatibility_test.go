package ai_test

import (
	"encoding/json"
	"os"
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
	if request.Parts[0].(ai.SystemPromptPart).Content != "be nice" {
		t.Fatalf("unexpected system prompt %+v", request.Parts[0])
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
	if toolReturn.Outcome != ai.ToolReturnOutcomeSuccess {
		t.Fatalf("unexpected tool outcome %q", toolReturn.Outcome)
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
