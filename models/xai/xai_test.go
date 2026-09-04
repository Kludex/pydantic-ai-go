package xai_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/models/xai"
)

func TestModelRequest(t *testing.T) {
	from := time.Date(2026, time.January, 2, 12, 0, 0, 0, time.UTC)
	to := time.Date(2026, time.February, 3, 12, 0, 0, 0, time.UTC)
	maximum := 4
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses" || request.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("unexpected request: %s %#v", request.URL, request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		tools := body["tools"].([]any)
		byType := map[string]map[string]any{}
		for _, value := range tools {
			tool := value.(map[string]any)
			byType[tool["type"].(string)] = tool
		}
		xSearch := byType["x_search"]
		collections := byType["collections_search"]
		if xSearch["from_date"] != "2026-01-02" || xSearch["to_date"] != "2026-02-03" ||
			len(xSearch["allowed_x_handles"].([]any)) != 1 || xSearch["enable_video_understanding"] != true {
			t.Errorf("unexpected X search tool: %#v", xSearch)
		}
		if collections["collection_ids"].([]any)[0] != "collection" ||
			collections["max_num_results"] != float64(maximum) || collections["retrieval_mode"] != "hybrid" {
			t.Errorf("unexpected collections tool: %#v", collections)
		}
		includes := body["include"].([]any)
		if len(includes) != 7 || body["seed"] != float64(7) || body["stop"].([]any)[0] != "stop" ||
			body["logprobs"] != true || body["top_logprobs"] != float64(3) || body["agent_count"] != float64(4) {
			t.Errorf("unexpected settings: %#v", body)
		}
		input := body["input"].([]any)
		var content []any
		for _, item := range input {
			message := item.(map[string]any)
			if message["role"] == "user" {
				content, _ = message["content"].([]any)
			}
		}
		if len(content) != 3 || content[2].(map[string]any)["file_id"] != "file-1" {
			t.Errorf("uploaded file was not rendered: %#v", input)
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(response, `{
			"id":"resp-1","model":"grok-4.3","created_at":100,"status":"completed",
			"usage":{"input_tokens":5,"output_tokens":7,"output_tokens_details":{"reasoning_tokens":2}},
			"output":[
				{"id":"reason","type":"reasoning","encrypted_content":"signature","summary":[{"text":"thinking"}]},
				{"id":"x-1","type":"x_search_call","status":"completed","action":{"query":"Go"},"output":{"citations":["https://x.com/post"]}},
				{"id":"c-1","type":"collections_search_call","status":"completed","queries":["docs"],"results":[{"text":"answer"}]},
				{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}
			]
		}`)
	}))
	defer server.Close()

	logprobs := true
	topLogprobs := 3
	store := false
	settings, err := (xai.Settings{
		Common:   ai.ModelSettings{Seed: intPointer(7), StopSequences: []string{"stop"}},
		Logprobs: &logprobs, TopLogprobs: &topLogprobs, User: "user", StoreMessages: &store,
		PreviousResponseID: "previous", IncludeEncryptedContent: true,
		IncludeCodeExecutionOutput: true, IncludeWebSearchOutput: true, IncludeInlineCitations: true,
		IncludeMCPOutput: true, IncludeXSearchOutput: true, IncludeCollectionsSearchOutput: true,
		IncludeAttachmentSearchOutput: true, ReasoningEffort: xai.ReasoningEffortMedium,
		MaxTurns: 5, AgentCount: 4,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := xai.NewModel(
		"grok-4.3", xai.WithBaseURL(server.URL), xai.WithAPIKey("key"),
		xai.WithHTTPClient(server.Client()), xai.WithDefaultSettings(ai.ModelSettings{MaxTokens: 10}),
	)
	result, err := model.Request(t.Context(), []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "Earlier"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "System"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "search"},
				ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
				ai.UploadedFile{FileID: "file-1", ProviderName: "xai", MediaType: "application/pdf"},
			}},
		}},
	}, ai.ModelRequestParams{
		Settings: settings, OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"},
		NativeTools: []ai.NativeTool{
			ai.XSearchTool{
				AllowedXHandles: []string{"pydantic"},
				FromDate:        &from, ToDate: &to, EnableImageUnderstanding: true,
				EnableVideoUnderstanding: true, IncludeOutput: true,
			},
			ai.FileSearchTool{
				FileStoreIDs: []string{"collection"}, MaxNumResults: &maximum,
				Instructions: "Use docs", RetrievalMode: ai.FileSearchRetrievalHybrid,
			},
			ai.WebSearchTool{BlockedDomains: []string{"spam.example"}}, ai.CodeExecutionTool{},
			ai.MCPServerTool{ID: "docs", URL: "https://example.com/mcp"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderName != "xai" || result.ProviderResponseID != "resp-1" || result.Text() != "done" ||
		result.Usage.ReasoningTokens != 2 || len(result.Parts) != 6 {
		t.Fatalf("unexpected response: %#v", result)
	}
	xCall := result.Parts[1].(ai.NativeToolCallPart)
	xReturn := result.Parts[2].(ai.NativeToolReturnPart)
	if xCall.ToolKind != ai.ToolPartKindXSearch || xReturn.ToolKind != ai.ToolPartKindXSearch ||
		xReturn.Content.(map[string]any)["output"] == nil {
		t.Fatalf("unexpected X search parts: %#v %#v", xCall, xReturn)
	}
	if model.Name() != "grok-4.3" || model.ProviderName() != "xai" || model.ProviderURL() != server.URL ||
		model.DefaultModelSettings().MaxTokens != 10 || model.ModelProfile().DefaultOutputMode != ai.OutputModeTool {
		t.Fatalf("unexpected model identity")
	}
}

func TestProviderAndGateway(t *testing.T) {
	t.Setenv("XAI_API_KEY", "environment-key")
	provider := xai.NewProviderConfig()
	if provider.Name != "xai" || provider.BaseURL != "https://api.x.ai/v1" || provider.APIKey != "environment-key" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
	model := xai.NewModel("grok-build-0.1", xai.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: "https://gateway.example/v1", APIKey: "gateway-key",
	}))
	if model.ProviderURL() != "https://gateway.example/v1" || model.ProviderName() != "xai" {
		t.Fatalf("unexpected gateway identity: %q %q", model.ProviderURL(), model.ProviderName())
	}
	if !model.SupportsNativeTool(ai.XSearchTool{}) || model.SupportsNativeTool(ai.ImageGenerationTool{}) {
		t.Fatalf("unexpected native support")
	}
	legacy := xai.NewModel("grok-2")
	if legacy.SupportsNativeTool(ai.WebSearchTool{}) || legacy.SupportsNativeTool(ai.XSearchTool{FromDate: timePointer(time.Now()), ToDate: timePointer(time.Now().Add(-time.Hour))}) {
		t.Fatalf("unexpected legacy native support")
	}
}

func intPointer(value int) *int              { return &value }
func timePointer(value time.Time) *time.Time { return &value }
