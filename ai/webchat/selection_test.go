package webchat_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/webchat"
)

func TestHandlerSelectsModelsAndNativeTools(t *testing.T) {
	var received []ai.NativeTool
	selected := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		received = ai.CloneNativeTools(params.NativeTools)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "selected model"}}}, nil
	})
	defaultModel := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "default model"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](defaultModel)
	handler := newTestHandler(t, agent, webchat.Config{
		Models: []webchat.ModelOption{
			{ID: "selected", Name: "Selected", Model: ai.WrapModel(selected)},
		},
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}, ai.CodeExecutionTool{}},
	})
	body := `{"trigger":"submit-message","id":"chat","model":"selected",` +
		`"builtinTools":["web_search","web_search"],` +
		`"messages":[{"id":"user","role":"user","parts":[{"type":"text","text":"hello"}]}]}`
	response := serve(handler, jsonRequest(http.MethodPost, "/api/chat", body))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "selected model") ||
		len(received) != 1 || received[0].UniqueID() != "web_search" {
		t.Fatalf("selection was not applied: status=%d tools=%#v body=%s", response.Code, received, response.Body.String())
	}

	for _, test := range []struct {
		name string
		body string
		want string
	}{
		{
			name: "unknown model",
			body: `{"trigger":"submit-message","id":"chat","model":"missing","messages":[]}`,
			want: `model \"missing\" is not in the allowed models list`,
		},
		{
			name: "unknown tool",
			body: `{"trigger":"submit-message","id":"chat","builtinTools":["missing"],"messages":[]}`,
			want: `native tool \"missing\" is not in the allowed tools list`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := serve(handler, jsonRequest(http.MethodPost, "/api/chat", test.body))
			if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), test.want) {
				t.Fatalf("unexpected validation response: %d %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestConfigurationReportsPerModelToolSupport(t *testing.T) {
	supported := selectiveModel{name: "supported", supported: "web_search"}
	unknown := namedModel{name: "unknown"}
	handler := newTestHandler(t, ai.NewAgent[struct{}, string](supported), webchat.Config{
		Models: []webchat.ModelOption{
			{ID: "wrapped", Model: ai.WrapModel(supported)},
			{ID: "unknown", Model: unknown},
		},
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}, ai.CodeExecutionTool{}},
	})
	response := serve(handler, request(http.MethodGet, "/api/configure", ""))
	var configuration struct {
		Models []struct {
			ID           string   `json:"id"`
			BuiltinTools []string `json:"builtinTools"`
		} `json:"models"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &configuration); err != nil {
		t.Fatal(err)
	}
	if len(configuration.Models) != 3 || strings.Join(configuration.Models[0].BuiltinTools, ",") != "web_search" ||
		strings.Join(configuration.Models[1].BuiltinTools, ",") != "web_search" ||
		strings.Join(configuration.Models[2].BuiltinTools, ",") != "web_search,code_execution" {
		t.Fatalf("unexpected per-model tools: %#v", configuration.Models)
	}
}

type selectiveModel struct {
	name      string
	supported string
}

func (model selectiveModel) Name() string { return model.name }

func (selectiveModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
}

func (model selectiveModel) SupportsNativeTool(tool ai.NativeTool) bool {
	return tool.UniqueID() == model.supported
}
