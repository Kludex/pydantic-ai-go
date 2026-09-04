package snowflake_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/snowflake"
)

func TestExistingFinishReasonAndStreamErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Accept") == "text/event-stream" {
			_, _ = io.WriteString(response, "data: {\n\n")
			return
		}
		_, _ = io.WriteString(response, `{"model":"model","choices":[{"message":{"content":"",
			"tool_calls":[{"id":"call","type":"function","function":{"name":"tool","arguments":"{}"}}]},
			"finish_reason":"tool_calls"}]}`)
	}))
	defer server.Close()
	model := snowflake.NewModel(
		"openai-gpt-5", snowflake.WithBaseURL(server.URL), snowflake.WithToken("token"),
		snowflake.WithHTTPClient(server.Client()),
	)
	result, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || result.FinishReason != ai.FinishReasonToolCall {
		t.Fatalf("unexpected response: %#v %v", result, err)
	}
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var streamErr error
	for _, err := range stream {
		streamErr = err
	}
	if streamErr == nil {
		t.Fatal("expected stream error")
	}
}

func TestStreamSetupErrors(t *testing.T) {
	model, _ := snowflakeValidationModel(t, "mistral-large")
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "tool"}},
	}); err == nil {
		t.Fatal("expected preparation error")
	}
	model = snowflake.NewModel(
		"openai-gpt-5", snowflake.WithBaseURL("://bad"), snowflake.WithToken("token"),
	)
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected stream setup error")
	}
}

func TestStreamConsumerStops(t *testing.T) {
	for _, stopAtFinish := range []bool{false, true} {
		t.Run(map[bool]string{false: "delta", true: "finish"}[stopAtFinish], func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(response,
					`data: {"choices":[{"delta":{"content":"hello"}}]}`+"\n\n"+
						"data: [DONE]\n\n")
			}))
			defer server.Close()
			model := snowflake.NewModel(
				"openai-gpt-5", snowflake.WithBaseURL(server.URL), snowflake.WithToken("token"),
				snowflake.WithHTTPClient(server.Client()),
			)
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			stopped := false
			stream(func(event ai.ModelStreamEvent, eventErr error) bool {
				if eventErr != nil {
					t.Fatal(eventErr)
				}
				_, finish := event.(ai.FinishEvent)
				if finish == stopAtFinish {
					stopped = true
					return false
				}
				return true
			})
			if !stopped {
				t.Fatal("stream did not stop")
			}
		})
	}
}
