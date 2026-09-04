package cli_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"iter"
	"os"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/cli"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type streamModel struct{}

func (streamModel) Name() string { return "stream" }
func (streamModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return nil, errors.New("unexpected request")
}
func (streamModel) StreamRequest(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		yield(ai.TextDeltaEvent{PartID: "text", Delta: "one"}, nil)
		yield(ai.TextDeltaEvent{PartID: "text", Delta: "two"}, nil)
		yield(ai.FinishEvent{}, nil)
	}, nil
}

func TestRunShowsToolsHistoryAndUsage(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{
		Name: "weather", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) { return "sunny", nil })
	output := &bytes.Buffer{}
	err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
		Input: strings.NewReader("hello\n/usage\n/clear\n\n/quit\n"), Output: output, ErrorOutput: output,
		Prompt: "prompt> ", History: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "old"},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	text := output.String()
	for _, expected := range []string{"[tool] weather", "[tool result] weather", "requests=2", "History cleared."} {
		if !strings.Contains(text, expected) {
			t.Fatalf("missing %q in %q", expected, text)
		}
	}
}

func TestRunStructuredOutputAndRetry(t *testing.T) {
	type output struct {
		Answer string `json:"answer"`
	}
	model := fakes.NewTestModel()
	model.CustomOutputArgs = []byte(`{"answer":"yes"}`)
	agent := ai.NewAgent[struct{}, output](model)
	buffer := &bytes.Buffer{}
	if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
		Input: strings.NewReader("answer\n/exit\n"), Output: buffer,
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buffer.String(), `{"answer":"yes"}`) {
		t.Fatalf("structured output missing: %q", buffer.String())
	}

	attempt := 0
	retryAgent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	retryAgent.AddRawTool(ai.ToolDefinition{
		Name: "retry", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) {
		attempt++
		if attempt == 1 {
			return nil, ai.Retryf("try again")
		}
		return "done", nil
	}, ai.WithToolMaxRetries(2))
	buffer.Reset()
	if err := cli.Run(context.Background(), retryAgent, struct{}{}, cli.Config{
		Input: strings.NewReader("run\n/exit\n"), Output: buffer,
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buffer.String(), "retry:") {
		t.Fatalf("retry result missing: %q", buffer.String())
	}

	unprintable := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	unprintable.AddRawTool(ai.ToolDefinition{
		Name: "channel", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) { return make(chan int), nil })
	buffer.Reset()
	if err := cli.Run(context.Background(), unprintable, struct{}{}, cli.Config{
		Input: strings.NewReader("run\n/exit\n"), Output: buffer,
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buffer.String(), "<unprintable>") {
		t.Fatalf("unprintable result missing: %q", buffer.String())
	}

	outputFunction := ai.NewOutputFunction[struct{}, struct {
		Value string `json:"value"`
	}, chan int]("output", func(context.Context, *ai.RunContext[struct{}], struct {
		Value string `json:"value"`
	}) (chan int, error) {
		return make(chan int), nil
	})
	channelAgent := ai.NewOutputFunctionAgent(fakes.NewTestModel(), outputFunction)
	if err := cli.Run(context.Background(), channelAgent, struct{}{}, cli.Config{
		Input: strings.NewReader("run\n"), Output: &bytes.Buffer{},
	}); err == nil || !strings.Contains(err.Error(), "encode output") {
		t.Fatalf("unexpected output encoding error: %v", err)
	}
}

func TestRunErrors(t *testing.T) {
	if err := cli.Run[struct{}, string](context.Background(), nil, struct{}{}, cli.Config{}); err == nil {
		t.Fatal("expected nil agent error")
	}
	missing := t.TempDir() + "/missing.json"
	if err := cli.Run(context.Background(), ai.NewAgent[struct{}, string](fakes.NewTestModel()), struct{}{}, cli.Config{
		Input: strings.NewReader("/exit\n"), MCPConfigPath: missing,
	}); err == nil || !strings.Contains(err.Error(), "load MCP configuration") {
		t.Fatalf("unexpected MCP error: %v", err)
	}
	configPath := t.TempDir() + "/mcp.json"
	if err := os.WriteFile(configPath, []byte(`{"mcpServers":{}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := cli.Run(context.Background(), ai.NewAgent[struct{}, string](fakes.NewTestModel()), struct{}{}, cli.Config{
		Input: strings.NewReader("/exit\n"), Output: &bytes.Buffer{}, MCPConfigPath: configPath,
	}); err != nil {
		t.Fatal(err)
	}

	modelErr := errors.New("model failed")
	agent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, modelErr
	}))
	buffer := &bytes.Buffer{}
	if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
		Input: strings.NewReader("run\n/exit\n"), Output: buffer, ErrorOutput: buffer,
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buffer.String(), "model failed") {
		t.Fatalf("model error missing: %q", buffer.String())
	}
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

type errorWriter struct{ writes int }

func (writer *errorWriter) Write([]byte) (int, error) {
	writer.writes++
	return 0, errors.New("write failed")
}

type failAfterWriter struct {
	writes int
	limit  int
}

func (writer *failAfterWriter) Write(data []byte) (int, error) {
	writer.writes++
	if writer.writes > writer.limit {
		return 0, errors.New("write failed")
	}
	return len(data), nil
}

func TestIOErrors(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	stdin, stdinWriter, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	_ = stdinWriter.Close()
	stdoutReader, stdout, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	stderrReader, stderr, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStdin, oldStdout, oldStderr := os.Stdin, os.Stdout, os.Stderr
	os.Stdin, os.Stdout, os.Stderr = stdin, stdout, stderr
	if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{}); err != nil {
		t.Fatal(err)
	}
	os.Stdin, os.Stdout, os.Stderr = oldStdin, oldStdout, oldStderr
	_ = stdin.Close()
	_ = stdout.Close()
	_ = stdoutReader.Close()
	_ = stderr.Close()
	_ = stderrReader.Close()

	if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
		Input: errorReader{}, Output: &bytes.Buffer{},
	}); err == nil || !strings.Contains(err.Error(), "read prompt") {
		t.Fatalf("unexpected read error: %v", err)
	}
	if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
		Input: strings.NewReader("/exit\n"), Output: &errorWriter{},
	}); err == nil || !strings.Contains(err.Error(), "write prompt") {
		t.Fatalf("unexpected write error: %v", err)
	}
	for _, command := range []string{"/clear\n", "/usage\n"} {
		err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
			Input: strings.NewReader(command), Output: &failAfterWriter{limit: 1},
		})
		if err == nil || !strings.Contains(err.Error(), "write output") {
			t.Fatalf("command=%q: unexpected error %v", command, err)
		}
	}
	if err := cli.Run(context.Background(), ai.NewAgent[struct{}, string](streamModel{}), struct{}{}, cli.Config{
		Input: strings.NewReader("run\n/exit\n"), Output: &bytes.Buffer{},
	}); err != nil {
		t.Fatal(err)
	}
	if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
		Input: strings.NewReader(""), Output: &bytes.Buffer{},
	}); err != nil {
		t.Fatal(err)
	}

	if err := cli.Run(context.Background(), agent, struct{}{}, cli.Config{
		Input: strings.NewReader("run\n"), Output: &failAfterWriter{limit: 1},
	}); err == nil || !strings.Contains(err.Error(), "write stream") {
		t.Fatalf("unexpected stream write error: %v", err)
	}
	textModel := fakes.NewTestModel()
	textModel.CustomOutputText = "text"
	if err := cli.Run(context.Background(), ai.NewAgent[struct{}, string](textModel), struct{}{}, cli.Config{
		Input: strings.NewReader("run\n"), Output: &failAfterWriter{limit: 2},
	}); err == nil || !strings.Contains(err.Error(), "write output") {
		t.Fatalf("unexpected newline write error: %v", err)
	}
	type output struct {
		Answer string `json:"answer"`
	}
	structuredModel := fakes.NewTestModel()
	structuredModel.CustomOutputArgs = []byte(`{"answer":"yes"}`)
	if err := cli.Run(
		context.Background(), ai.NewAgent[struct{}, output](structuredModel), struct{}{}, cli.Config{
			Input: strings.NewReader("run\n"), Output: &failAfterWriter{limit: 1},
		},
	); err == nil || !strings.Contains(err.Error(), "write output") {
		t.Fatalf("unexpected structured write error: %v", err)
	}
	failedAgent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("failed")
	}))
	if err := cli.Run(context.Background(), failedAgent, struct{}{}, cli.Config{
		Input: strings.NewReader("run\n"), Output: &bytes.Buffer{}, ErrorOutput: &errorWriter{},
	}); err == nil || !strings.Contains(err.Error(), "write error") {
		t.Fatalf("unexpected error output failure: %v", err)
	}
}
