package ai_test

import (
	"context"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

type templateDeps struct {
	Name  string
	Facts map[string]any
}

func TestPromptTemplateRendersAgentInstructions(t *testing.T) {
	prompt, err := ai.ParsePromptTemplate[templateDeps]("Hello {{.Name}}. Facts: {{xml .Facts}}")
	if err != nil {
		t.Fatal(err)
	}
	if prompt.String() != "Hello {{.Name}}. Facts: {{xml .Facts}}" {
		t.Fatalf("unexpected source %q", prompt.String())
	}
	values := templateDeps{Name: "Ada", Facts: map[string]any{"role": "admin"}}
	want := "Hello Ada. Facts: <role>admin</role>"
	rendered, err := prompt.Render(values)
	if err != nil || rendered != want {
		t.Fatalf("unexpected rendering %q: %v", rendered, err)
	}
	rendered, err = prompt.Instructions(context.Background(), &ai.RunContext[templateDeps]{Deps: values})
	if err != nil || rendered != want {
		t.Fatalf("unexpected instructions %q: %v", rendered, err)
	}
	rendered, err = prompt.Description(context.Background(), values)
	if err != nil || rendered != want {
		t.Fatalf("unexpected description %q: %v", rendered, err)
	}

}

func TestPromptTemplateFailures(t *testing.T) {
	if _, err := ai.ParsePromptTemplate[map[string]string]("{{"); err == nil {
		t.Fatal("expected parse error")
	}
	prompt := ai.MustParsePromptTemplate[map[string]string]("{{.missing}}")
	if _, err := prompt.Render(map[string]string{}); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("expected missing-key error, got %v", err)
	}
	var nilPrompt *ai.PromptTemplate[map[string]string]
	if nilPrompt.String() != "" {
		t.Fatal("nil template must have an empty source")
	}
	if _, err := nilPrompt.Render(nil); err == nil {
		t.Fatal("expected nil-template error")
	}

	defer func() {
		if recover() == nil {
			t.Fatal("expected MustParsePromptTemplate panic")
		}
	}()
	ai.MustParsePromptTemplate[struct{}]("{{")
}
