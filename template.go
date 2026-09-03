package ai

import (
	"bytes"
	"context"
	"fmt"
	"text/template"
)

// PromptTemplate is a compiled Go template rendered against run dependencies.
type PromptTemplate[Deps any] struct {
	template *template.Template
	source   string
}

// ParsePromptTemplate compiles source with strict missing-key validation.
// Templates use the standard library text/template syntax and can call the xml
// function to format a value with FormatAsXML defaults.
func ParsePromptTemplate[Deps any](source string) (*PromptTemplate[Deps], error) {
	parsed, err := template.New("prompt").Option("missingkey=error").Funcs(template.FuncMap{
		"xml": func(value any) (string, error) { return FormatAsXML(value) },
	}).Parse(source)
	if err != nil {
		return nil, fmt.Errorf("ai: parse prompt template: %w", err)
	}
	return &PromptTemplate[Deps]{template: parsed, source: source}, nil
}

// MustParsePromptTemplate compiles source and panics on an invalid template.
func MustParsePromptTemplate[Deps any](source string) *PromptTemplate[Deps] {
	prompt, err := ParsePromptTemplate[Deps](source)
	if err != nil {
		panic(err)
	}
	return prompt
}

// String returns the original template source.
func (prompt *PromptTemplate[Deps]) String() string {
	if prompt == nil {
		return ""
	}
	return prompt.source
}

// Render executes the template against deps.
func (prompt *PromptTemplate[Deps]) Render(deps Deps) (string, error) {
	if prompt == nil || prompt.template == nil {
		return "", fmt.Errorf("ai: prompt template is nil")
	}
	var rendered bytes.Buffer
	if err := prompt.template.Execute(&rendered, deps); err != nil {
		return "", fmt.Errorf("ai: render prompt template: %w", err)
	}
	return rendered.String(), nil
}

// Instructions renders the template as an InstructionsFunc.
func (prompt *PromptTemplate[Deps]) Instructions(_ context.Context, rc *RunContext[Deps]) (string, error) {
	return prompt.Render(rc.Deps)
}

// Description renders the template as an AgentDescriptionFunc.
func (prompt *PromptTemplate[Deps]) Description(_ context.Context, deps Deps) (string, error) {
	return prompt.Render(deps)
}
