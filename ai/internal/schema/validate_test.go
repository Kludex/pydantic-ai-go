package schema_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/internal/schema"
)

func TestValidator(t *testing.T) {
	validator, err := schema.Compile(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count": map[string]any{"type": "integer", "minimum": 2},
			"state": map[string]any{"type": "string", "enum": []string{"ready"}},
			"email": map[string]any{"type": "string", "format": "email"},
		},
		"required":             []string{"count", "state"},
		"additionalProperties": false,
		"allOf":                []any{map[string]any{"type": "object"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"count":2,"state":"ready"}`)); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{
		"constraint":       `{"count":1,"state":"ready"}`,
		"malformed":        `{"count":`,
		"multiple values":  `{"count":2,"state":"ready"} {}`,
		"trailing garbage": `{"count":2,"state":"ready"} x`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := validator.ValidateJSON([]byte(data)); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	if err := validator.Validate(map[string]any{"count": 2, "state": "other"}); err == nil {
		t.Fatal("expected decoded-value validation error")
	}
	if err := validator.Validate(map[string]any{"count": 2, "state": "ready", "email": "bad"}); err == nil {
		t.Fatal("expected format validation error")
	}
}

func TestValidationIssues(t *testing.T) {
	validator, err := schema.Compile(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"a/b~c": map[string]any{"type": "integer"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	err = validator.ValidateJSON([]byte(`{"a/b~c":"invalid"}`))
	issues := schema.ValidationIssues(err)
	if len(issues) != 1 || issues[0].Keyword != "type" || len(issues[0].Location) != 1 ||
		issues[0].Location[0] != "a/b~c" || issues[0].Message == "" {
		t.Fatalf("unexpected validation issues: %+v", issues)
	}
	if issues := schema.ValidationIssues(errors.New("decode failure")); issues != nil {
		t.Fatalf("non-validation error produced issues: %+v", issues)
	}
	required, err := schema.Compile(map[string]any{
		"type": "object", "required": []string{"value"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if issues := schema.ValidationIssues(required.Validate(map[string]any{})); len(issues) != 1 || issues[0].Location != nil || issues[0].Keyword != "required" {
		t.Fatalf("unexpected root validation issue: %+v", issues)
	}
}

func TestCompileValidatorErrors(t *testing.T) {
	_, err := schema.Compile(map[string]any{"type": "not-a-type"})
	if err == nil || !strings.Contains(err.Error(), "compile schema") {
		t.Fatalf("unexpected compile error: %v", err)
	}
	_, err = schema.Compile(map[string]any{"$id": "http://[invalid"})
	if err == nil {
		t.Fatal("expected resource error")
	}
}
