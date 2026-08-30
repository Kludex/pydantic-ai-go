package schema

import (
	"strings"
	"testing"
)

func TestCompileRejectsInvalidResourceLocation(t *testing.T) {
	_, err := compileAt(map[string]any{"type": "object"}, "://invalid")
	if err == nil || !strings.Contains(err.Error(), "add schema resource") {
		t.Fatalf("unexpected resource error: %v", err)
	}
}
