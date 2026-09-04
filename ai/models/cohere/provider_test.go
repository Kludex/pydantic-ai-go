package cohere_test

import (
	"net/http"
	"reflect"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/models/cohere"
)

func TestProviderConfigClone(t *testing.T) {
	config := cohere.ProviderConfig{Headers: http.Header{"X-Test": {"one", "two"}}}
	cloned := config.Clone()
	cloned.Headers["X-Test"][0] = "changed"
	if !reflect.DeepEqual(config.Headers, http.Header{"X-Test": {"one", "two"}}) {
		t.Fatalf("provider config was not detached: %#v", config.Headers)
	}
	if (cohere.ProviderConfig{}).Clone().Headers != nil {
		t.Fatal("nil headers became non-nil")
	}
}

func TestAPIError(t *testing.T) {
	err := &cohere.APIError{ProviderName: "cohere", StatusCode: http.StatusBadRequest, Body: `{"message":"bad"}`}
	if got, want := err.Error(), `cohere API returned status 400: {"message":"bad"}`; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}
