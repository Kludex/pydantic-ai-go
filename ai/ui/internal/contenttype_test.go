package internal

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCheckContentTypeDefaultsToJSON(t *testing.T) {
	allowed := DefaultAllowedContentTypes
	if len(allowed) != 1 || allowed[0] != "application/json" {
		t.Fatalf("default allowed types wrong: %v", allowed)
	}
}

func TestCheckContentTypeAccepts(t *testing.T) {
	for _, raw := range []string{
		"application/json",
		"application/json; charset=utf-8",
		"APPLICATION/JSON",
	} {
		request := httptest.NewRequest(http.MethodPost, "/", nil)
		request.Header.Set("Content-Type", raw)
		if err := CheckContentType(request, nil); err != nil {
			t.Fatalf("nil allowlist should default to JSON, but rejected %q: %v", raw, err)
		}
	}
}

func TestCheckContentTypeRejects(t *testing.T) {
	for _, raw := range []string{
		"",
		"text/plain",
		"application/octet-stream",
		"application/xml",
		"text/plain; charset=utf-8",
	} {
		request := httptest.NewRequest(http.MethodPost, "/", nil)
		if raw != "" {
			request.Header.Set("Content-Type", raw)
		}
		err := CheckContentType(request, nil)
		var typeErr *ContentTypeError
		if !errors.As(err, &typeErr) {
			t.Fatalf("expected ContentTypeError for %q, got %v", raw, err)
		}
		if typeErr.Got == "" {
			t.Fatalf("expected a populated Got value for %q", raw)
		}
	}
}

func TestCheckContentTypeSkippedOnExplicitEmpty(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/", nil)
	request.Header.Set("Content-Type", "text/plain")
	if err := CheckContentType(request, []string{}); err != nil {
		t.Fatalf("explicit empty allowlist should skip the check, got %v", err)
	}
}

func TestCheckContentTypeCustomAllowlist(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/", nil)
	request.Header.Set("Content-Type", "text/plain; charset=utf-8")
	if err := CheckContentType(request, []string{" TEXT/PLAIN "}); err != nil {
		t.Fatalf("normalised allowlist should accept text/plain, got %v", err)
	}
	if err := CheckContentType(request, []string{"application/json"}); err == nil {
		t.Fatalf("expected rejection when allowlist excludes the request")
	}
}

func TestCheckContentTypeSkippedWhenAllowlistNormalisesEmpty(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/", nil)
	request.Header.Set("Content-Type", "application/json")
	if err := CheckContentType(request, []string{" ", "", "\t"}); err != nil {
		t.Fatalf("allowlist of only blank entries should skip, got %v", err)
	}
}

func TestCheckContentTypeHandlesMalformedHeader(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/", nil)
	request.Header.Set("Content-Type", ";;;")
	err := CheckContentType(request, []string{"application/json"})
	var typeErr *ContentTypeError
	if !errors.As(err, &typeErr) {
		t.Fatalf("expected rejection on malformed header, got %v", err)
	}
	if typeErr.Got != "no content type" {
		t.Fatalf("malformed header should fall back to empty got, got %q", typeErr.Got)
	}
}

func TestContentTypeErrorMessage(t *testing.T) {
	err := &ContentTypeError{Got: "text/plain", Expected: []string{"application/json"}}
	if msg := err.Error(); !strings.Contains(msg, "application/json") || !strings.Contains(msg, "text/plain") {
		t.Fatalf("error message missing expected detail: %q", msg)
	}
}

func TestCheckContentTypeNormalisesAllowlistParams(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/", nil)
	request.Header.Set("Content-Type", "application/json; charset=utf-8")
	if err := CheckContentType(request, []string{" Application/Json ; charset=utf-8 "}); err != nil {
		t.Fatalf("allowlist with parameters should normalise, got %v", err)
	}
	if err := CheckContentType(request, []string{";;;invalid"}); err == nil {
		t.Fatalf("invalid allowlist entries should still require match")
	}
}
