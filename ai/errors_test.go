package ai_test

import (
	"errors"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestModelRateLimitErrorFormatting(t *testing.T) {
	withoutRetry := &ai.ModelRateLimitError{
		ModelName: "gpt-5", ProviderName: "openai",
	}
	if !errors.Is(withoutRetry, ai.ErrModelRateLimit) {
		t.Fatalf("expected ErrModelRateLimit, got %v", withoutRetry)
	}
	if got := withoutRetry.Error(); got != ai.ErrModelRateLimit.Error() {
		t.Fatalf("unexpected error %q", got)
	}
	cases := []struct {
		name string
		err  *ai.ModelRateLimitError
		want string
	}{
		{
			name: "retry after matches",
			err: &ai.ModelRateLimitError{
				ModelName: "gpt-5", RetryAfter: 2 * time.Second,
			},
			want: "retry after 2s",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.err.Error(); !contains(got, tc.want) {
				t.Fatalf("got %q want contains %q", got, tc.want)
			}
		})
	}

	wrapped := &ai.ModelRateLimitError{
		RetryAfter: time.Second, Cause: errors.New("429"),
	}
	if !errors.Is(wrapped, ai.ErrModelRateLimit) || wrapped.Error() == "" {
		t.Fatalf("wrapped rate limit error lost identity: %v", wrapped)
	}
	if !errors.Is(wrapped, errors.Unwrap(wrapped)) {
		t.Fatalf("rate limit error did not unwrap to cause")
	}

	var api ai.ModelAPIError
	if !errors.As(wrapped, &api) || !api.IsModelAPIError() {
		t.Fatalf("rate limit error must satisfy ModelAPIError fallback contract")
	}
}

func TestModelContextWindowExceededErrorFormatting(t *testing.T) {
	plain := &ai.ModelContextWindowExceededError{}
	if !errors.Is(plain, ai.ErrModelContextWindowExceeded) {
		t.Fatalf("expected context-window sentinel")
	}
	if plain.Error() != ai.ErrModelContextWindowExceeded.Error() {
		t.Fatalf("unexpected error %q", plain.Error())
	}
	detailed := &ai.ModelContextWindowExceededError{
		ModelName: "gpt-5", ProviderName: "openai",
		Tokens: 9000, MaxTokens: 8000,
	}
	if !contains(detailed.Error(), "9000") || !contains(detailed.Error(), "8000") {
		t.Fatalf("detailed error missing counts: %q", detailed.Error())
	}
	var api ai.ModelAPIError
	if !errors.As(detailed, &api) || !api.IsModelAPIError() {
		t.Fatalf("context-window error must satisfy ModelAPIError fallback contract")
	}
	wrapped := &ai.ModelContextWindowExceededError{Cause: errors.New("boom")}
	if !errors.Is(wrapped, errors.Unwrap(wrapped)) {
		t.Fatalf("context-window error should unwrap to its cause")
	}
}

func TestModelConnectionErrorPhases(t *testing.T) {
	for _, phase := range []ai.ConnectionErrorPhase{
		ai.ConnectionPhasePool, ai.ConnectionPhaseConnect,
		ai.ConnectionPhaseWrite, ai.ConnectionPhaseRead,
	} {
		err := &ai.ModelConnectionError{
			ModelName: "gpt-5", ProviderName: "openai",
			Phase: phase, Cause: errors.New("dial"),
		}
		if !errors.Is(err, ai.ErrModelConnection) {
			t.Fatalf("phase %q did not match ErrModelConnection", phase)
		}
		if err.Error() == "" {
			t.Fatalf("phase %q produced empty error", phase)
		}
	}
	missing := &ai.ModelConnectionError{}
	if !errors.Is(missing, ai.ErrModelConnection) {
		t.Fatalf("missing-phase error lost identity")
	}
	if !contains(missing.Error(), "transport") {
		t.Fatalf("missing-phase error should default to transport, got %q", missing.Error())
	}
	var api ai.ModelAPIError
	if !errors.As(missing, &api) || !api.IsModelAPIError() {
		t.Fatalf("connection error must satisfy ModelAPIError fallback contract")
	}
	streamed := &ai.ModelConnectionError{InStream: true}
	if !errors.Is(streamed, ai.ErrModelConnection) {
		t.Fatalf("in-stream connection error lost identity")
	}
	wrapped := &ai.ModelConnectionError{Cause: errors.New("dial")}
	if !errors.Is(wrapped, errors.Unwrap(wrapped)) {
		t.Fatalf("connection error should unwrap to its cause")
	}
}

func TestUsageLimitUnavailableWarningMatchesIs(t *testing.T) {
	err := ai.UsageLimitUnavailableWarning
	if !errors.Is(err, err) {
		t.Fatalf("UsageLimitUnavailableWarning should match itself")
	}
	if errors.Is(errors.New("other"), ai.UsageLimitUnavailableWarning) {
		t.Fatalf("unrelated error should not match")
	}
}

func TestOtherModelErrorSentinelsExist(t *testing.T) {
	for _, sentinel := range []error{
		ai.ErrModelQuotaExceeded, ai.ErrModelUnavailable, ai.ErrModelOverloaded,
		ai.ErrModelServerError, ai.ErrModelTimeout,
	} {
		if sentinel == nil {
			t.Fatal("nil sentinel in model error categories")
		}
		if !errors.Is(sentinel, sentinel) {
			t.Fatalf("%v did not match itself", sentinel)
		}
	}
}

func contains(haystack, needle string) bool {
	return len(needle) == 0 || len(haystack) >= len(needle) && (haystack == needle ||
		indexOf(haystack, needle) >= 0)
}

func indexOf(haystack, needle string) int {
outer:
	for start := 0; start+len(needle) <= len(haystack); start++ {
		for offset := 0; offset < len(needle); offset++ {
			if haystack[start+offset] != needle[offset] {
				continue outer
			}
		}
		return start
	}
	return -1
}
