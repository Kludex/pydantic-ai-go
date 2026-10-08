package openrouter_test

import (
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/models/openrouter"
)

func TestDownstreamCacheProfiles(t *testing.T) {
	for _, test := range []struct {
		name string
		want time.Duration
	}{
		{name: "unqualified"},
		{name: "google/gemini-3", want: 5 * time.Minute},
		{name: "anthropic/claude-sonnet-5-5", want: 5 * time.Minute},
		{name: "~openai/gpt-5.6", want: 30 * time.Minute},
		{name: "openai/gpt-6-astra", want: 30 * time.Minute},
		{name: "openai/gpt-6-sol", want: 30 * time.Minute},
		{name: "openai/gpt-6-luna", want: 30 * time.Minute},
		{name: "openai/gpt-6.1-sol", want: 30 * time.Minute},
		{name: "openai/gpt-5"},
	} {
		if got := openrouter.NewModel(test.name).ModelProfile().DefaultCacheRetention; got != test.want {
			t.Fatalf("%s retention=%s want=%s", test.name, got, test.want)
		}
	}
}
