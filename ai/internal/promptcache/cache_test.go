package promptcache

import (
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestRaiseEarlierCacheTTLs(t *testing.T) {
	cases := []struct {
		name string
		in   []CacheTTL
		want []CacheTTL
	}{
		{name: "empty", in: nil, want: nil},
		{
			name: "all five minutes",
			in:   []CacheTTL{CacheTTL5Minutes, CacheTTL5Minutes, CacheTTL5Minutes},
			want: []CacheTTL{CacheTTL5Minutes, CacheTTL5Minutes, CacheTTL5Minutes},
		},
		{
			name: "trailing one hour raises earlier breakpoints",
			in:   []CacheTTL{CacheTTL5Minutes, CacheTTL5Minutes, CacheTTL1Hour},
			want: []CacheTTL{CacheTTL1Hour, CacheTTL1Hour, CacheTTL1Hour},
		},
		{
			name: "leading one hour preserved",
			in:   []CacheTTL{CacheTTL1Hour, CacheTTL5Minutes, CacheTTL5Minutes},
			want: []CacheTTL{CacheTTL1Hour, CacheTTL5Minutes, CacheTTL5Minutes},
		},
		{
			name: "intermediate one hour widens each prefix it ends",
			in: []CacheTTL{
				CacheTTL5Minutes, CacheTTL1Hour, CacheTTL5Minutes,
				CacheTTL5Minutes, CacheTTL1Hour, CacheTTL5Minutes,
			},
			want: []CacheTTL{
				CacheTTL1Hour, CacheTTL1Hour, CacheTTL1Hour,
				CacheTTL1Hour, CacheTTL1Hour, CacheTTL5Minutes,
			},
		},
		{
			name: "single one hour",
			in:   []CacheTTL{CacheTTL1Hour},
			want: []CacheTTL{CacheTTL1Hour},
		},
		{
			name: "single five minutes",
			in:   []CacheTTL{CacheTTL5Minutes},
			want: []CacheTTL{CacheTTL5Minutes},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := RaiseEarlierCacheTTLs(tc.in)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("RaiseEarlierCacheTTLs(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestBedrockRetentionsClaudeHaiku55(t *testing.T) {
	for _, name := range []string{
		"claude-haiku-5-5", "anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-5-5",
	} {
		got := BedrockRetentions(name)
		if len(got) != 2 || got[1] != ai.CacheRetention1Hour {
			t.Fatalf("BedrockRetentions(%q) = %v, want [5m 1h]", name, got)
		}
	}
}
