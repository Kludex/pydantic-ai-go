package ai

import (
	"os"
	"testing"
)

// TestDetectCodingAgentSignals exercises every signal path in the coding agent detector
// by setting one env var per agent and asserting the returned agent name. The banner
// production path is suppressed under `go test`, so this test reaches the detector directly.
func TestDetectCodingAgentSignals(t *testing.T) {
	t.Setenv("PYDANTIC_AI_NO_BANNER", "1")
	for _, tc := range []struct {
		name   string
		key    string
		value  string
		expect string
	}{
		{name: "claude-code", key: "CLAUDECODE", value: "1", expect: "claude-code"},
		{name: "codex", key: "CODEX_THREAD_ID", value: "1", expect: "codex"},
		{name: "codex-ci", key: "CODEX_CI", value: "1", expect: "codex"},
		{name: "codex-sandbox", key: "CODEX_SANDBOX", value: "1", expect: "codex"},
		{name: "gemini-cli", key: "GEMINI_CLI", value: "1", expect: "gemini-cli"},
		{name: "gemini-agent", key: "GEMINI_AGENT", value: "1", expect: "gemini-cli"},
		{name: "cursor", key: "CURSOR_AGENT", value: "1", expect: "cursor"},
		{name: "opencode", key: "OPENCODE", value: "1", expect: "opencode"},
		{name: "opencode-bin", key: "OPENCODE_BIN_PATH", value: "1", expect: "opencode"},
		{name: "opencode-server", key: "OPENCODE_SERVER", value: "1", expect: "opencode"},
		{name: "opencode-app", key: "OPENCODE_APP_INFO", value: "1", expect: "opencode"},
		{name: "opencode-modes", key: "OPENCODE_MODES", value: "1", expect: "opencode"},
		{name: "opencode-client", key: "OPENCODE_CLIENT", value: "1", expect: "opencode"},
		{name: "pi", key: "PI_CODING_AGENT", value: "1", expect: "pi"},
		{name: "amp", key: "AMP_CURRENT_THREAD_ID", value: "1", expect: "amp"},
		{name: "antigravity", key: "ANTIGRAVITY_AGENT", value: "1", expect: "antigravity"},
		{name: "antigravity-project", key: "ANTIGRAVITY_PROJECT_ID", value: "1", expect: "antigravity"},
		{name: "crush", key: "CRUSH", value: "1", expect: "crush"},
		{name: "qwen-code", key: "QWEN_CODE", value: "1", expect: "qwen-code"},
		{name: "windsurf", key: "CODEIUM_EDITOR_APP_ROOT", value: "1", expect: "windsurf"},
		{name: "warp", key: "OZ_RUN_ID", value: "1", expect: "warp"},
		{name: "copilot-prefix", key: "COPILOT_TEST_VAR", value: "1", expect: "copilot"},
		{name: "copilot-prefix-github", key: "GITHUB_COPILOT_TOKEN", value: "1", expect: "copilot"},
		{name: "aider", key: "AIDER_TEST_VAR", value: "1", expect: "aider"},
		{name: "replit-exact", key: "REPLIT_MODE", value: "assistant", expect: "replit"},
		{name: "swe-agent", key: "SWE_AGENT", value: "1", expect: "swe-agent"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resetAgentEnv(t)
			t.Setenv(tc.key, tc.value)
			if got := detectCodingAgent(); got != tc.expect {
				t.Fatalf("detectCodingAgent()=%q want=%q", got, tc.expect)
			}
		})
	}

	t.Run("replit-mismatch", func(t *testing.T) {
		resetAgentEnv(t)
		t.Setenv("REPLIT_MODE", "no")
		if detectCodingAgent() != "" {
			t.Fatal("REPLIT_MODE=other should not match")
		}
	})

	t.Run("ai-agent-true", func(t *testing.T) {
		resetAgentEnv(t)
		t.Setenv("AI_AGENT", "true")
		if detectCodingAgent() != "agent" {
			t.Fatalf("AI_AGENT=true should match agent")
		}
	})

	t.Run("ai-agent-other", func(t *testing.T) {
		resetAgentEnv(t)
		t.Setenv("AI_AGENT", "harness")
		if detectCodingAgent() != "harness" {
			t.Fatalf("AI_AGENT=harness should match harness")
		}
	})

	t.Run("none", func(t *testing.T) {
		resetAgentEnv(t)
		if detectCodingAgent() != "" {
			t.Fatal("no signal env vars should return empty")
		}
	})

	t.Run("bannerAgentSignalMatches-prefix-miss", func(t *testing.T) {
		resetAgentEnv(t)
		if bannerAgentSignalMatches("CLAUDECODE_*") {
			t.Fatal("CLAUDECODE_* with no env prefix match should be false")
		}
	})
}

// resetAgentEnv unsets every env var the coding agent detector inspects. We rely on
// os.Unsetenv rather than `t.Setenv("", "")` because LookupEnv returns present=true for
// empty values, defeating the detector.
func resetAgentEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_IS_COWORK",
		"CODEX_THREAD_ID", "CODEX_CI", "CODEX_SANDBOX",
		"GEMINI_CLI", "GEMINI_AGENT",
		"CURSOR_AGENT",
		"OPENCODE", "OPENCODE_BIN_PATH", "OPENCODE_SERVER", "OPENCODE_APP_INFO", "OPENCODE_MODES", "OPENCODE_CLIENT",
		"PI_CODING_AGENT", "AMP_CURRENT_THREAD_ID",
		"ANTIGRAVITY_AGENT", "ANTIGRAVITY_PROJECT_ID",
		"CRUSH", "QWEN_CODE", "CODEIUM_EDITOR_APP_ROOT", "OZ_RUN_ID",
		"SWE_AGENT", "REPLIT_MODE",
		"AI_AGENT", "AGENT",
		// Synthetic prefix-var leftovers from previous test runs.
		"COPILOT_TEST_VAR", "GITHUB_COPILOT_TOKEN", "AIDER_TEST_VAR",
	} {
		previous, exists := os.LookupEnv(key)
		if err := os.Unsetenv(key); err != nil {
			t.Fatal(err)
		}
		if exists {
			previous := previous
			key := key
			t.Cleanup(func() { _ = os.Setenv(key, previous) })
		}
	}
}
