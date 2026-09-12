package ai

import (
	"io"
	"os"
	"strings"
)

var bannerCodingAgents = []struct {
	name    string
	signals []string
}{
	{name: "claude-code", signals: []string{"CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_IS_COWORK"}},
	{name: "codex", signals: []string{"CODEX_THREAD_ID", "CODEX_CI", "CODEX_SANDBOX"}},
	{name: "gemini-cli", signals: []string{"GEMINI_CLI", "GEMINI_AGENT"}},
	{name: "cursor", signals: []string{"CURSOR_AGENT"}},
	{name: "opencode", signals: []string{
		"OPENCODE", "OPENCODE_BIN_PATH", "OPENCODE_SERVER", "OPENCODE_APP_INFO", "OPENCODE_MODES", "OPENCODE_CLIENT",
	}},
	{name: "pi", signals: []string{"PI_CODING_AGENT"}},
	{name: "amp", signals: []string{"AMP_CURRENT_THREAD_ID"}},
	{name: "augment", signals: []string{"AUGMENT_AGENT"}},
	{name: "antigravity", signals: []string{"ANTIGRAVITY_AGENT", "ANTIGRAVITY_PROJECT_ID"}},
	{name: "crush", signals: []string{"CRUSH"}},
	{name: "qwen-code", signals: []string{"QWEN_CODE"}},
	{name: "windsurf", signals: []string{"CODEIUM_EDITOR_APP_ROOT"}},
	{name: "warp", signals: []string{"OZ_RUN_ID"}},
	{name: "copilot", signals: []string{"COPILOT_*", "GITHUB_COPILOT*"}},
	{name: "aider", signals: []string{"AIDER_*"}},
	{name: "replit", signals: []string{"REPLIT_MODE=assistant"}},
	{name: "swe-agent", signals: []string{"SWE_AGENT"}},
}

func runningUnderGoTest() bool {
	return len(os.Args) > 0 && strings.HasSuffix(strings.TrimSuffix(os.Args[0], ".exe"), ".test")
}

func environmentSet(name string) bool {
	_, ok := os.LookupEnv(name)
	return ok
}

func writerIsTerminal(writer io.Writer, honorForceColor bool) bool {
	file, ok := writer.(*os.File)
	terminal := false
	if ok {
		info, err := file.Stat()
		terminal = err == nil && info.Mode()&os.ModeCharDevice != 0
	}
	return terminal || honorForceColor && environmentSet("FORCE_COLOR")
}

func detectCodingAgent() string {
	for _, agent := range bannerCodingAgents {
		for _, signal := range agent.signals {
			if bannerAgentSignalMatches(signal) {
				return agent.name
			}
		}
	}
	for _, name := range []string{"AI_AGENT", "AGENT"} {
		if value := os.Getenv(name); value != "" {
			switch strings.ToLower(value) {
			case "1", "true", "yes":
				return "agent"
			default:
				return value
			}
		}
	}
	return ""
}

func bannerAgentSignalMatches(signal string) bool {
	if prefix, ok := strings.CutSuffix(signal, "*"); ok {
		for _, entry := range os.Environ() {
			if strings.HasPrefix(entry, prefix) {
				return true
			}
		}
		return false
	}
	name, value, exact := strings.Cut(signal, "=")
	actual, present := os.LookupEnv(name)
	return present && (!exact || actual == value)
}
