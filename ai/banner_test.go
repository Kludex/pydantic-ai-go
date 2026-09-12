package ai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type bannerOutputWithANameLongEnoughToRequireTruncation struct {
	Answer string `json:"answer"`
}

type bannerModel struct {
	*fakes.TestModel
	name string
}

func (model bannerModel) Name() string { return model.name }

type agentBannerCapability struct{}

func (agentBannerCapability) Setup(*ai.CapabilityRegistry) error { return nil }

type runBannerCapability struct{}

func (runBannerCapability) Setup(*ai.CapabilityRegistry) error { return nil }

type identifiedBannerCapability struct{ id string }

func (capability identifiedBannerCapability) Setup(*ai.CapabilityRegistry) error { return nil }
func (capability identifiedBannerCapability) CapabilityID() string               { return capability.id }

func TestFirstRunBanner(t *testing.T) {
	oldArgs := os.Args
	os.Args = []string{"banner-app"}
	t.Cleanup(func() { os.Args = oldArgs })
	for _, name := range []string{
		"CI", "PYDANTIC_AI_NO_BANNER", "FORCE_COLOR", "NO_COLOR", "AI_AGENT", "AGENT", "PI_CODING_AGENT",
	} {
		unsetenv(t, name)
	}

	probe := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	suppressed := &bytes.Buffer{}
	t.Setenv("CI", "")
	probe.WriteBanner(suppressed)
	unsetenv(t, "CI")
	t.Setenv("PYDANTIC_AI_NO_BANNER", "")
	probe.WriteBanner(suppressed)
	unsetenv(t, "PYDANTIC_AI_NO_BANNER")
	if suppressed.Len() != 0 {
		t.Fatalf("suppressed banner was written: %q", suppressed.String())
	}

	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStderr := os.Stderr
	os.Stderr = writer
	t.Cleanup(func() { os.Stderr = oldStderr })

	ai.SetBannerEnabled(false)
	if ai.BannerEnabled() {
		t.Fatal("banner remained enabled")
	}
	if _, err := ai.NewAgent[struct{}, string](fakes.NewTestModel()).Run(
		context.Background(), "hello", struct{}{},
	); err != nil {
		t.Fatal(err)
	}
	ai.SetBannerEnabled(true)
	if !ai.BannerEnabled() {
		t.Fatal("banner remained disabled")
	}

	instrumented := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithCapabilities(ai.NewInstrumentation()),
	)
	if _, err := instrumented.Run(context.Background(), "hello", struct{}{}); err != nil {
		t.Fatal(err)
	}

	probe.WriteBanner(nil)
	withCapability := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithCapabilities(agentBannerCapability{}),
	)
	withCapability.WriteBanner(nil)
	t.Setenv("AI_AGENT", "1")
	probe.WriteBanner(nil)
	t.Setenv("AI_AGENT", "test-harness")
	probe.WriteBanner(nil)
	unsetenv(t, "AI_AGENT")
	t.Setenv("COPILOT_TEST", "1")
	probe.WriteBanner(nil)
	unsetenv(t, "COPILOT_TEST")
	t.Setenv("REPLIT_MODE", "assistant")
	probe.WriteBanner(nil)
	unsetenv(t, "REPLIT_MODE")

	modelLess := ai.NewAgent[struct{}, string](nil)
	modelLess.WriteBanner(nil)
	probe.WriteBanner(nil, ai.WithRunModel(fakes.NewTestModel()), ai.WithRunModelID("conflict"))
	probe.WriteBanner(nil, ai.WithRunModelSelector(func(
		context.Context, ai.ModelSelectionContext[struct{}],
	) (ai.ModelSelection, error) {
		return ai.ModelSelection{}, nil
	}))
	duplicates := ai.WithRunCapabilities(
		identifiedBannerCapability{id: "duplicate"}, identifiedBannerCapability{id: "duplicate"},
	)
	probe.WriteBanner(nil, ai.WithRunModel(fakes.NewTestModel()), duplicates)
	probe.WriteBanner(nil, ai.WithRunModelID("application-model"), duplicates)
	replaceable := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithCapabilities(identifiedBannerCapability{id: "replaceable"}),
	)
	replaceable.WriteBanner(nil, ai.WithRunCapabilities(identifiedBannerCapability{id: "replaceable"}))

	t.Setenv("PI_CODING_AGENT", "1")
	t.Setenv("FORCE_COLOR", "1")
	fake := fakes.NewTestModel()
	fake.CustomOutputArgs = json.RawMessage(`{"answer":"yes"}`)
	model := bannerModel{
		TestModel: fake,
		name:      "bedrock:arn:aws:bedrock:us-east-1:123456789012:inference-profile/example-very-long-model-name",
	}
	agent := ai.NewAgent[struct{}, bannerOutputWithANameLongEnoughToRequireTruncation](
		model,
		ai.WithAgentName("support-agent-with-a-name-long-enough-to-wrap-banner-details"),
		ai.WithCapabilities(agentBannerCapability{}),
	)
	agent.AddRawTool(ai.ToolDefinition{
		Name: "ping", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) { return "pong", nil })
	if _, err := agent.Run(
		context.Background(), "hello", struct{}{}, ai.WithRunCapabilities(runBannerCapability{}),
	); err != nil {
		t.Fatal(err)
	}
	agent.WriteBanner(writer)
	if _, err := ai.NewAgent[struct{}, string](fakes.NewTestModel()).Run(
		context.Background(), "hello", struct{}{},
	); err != nil {
		t.Fatal(err)
	}

	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	output, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	banner := string(output)
	if !strings.Contains(banner, "\x1b[35m") || !strings.Contains(banner, "\x1b[32m") {
		t.Fatalf("terminal banner is missing color: %q", banner)
	}
	plainBanner := strings.NewReplacer("\x1b[35m", "", "\x1b[32m", "", "\x1b[0m", "").Replace(banner)
	for _, expected := range []string{
		"pydantic-ai-go", "Go go", "agent: support-agent-with-a-name-long-enough-to-wrap-banner-details",
		"model: bedrock:arn", "example-very-long-model-name", "output: bannerOutputWithANameLongEnough",
		"tools: 1", "capabilities: 2", "observability: off", "docs/observability.md", "PYDANTIC_AI_NO_BANNER=1",
	} {
		if !strings.Contains(plainBanner, expected) {
			t.Fatalf("banner is missing %q:\n%s", expected, plainBanner)
		}
	}
	if strings.Count(plainBanner, "`---.._|_..---'") != 1 {
		t.Fatalf("banner was not written exactly once:\n%s", plainBanner)
	}
	for _, line := range strings.Split(plainBanner, "\n") {
		if len(line) > 100 {
			t.Fatalf("banner line exceeds 100 columns: %q", line)
		}
	}
}

func unsetenv(t *testing.T, name string) {
	t.Helper()
	value, exists := os.LookupEnv(name)
	if err := os.Unsetenv(name); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if exists {
			_ = os.Setenv(name, value)
		} else {
			_ = os.Unsetenv(name)
		}
	})
}
