package realtime_test

import (
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	azurert "github.com/Kludex/pydantic-ai-go/ai/realtime/azure"
	googlert "github.com/Kludex/pydantic-ai-go/ai/realtime/google"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	xairt "github.com/Kludex/pydantic-ai-go/ai/realtime/xai"
)

func TestBundledRealtimeContextWindows(t *testing.T) {
	azure, err := azurert.NewModel("gpt-4o-realtime-preview", azurert.Config{
		Endpoint: "https://example.openai.azure.com",
		APIKey:   "key",
	})
	if err != nil {
		t.Fatal(err)
	}
	models := []struct {
		model realtime.Model
		want  int
	}{
		{model: openairt.NewModel("gpt-realtime"), want: 32_000},
		{model: openairt.NewModel("gpt-realtime-2"), want: 128_000},
		{model: openairt.NewModel("gpt-realtime-mini"), want: 0},
		{model: googlert.NewModel("gemini-2.5-flash-native-audio-latest"), want: 0},
		{model: xairt.NewModel("grok-voice-latest"), want: 0},
		{model: azure, want: 32_000},
	}
	for _, test := range models {
		if got := test.model.Profile().ContextWindow; got != test.want {
			t.Fatalf("unexpected %s context window: got %d want %d", test.model.Name(), got, test.want)
		}
	}
}

func TestRealtimeContextWindowOverride(t *testing.T) {
	window := 96_000
	model := openairt.NewModel("unknown-model", openairt.WithProfile(realtime.ProfileOverride{ContextWindow: &window}))
	if got := model.Profile().ContextWindow; got != window {
		t.Fatalf("unexpected explicit context window %d", got)
	}

	unknown := 0
	model = openairt.NewModel("gpt-realtime", openairt.WithProfile(realtime.ProfileOverride{ContextWindow: &unknown}))
	if got := model.Profile().ContextWindow; got != 0 {
		t.Fatalf("explicit unknown context window was replaced with %d", got)
	}
}

func TestRealtimeRejectsNegativeContextWindow(t *testing.T) {
	profile := fullProfile()
	profile.ContextWindow = -1
	if _, err := realtime.Open(
		t.Context(), &fakeModel{connection: newFakeConnection(), profile: profile}, realtime.ConnectParams{},
	); err == nil {
		t.Fatal("expected negative context-window error")
	}
}
