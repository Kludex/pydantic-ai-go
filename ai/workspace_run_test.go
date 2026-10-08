package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type workspaceProvider struct {
	backend ai.WorkspaceBackend
	err     error
	calls   int
}

func (*workspaceProvider) Setup(*ai.CapabilityRegistry) error { return nil }
func (p *workspaceProvider) GetWorkspace(_ context.Context, info *ai.RunInfo, ref *ai.WorkspaceRef) (ai.WorkspaceBackend, error) {
	p.calls++
	if info.Workspace().Attached() {
		return nil, errors.New("workspace selected twice")
	}
	if p.err != nil {
		return nil, p.err
	}
	if p.backend != nil && (ref == nil || *ref == *p.backend.Ref()) {
		return p.backend, nil
	}
	return nil, nil
}

func TestWorkspaceSelection(t *testing.T) {
	ctx := context.Background()
	_, w, root := localWorkspace(t)
	other, _, otherRoot := localWorkspace(t)
	local, err := ai.NewLocalWorkspace(root, map[string]string{"CONFIG": "original"})
	if err != nil {
		t.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(local))
	first, err := agent.Run(ctx, "one", struct{}{})
	if err != nil || first.Workspace().Ref().ID != root {
		t.Fatalf("local selection: %#v %v", first, err)
	}
	history := first.Messages()
	ref := history[len(history)-1].(ai.ModelResponse).WorkspaceRef
	if ref == nil || *ref != *first.Workspace().Ref() {
		t.Fatalf("response reference: %#v", ref)
	}
	saved, err := ai.MarshalMessages(history)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := ai.UnmarshalMessages(saved)
	if err != nil || *restored[len(restored)-1].(ai.ModelResponse).WorkspaceRef != *ref {
		t.Fatalf("restored ref: %#v %v", restored, err)
	}
	second, err := agent.Run(ctx, "two", struct{}{}, ai.WithMessageHistory(restored))
	if err != nil || *second.Workspace().Ref() != *ref {
		t.Fatalf("historical selection: %#v %v", second, err)
	}
	for _, options := range [][]ai.RunOption{
		{ai.WithRunWorkspaceRef(*ref)},
		{ai.WithNewWorkspace(), ai.WithMessageHistory(restored)},
		{ai.WithRunWorkspace(&ai.Workspace{}), ai.WithMessageHistory(restored)},
	} {
		result, err := agent.Run(ctx, "run", struct{}{}, options...)
		if err != nil || result.Workspace().Ref().ID != root {
			t.Fatalf("local run option: %#v %v", result, err)
		}
	}
	explicit, err := agent.Run(ctx, "different", struct{}{}, ai.WithRunWorkspace(other), ai.WithMessageHistory(restored))
	if err != nil || explicit.Workspace().Ref().ID != otherRoot {
		t.Fatalf("explicit precedence: %#v %v", explicit, err)
	}
	readonly, err := agent.Run(ctx, "review", struct{}{}, ai.WithRunWorkspace(ai.ReadOnlyWorkspace(w)))
	if err != nil || !readonly.Workspace().ReadOnly() {
		t.Fatalf("explicit policy: %#v %v", readonly, err)
	}
	decline := &workspaceProvider{}
	prefer := &workspaceProvider{backend: other}
	chosen, err := agent.Run(ctx, "priority", struct{}{}, ai.WithRunCapabilities(decline, prefer))
	if err != nil || chosen.Workspace().Ref().ID != otherRoot || decline.calls != 1 || prefer.calls != 1 {
		t.Fatalf("run providers first: %#v %v", chosen, err)
	}
	failure := errors.New("provider failed")
	if _, err := agent.Run(ctx, "typed nil", struct{}{}, ai.WithRunCapabilities(&workspaceProvider{backend: (*ai.Workspace)(nil)})); err == nil {
		t.Fatal("typed nil provider accepted")
	}
	if _, err := agent.Run(ctx, "error", struct{}{}, ai.WithRunCapabilities(&workspaceProvider{err: failure})); !errors.Is(err, failure) {
		t.Fatalf("provider error: %v", err)
	}
	for _, options := range [][]ai.RunOption{
		{ai.WithRunWorkspaceRef(*other.Ref())},
		{ai.WithMessageHistory(explicit.Messages())},
	} {
		if _, err := agent.Run(ctx, "wrong directory", struct{}{}, options...); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
			t.Fatalf("redirected local directory: %v", err)
		}
	}
	summary := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	carried, err := summary.Run(ctx, "summarize", struct{}{}, ai.WithMessageHistory(restored))
	if err != nil || carried.Workspace().Attached() || *carried.Messages()[len(carried.Messages())-1].(ai.ModelResponse).WorkspaceRef != *ref {
		t.Fatalf("carried ref: %#v %v", carried, err)
	}
	blocked, err := agent.Run(ctx, "blocked", struct{}{}, ai.WithRunWorkspace(ai.UnavailableWorkspace("disabled")), ai.WithMessageHistory(restored))
	if err != nil || blocked.Workspace().Attached() || *blocked.Messages()[len(blocked.Messages())-1].(ai.ModelResponse).WorkspaceRef != *ref {
		t.Fatalf("explicit unavailable: %#v %v", blocked, err)
	}
	for _, options := range [][]ai.RunOption{
		{ai.WithNewWorkspace()}, {ai.WithRunWorkspaceRef(*ref)},
		{ai.WithNewWorkspace(), ai.WithRunCapabilities(decline)},
	} {
		if _, err := summary.Run(ctx, "unresolved", struct{}{}, options...); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
			t.Fatalf("unresolved ref: %v", err)
		}
	}
	latestNil := append(slices.Clone(restored), ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "new conversation"}}})
	fresh, err := summary.Run(ctx, "no old ref", struct{}{}, ai.WithMessageHistory(latestNil))
	if err != nil || fresh.Messages()[len(fresh.Messages())-1].(ai.ModelResponse).WorkspaceRef != nil {
		t.Fatalf("latest nil did not suppress old ref: %v", err)
	}
	if _, err := summary.Run(ctx, "no workspace", struct{}{}, ai.WithRunCapabilities(
		ai.BeforeRunFunc(func(context.Context, *ai.RunInfo) error { return nil }),
	)); err != nil {
		t.Fatal(err)
	}
	otherLocal, err := ai.NewLocalWorkspace(otherRoot, nil)
	if err != nil {
		t.Fatal(err)
	}
	otherLocal.ReadOnly = true
	overridden, err := agent.Run(ctx, "override", struct{}{}, ai.WithRunCapabilities(otherLocal), ai.WithNewWorkspace())
	if err != nil || overridden.Workspace().Ref().ID != otherRoot || !overridden.Workspace().ReadOnly() {
		t.Fatalf("capability replacement: %#v %v", overridden, err)
	}
	combined := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(local, otherLocal))
	combinedResult, err := combined.Run(ctx, "last", struct{}{})
	if err != nil || !combinedResult.Workspace().ReadOnly() || combinedResult.Workspace().Ref().ID != otherRoot {
		t.Fatalf("whole replacement: %#v %v", combinedResult, err)
	}
	otherLocal.ID = "second_local"
	if otherLocal.CapabilityID() != "second_local" {
		t.Fatal("custom identity ignored")
	}
	if _, err := (&ai.LocalWorkspace{}).GetWorkspace(ctx, nil, nil); err == nil {
		t.Fatal("zero capability accepted")
	}
	if _, err := ai.NewLocalWorkspace("\x00", nil); err == nil {
		t.Fatal("invalid local capability")
	}
}

func TestWorkspaceRunAccessAndPersistence(t *testing.T) {
	ctx := context.Background()
	_, w, _ := localWorkspace(t)
	var uses atomic.Int64
	model := fakes.NewTestModel()
	capability := ai.BeforeRunFunc(func(ctx context.Context, info *ai.RunInfo) error {
		uses.Add(1)
		return info.Workspace().WriteText(ctx, "hook", "ready")
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	ai.AddTool(agent, "write", func(ctx context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		uses.Add(1)
		return "written", rc.Workspace.WriteText(ctx, "tool", "ready")
	})
	driver, err := agent.StartRun(ctx, "write", struct{}{}, ai.WithRunWorkspace(w))
	if err != nil || driver.Workspace().Backend() != w.Backend() {
		t.Fatalf("driver workspace: %#v %v", driver, err)
	}
	for _, err := range driver.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	result, err := driver.Result(), driver.Err()
	if err != nil || result.Workspace().Backend() != w.Backend() || uses.Load() != 2 {
		t.Fatalf("run access: %#v %v uses %d", result, err, uses.Load())
	}
	for _, name := range []string{"hook", "tool"} {
		if text, err := result.Workspace().ReadText(ctx, name); err != nil || text != "ready" {
			t.Fatalf("post-run file: %q %v", text, err)
		}
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ai.RunResult[string]
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Workspace().Attached() || decoded.Messages()[len(decoded.Messages())-1].(ai.ModelResponse).WorkspaceRef == nil {
		t.Fatal("persistence stored live backend or lost reference")
	}
	history := result.Messages()
	sanitized, report, err := ai.SanitizeMessages(history, ai.MessageSanitizationOptions{})
	if err != nil || !report.Changed() || report.StrippedWorkspaceRefs != 2 {
		t.Fatalf("sanitize references: %#v %v", report, err)
	}
	for _, message := range sanitized {
		if response, ok := message.(ai.ModelResponse); ok && response.WorkspaceRef != nil {
			t.Fatal("client can redirect workspace")
		}
	}
	trusted, _, err := ai.SanitizeMessages(history, ai.MessageSanitizationOptions{AllowWorkspaceRefs: true})
	if err != nil {
		t.Fatal(err)
	}
	trusted[len(trusted)-1].(ai.ModelResponse).WorkspaceRef.ID = "changed"
	if history[len(history)-1].(ai.ModelResponse).WorkspaceRef.ID == "changed" {
		t.Fatal("sanitized reference aliases input")
	}
	if err := result.Workspace().Remove(ctx, filepath.Dir("hook")); err == nil {
		t.Fatal("removed workspace root")
	}
}

func TestWorkspaceOptionValidation(t *testing.T) {
	for name, option := range map[string]func(){
		"typed nil facade": func() { ai.NewWorkspace((*ai.LocalWorkspaceBackend)(nil)) },
		"nil":              func() { ai.WithRunWorkspace(nil) },
		"typed nil":        func() { ai.WithRunWorkspace((*ai.Workspace)(nil)) },
		"empty provider":   func() { ai.WithRunWorkspaceRef(ai.WorkspaceRef{ID: "id"}) },
		"empty id":         func() { ai.WithRunWorkspaceRef(ai.WorkspaceRef{Provider: "test"}) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid option accepted")
				}
			}()
			option()
		})
	}
}
