package ai_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type lazyWorkspace struct {
	mu      sync.Mutex
	created bool
	creates int
	backend ai.WorkspaceBackend
}

func (b *lazyWorkspace) Ref() *ai.WorkspaceRef {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.created {
		return nil
	}
	return b.backend.Ref()
}

func (b *lazyWorkspace) WorkingDir(ctx context.Context) (string, error) {
	b.mu.Lock()
	if !b.created {
		b.created = true
		b.creates++
	}
	b.mu.Unlock()
	return b.backend.WorkingDir(ctx)
}

func TestWorkspaceLazyCreationAndOwnership(t *testing.T) {
	ctx := context.Background()
	b, _, root := localWorkspace(t)
	lazy := &lazyWorkspace{backend: b}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	unused, err := agent.Run(ctx, "no tools", struct{}{}, ai.WithRunWorkspace(lazy))
	if err != nil || lazy.creates != 0 || unused.Workspace().Ref() != nil {
		t.Fatalf("eager workspace creation: %#v %v", unused, err)
	}
	if unused.Messages()[len(unused.Messages())-1].(ai.ModelResponse).WorkspaceRef != nil {
		t.Fatal("unused workspace got a reference")
	}
	var group sync.WaitGroup
	for range 20 {
		group.Go(func() {
			if _, err := unused.Workspace().WorkingDir(ctx); err != nil {
				t.Error(err)
			}
		})
	}
	group.Wait()
	if lazy.creates != 1 || unused.Workspace().Ref().ID != root {
		t.Fatalf("concurrent creation: %d %#v", lazy.creates, unused.Workspace().Ref())
	}
	if unused.Messages()[len(unused.Messages())-1].(ai.ModelResponse).WorkspaceRef != nil {
		t.Fatal("post-run creation changed persisted messages")
	}
	fresh := &lazyWorkspace{backend: b}
	writer := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	ai.AddTool(writer, "touch", func(ctx context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		return rc.Workspace.WorkingDir(ctx)
	})
	used, err := writer.Run(ctx, "touch", struct{}{}, ai.WithRunWorkspace(fresh))
	if err != nil || fresh.creates != 1 || used.Workspace().Ref() == nil {
		t.Fatalf("tool creation: %#v %v", used, err)
	}
	for _, message := range used.Messages() {
		if response, ok := message.(ai.ModelResponse); ok && (response.WorkspaceRef == nil || response.WorkspaceRef.ID != root) {
			t.Fatal("lazy reference not refreshed after tools")
		}
	}
	pending := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	ai.AddTool(pending, "approved_touch", func(ctx context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		return rc.Workspace.WorkingDir(ctx)
	}, ai.WithApprovalRequired())
	paused, err := pending.Run(ctx, "approve", struct{}{}, ai.WithRunWorkspace(used.Workspace()))
	if err != nil || paused.Deferred() == nil || paused.Workspace().Ref() == nil ||
		paused.Messages()[len(paused.Messages())-1].(ai.ModelResponse).WorkspaceRef == nil {
		t.Fatalf("deferred workspace: %#v %v", paused, err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	createdOnCancel := &lazyWorkspace{backend: b}
	if _, err := ai.NewWorkspace(createdOnCancel).WorkingDir(cancelled); !errors.Is(err, context.Canceled) || createdOnCancel.Ref() == nil {
		t.Fatalf("cancelled creation lost reference: %v", err)
	}
	if _, err := used.Workspace().WorkingDir(ctx); err != nil {
		t.Fatal("a run closed its environment", err)
	}
}
