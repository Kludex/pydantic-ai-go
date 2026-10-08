//go:build unix

package ai_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai"
)

func TestWorkspaceRelativeDirectory(t *testing.T) {
	ctx := context.Background()
	parent := t.TempDir()
	root := filepath.Join(parent, "root")
	target := filepath.Join(parent, "other", "nested")
	for _, dir := range []string{root, target} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(target, filepath.Join(root, "link")); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)
	backend, err := ai.NewLocalWorkspaceBackend("link/..", nil)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := backend.WorkingDir(ctx)
	want, resolveErr := filepath.EvalSymlinks(filepath.Dir(target))
	if err != nil || resolveErr != nil || resolved != want {
		t.Fatalf("symlink followed by ..: %q want %q, %v %v", resolved, want, err, resolveErr)
	}
	if backend.Ref().ID != root {
		t.Fatalf("lexical ref: %#v", backend.Ref())
	}
	t.Chdir(parent)
	if resolvedAgain, err := backend.WorkingDir(ctx); err != nil || resolvedAgain != resolved {
		t.Fatalf("changing cwd moved backend: %q %v", resolvedAgain, err)
	}
	deleted := t.TempDir()
	t.Chdir(deleted)
	if err := os.Remove(deleted); err != nil {
		t.Fatal(err)
	}
	if backend, err := ai.NewLocalWorkspaceBackend(".", nil); err == nil {
		if _, err := backend.WorkingDir(ctx); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
			t.Fatalf("missing cwd operation: %v", err)
		}
	}
}

func TestWorkspaceSpecialFilesAndPermissions(t *testing.T) {
	ctx := context.Background()
	_, w, root := localWorkspace(t)
	fifo := filepath.Join(root, "fifo")
	if err := syscall.Mkfifo(fifo, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := w.ReadBytes(ctx, "fifo"); err == nil {
		t.Fatal("read FIFO")
	}
	if err := w.WriteBytes(ctx, "fifo", nil); err == nil {
		t.Fatal("write FIFO")
	}
	if err := w.WriteText(ctx, "denied", "content"); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(filepath.Join(root, "denied"), 0); err != nil {
		t.Fatal(err)
	}
	if os.Geteuid() == 0 {
		t.Skip("root bypasses file permission checks")
	}
	if err := w.WriteBytes(ctx, "denied", nil); !errors.Is(err, os.ErrPermission) {
		t.Fatalf("write permission: %v", err)
	}
	if _, err := w.ReadBytes(ctx, "denied"); !errors.Is(err, os.ErrPermission) {
		t.Fatalf("read permission: %v", err)
	}
	if err := w.MakeDir(ctx, "hidden"); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(filepath.Join(root, "hidden"), 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(filepath.Join(root, "hidden"), 0o700); err != nil {
			t.Error(err)
		}
	})
	if _, err := w.Exists(ctx, "hidden/file"); !errors.Is(err, os.ErrPermission) {
		t.Fatalf("exists permission: %v", err)
	}
}
