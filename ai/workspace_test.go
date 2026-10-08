package ai_test

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai"
)

func localWorkspace(t *testing.T) (*ai.LocalWorkspaceBackend, *ai.Workspace, string) {
	t.Helper()
	root := t.TempDir()
	backend, err := ai.NewLocalWorkspaceBackend(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	return backend, ai.NewWorkspace(backend), root
}

func TestWorkspaceFiles(t *testing.T) {
	ctx := context.Background()
	backend, w, root := localWorkspace(t)
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	if !w.Attached() || w.ReadOnly() || w.Backend() != backend || w.Ref().ID != root {
		t.Fatalf("workspace: %#v", w.Ref())
	}
	ref := w.Ref()
	ref.ID = "changed"
	if w.Ref().ID != root {
		t.Fatal("reference aliases backend")
	}
	if err := w.WriteText(ctx, "dir/file.txt", "hello"); err != nil {
		t.Fatal(err)
	}
	if text, err := w.ReadText(ctx, "dir/file.txt"); err != nil || text != "hello" {
		t.Fatalf("read: %q %v", text, err)
	}
	if err := w.WriteBytes(ctx, "dir/bytes", []byte{0, 0xff}); err != nil {
		t.Fatal(err)
	}
	if _, err := w.ReadText(ctx, "dir/bytes"); err == nil {
		t.Fatal("invalid UTF-8 accepted")
	}
	if err := w.WriteText(ctx, "bad", string([]byte{0xff})); err == nil {
		t.Fatal("invalid text accepted")
	}
	if entry, err := w.Stat(ctx, "dir/file.txt"); err != nil || entry.Size == nil || *entry.Size != 5 || entry.IsDir {
		t.Fatalf("stat: %#v %v", entry, err)
	}
	if entry, err := w.Stat(ctx, "dir"); err != nil || !entry.IsDir || entry.Size != nil {
		t.Fatalf("directory: %#v %v", entry, err)
	}
	if err := w.MakeDir(ctx, "dir/nested"); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("missing", filepath.Join(root, "dir/broken")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("nested", filepath.Join(root, "dir/link")); err != nil {
		t.Fatal(err)
	}
	entries, err := w.ListDir(ctx, "dir")
	if err != nil || len(entries) != 5 || entries[0].Name != "broken" || entries[0].Size != nil || !entries[3].IsDir {
		t.Fatalf("listing: %#v %v", entries, err)
	}
	if exists, err := w.Exists(ctx, "dir/broken"); err != nil || exists {
		t.Fatalf("broken link exists: %v %v", exists, err)
	}
	if exists, err := w.Exists(ctx, "dir/file.txt"); err != nil || !exists {
		t.Fatalf("file exists: %v %v", exists, err)
	}
	if real, err := w.Realpath(ctx, "dir/link/../missing"); err != nil || real != filepath.Join(resolvedRoot, "dir/missing") {
		t.Fatalf("realpath: %q %v", real, err)
	}
	if real, err := w.Realpath(ctx, resolvedRoot+"/dir/./file.txt"); err != nil || real != filepath.Join(resolvedRoot, "dir/file.txt") {
		t.Fatalf("absolute realpath: %q %v", real, err)
	}
	if err := w.Remove(ctx, "dir/link"); err != nil {
		t.Fatal(err)
	}
	if exists, err := w.Exists(ctx, "dir/nested"); err != nil || !exists {
		t.Fatal("removing a symlink removed its target")
	}
	for _, name := range []string{".", "..", "/"} {
		if err := w.Remove(ctx, name); err == nil {
			t.Fatalf("removed ancestor %q", name)
		}
	}
	if err := w.Remove(ctx, "dir"); err != nil {
		t.Fatal(err)
	}
	if err := w.Remove(ctx, "dir"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("missing remove: %v", err)
	}
	if _, err := w.ReadBytes(ctx, "."); err == nil {
		t.Fatal("read directory succeeded")
	}
	if err := w.WriteBytes(ctx, ".", nil); err == nil {
		t.Fatal("write directory succeeded")
	}
	if err := w.WriteBytes(ctx, "file", nil); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteBytes(ctx, "file/child", nil); err == nil {
		t.Fatal("created parent over a file")
	}
	if _, err := w.Exists(ctx, "file/child"); err == nil {
		t.Fatal("not-directory error lost")
	}
	if _, err := w.Realpath(ctx, "file/child"); err == nil {
		t.Fatal("not-directory realpath error lost")
	}
	if _, err := w.Stat(ctx, "missing"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("stat missing: %v", err)
	}
	if _, err := w.ListDir(ctx, "missing"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("list missing: %v", err)
	}
	if _, err := w.ReadText(ctx, "missing"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("read text missing: %v", err)
	}
	if _, err := w.ReadBytes(ctx, "missing"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("read missing: %v", err)
	}
	if err := w.MakeDir(ctx, "file/child"); err == nil {
		t.Fatal("made a directory under file")
	}
	if err := os.Symlink("loop", filepath.Join(root, "loop")); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Realpath(ctx, "loop"); err == nil {
		t.Fatal("symlink loop succeeded")
	}
	if err := os.Symlink(filepath.Join(resolvedRoot, "file"), filepath.Join(root, "absolute")); err != nil {
		t.Fatal(err)
	}
	if real, err := w.Realpath(ctx, "absolute"); err != nil || real != filepath.Join(resolvedRoot, "file") {
		t.Fatalf("absolute symlink: %q %v", real, err)
	}
	if err := w.WriteText(ctx, "absolute", "through link"); err != nil {
		t.Fatal(err)
	}
	if text, err := w.ReadText(ctx, "file"); err != nil || text != "through link" {
		t.Fatalf("write through symlink: %q %v", text, err)
	}
	if err := w.WriteBytes(ctx, "/dev/null", nil); err == nil {
		t.Fatal("wrote device")
	}
	if _, err := w.ReadBytes(ctx, "/dev/null"); err == nil {
		t.Fatal("read device")
	}
}

type directoryWorkspace struct{}

func (directoryWorkspace) Ref() *ai.WorkspaceRef                      { return nil }
func (directoryWorkspace) WorkingDir(context.Context) (string, error) { return "/project", nil }

func TestWorkspacePolicies(t *testing.T) {
	ctx := context.Background()
	_, w, _ := localWorkspace(t)
	readonly := ai.NewWorkspace(ai.ReadOnlyWorkspace(w))
	if !readonly.ReadOnly() || !readonly.Attached() || readonly.Backend() != w.Backend() {
		t.Fatal("policy lost through wrapping")
	}
	if err := w.WriteText(ctx, "file", "text"); err != nil {
		t.Fatal(err)
	}
	if text, err := readonly.ReadText(ctx, "file"); err != nil || text != "text" {
		t.Fatalf("readonly read: %q %v", text, err)
	}
	for name, operation := range map[string]func() error{
		"run":    func() error { _, err := readonly.Run(ctx, ai.WorkspaceCommand{Shell: "true"}); return err },
		"write":  func() error { return readonly.WriteText(ctx, "file", "changed") },
		"mkdir":  func() error { return readonly.MakeDir(ctx, "new") },
		"remove": func() error { return readonly.Remove(ctx, "file") },
	} {
		if err := operation(); !errors.Is(err, ai.ErrWorkspaceReadOnly) || !errors.Is(err, ai.ErrWorkspace) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	if entries, err := readonly.ListDir(ctx, "."); err != nil || len(entries) != 1 {
		t.Fatalf("readonly listing: %#v %v", entries, err)
	}
	if _, err := readonly.Stat(ctx, "file"); err != nil {
		t.Fatal(err)
	}
	if _, err := readonly.Exists(ctx, "file"); err != nil {
		t.Fatal(err)
	}
	if _, err := readonly.Realpath(ctx, "file"); err != nil {
		t.Fatal(err)
	}
	unsupported := ai.NewWorkspace(directoryWorkspace{})
	if _, err := unsupported.Run(ctx, ai.WorkspaceCommand{Shell: "true"}); !errors.Is(err, ai.ErrWorkspaceUnsupported) {
		t.Fatalf("commands unsupported: %v", err)
	}
	if _, err := unsupported.ReadBytes(ctx, "file"); !errors.Is(err, ai.ErrWorkspaceUnsupported) {
		t.Fatalf("filesystem unsupported: %v", err)
	}
	if real, err := unsupported.Realpath(ctx, "dir/../file"); err != nil || real != "/project/file" {
		t.Fatalf("lexical realpath: %q %v", real, err)
	}
	if name, err := unsupported.Resolve(ctx, "/file/../absolute"); err != nil || name != "/absolute" {
		t.Fatalf("absolute resolve: %q %v", name, err)
	}
	if _, err := (*ai.Workspace)(nil).Resolve(ctx, "file"); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
		t.Fatalf("unattached resolve: %v", err)
	}
	if _, err := w.Resolve(ctx, "\x00"); err == nil {
		t.Fatal("NUL path accepted")
	}
	if _, err := w.Realpath(ctx, "\x00"); err == nil {
		t.Fatal("NUL realpath accepted")
	}
	for _, disabled := range []*ai.Workspace{nil, {}, ai.UnavailableWorkspace("gone")} {
		if disabled.Attached() || disabled.ReadOnly() || disabled.Ref() != nil {
			t.Fatal("unattached workspace state")
		}
		operations := []func() error{
			func() error { _, err := disabled.ReadBytes(ctx, "file"); return err },
			func() error { return disabled.WriteBytes(ctx, "file", nil) },
			func() error { _, err := disabled.Stat(ctx, "file"); return err },
			func() error { _, err := disabled.ListDir(ctx, "."); return err },
			func() error { return disabled.MakeDir(ctx, "dir") },
			func() error { return disabled.Remove(ctx, "file") },
			func() error { _, err := disabled.Exists(ctx, "file"); return err },
			func() error { _, err := disabled.Realpath(ctx, "file"); return err },
			func() error { _, err := disabled.Run(ctx, ai.WorkspaceCommand{Shell: "true"}); return err },
		}
		for _, operation := range operations {
			if err := operation(); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
				t.Fatalf("unattached operation: %v", err)
			}
		}
	}
	for _, command := range []ai.WorkspaceCommand{
		{}, {Args: []string{"true"}, Shell: "true"}, {Shell: "true", Timeout: -time.Second},
	} {
		if _, err := w.Run(ctx, command); err == nil {
			t.Fatal("invalid command accepted")
		}
	}
}

func TestWorkspaceFailureAndConcurrency(t *testing.T) {
	ctx := context.Background()
	b, w, root := localWorkspace(t)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := b.WorkingDir(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled working dir: %v", err)
	}
	if err := b.WriteBytes(ctx, "relative", nil); err == nil {
		t.Fatal("relative backend path accepted")
	}
	if _, err := w.ReadBytes(ctx, "\x00"); err == nil {
		t.Fatal("NUL read accepted")
	}
	if err := w.WriteText(ctx, "shared", "initial"); err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	for i := range 20 {
		group.Go(func() {
			text := strings.Repeat(string(rune('a'+i)), 10000)
			if err := w.WriteText(ctx, "shared", text); err != nil {
				t.Error(err)
			}
			data, err := w.ReadText(ctx, "shared")
			if err != nil || len(data) != 10000 || strings.Trim(data, data[:1]) != "" {
				t.Errorf("interleaved read: %d %v", len(data), err)
			}
		})
	}
	group.Wait()
	if err := os.RemoveAll(root); err != nil {
		t.Fatal(err)
	}
	for _, operation := range []func() error{
		func() error { _, err := b.ReadBytes(ctx, root+"/file"); return err },
		func() error { return b.WriteBytes(ctx, root+"/file", nil) },
		func() error { _, err := b.Stat(ctx, root+"/file"); return err },
		func() error { _, err := b.ListDir(ctx, root); return err },
		func() error { return b.MakeDir(ctx, root+"/dir") },
		func() error { return b.Remove(ctx, root+"/file") },
		func() error { _, err := b.Exists(ctx, root+"/file"); return err },
		func() error { _, err := b.Realpath(ctx, root+"/file"); return err },
	} {
		if err := operation(); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
			t.Fatalf("dead workspace: %v", err)
		}
	}
	missing, err := ai.NewLocalWorkspaceBackend(root, nil)
	if err != nil || missing.Ref() == nil {
		t.Fatal("construction did I/O")
	}
	if _, err := missing.WorkingDir(ctx); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
		t.Fatalf("missing directory: %v", err)
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	notDir, err := ai.NewLocalWorkspaceBackend(file, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := notDir.WorkingDir(ctx); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
		t.Fatalf("not a directory: %v", err)
	}
	if _, err := ai.NewLocalWorkspaceBackend("\x00", nil); err == nil {
		t.Fatal("invalid directory accepted")
	}
	t.Setenv("HOME", t.TempDir())
	if home, err := ai.NewLocalWorkspaceBackend("~/dir", nil); err != nil || home.Ref().ID != filepath.Join(os.Getenv("HOME"), "dir") {
		t.Fatalf("home expansion: %#v %v", home, err)
	}
	if _, err := ai.NewLocalWorkspaceBackend(root, map[string]string{"INVALID=KEY": "value"}); err == nil {
		t.Fatal("invalid backend environment accepted")
	}
	t.Setenv("HOME", "")
	if _, err := ai.NewLocalWorkspaceBackend("~", nil); err == nil {
		t.Fatal("missing home accepted")
	}
}
