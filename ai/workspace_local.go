package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// LocalWorkspaceBackend uses host files and subprocesses. It is not a sandbox:
// paths and commands have the full permissions of this process. The caller owns
// the directory. A missing directory is reported on first use, never created.
type LocalWorkspaceBackend struct {
	directory string
	resolved  string
	env       map[string]string
	mu        sync.Mutex
}

// NewLocalWorkspaceBackend pins directory to an absolute path without I/O.
// Commands inherit only PATH, HOME, LANG, LC_ALL, LC_CTYPE and the supplied env.
// Only POSIX hosts support the local command termination contract.
func NewLocalWorkspaceBackend(directory string, env map[string]string) (*LocalWorkspaceBackend, error) {
	if err := localWorkspacePlatform(); err != nil { // pragma: no cover - exercised only on non-POSIX hosts.
		return nil, err
	}
	if directory == "~" || strings.HasPrefix(directory, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		directory = filepath.Join(home, strings.TrimPrefix(directory, "~"))
	}
	if strings.ContainsRune(directory, 0) {
		return nil, fmt.Errorf("ai: workspace directory contains NUL")
	}
	absolute := directory
	if !filepath.IsAbs(absolute) {
		cwd, err := os.Getwd()
		if err != nil { // pragma: no cover - getcwd failure for a deleted directory is host-dependent.
			return nil, err
		}
		absolute = cwd + "/" + directory
	}
	if err := validateWorkspaceCommand(WorkspaceCommand{Shell: "true", Env: env}); err != nil {
		return nil, err
	}
	merged := map[string]string{}
	for _, name := range []string{"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE"} {
		if value, ok := os.LookupEnv(name); ok {
			merged[name] = value
		}
	}
	maps.Copy(merged, env)
	return &LocalWorkspaceBackend{directory: absolute, env: merged}, nil
}

// Ref returns the directory identity, available even before first use.
func (b *LocalWorkspaceBackend) Ref() *WorkspaceRef {
	return &WorkspaceRef{Provider: "local", ID: filepath.Clean(b.directory)}
}

// WorkingDir validates and resolves the directory on first use.
func (b *LocalWorkspaceBackend) WorkingDir(ctx context.Context) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.workingDir(ctx)
}

func (b *LocalWorkspaceBackend) workingDir(ctx context.Context) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	root := b.resolved
	if root == "" {
		var err error
		root, err = filepath.EvalSymlinks(b.directory)
		if err != nil {
			return "", fmt.Errorf("%w: %s: %w", ErrWorkspaceUnavailable, b.directory, err)
		}
	}
	info, err := os.Stat(root)
	if err != nil {
		return "", fmt.Errorf("%w: %s: %w", ErrWorkspaceUnavailable, root, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%w: %s is not a directory", ErrWorkspaceUnavailable, root)
	}
	b.resolved = root
	return root, nil
}

func (b *LocalWorkspaceBackend) filePath(ctx context.Context, name string) error {
	if _, err := b.workingDir(ctx); err != nil {
		return err
	}
	if !filepath.IsAbs(name) || strings.ContainsRune(name, 0) {
		return fmt.Errorf("ai: workspace backend path must be absolute and contain no NUL")
	}
	return nil
}

// ReadBytes reads regular files, refusing devices and FIFOs without blocking.
func (b *LocalWorkspaceBackend) ReadBytes(ctx context.Context, name string) (data []byte, err error) {
	// ponytail: serialize file operations per backend; use per-path locks if throughput matters.
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(name, os.O_RDONLY|localWorkspaceNonblock, 0)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	info, err := file.Stat()
	if err != nil { // pragma: no cover - a valid open descriptor cannot disappear before fstat.
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, &fs.PathError{Op: "read", Path: name, Err: fmt.Errorf("not a regular file")}
	}
	return io.ReadAll(file)
}

// WriteBytes creates parents and rewrites regular files, preserving permissions
// and following existing symlinks. Commands and other processes are not locked.
func (b *LocalWorkspaceBackend) WriteBytes(ctx context.Context, name string, data []byte) (err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(name, os.O_WRONLY|os.O_CREATE|localWorkspaceNonblock, 0o666)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	info, err := file.Stat()
	if err != nil { // pragma: no cover - a valid open descriptor cannot disappear before fstat.
		return err
	}
	if !info.Mode().IsRegular() {
		return &fs.PathError{Op: "write", Path: name, Err: fmt.Errorf("not a regular file")}
	}
	if err := file.Truncate(0); err != nil { // pragma: no cover - requires a host filesystem failure after opening for writing.
		return err
	}
	_, err = file.Write(data)
	return err
}

// Stat returns target metadata for files and directories.
func (b *LocalWorkspaceBackend) Stat(ctx context.Context, name string) (FileEntry, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return FileEntry{}, err
	}
	info, err := os.Stat(name)
	if err != nil {
		return FileEntry{}, err
	}
	return localFileEntry(name, info), nil
}

// ListDir lists sorted entries, retaining dangling symlinks with an unknown size.
func (b *LocalWorkspaceBackend) ListDir(ctx context.Context, name string) ([]FileEntry, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return nil, err
	}
	children, err := os.ReadDir(name)
	if err != nil {
		return nil, err
	}
	entries := make([]FileEntry, 0, len(children))
	for _, child := range children {
		childPath := filepath.Join(name, child.Name())
		info, err := os.Stat(childPath)
		entry := FileEntry{Name: child.Name(), Path: childPath}
		if err == nil {
			entry = localFileEntry(childPath, info)
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// MakeDir creates a directory and all missing parents.
func (b *LocalWorkspaceBackend) MakeDir(ctx context.Context, name string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return err
	}
	return os.MkdirAll(name, 0o755)
}

// Remove recursively deletes a directory or unlinks a file or symlink.
// It refuses to delete the workspace's resolved root or any ancestor.
func (b *LocalWorkspaceBackend) Remove(ctx context.Context, name string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return err
	}
	info, err := os.Lstat(name)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink == 0 {
		target, err := filepath.EvalSymlinks(name)
		if err != nil { // pragma: no cover - requires an external deletion between lstat and symlink resolution.
			return err
		}
		if target == b.resolved || strings.HasPrefix(b.resolved, strings.TrimSuffix(target, "/")+"/") {
			return fmt.Errorf("ai: cannot remove the workspace root or an ancestor")
		}
	}
	return os.RemoveAll(name)
}

// Exists follows symlinks and returns false for a missing path.
func (b *LocalWorkspaceBackend) Exists(ctx context.Context, name string) (bool, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return false, err
	}
	_, err := os.Stat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	return err == nil, err
}

// Realpath resolves symlinks before .., leaving missing components in the path.
func (b *LocalWorkspaceBackend) Realpath(ctx context.Context, name string) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.filePath(ctx, name); err != nil {
		return "", err
	}
	resolved := "/"
	parts := strings.Split(name, "/")
	links := 0
	for len(parts) > 0 {
		part := parts[0]
		parts = parts[1:]
		if part == "" || part == "." {
			continue
		}
		if part == ".." {
			resolved = filepath.Dir(resolved)
			continue
		}
		candidate := filepath.Join(resolved, part)
		info, err := os.Lstat(candidate)
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return "", err
		}
		if err == nil && info.Mode()&os.ModeSymlink != 0 {
			links++
			if links > 40 {
				return "", fmt.Errorf("ai: too many workspace symlinks")
			}
			target, err := os.Readlink(candidate)
			if err != nil { // pragma: no cover - requires an external symlink replacement between lstat and readlink.
				return "", err
			}
			if filepath.IsAbs(target) {
				resolved = "/"
			}
			parts = append(strings.Split(target, "/"), parts...)
			continue
		}
		resolved = candidate
	}
	return resolved, nil
}

func localFileEntry(name string, info fs.FileInfo) FileEntry {
	entry := FileEntry{Name: filepath.Base(name), Path: name, IsDir: info.IsDir()}
	if !entry.IsDir {
		size := info.Size()
		entry.Size = &size
	}
	return entry
}
