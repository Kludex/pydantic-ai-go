package ai

import (
	"context"
	"errors"
	"fmt"
	"path"
	"reflect"
	"strings"
	"time"
	"unicode/utf8"
)

// WorkspaceRef identifies an environment, not its contents or credentials.
type WorkspaceRef struct {
	Provider string `json:"provider"`
	ID       string `json:"id"`
}

// WorkspaceBackend supplies one environment. Construction must do no I/O.
// Ref is nil until creation, then stable. Backends must synchronize lazy creation,
// retain the ref on cancellation, and never replace or destroy a missing environment.
type WorkspaceBackend interface {
	Ref() *WorkspaceRef
	WorkingDir(context.Context) (string, error)
}

// WorkspaceCommands optionally supplies command execution in the same environment.
// Nonzero exits are results. Cancellation and timeout must stop the foreground tree
// best-effort, with bounded pipe draining. Output replaces invalid UTF-8 bytes.
type WorkspaceCommands interface {
	Run(context.Context, WorkspaceCommand) (CommandResult, error)
}

// WorkspaceCommand selects either Args (without a shell) or Shell, never both.
// Env overlays backend defaults. A zero Timeout means no command deadline.
type WorkspaceCommand struct {
	Args    []string
	Shell   string
	Env     map[string]string
	Timeout time.Duration
}

// CommandResult contains the complete output of a completed command.
type CommandResult struct {
	ExitCode int    `json:"exit_code"`
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
}

// FileEntry describes an absolute POSIX path, following symlinks.
// Size is nil for directories or when the backend cannot determine it.
type FileEntry struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	IsDir bool   `json:"is_dir"`
	Size  *int64 `json:"size,omitempty"`
}

// WorkspaceFilesystem optionally supplies native file operations on absolute paths.
// Writes create parents and follow symlinks. Remove deletes symlinks themselves,
// recursively deletes directories, and refuses the working directory and ancestors.
// Path failures preserve errors.Is compatibility with io/fs errors.
type WorkspaceFilesystem interface {
	ReadBytes(context.Context, string) ([]byte, error)
	WriteBytes(context.Context, string, []byte) error
	Stat(context.Context, string) (FileEntry, error)
	ListDir(context.Context, string) ([]FileEntry, error)
	MakeDir(context.Context, string) error
	Remove(context.Context, string) error
	Exists(context.Context, string) (bool, error)
}

// WorkspaceRealpath optionally resolves symlinks, including missing trailing paths.
// A filesystem backend with symlinks should implement it for reliable path checks.
type WorkspaceRealpath interface {
	Realpath(context.Context, string) (string, error)
}

var (
	// ErrWorkspace identifies deliberate workspace failures.
	ErrWorkspace = errors.New("ai: workspace operation failed")
	// ErrWorkspaceUnavailable indicates a missing or unattached environment.
	ErrWorkspaceUnavailable = fmt.Errorf("%w: environment unavailable", ErrWorkspace)
	// ErrWorkspaceReadOnly indicates a refused command or mutation.
	ErrWorkspaceReadOnly = fmt.Errorf("%w: workspace is read-only", ErrWorkspace)
	// ErrWorkspaceTimeout indicates an expired command timeout.
	ErrWorkspaceTimeout = fmt.Errorf("%w: command timed out", ErrWorkspace)
	// ErrWorkspaceOutputLimit indicates excessive command output.
	ErrWorkspaceOutputLimit = fmt.Errorf("%w: command output limit exceeded", ErrWorkspace)
	// ErrWorkspaceUnsupported indicates an operation the backend does not support.
	ErrWorkspaceUnsupported = fmt.Errorf("%w: operation unsupported", ErrWorkspace)
)

// WorkspaceCommandError carries partial command output on timeout or overflow.
type WorkspaceCommandError struct {
	Err    error
	Stdout string
	Stderr string
	Limit  int
}

// Error returns the operation failure.
func (e *WorkspaceCommandError) Error() string { return e.Err.Error() }

// Unwrap exposes the failure to errors.Is and errors.As.
func (e *WorkspaceCommandError) Unwrap() error { return e.Err }

// Workspace adds relative paths and UTF-8 text to a backend. Its zero value is
// unattached. It owns neither the backend nor the environment; runs never close it.
// Share a workspace between agents to share files, including its read-only policy.
type Workspace struct {
	backend  WorkspaceBackend
	readOnly bool
	reason   string
}

// NewWorkspace wraps a backend without I/O, preserving existing workspace policies.
func NewWorkspace(backend WorkspaceBackend) *Workspace {
	if backend != nil && isNilCapabilityField(reflect.ValueOf(backend)) {
		panic("ai: workspace backend must not be a typed nil")
	}
	return &Workspace{backend: backend}
}

// ReadOnlyWorkspace refuses all commands and file changes, not just known writes.
// It restricts this API, not host permissions or access through Backend.
func ReadOnlyWorkspace(workspace *Workspace) *Workspace {
	return &Workspace{backend: workspace, readOnly: true}
}

// UnavailableWorkspace explicitly disables workspace access with a reason.
func UnavailableWorkspace(reason string) *Workspace {
	return &Workspace{reason: reason}
}

// Backend returns the underlying provider, bypassing policy wrappers.
func (w *Workspace) Backend() WorkspaceBackend {
	if w == nil {
		return nil
	}
	if wrapped, ok := w.backend.(*Workspace); ok {
		return wrapped.Backend()
	}
	return w.backend
}

// Attached reports whether an environment backend was supplied, without I/O.
func (w *Workspace) Attached() bool { return w.Backend() != nil }

// ReadOnly reports whether this facade or an inner facade refuses changes.
func (w *Workspace) ReadOnly() bool {
	if w == nil {
		return false
	}
	if wrapped, ok := w.backend.(*Workspace); ok {
		return w.readOnly || wrapped.ReadOnly()
	}
	return w.readOnly
}

// Ref returns a detached identity, or nil before lazy creation.
func (w *Workspace) Ref() *WorkspaceRef {
	if w == nil || w.backend == nil {
		return nil
	}
	return clonePointer(w.backend.Ref())
}

// WorkingDir returns the backend's stable, absolute, resolved working directory.
func (w *Workspace) WorkingDir(ctx context.Context) (string, error) {
	if w == nil || w.backend == nil {
		reason := "attach a workspace capability or use WithRunWorkspace"
		if w != nil && w.reason != "" {
			reason = w.reason
		}
		return "", fmt.Errorf("%w: %s", ErrWorkspaceUnavailable, reason)
	}
	return w.backend.WorkingDir(ctx)
}

// Resolve lexically joins a relative path to the working directory. It confines
// nothing: absolute paths and .. can escape it. It does not resolve symlinks.
func (w *Workspace) Resolve(ctx context.Context, name string) (string, error) {
	if strings.ContainsRune(name, 0) {
		return "", fmt.Errorf("ai: workspace path contains NUL")
	}
	if path.IsAbs(name) {
		return path.Clean(name), nil
	}
	base, err := w.WorkingDir(ctx)
	if err != nil {
		return "", err
	}
	return path.Join(base, name), nil
}

// Run executes Args directly or Shell through the backend's shell.
func (w *Workspace) Run(ctx context.Context, command WorkspaceCommand) (CommandResult, error) {
	if w.ReadOnly() {
		return CommandResult{}, ErrWorkspaceReadOnly
	}
	if err := validateWorkspaceCommand(command); err != nil {
		return CommandResult{}, err
	}
	if _, err := w.WorkingDir(ctx); err != nil {
		return CommandResult{}, err
	}
	backend, ok := w.backend.(WorkspaceCommands)
	if !ok {
		return CommandResult{}, ErrWorkspaceUnsupported
	}
	return backend.Run(ctx, command)
}

func validateWorkspaceCommand(command WorkspaceCommand) error {
	if command.Timeout < 0 || (len(command.Args) == 0) == (command.Shell == "") {
		return fmt.Errorf("ai: provide exactly one of Args or Shell and a non-negative timeout")
	}
	if len(command.Args) > 0 && command.Args[0] == "" {
		return fmt.Errorf("ai: command program must not be empty")
	}
	for _, argument := range append([]string{command.Shell}, command.Args...) {
		if strings.ContainsRune(argument, 0) {
			return fmt.Errorf("ai: command contains NUL")
		}
	}
	for name, value := range command.Env {
		if name == "" || strings.ContainsAny(name, "=\x00") || strings.ContainsRune(value, 0) {
			return fmt.Errorf("ai: invalid command environment variable %q", name)
		}
	}
	return nil
}

func (w *Workspace) filesystem(ctx context.Context, name string, mutation bool) (WorkspaceFilesystem, string, error) {
	if mutation && w.ReadOnly() {
		return nil, "", ErrWorkspaceReadOnly
	}
	if _, err := w.WorkingDir(ctx); err != nil {
		return nil, "", err
	}
	resolved, err := w.Resolve(ctx, name)
	if err != nil {
		return nil, "", err
	}
	backend, ok := w.backend.(WorkspaceFilesystem)
	if !ok {
		return nil, "", ErrWorkspaceUnsupported
	}
	return backend, resolved, nil
}

// ReadBytes reads a complete file.
func (w *Workspace) ReadBytes(ctx context.Context, name string) ([]byte, error) {
	fs, resolved, err := w.filesystem(ctx, name, false)
	if err != nil {
		return nil, err
	}
	return fs.ReadBytes(ctx, resolved)
}

// WriteBytes replaces a file, creating missing parents.
func (w *Workspace) WriteBytes(ctx context.Context, name string, data []byte) error {
	fs, resolved, err := w.filesystem(ctx, name, true)
	if err != nil {
		return err
	}
	return fs.WriteBytes(ctx, resolved, data)
}

// ReadText reads UTF-8, rejecting malformed text instead of returning partial data.
func (w *Workspace) ReadText(ctx context.Context, name string) (string, error) {
	data, err := w.ReadBytes(ctx, name)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("ai: file %q is not UTF-8", name)
	}
	return string(data), nil
}

// WriteText writes UTF-8 text.
func (w *Workspace) WriteText(ctx context.Context, name, text string) error {
	if !utf8.ValidString(text) {
		return fmt.Errorf("ai: text is not UTF-8")
	}
	return w.WriteBytes(ctx, name, []byte(text))
}

// Stat returns metadata following symlinks.
func (w *Workspace) Stat(ctx context.Context, name string) (FileEntry, error) {
	fs, resolved, err := w.filesystem(ctx, name, false)
	if err != nil {
		return FileEntry{}, err
	}
	return fs.Stat(ctx, resolved)
}

// ListDir lists one directory, sorted by path.
func (w *Workspace) ListDir(ctx context.Context, name string) ([]FileEntry, error) {
	fs, resolved, err := w.filesystem(ctx, name, false)
	if err != nil {
		return nil, err
	}
	return fs.ListDir(ctx, resolved)
}

// MakeDir creates a directory and missing parents.
func (w *Workspace) MakeDir(ctx context.Context, name string) error {
	fs, resolved, err := w.filesystem(ctx, name, true)
	if err != nil {
		return err
	}
	return fs.MakeDir(ctx, resolved)
}

// Remove deletes a file, symlink, or directory tree.
func (w *Workspace) Remove(ctx context.Context, name string) error {
	fs, resolved, err := w.filesystem(ctx, name, true)
	if err != nil {
		return err
	}
	return fs.Remove(ctx, resolved)
}

// Exists checks a path, following symlinks.
func (w *Workspace) Exists(ctx context.Context, name string) (bool, error) {
	fs, resolved, err := w.filesystem(ctx, name, false)
	if err != nil {
		return false, err
	}
	return fs.Exists(ctx, resolved)
}

// Realpath resolves symlinks when supported. Without WorkspaceRealpath it only
// normalizes text, so you cannot use that fallback as a security boundary.
func (w *Workspace) Realpath(ctx context.Context, name string) (string, error) {
	base, err := w.WorkingDir(ctx)
	if err != nil {
		return "", err
	}
	if strings.ContainsRune(name, 0) {
		return "", fmt.Errorf("ai: workspace path contains NUL")
	}
	if !path.IsAbs(name) {
		name = base + "/" + name
	}
	if backend, ok := w.backend.(WorkspaceRealpath); ok {
		return backend.Realpath(ctx, name)
	}
	return path.Clean(name), nil
}
