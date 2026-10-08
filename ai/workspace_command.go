package ai

import (
	"bytes"
	"context"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const workspaceCaptureLimit = 10 * 1024 * 1024

// Run starts a host command with stdin at EOF and a sanitized environment.
// Missing programs exit 127; non-executable programs exit 126. Timeout or
// cancellation kills the foreground process group. Detached jobs are caller-owned.
func (b *LocalWorkspaceBackend) Run(ctx context.Context, command WorkspaceCommand) (CommandResult, error) {
	if err := validateWorkspaceCommand(command); err != nil {
		return CommandResult{}, err
	}
	directory, err := b.WorkingDir(ctx)
	if err != nil {
		return CommandResult{}, err
	}
	commandCtx := ctx
	if command.Timeout > 0 {
		var cancel context.CancelFunc
		commandCtx, cancel = context.WithTimeout(ctx, command.Timeout)
		defer cancel()
	}
	commandCtx, cancel := context.WithCancelCause(commandCtx)
	defer cancel(nil)
	args := command.Args
	if command.Shell != "" {
		args = []string{"/bin/sh", "-c", command.Shell}
	}
	cmd := exec.CommandContext(commandCtx, args[0], args[1:]...)
	cmd.Dir = directory
	cmd.Env = make([]string, 0, len(b.env)+len(command.Env))
	for name, value := range b.env {
		if _, overridden := command.Env[name]; !overridden {
			cmd.Env = append(cmd.Env, name+"="+value)
		}
	}
	for name, value := range command.Env {
		cmd.Env = append(cmd.Env, name+"="+value)
	}
	if !strings.ContainsRune(args[0], '/') {
		searchPath := b.env["PATH"]
		if override, ok := command.Env["PATH"]; ok {
			searchPath = override
		}
		cmd.Err = exec.ErrNotFound
		for _, directoryEntry := range strings.Split(searchPath, ":") {
			if !filepath.IsAbs(directoryEntry) {
				directoryEntry = filepath.Join(directory, directoryEntry)
			}
			candidate := filepath.Join(directoryEntry, args[0])
			info, statErr := os.Stat(candidate)
			if errors.Is(statErr, fs.ErrPermission) || statErr == nil && (info.IsDir() || info.Mode()&0o111 == 0) {
				cmd.Err = fs.ErrPermission
				continue
			}
			if statErr == nil {
				cmd.Path = candidate
				cmd.Err = nil
				break
			}
		}
	}
	configureWorkspaceProcess(cmd)
	cmd.WaitDelay = 2 * time.Second
	capture := &workspaceCapture{cancel: cancel}
	cmd.Stdout = &workspaceCaptureWriter{capture: capture, stdout: true}
	cmd.Stderr = &workspaceCaptureWriter{capture: capture}
	err = cmd.Run()
	result := CommandResult{
		Stdout: string(bytes.Runes(capture.stdout.Bytes())),
		Stderr: string(bytes.Runes(capture.stderr.Bytes())),
	}
	if _, aliveErr := b.WorkingDir(ctx); aliveErr != nil && ctx.Err() == nil {
		return CommandResult{}, aliveErr
	}
	if capture.overflow || commandCtx.Err() != nil {
		cause := context.Cause(commandCtx)
		limit := 0
		if capture.overflow {
			cause = ErrWorkspaceOutputLimit
			limit = workspaceCaptureLimit
			result.Stdout = string(bytes.Runes(capture.stdout.Bytes()[:min(capture.stdout.Len(), 64*1024)]))
			result.Stderr = string(bytes.Runes(capture.stderr.Bytes()[:min(capture.stderr.Len(), 64*1024)]))
		} else if ctx.Err() == nil {
			cause = errors.Join(ErrWorkspaceTimeout, cause)
		}
		return CommandResult{}, &WorkspaceCommandError{Err: cause, Stdout: result.Stdout, Stderr: result.Stderr, Limit: limit}
	}
	if err == nil || errors.Is(err, exec.ErrWaitDelay) {
		return result, nil
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		result.ExitCode = workspaceExitCode(exit)
		return result, nil
	}
	if errors.Is(err, exec.ErrNotFound) || errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrPermission) {
		result.ExitCode = 127
		if errors.Is(err, fs.ErrPermission) {
			result.ExitCode = 126
		}
		result.Stderr = err.Error() + "\n"
		return result, nil
	}
	return CommandResult{}, err
}

type workspaceCapture struct {
	mu       sync.Mutex
	stdout   bytes.Buffer
	stderr   bytes.Buffer
	overflow bool
	cancel   context.CancelCauseFunc
}

type workspaceCaptureWriter struct {
	capture *workspaceCapture
	stdout  bool
}

func (w *workspaceCaptureWriter) Write(data []byte) (int, error) {
	c := w.capture
	c.mu.Lock()
	defer c.mu.Unlock()
	remaining := workspaceCaptureLimit - c.stdout.Len() - c.stderr.Len()
	buffer := &c.stderr
	if w.stdout {
		buffer = &c.stdout
	}
	buffer.Write(data[:min(len(data), remaining)])
	if len(data) > remaining {
		c.overflow = true
		c.cancel(ErrWorkspaceOutputLimit)
	}
	return len(data), nil
}
