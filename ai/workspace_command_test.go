package ai_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai"
)

func TestWorkspaceCommands(t *testing.T) {
	ctx := context.Background()
	t.Setenv("WORKSPACE_SECRET", "not-inherited")
	t.Setenv("LANG", "C")
	_, w, root := localWorkspace(t)
	for name, command := range map[string]ai.WorkspaceCommand{
		"argv":  {Args: []string{"/bin/sh", "-c", "printf '%s' \"$PWD\"; printf err >&2; exit 3"}},
		"shell": {Shell: "printf '%s' \"$PWD\"; printf err >&2; exit 3"},
	} {
		t.Run(name, func(t *testing.T) {
			result, err := w.Run(ctx, command)
			if err != nil || result.ExitCode != 3 || result.Stderr != "err" || !strings.HasSuffix(result.Stdout, filepath.Base(root)) {
				t.Fatalf("command: %#v %v", result, err)
			}
		})
	}
	env := map[string]string{"WORKSPACE_CONFIG": "base", "WORKSPACE_OVERRIDE": "base"}
	backend, err := ai.NewLocalWorkspaceBackend(root, env)
	if err != nil {
		t.Fatal(err)
	}
	env["WORKSPACE_CONFIG"] = "mutated"
	result, err := backend.Run(ctx, ai.WorkspaceCommand{
		Shell: "printf '%s|%s|%s|%s' \"$WORKSPACE_SECRET\" \"$WORKSPACE_CONFIG\" \"$WORKSPACE_OVERRIDE\" \"$LANG\"; read value",
		Env:   map[string]string{"WORKSPACE_OVERRIDE": "call"},
	})
	if err != nil || result.Stdout != "|base|call|C" || result.ExitCode != 1 {
		t.Fatalf("environment/stdin: %#v %v", result, err)
	}
	result, err = w.Run(ctx, ai.WorkspaceCommand{Args: []string{"/bin/sh", "-c", "printf '\\377'"}})
	if err != nil || result.Stdout != "\uFFFD" {
		t.Fatalf("invalid command UTF-8: %#v %v", result, err)
	}
	result, err = w.Run(ctx, ai.WorkspaceCommand{Shell: "kill -TERM $$"})
	if err != nil || result.ExitCode != 143 {
		t.Fatalf("signal death: %#v %v", result, err)
	}
	for _, program := range []string{"pydantic-ai-no-such-workspace-program", root + "/missing"} {
		result, err = w.Run(ctx, ai.WorkspaceCommand{Args: []string{program}})
		if err != nil || result.ExitCode != 127 || result.Stderr == "" {
			t.Fatalf("missing program: %#v %v", result, err)
		}
	}
	file := filepath.Join(root, "program")
	if err := os.WriteFile(file, []byte("#!/bin/sh\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err = w.Run(ctx, ai.WorkspaceCommand{Args: []string{file}})
	if err != nil || result.ExitCode != 126 {
		t.Fatalf("permission denied: %#v %v", result, err)
	}
	result, err = w.Run(ctx, ai.WorkspaceCommand{Args: []string{"program"}, Env: map[string]string{"PATH": root}})
	if err != nil || result.ExitCode != 126 {
		t.Fatalf("PATH permission: %#v %v", result, err)
	}
	if err := os.Chmod(file, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("#!/bin/sh\nprintf configured-path"), 0o700); err != nil {
		t.Fatal(err)
	}
	for _, searchPath := range []string{root, "", "."} {
		result, err = w.Run(ctx, ai.WorkspaceCommand{Args: []string{"program"}, Env: map[string]string{"PATH": searchPath}})
		if err != nil || result.Stdout != "configured-path" || result.ExitCode != 0 {
			t.Fatalf("configured PATH: %#v %v", result, err)
		}
	}
	result, err = w.Run(ctx, ai.WorkspaceCommand{Args: []string{"sh", "-c", "printf found"}})
	if err != nil || result.Stdout != "found" {
		t.Fatalf("inherited PATH: %#v %v", result, err)
	}
	if err := os.WriteFile(file, []byte("not a program"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Run(ctx, ai.WorkspaceCommand{Args: []string{file}}); err == nil {
		t.Fatal("invalid executable did not fail")
	}
	for _, command := range []ai.WorkspaceCommand{{}, {Args: []string{"true"}, Shell: "true"}, {Shell: "true", Timeout: -1}} {
		if _, err := backend.Run(ctx, command); err == nil {
			t.Fatal("backend accepted invalid command")
		}
	}
	for _, command := range []ai.WorkspaceCommand{
		{Args: []string{""}},
		{Shell: "true", Env: map[string]string{"": "value"}},
		{Shell: "true", Env: map[string]string{"KEY": "\x00"}},
	} {
		if _, err := w.Run(ctx, command); err == nil {
			t.Fatal("invalid command input accepted")
		}
	}
	if _, err := backend.Run(ctx, ai.WorkspaceCommand{Args: []string{"/bin/echo", "\x00"}}); err == nil {
		t.Fatal("NUL argument accepted")
	}
}

func TestWorkspaceCommandLimits(t *testing.T) {
	ctx := context.Background()
	backend, w, root := localWorkspace(t)
	result, err := w.Run(ctx, ai.WorkspaceCommand{
		Shell:   "printf before; printf error >&2; (sleep 0.3; echo survived > escaped) & wait",
		Timeout: 30 * time.Millisecond,
	})
	var commandErr *ai.WorkspaceCommandError
	if !errors.Is(err, ai.ErrWorkspaceTimeout) || !errors.Is(err, context.DeadlineExceeded) ||
		!errors.As(err, &commandErr) || commandErr.Stdout != "" && commandErr.Stdout != "before" ||
		commandErr.Stderr != "" && commandErr.Stderr != "error" ||
		commandErr.Error() == "" || result.Stdout != "" {
		t.Fatalf("timeout: %#v %v", commandErr, err)
	}
	time.Sleep(350 * time.Millisecond)
	if _, err := os.Stat(filepath.Join(root, "escaped")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("foreground tree survived timeout: %v", err)
	}
	cancelled, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
	defer cancel()
	if _, err := w.Run(cancelled, ai.WorkspaceCommand{Shell: "sleep 10"}); !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ai.ErrWorkspaceTimeout) {
		t.Fatalf("parent deadline: %v", err)
	}
	alreadyCancelled, stop := context.WithCancel(ctx)
	stop()
	if _, err := backend.Run(alreadyCancelled, ai.WorkspaceCommand{Shell: "true"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled backend: %v", err)
	}
	if _, err := w.Run(alreadyCancelled, ai.WorkspaceCommand{Shell: "true"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("already cancelled: %v", err)
	}
	if _, err := w.Run(ctx, ai.WorkspaceCommand{
		Shell: "dd if=/dev/zero bs=1048576 count=11 2>/dev/null",
	}); !errors.Is(err, ai.ErrWorkspaceOutputLimit) || !errors.As(err, &commandErr) || commandErr.Limit != 10*1024*1024 || len(commandErr.Stdout) > 64*1024 {
		t.Fatalf("output overflow: %v", err)
	}
	start := time.Now()
	result, err = w.Run(ctx, ai.WorkspaceCommand{Shell: "sleep 3 & printf done"})
	if err != nil || result.ExitCode != 0 || result.Stdout != "done" || time.Since(start) > 2900*time.Millisecond {
		t.Fatalf("background pipe: %#v %v, took %s", result, err, time.Since(start))
	}
	if _, err := w.Run(ctx, ai.WorkspaceCommand{Shell: "rmdir \"$PWD\""}); !errors.Is(err, ai.ErrWorkspaceUnavailable) {
		t.Fatalf("workspace destroyed by command: %v", err)
	}
}
