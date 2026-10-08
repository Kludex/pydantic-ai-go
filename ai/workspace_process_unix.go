//go:build unix

package ai

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
)

const localWorkspaceNonblock = syscall.O_NONBLOCK

func localWorkspacePlatform() error { return nil }

func configureWorkspaceProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if errors.Is(err, syscall.ESRCH) { // pragma: no cover - requires cancellation racing with process exit after reaping.
			return os.ErrProcessDone
		}
		return err
	}
}

func workspaceExitCode(exit *exec.ExitError) int {
	status := exit.Sys().(syscall.WaitStatus)
	if status.Signaled() {
		return 128 + int(status.Signal())
	}
	return exit.ExitCode()
}
