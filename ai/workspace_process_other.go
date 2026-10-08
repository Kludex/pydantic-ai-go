//go:build !unix

package ai

import "os/exec"

func localWorkspacePlatform() error { return ErrWorkspaceUnsupported }

const localWorkspaceNonblock = 0

func configureWorkspaceProcess(*exec.Cmd) {} // pragma: no cover - local construction rejects non-POSIX hosts.

func workspaceExitCode(exit *exec.ExitError) int { return exit.ExitCode() } // pragma: no cover - local construction rejects non-POSIX hosts.
