# Workspaces

## Give your tools files and commands

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func main() {
	ctx := context.Background()
	directory, err := os.MkdirTemp("", "agent-workspace-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(directory)
	local, err := ai.NewLocalWorkspace(directory, nil)
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(local))
	ai.AddTool(agent, "save_note", func(ctx context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		return "saved", rc.Workspace.WriteText(ctx, "notes/today.txt", "ready")
	})
	result, err := agent.Run(ctx, "Save a note.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	text, err := result.Workspace().ReadText(ctx, "notes/today.txt")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)
	reviewer := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	review, err := reviewer.Run(ctx, "Review the note.", struct{}{},
		ai.WithRunWorkspace(ai.ReadOnlyWorkspace(result.Workspace())),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(review.Workspace().ReadOnly())
}
```

A workspace is the environment your tools use for files and commands.
`RunContext.Workspace` gives typed tools access to it.
Capabilities use `RunInfo.Workspace()`.
`AgentRun.Workspace()` is available before you advance the run.
`RunResult.Workspace()` keeps the live workspace available after completion.

`LocalWorkspace` supplies the environment, not model tools or instructions.
You register the tools your application permits.
The example uses a fake model, so you can run it without credentials.
It prints `ready` and `true`.

!!! warning "A local workspace isolates nothing"
    Local commands run as your process's user on a POSIX host.
    The directory only sets where commands start and relative file paths resolve.
    Absolute paths and `..` can reach outside it.
    Use a container or VM backend for untrusted commands.

`ReadOnlyWorkspace` refuses every command and file mutation.
Commands are refused because even a command intended to read can change files.
Reads, directory listings, metadata, and path checks remain available.
The policy survives wrapping or sharing a workspace with another agent.
`Backend()` deliberately bypasses that policy for trusted application code.
It is not an authorization or isolation boundary.

## Run a command

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai"
)

func main() {
	backend, err := ai.NewLocalWorkspaceBackend(".", map[string]string{"MODE": "offline"})
	if err != nil {
		log.Fatal(err)
	}
	workspace := ai.NewWorkspace(backend)
	result, err := workspace.Run(context.Background(), ai.WorkspaceCommand{
		Args:    []string{"/bin/sh", "-c", "printf '%s' \"$MODE\""},
		Timeout: 5 * time.Second,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.ExitCode, result.Stdout)
}
```

Pass `Args` to execute a program directly.
Pass `Shell` instead to interpret a command through the backend's shell.
You cannot combine them.
A zero `Timeout` means no command deadline; a negative duration is invalid.
Pass `context.Context` to every operation to carry cancellation.

| Contract | Local behavior |
|---|---|
| Working directory | Must already exist. Pinned at construction and resolved on first use. |
| Environment | Inherit only `PATH`, `HOME`, `LANG`, `LC_ALL`, and `LC_CTYPE`. Constructor values override these; per-command values override both. |
| Program lookup | Use the command's merged `PATH`, including relative entries from the working directory. |
| Input | Standard input starts at EOF. |
| Output | Capture stdout and stderr independently. Replace invalid UTF-8 bytes. |
| Nonzero exit | Return `CommandResult`, not an error. |
| Missing or non-executable program | Return exit code 127 or 126. |
| Signal death | Return exit code `128 + signal`. |
| Timeout or cancellation | Kill the foreground process group best-effort and reap the direct child. |
| Background output pipes | Drain for at most two seconds after the direct child exits. |
| Output cap | Stop commands exceeding 10 MiB combined output. Keep at most 64 KiB from each stream in the error. |

!!! warning "Do not copy the host environment"
    Pass only the environment variables commands need.
    Passing all process variables can expose provider credentials to model-generated commands.
    Detached background jobs remain your responsibility.

Use `errors.Is` with `ErrWorkspaceUnavailable`, `ErrWorkspaceReadOnly`, `ErrWorkspaceTimeout`,
`ErrWorkspaceOutputLimit`, or `ErrWorkspaceUnsupported`.
These errors also match `ErrWorkspace`.
A caller's cancellation remains inspectable as `context.Canceled` or `context.DeadlineExceeded`.
`WorkspaceCommandError` exposes partial `Stdout`, `Stderr`, and the output `Limit` through `errors.As`.
Path errors preserve standard `io/fs` error identities.

## Continue in the same workspace

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func main() {
	ctx := context.Background()
	local, err := ai.NewLocalWorkspace(".", nil)
	if err != nil {
		log.Fatal(err)
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(local))
	first, err := agent.Run(ctx, "Inspect the project.", struct{}{})
	if err != nil {
		log.Fatal(err)
	}
	stored, err := ai.MarshalMessages(first.Messages())
	if err != nil {
		log.Fatal(err)
	}
	history, err := ai.UnmarshalMessages(stored)
	if err != nil {
		log.Fatal(err)
	}
	second, err := agent.Run(ctx, "Continue.", struct{}{}, ai.WithMessageHistory(history))
	if err != nil {
		log.Fatal(err)
	}
	ref := second.Workspace().Ref()
	third, err := agent.Run(ctx, "Start another conversation in this project.", struct{}{},
		ai.WithRunWorkspaceRef(*ref),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(*third.Workspace().Ref() == *ref)
}
```

`WorkspaceRef` contains the provider and environment ID.
Every model response records the run's reference as `workspace_ref`.
`MarshalMessages` and `UnmarshalMessages` preserve it.
Serialized run results preserve references, not live backends.
Reconnect through your capabilities after decoding a result.

| Selection | Precedence and behavior |
|---|---|
| `WithRunWorkspace` | Use the supplied backend or facade. Ignore history and providers. |
| `WithRunWorkspaceRef` | Ask capabilities to reconnect to this reference. An unrecognized reference fails the run. |
| `WithNewWorkspace` | Ignore history and ask capabilities for a fresh environment. A local environment still names the configured directory. |
| No workspace option | Offer the latest model response's reference to capabilities. A latest nil reference suppresses older ones. |
| Workspace providers | Try run capabilities before agent capabilities. The first non-nil backend wins. Normal capability ID replacement still applies. |
| No provider | Use an unattached placeholder whose operations return `ErrWorkspaceUnavailable`. Preserve a historical reference for later turns. |

Selection happens before dynamic metadata, toolset lifecycle callbacks, and run hooks.
It does no provider I/O.
A backend can create its environment lazily on the first operation.
Its reference stays nil until then.
If you first use it after the run ends, that new reference cannot appear in the completed messages.
Save it server-side yourself.

`LocalWorkspace` accepts only its configured directory's reference.
History cannot redirect it to another directory.
Repeated local capabilities with the same ID replace the earlier configuration whole.
A run's matching capability replaces the agent's configuration.
Secrets and read-only policy from the replaced configuration do not carry over.

!!! warning "Workspace references are trusted server state"
    `SanitizeMessages` strips client-supplied references by default.
    Save and authorize the reference on your server before passing `WithRunWorkspaceRef`.
    Enable `AllowWorkspaceRefs` only for history you already trust.
    A reference is an environment selector, not proof of permission to access it.

## Filesystem and backend contracts

You implement `WorkspaceBackend` with `Ref()` and `WorkingDir(context.Context)`.
The agent discovers optional `WorkspaceCommands`, `WorkspaceFilesystem`, and `WorkspaceRealpath` interfaces.
A missing primitive returns `ErrWorkspaceUnsupported`.
The Go API requires native filesystem operations; it does not synthesize filesystem shell scripts.

File paths passed to backends are absolute POSIX paths.
The facade resolves relative paths lexically against the working directory.
`Realpath` resolves symlinks before `..` when your backend implements `WorkspaceRealpath`.
Without it, the fallback only normalizes text and must not enforce a security boundary.

`ReadBytes` returns the complete file.
`ReadText` rejects invalid UTF-8.
`WriteBytes` and `WriteText` create missing parents and follow existing symlinks.
Local writes preserve existing permissions.
Local reads and writes refuse devices and FIFOs.
`ListDir` returns sorted entries, including dangling symlinks with an unknown size.
`Remove` unlinks a symlink itself and recursively deletes a directory.
It refuses the workspace root and its ancestors.

Backends must synchronize lazy creation so concurrent first operations create at most one environment.
They must set the reference as soon as creation succeeds, retain it on cancellation, and never change it.
An expired or deleted environment must raise `ErrWorkspaceUnavailable`, not create a replacement.
Commands and file operations must address the same filesystem.

Runs never close a backend or destroy its environment.
The application or provider that owns the reference decides when to remove it.
Local filesystem operations are serialized per backend.
This prevents mixed writes and partial reads through that backend, including shared facades and concurrent runs.
Commands, independent backends, and external processes are not covered by that lock.
Workspace capability configuration must remain immutable during runs.

Workspaces do not add Python workflow runtimes, durable workspace journals, or harness shell and filesystem tools.
You supply those policies and lifecycle integrations in your application.
