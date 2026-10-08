package ai

import (
	"context"
	"fmt"
	"reflect"
)

// WorkspaceProvider optionally supplies a run's workspace. Return nil to decline
// a ref, or to defer fresh selection. Selection must perform no I/O. Run-layer
// providers are tried before agent-layer providers; the first non-nil backend wins.
// The backend creates or attaches on first use, not during selection.
type WorkspaceProvider interface {
	GetWorkspace(context.Context, *RunInfo, *WorkspaceRef) (WorkspaceBackend, error)
}

// WithRunWorkspace shares a live backend or workspace for this run, preserving
// facade policies. It overrides capabilities and any reference in history.
func WithRunWorkspace(backend WorkspaceBackend) RunOption {
	if backend == nil || isNilCapabilityField(reflect.ValueOf(backend)) {
		panic("ai: run workspace must not be nil")
	}
	return func(c *runConfig) {
		c.workspace = backend
		c.workspaceRef = nil
		c.newWorkspace = false
	}
}

// WithRunWorkspaceRef reconnects through this run's workspace capabilities.
// Keep refs server-side and authorize them before use; they are not credentials.
func WithRunWorkspaceRef(ref WorkspaceRef) RunOption {
	if ref.Provider == "" || ref.ID == "" {
		panic("ai: workspace reference needs a provider and ID")
	}
	return func(c *runConfig) {
		c.workspace = nil
		c.workspaceRef = clonePointer(&ref)
		c.newWorkspace = false
	}
}

// WithNewWorkspace ignores history and asks capabilities for a fresh workspace.
// For a local workspace, fresh selection still names the configured directory.
func WithNewWorkspace() RunOption {
	return func(c *runConfig) {
		c.workspace = nil
		c.workspaceRef = nil
		c.newWorkspace = true
	}
}

func (r *run[Deps, Output]) selectWorkspace(
	ctx context.Context, cfg runConfig, agentCapabilities, runCapabilities []Capability,
) error {
	var historical *WorkspaceRef
	for index := len(cfg.history) - 1; index >= 0; index-- {
		if response, ok := cfg.history[index].(ModelResponse); ok {
			historical = clonePointer(response.WorkspaceRef)
			break
		}
	}
	if !cfg.newWorkspace {
		r.carriedWorkspaceRef = historical
	}
	workspace := &Workspace{}
	r.rc.Workspace = workspace
	r.info.workspace = workspace
	explicit := cfg.workspace != nil
	if supplied, ok := cfg.workspace.(*Workspace); ok && supplied.backend == nil && supplied.reason == "" {
		explicit = false
	}
	if explicit {
		workspace = NewWorkspace(cfg.workspace)
	} else {
		ref := historical
		if cfg.newWorkspace {
			ref = nil
		} else if cfg.workspaceRef != nil {
			ref = cfg.workspaceRef
		}
		hasProviders := false
		providers := append(append([]Capability{}, runCapabilities...), agentCapabilities...)
		for _, capability := range providers {
			provider, ok := capability.(WorkspaceProvider)
			if !ok {
				continue
			}
			hasProviders = true
			backend, err := provider.GetWorkspace(ctx, r.info, clonePointer(ref))
			if err != nil {
				return err
			}
			if backend != nil {
				if isNilCapabilityField(reflect.ValueOf(backend)) {
					return fmt.Errorf("ai: workspace provider returned a typed nil backend")
				}
				workspace = NewWorkspace(backend)
				break
			}
		}
		if workspace.backend == nil && (cfg.newWorkspace || cfg.workspaceRef != nil || ref != nil && hasProviders) {
			return fmt.Errorf("%w: no capability recognized the requested workspace; use WithNewWorkspace to start over", ErrWorkspaceUnavailable)
		}
	}
	r.rc.Workspace = workspace
	r.info.workspace = workspace
	return nil
}

func (r *run[Deps, Output]) recordedWorkspaceRef() *WorkspaceRef {
	if r.rc.Workspace.Attached() {
		return r.rc.Workspace.Ref()
	}
	return clonePointer(r.carriedWorkspaceRef)
}

func (r *run[Deps, Output]) refreshWorkspaceRef() {
	for index := len(r.messages) - 1; index >= r.newMessages; index-- {
		if response, ok := r.messages[index].(ModelResponse); ok {
			response.WorkspaceRef = r.recordedWorkspaceRef()
			r.messages[index] = response
			break
		}
	}
}

// Workspace returns the environment selected before run hooks and tools execute.
func (ri *RunInfo) Workspace() *Workspace { return ri.workspace }

// Workspace returns the live workspace. Runs never destroy it on completion.
// A decoded result has an unattached workspace; reconnect using its message ref.
func (r *RunResult[Output]) Workspace() *Workspace { return r.workspace }

// Workspace returns the manually driven run's workspace, including before Next.
func (r *AgentRun[Deps, Output]) Workspace() *Workspace { return r.run.rc.Workspace }

// LocalWorkspace supplies host files and commands without adding any model tools.
// Configure ReadOnly and ID before registering it; do not mutate it during runs.
type LocalWorkspace struct {
	// ReadOnly refuses commands and file mutations.
	ReadOnly bool
	// ID defaults to local_workspace. Distinct IDs retain multiple providers.
	ID string

	backend *LocalWorkspaceBackend
}

// NewLocalWorkspace pins a host directory and copies its command environment.
// It does no I/O; the caller creates and removes the directory.
func NewLocalWorkspace(directory string, env map[string]string) (*LocalWorkspace, error) {
	backend, err := NewLocalWorkspaceBackend(directory, env)
	if err != nil {
		return nil, err
	}
	return &LocalWorkspace{backend: backend}, nil
}

// Setup contributes no tools or instructions.
func (*LocalWorkspace) Setup(*CapabilityRegistry) error { return nil }

// CapabilityID returns the identity used for run overrides.
func (c *LocalWorkspace) CapabilityID() string {
	if c.ID != "" {
		return c.ID
	}
	return "local_workspace"
}

// CombineCapabilities replaces earlier configuration whole, without merging env or policy.
func (*LocalWorkspace) CombineCapabilities(capabilities []Capability) (Capability, error) {
	return capabilities[len(capabilities)-1], nil
}

// GetWorkspace declines refs for any other directory, never redirecting host access.
func (c *LocalWorkspace) GetWorkspace(_ context.Context, _ *RunInfo, ref *WorkspaceRef) (WorkspaceBackend, error) {
	if c.backend == nil {
		return nil, fmt.Errorf("ai: construct LocalWorkspace with NewLocalWorkspace")
	}
	if ref != nil && *ref != *c.backend.Ref() {
		return nil, nil
	}
	if c.ReadOnly {
		return ReadOnlyWorkspace(NewWorkspace(c.backend)), nil
	}
	return c.backend, nil
}
