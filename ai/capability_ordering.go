package ai

import (
	"fmt"
	"reflect"
	"slices"
)

// CapabilityPosition selects a fixed middleware tier.
type CapabilityPosition string

const (
	// CapabilityOutermost places a capability before the default and innermost tiers.
	CapabilityOutermost CapabilityPosition = "outermost"
	// CapabilityInnermost places a capability after the outermost and default tiers.
	CapabilityInnermost CapabilityPosition = "innermost"
)

// CapabilityReference identifies a capability type or a specific pointer instance.
type CapabilityReference struct {
	typ      reflect.Type
	instance Capability
}

// CapabilityType references every capability assignable to T.
func CapabilityType[T Capability]() CapabilityReference {
	return CapabilityReference{typ: reflect.TypeFor[T]()}
}

// CapabilityInstance references one capability pointer by identity.
func CapabilityInstance(capability Capability) CapabilityReference {
	value := reflect.ValueOf(capability)
	if !value.IsValid() || value.Kind() != reflect.Pointer || value.IsNil() {
		panic("ai: capability instance reference requires a non-nil pointer")
	}
	return CapabilityReference{typ: value.Type(), instance: capability}
}

// CapabilityOrdering declares middleware position, relative order, and dependencies.
// Agent and per-run registrations are sorted independently; agent capabilities
// remain outside per-run capabilities.
type CapabilityOrdering struct {
	// Position selects an outermost or innermost fixed tier.
	Position CapabilityPosition
	// Wraps places this capability outside matching references.
	Wraps []CapabilityReference
	// WrappedBy places this capability inside matching references.
	WrappedBy []CapabilityReference
	// Requires rejects registration when a reference is unavailable.
	Requires []CapabilityReference
}

// CapabilityOrderingProvider supplies ordering constraints for a capability.
type CapabilityOrderingProvider interface {
	// CapabilityOrdering returns detached ordering and dependency constraints.
	CapabilityOrdering() CapabilityOrdering
}

type capabilityEntry struct {
	capability Capability
	rootID     string
}

func sortCapabilityEntries(entries []capabilityEntry, available []Capability) ([]capabilityEntry, error) {
	entries = slices.Clone(entries)
	orderings := make([]CapabilityOrdering, len(entries))
	for index, entry := range entries {
		capability := entry.capability
		provider, ok := capability.(CapabilityOrderingProvider)
		if !ok {
			continue
		}
		ordering := provider.CapabilityOrdering()
		if ordering.Position != "" && ordering.Position != CapabilityOutermost && ordering.Position != CapabilityInnermost {
			return nil, fmt.Errorf("capability %T has invalid position %q", capability, ordering.Position)
		}
		orderings[index] = ordering
	}
	for index, ordering := range orderings {
		for _, required := range ordering.Requires {
			if !matchesAnyCapability(required, available) {
				return nil, fmt.Errorf("capability %T requires %s, but it is not registered", entries[index].capability, required)
			}
		}
	}
	if len(entries) < 2 {
		return entries, nil
	}

	edges := make([][]bool, len(entries))
	for index := range edges {
		edges[index] = make([]bool, len(entries))
	}
	for index, ordering := range orderings {
		switch ordering.Position {
		case CapabilityOutermost:
			for other, otherOrdering := range orderings {
				if other != index && otherOrdering.Position != CapabilityOutermost {
					edges[index][other] = true
				}
			}
		case CapabilityInnermost:
			for other, otherOrdering := range orderings {
				if other != index && otherOrdering.Position != CapabilityInnermost {
					edges[other][index] = true
				}
			}
		}
		for _, wrapped := range ordering.Wraps {
			for other, entry := range entries {
				if other != index && wrapped.matches(entry.capability) {
					edges[index][other] = true
				}
			}
		}
		for _, wrapper := range ordering.WrappedBy {
			for other, entry := range entries {
				if other != index && wrapper.matches(entry.capability) {
					edges[other][index] = true
				}
			}
		}
	}

	indegree := make([]int, len(entries))
	for _, outgoing := range edges {
		for target, edge := range outgoing {
			if edge {
				indegree[target]++
			}
		}
	}
	ordered := make([]capabilityEntry, 0, len(entries))
	used := make([]bool, len(entries))
	for len(ordered) < len(entries) {
		next := -1
		for index := range entries {
			if !used[index] && indegree[index] == 0 {
				next = index
				break
			}
		}
		if next < 0 {
			return nil, fmt.Errorf("circular capability ordering constraints")
		}
		used[next] = true
		ordered = append(ordered, entries[next])
		for target, edge := range edges[next] {
			if edge {
				indegree[target]--
			}
		}
	}
	return ordered, nil
}

func matchesAnyCapability(reference CapabilityReference, capabilities []Capability) bool {
	for _, capability := range capabilities {
		if reference.matches(capability) {
			return true
		}
	}
	return false
}

func (reference CapabilityReference) matches(capability Capability) bool {
	capabilityType := reflect.TypeOf(capability)
	if reference.instance != nil {
		return capabilityType == reference.typ && reflect.ValueOf(capability).Pointer() == reflect.ValueOf(reference.instance).Pointer()
	}
	if reference.typ == nil || capabilityType == nil {
		return false
	}
	if reference.typ.Kind() == reflect.Interface {
		return capabilityType.Implements(reference.typ)
	}
	return capabilityType == reference.typ
}

// String returns a diagnostic capability type or instance description.
func (reference CapabilityReference) String() string {
	if reference.typ == nil {
		return "<invalid capability reference>"
	}
	if reference.instance != nil {
		return fmt.Sprintf("instance of %v", reference.typ)
	}
	return reference.typ.String()
}
