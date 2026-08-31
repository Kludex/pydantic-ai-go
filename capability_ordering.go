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
	Position  CapabilityPosition
	Wraps     []CapabilityReference
	WrappedBy []CapabilityReference
	Requires  []CapabilityReference
}

// CapabilityOrderingProvider supplies ordering constraints for a capability.
type CapabilityOrderingProvider interface {
	CapabilityOrdering() CapabilityOrdering
}

func sortCapabilities(capabilities []Capability, available []Capability) ([]Capability, error) {
	capabilities = slices.Clone(capabilities)
	orderings := make([]CapabilityOrdering, len(capabilities))
	for index, capability := range capabilities {
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
				return nil, fmt.Errorf("capability %T requires %s, but it is not registered", capabilities[index], required)
			}
		}
	}
	if len(capabilities) < 2 {
		return capabilities, nil
	}

	edges := make([][]bool, len(capabilities))
	for index := range edges {
		edges[index] = make([]bool, len(capabilities))
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
			for other, capability := range capabilities {
				if other != index && wrapped.matches(capability) {
					edges[index][other] = true
				}
			}
		}
		for _, wrapper := range ordering.WrappedBy {
			for other, capability := range capabilities {
				if other != index && wrapper.matches(capability) {
					edges[other][index] = true
				}
			}
		}
	}

	indegree := make([]int, len(capabilities))
	for _, outgoing := range edges {
		for target, edge := range outgoing {
			if edge {
				indegree[target]++
			}
		}
	}
	ordered := make([]Capability, 0, len(capabilities))
	used := make([]bool, len(capabilities))
	for len(ordered) < len(capabilities) {
		next := -1
		for index := range capabilities {
			if !used[index] && indegree[index] == 0 {
				next = index
				break
			}
		}
		if next < 0 {
			return nil, fmt.Errorf("circular capability ordering constraints")
		}
		used[next] = true
		ordered = append(ordered, capabilities[next])
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

func (reference CapabilityReference) String() string {
	if reference.typ == nil {
		return "<invalid capability reference>"
	}
	if reference.instance != nil {
		return fmt.Sprintf("instance of %v", reference.typ)
	}
	return reference.typ.String()
}
