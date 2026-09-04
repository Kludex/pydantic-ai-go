package ai

import (
	"fmt"
	"reflect"
)

// WrappedCapability is embedded by a capability decorator. Registration expands
// it into the decorator followed by the wrapped capability, so every optional
// interface not overridden by the decorator remains active.
type WrappedCapability struct {
	wrapped Capability
}

// WrapCapability creates the embeddable base for a decorator around one capability.
// Combine decorated capabilities after wrapping individual leaves.
func WrapCapability(capability Capability) WrappedCapability {
	if capabilityIsNil(capability) {
		panic("ai: wrapped capability must not be nil")
	}
	if _, combined := capability.(combinedCapability); combined {
		panic("ai: wrap individual capabilities before combining them")
	}
	return WrappedCapability{wrapped: capability}
}

// Setup validates a zero-value wrapper. The wrapped capability is set up separately.
func (wrapper WrappedCapability) Setup(*CapabilityRegistry) error {
	if capabilityIsNil(wrapper.wrapped) {
		return fmt.Errorf("ai: wrapped capability must not be nil")
	}
	return nil
}

// Wrapped returns the decorated capability.
func (wrapper WrappedCapability) Wrapped() Capability { return wrapper.wrapped }

// CapabilityOrdering adopts the wrapped capability's ordering by default.
func (wrapper WrappedCapability) CapabilityOrdering() CapabilityOrdering {
	if provider, ok := wrapper.wrapped.(CapabilityOrderingProvider); ok {
		return provider.CapabilityOrdering()
	}
	return CapabilityOrdering{}
}

func (wrapper WrappedCapability) wrappedCapability() Capability { return wrapper.wrapped }

func capabilityIsNil(capability Capability) bool {
	if capability == nil {
		return true
	}
	value := reflect.ValueOf(capability)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
