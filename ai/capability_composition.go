package ai

import (
	"fmt"
	"reflect"
	"strings"
)

// MergeCapabilities applies the default merge for repeated capabilities. All
// values must have the same concrete struct type. Nil pointer, map, slice, and
// interface fields are unstated. Maps merge by key, slices form an ordered
// union, and later scalar values win.
//
// The default merge rejects unexported fields because it cannot distinguish
// configuration from derived state. Implement CapabilityCombiner directly
// when a capability needs to rebuild derived state or enforce invariants.
func MergeCapabilities(capabilities ...Capability) (Capability, error) {
	if len(capabilities) == 0 {
		return nil, fmt.Errorf("ai: cannot merge an empty capability collection")
	}
	firstType := reflect.TypeOf(capabilities[0])
	if firstType == nil {
		return nil, fmt.Errorf("ai: cannot merge a nil capability")
	}
	for _, capability := range capabilities {
		if capabilityIsNil(capability) {
			return nil, fmt.Errorf("ai: cannot merge a nil capability")
		}
		if reflect.TypeOf(capability) != firstType {
			return nil, fmt.Errorf("ai: cannot merge capabilities of different types %T and %T", capabilities[0], capability)
		}
	}

	pointer := firstType.Kind() == reflect.Pointer
	structType := firstType
	if pointer {
		structType = firstType.Elem()
	}
	if structType.Kind() != reflect.Struct {
		return nil, fmt.Errorf("ai: default capability merge requires a struct, got %v", firstType)
	}
	for index := 0; index < structType.NumField(); index++ {
		field := structType.Field(index)
		if field.PkgPath != "" {
			return nil, fmt.Errorf(
				"ai: cannot merge capability %v with unexported field %q; implement CapabilityCombiner to preserve derived state",
				firstType, field.Name,
			)
		}
	}

	values := make([]reflect.Value, len(capabilities))
	for index, capability := range capabilities {
		value := reflect.ValueOf(capability)
		if pointer {
			value = value.Elem()
		}
		values[index] = value
	}
	merged := reflect.New(structType).Elem()
	merged.Set(values[len(values)-1])
	for fieldIndex := 0; fieldIndex < structType.NumField(); fieldIndex++ {
		fieldValues := make([]reflect.Value, len(values))
		for index, value := range values {
			fieldValues[index] = value.Field(fieldIndex)
		}
		value, err := mergeCapabilityField(fieldValues)
		if err != nil {
			return nil, fmt.Errorf("ai: merge capability field %q: %w", structType.Field(fieldIndex).Name, err)
		}
		merged.Field(fieldIndex).Set(value)
	}
	if pointer {
		result := reflect.New(structType)
		result.Elem().Set(merged)
		return result.Interface().(Capability), nil
	}
	return merged.Interface().(Capability), nil
}

func combineCapabilityLayer(capabilities []Capability) ([]Capability, error) {
	roots := expandCombinedCapabilities(capabilities)
	byID := make(map[string][]int)
	for index, capability := range roots {
		if capabilityIsNil(capability) {
			return nil, fmt.Errorf("capability must not be nil")
		}
		if id := capabilityIdentity(capability); id != "" {
			byID[id] = append(byID[id], index)
		}
	}
	removed := make([]bool, len(roots))
	for id, indexes := range byID {
		if len(indexes) < 2 {
			continue
		}
		identityType := reflect.TypeOf(roots[indexes[0]])
		for _, index := range indexes[1:] {
			if reflect.TypeOf(roots[index]) != identityType {
				return nil, fmt.Errorf(
					"capability ID %q is used by capabilities of different types (%T, %T)",
					id, roots[indexes[0]], roots[index],
				)
			}
		}
		last := indexes[len(indexes)-1]
		combiner, ok := roots[last].(CapabilityCombiner)
		if !ok {
			return nil, fmt.Errorf(
				"capability ID %q is used by multiple capabilities; implement CapabilityCombiner or use distinct IDs", id,
			)
		}
		group := make([]Capability, len(indexes))
		for groupIndex, index := range indexes {
			group[groupIndex] = roots[index]
			if index != last {
				removed[index] = true
			}
		}
		combined, err := combiner.CombineCapabilities(group)
		if err != nil {
			return nil, fmt.Errorf("combine capability ID %q: %w", id, err)
		}
		if capabilityIsNil(combined) {
			return nil, fmt.Errorf("combine capability ID %q returned nil", id)
		}
		if combinedID := capabilityIdentity(combined); combinedID != id {
			return nil, fmt.Errorf("combine capability ID %q returned capability ID %q", id, combinedID)
		}
		roots[last] = combined
	}
	result := make([]Capability, 0, len(roots))
	for index, capability := range roots {
		if !removed[index] {
			result = append(result, capability)
		}
	}
	return result, nil
}

func capabilityEntriesForRoots(roots []Capability) []capabilityEntry {
	var entries []capabilityEntry
	for _, root := range roots {
		rootID := capabilityIdentity(root)
		for _, capability := range flattenCapabilities([]Capability{root}) {
			entries = append(entries, capabilityEntry{capability: capability, rootID: rootID})
		}
	}
	return entries
}

func expandCombinedCapabilities(capabilities []Capability) []Capability {
	var roots []Capability
	for _, capability := range capabilities {
		if combined, ok := capability.(combinedCapability); ok {
			roots = append(roots, expandCombinedCapabilities(combined.capabilities)...)
			continue
		}
		roots = append(roots, capability)
	}
	return roots
}

func capabilityIdentity(capability Capability) string {
	if provider, ok := capability.(CapabilityIDProvider); ok {
		if id := strings.TrimSpace(provider.CapabilityID()); id != "" {
			return id
		}
	}
	if wrapper, ok := capability.(interface{ wrappedCapability() Capability }); ok {
		return capabilityIdentity(wrapper.wrappedCapability())
	}
	return ""
}

func capabilityIdentityType(capability Capability) reflect.Type {
	if provider, ok := capability.(CapabilityIDProvider); ok && strings.TrimSpace(provider.CapabilityID()) != "" {
		return reflect.TypeOf(capability)
	}
	if wrapper, ok := capability.(interface{ wrappedCapability() Capability }); ok {
		return capabilityIdentityType(wrapper.wrappedCapability())
	}
	return reflect.TypeOf(capability)
}

func incompatibleCapabilityIDError(id string, first, second Capability) error {
	return fmt.Errorf(
		"capability ID %q is used by capabilities of different types (%v, %v)",
		id, capabilityIdentityType(first), capabilityIdentityType(second),
	)
}

func validateCapabilityLayers(agent, run []Capability) error {
	agentByID := make(map[string]Capability)
	for _, capability := range agent {
		if id := capabilityIdentity(capability); id != "" {
			agentByID[id] = capability
		}
	}
	for _, capability := range run {
		id := capabilityIdentity(capability)
		previous, ok := agentByID[id]
		if id == "" || !ok {
			continue
		}
		if capabilityIdentityType(previous) != capabilityIdentityType(capability) {
			return incompatibleCapabilityIDError(id, previous, capability)
		}
	}
	return nil
}

func mergeCapabilityField(values []reflect.Value) (reflect.Value, error) {
	stated := values[:0]
	for _, value := range values {
		if isNilCapabilityField(value) {
			continue
		}
		stated = append(stated, value)
	}
	if len(stated) == 0 {
		return reflect.Zero(values[0].Type()), nil
	}
	allEqual := true
	for _, value := range stated[1:] {
		if !reflect.DeepEqual(stated[0].Interface(), value.Interface()) {
			allEqual = false
			break
		}
	}
	if allEqual {
		return cloneReflectValue(stated[0]), nil
	}

	kind := stated[0].Kind()
	if kind == reflect.Interface {
		dynamic := make([]reflect.Value, len(stated))
		collection := true
		for index, value := range stated {
			dynamic[index] = value.Elem()
			if dynamic[index].Kind() != reflect.Map && dynamic[index].Kind() != reflect.Slice {
				collection = false
			}
		}
		if collection {
			for _, value := range dynamic[1:] {
				if value.Type() != dynamic[0].Type() {
					return reflect.Value{}, fmt.Errorf(
						"collection types %v and %v cannot be rebuilt as one declared value",
						dynamic[0].Type(), value.Type(),
					)
				}
			}
			merged, _ := mergeCapabilityField(dynamic)
			result := reflect.New(stated[0].Type()).Elem()
			result.Set(merged)
			return result, nil
		}
		return cloneReflectValue(stated[len(stated)-1]), nil
	}
	switch kind {
	case reflect.Map:
		merged := reflect.MakeMapWithSize(stated[0].Type(), 0)
		for _, value := range stated {
			iterator := value.MapRange()
			for iterator.Next() {
				merged.SetMapIndex(iterator.Key(), iterator.Value())
			}
		}
		return merged, nil
	case reflect.Slice:
		merged := reflect.MakeSlice(stated[0].Type(), 0, 0)
		for _, value := range stated {
			for index := 0; index < value.Len(); index++ {
				candidate := value.Index(index)
				found := false
				for kept := 0; kept < merged.Len(); kept++ {
					if reflect.DeepEqual(candidate.Interface(), merged.Index(kept).Interface()) {
						found = true
						break
					}
				}
				if !found {
					merged = reflect.Append(merged, candidate)
				}
			}
		}
		return merged, nil
	default:
		return cloneReflectValue(stated[len(stated)-1]), nil
	}
}

func isNilCapabilityField(value reflect.Value) bool {
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func cloneReflectValue(value reflect.Value) reflect.Value {
	switch value.Kind() {
	case reflect.Map:
		cloned := reflect.MakeMapWithSize(value.Type(), value.Len())
		iterator := value.MapRange()
		for iterator.Next() {
			cloned.SetMapIndex(iterator.Key(), iterator.Value())
		}
		return cloned
	case reflect.Slice:
		cloned := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		reflect.Copy(cloned, value)
		return cloned
	default:
		return value
	}
}
