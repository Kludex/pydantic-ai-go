package ai

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// InstructionSourceKind identifies the framework component that authored an
// addressable instruction.
type InstructionSourceKind string

const (
	// InstructionSourceAgent identifies agent-level instructions.
	InstructionSourceAgent InstructionSourceKind = "agent"
	// InstructionSourceToolset identifies toolset-contributed instructions.
	InstructionSourceToolset InstructionSourceKind = "toolset"
	// InstructionSourceCapability identifies capability-contributed instructions.
	InstructionSourceCapability InstructionSourceKind = "capability"
)

// InstructionSource identifies the agent, toolset, or capability that authored
// an instruction. ID is empty only for the singleton agent source.
type InstructionSource struct {
	// Kind identifies the framework component category.
	Kind InstructionSourceKind
	// ID identifies one toolset or capability and is empty for the agent.
	ID string
}

// String returns the stable serialized source key.
func (source InstructionSource) String() string {
	if source.Kind == InstructionSourceAgent {
		return string(InstructionSourceAgent)
	}
	if source.ID == "" {
		return ""
	}
	return string(source.Kind) + ":" + source.ID
}

// InstructionID is the stable address of one instruction block. An empty Name
// addresses every unnamed block contributed by the source.
type InstructionID struct {
	// Source identifies the component that authored the instruction.
	Source InstructionSource
	// Name identifies one block relative to its source.
	Name string
}

// String returns the colon-delimited persisted address.
func (id InstructionID) String() string {
	base := id.Source.String()
	if base == "" || id.Name == "" {
		return base
	}
	return base + ":" + id.Name
}

// MarshalJSON serializes an instruction ID as its stable string form.
func (id InstructionID) MarshalJSON() ([]byte, error) {
	if err := validateInstructionID(id); err != nil {
		return nil, err
	}
	return json.Marshal(id.String())
}

// UnmarshalJSON accepts stable IDs emitted by this package. Unknown source
// namespaces return an error when decoding an ID directly. InstructionPart
// treats them as unaddressable for forward compatibility.
func (id *InstructionID) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	parsed, ok := ParseInstructionID(value)
	if !ok {
		return fmt.Errorf("ai: invalid instruction ID %q", value)
	}
	*id = *parsed
	return nil
}

// ParseInstructionID parses a persisted instruction address. It returns false
// for malformed values and source namespaces unknown to this version.
func ParseInstructionID(value string) (*InstructionID, bool) {
	segments := strings.Split(value, ":")
	var id InstructionID
	switch {
	case len(segments) == 1 && segments[0] == string(InstructionSourceAgent):
		id.Source = InstructionSource{Kind: InstructionSourceAgent}
	case len(segments) == 2 && segments[0] == string(InstructionSourceAgent) && segments[1] != "":
		id.Source = InstructionSource{Kind: InstructionSourceAgent}
		id.Name = segments[1]
	case len(segments) == 2 && isIdentifiedInstructionSource(segments[0]) && segments[1] != "":
		id.Source = InstructionSource{Kind: InstructionSourceKind(segments[0]), ID: segments[1]}
	case len(segments) == 3 && isIdentifiedInstructionSource(segments[0]) && segments[1] != "" && segments[2] != "":
		id.Source = InstructionSource{Kind: InstructionSourceKind(segments[0]), ID: segments[1]}
		id.Name = segments[2]
	default:
		return nil, false
	}
	if validateInstructionID(id) != nil {
		return nil, false
	}
	return &id, true
}

// AgentInstructionID returns an address under the agent's instruction source.
func AgentInstructionID(name ...string) *InstructionID {
	return newInstructionID(InstructionSource{Kind: InstructionSourceAgent}, name)
}

// ToolsetInstructionID returns an address under an identified toolset.
func ToolsetInstructionID(toolsetID string, name ...string) *InstructionID {
	return newInstructionID(InstructionSource{Kind: InstructionSourceToolset, ID: toolsetID}, name)
}

// CapabilityInstructionID returns an address under an identified capability.
func CapabilityInstructionID(capabilityID string, name ...string) *InstructionID {
	return newInstructionID(InstructionSource{Kind: InstructionSourceCapability, ID: capabilityID}, name)
}

func newInstructionID(source InstructionSource, names []string) *InstructionID {
	if len(names) > 1 {
		panic("ai: instruction ID accepts at most one name")
	}
	id := &InstructionID{Source: source}
	if len(names) == 1 {
		id.Name = names[0]
	}
	if err := validateInstructionID(*id); err != nil {
		panic(err.Error())
	}
	return id
}

func isIdentifiedInstructionSource(value string) bool {
	return value == string(InstructionSourceToolset) || value == string(InstructionSourceCapability)
}

func validateInstructionID(id InstructionID) error {
	switch id.Source.Kind {
	case InstructionSourceAgent:
		if id.Source.ID != "" {
			return fmt.Errorf("ai: agent instruction source must not have an ID")
		}
	case InstructionSourceToolset, InstructionSourceCapability:
		if err := validateInstructionSegment(id.Source.ID, string(id.Source.Kind)+" ID"); err != nil {
			return err
		}
	default:
		return fmt.Errorf("ai: unknown instruction source %q", id.Source.Kind)
	}
	return validateInstructionName(id.Name, true)
}

func validateInstructionName(name string, allowEmpty bool) error {
	if name == "" && allowEmpty {
		return nil
	}
	if err := validateInstructionSegment(name, "instruction name"); err != nil {
		return err
	}
	if name == string(InstructionSourceAgent) {
		return fmt.Errorf("ai: instruction name %q is reserved for the agent's own instructions", name)
	}
	return nil
}

func validateInstructionSegment(value, kind string) error {
	if value == "" {
		return fmt.Errorf("ai: %s must not be empty", kind)
	}
	if strings.Contains(value, ":") {
		return fmt.Errorf("ai: %s %q cannot contain ':' because it is reserved as an instruction ID delimiter", kind, value)
	}
	return nil
}

// MarshalJSON emits the upstream-compatible instruction part shape.
func (part InstructionPart) MarshalJSON() ([]byte, error) {
	if err := validateInstructionName(part.Name, true); err != nil {
		return nil, err
	}
	var id *string
	if part.ID != nil {
		if err := validateInstructionID(*part.ID); err != nil {
			return nil, err
		}
		value := part.ID.String()
		id = &value
	}
	return json.Marshal(struct {
		Content  string  `json:"content"`
		Dynamic  bool    `json:"dynamic"`
		Name     *string `json:"name"`
		ID       *string `json:"id"`
		PartKind string  `json:"part_kind"`
	}{
		Content: part.Content, Dynamic: part.Dynamic, Name: optionalInstructionName(part.Name), ID: id,
		PartKind: "instruction",
	})
}

// UnmarshalJSON accepts instruction parts from current and older versions.
// IDs in unknown future namespaces are retained as unaddressable nil values.
func (part *InstructionPart) UnmarshalJSON(data []byte) error {
	var wire struct {
		Content  string  `json:"content"`
		Dynamic  bool    `json:"dynamic"`
		Name     *string `json:"name"`
		ID       *string `json:"id"`
		PartKind string  `json:"part_kind"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if wire.PartKind != "" && wire.PartKind != "instruction" {
		return fmt.Errorf("ai: invalid instruction part kind %q", wire.PartKind)
	}
	name := ""
	if wire.Name != nil {
		name = *wire.Name
	}
	if err := validateInstructionName(name, true); err != nil {
		return err
	}
	*part = InstructionPart{Content: wire.Content, Dynamic: wire.Dynamic, Name: name}
	if wire.ID != nil {
		if id, ok := ParseInstructionID(*wire.ID); ok {
			part.ID = id
		}
	}
	return nil
}

func optionalInstructionName(name string) *string {
	if name == "" {
		return nil
	}
	return &name
}

func cloneInstructionID(id *InstructionID) *InstructionID {
	if id == nil {
		return nil
	}
	cloned := *id
	return &cloned
}

func cloneInstructionParts(parts []InstructionPart) []InstructionPart {
	if parts == nil {
		return nil
	}
	cloned := make([]InstructionPart, len(parts))
	for index, part := range parts {
		part.ID = cloneInstructionID(part.ID)
		cloned[index] = part
	}
	return cloned
}

func qualifyInstructionPart(part InstructionPart, source *InstructionSource) (InstructionPart, error) {
	part.ID = cloneInstructionID(part.ID)
	if err := validateInstructionName(part.Name, true); err != nil {
		return InstructionPart{}, err
	}
	if source == nil {
		return part, nil
	}
	if err := validateInstructionID(InstructionID{Source: *source}); err != nil {
		return InstructionPart{}, err
	}
	if part.ID != nil && part.ID.Source == *source && (part.Name == "" || part.ID.Name == part.Name) {
		return part, nil
	}
	part.ID = &InstructionID{Source: *source, Name: part.Name}
	return part, nil
}

func capabilityInstructionSource(capability Capability) (*InstructionSource, error) {
	provider, ok := capability.(CapabilityIDProvider)
	if !ok {
		return nil, nil
	}
	id := provider.CapabilityID()
	if id == "" {
		return nil, nil
	}
	source := &InstructionSource{Kind: InstructionSourceCapability, ID: id}
	if err := validateInstructionID(InstructionID{Source: *source}); err != nil {
		return nil, err
	}
	return source, nil
}

func capabilityContributesInstructions(capability Capability, static []InstructionPart) bool {
	if len(static) > 0 {
		return true
	}
	if _, ok := capability.(InstructionPartsProvider); ok {
		return true
	}
	_, ok := capability.(InstructionsProvider)
	return ok
}

func qualifyInstructionParts(parts []InstructionPart, source *InstructionSource) ([]InstructionPart, error) {
	qualified := make([]InstructionPart, 0, len(parts))
	for _, part := range parts {
		if strings.TrimSpace(part.Content) == "" {
			continue
		}
		resolved, err := qualifyInstructionPart(part, source)
		if err != nil {
			return nil, err
		}
		resolved.Content = strings.TrimSpace(resolved.Content)
		qualified = append(qualified, resolved)
	}
	return qualified, nil
}

func recordToolsetInstructionOwners(owners map[string]int, owner int, parts []InstructionPart) error {
	for _, part := range parts {
		if part.ID == nil || part.ID.Source.Kind != InstructionSourceToolset {
			continue
		}
		sourceID := part.ID.Source.ID
		if previous, exists := owners[sourceID]; exists && previous != owner {
			return fmt.Errorf("toolset ID %q is used by multiple toolsets that contribute instructions", sourceID)
		}
		owners[sourceID] = owner
	}
	return nil
}

// JoinInstructionParts renders instruction blocks with one blank line between
// non-empty blocks. IDs do not affect rendering.
func JoinInstructionParts(parts []InstructionPart) string {
	instructions := make([]string, 0, len(parts))
	for _, part := range parts {
		if content := strings.TrimSpace(part.Content); content != "" {
			instructions = append(instructions, content)
		}
	}
	return strings.Join(instructions, "\n\n")
}

// SortInstructionParts returns a detached stable ordering with cacheable
// static blocks before dynamic blocks. IDs do not affect ordering.
func SortInstructionParts(parts []InstructionPart) []InstructionPart {
	parts = cloneInstructionParts(parts)
	slices.SortStableFunc(parts, func(left, right InstructionPart) int {
		switch {
		case left.Dynamic == right.Dynamic:
			return 0
		case left.Dynamic:
			return 1
		default:
			return -1
		}
	})
	return parts
}

func joinInstructionParts(parts []InstructionPart) string { return JoinInstructionParts(parts) }
