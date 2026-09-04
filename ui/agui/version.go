package agui

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// DefaultVersion is the newest AG-UI version with explicit adapter behavior.
const DefaultVersion = "0.1.19"

type protocolVersion struct {
	major int
	minor int
	patch int
}

func parseVersion(value string) (protocolVersion, error) {
	if value == "" {
		value = DefaultVersion
	}
	end := 0
	for end < len(value) && (unicode.IsDigit(rune(value[end])) || value[end] == '.') {
		end++
	}
	parts := strings.Split(value[:end], ".")
	if end == 0 || len(parts) > 3 {
		return protocolVersion{}, fmt.Errorf("agui: invalid protocol version %q", value)
	}
	numbers := [3]int{}
	for index, part := range parts {
		if part == "" {
			return protocolVersion{}, fmt.Errorf("agui: invalid protocol version %q", value)
		}
		number, err := strconv.Atoi(part)
		if err != nil {
			return protocolVersion{}, fmt.Errorf("agui: invalid protocol version %q", value)
		}
		numbers[index] = number
	}
	return protocolVersion{major: numbers[0], minor: numbers[1], patch: numbers[2]}, nil
}

func (version protocolVersion) atLeast(major int, minor int, patch int) bool {
	if version.major != major {
		return version.major > major
	}
	if version.minor != minor {
		return version.minor > minor
	}
	return version.patch >= patch
}
