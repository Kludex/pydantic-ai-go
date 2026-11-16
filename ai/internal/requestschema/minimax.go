package requestschema

import "strings"

// MiniMax reports whether name is a published MiniMax model id, such as minimax-m3.1-flash.
// These models stringify nested tool arguments whose schema is a $ref into $defs.
func MiniMax(name string) bool {
	switch strings.ToLower(name) {
	case "minimax-m3", "minimax-m3.1-flash", "minimax-m2.7", "minimax-m2.7-highspeed", "minimax-m2.5",
		"minimax-m2.5-highspeed", "minimax-m2.1", "minimax-m2.1-highspeed", "minimax-m2", "minimax-m1", "minimax-m1-80k":
		return true
	}
	return false
}
