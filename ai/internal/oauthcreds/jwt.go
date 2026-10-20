package oauthcreds

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"strings"
	"time"
)

// ExpiresAt returns the unverified exp claim, a refresh hint and never an authority.
func ExpiresAt(token string) (time.Time, bool) {
	payload, ok := Claims(token)
	if !ok {
		return time.Time{}, false
	}
	raw, exists := payload["exp"]
	if !exists {
		return time.Time{}, false
	}
	var seconds float64
	if err := json.Unmarshal(raw, &seconds); err != nil || math.IsInf(seconds, 0) || math.IsNaN(seconds) {
		return time.Time{}, false
	}
	whole, fraction := math.Modf(seconds)
	expires := time.Unix(int64(whole), int64(fraction*float64(time.Second))).UTC()
	if expires.Year() < 1 || expires.Year() > 9999 {
		return time.Time{}, false
	}
	return expires, true
}

// Claims decodes a JWT payload without verifying its signature.
func Claims(token string) (map[string]json.RawMessage, bool) {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return nil, false
	}
	data, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, false
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal(data, &payload) != nil || payload == nil {
		return nil, false
	}
	return payload, true
}
