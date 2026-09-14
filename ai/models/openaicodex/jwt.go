package openaicodex

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"strings"
	"time"
)

func jwtExpiresAt(token string) (time.Time, bool) {
	payload, ok := jwtPayload(token)
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

func accountIDFromJWT(token string) string {
	payload, ok := jwtPayload(token)
	if !ok {
		return ""
	}
	var nested struct {
		AccountID string `json:"chatgpt_account_id"`
	}
	if raw := payload["https://api.openai.com/auth"]; len(raw) > 0 {
		_ = json.Unmarshal(raw, &nested)
	}
	if nested.AccountID != "" {
		return nested.AccountID
	}
	for _, name := range []string{"chatgpt_account_id", "account_id"} {
		var accountID string
		if json.Unmarshal(payload[name], &accountID) == nil && accountID != "" {
			return accountID
		}
	}
	return ""
}

func jwtPayload(token string) (map[string]json.RawMessage, bool) {
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
