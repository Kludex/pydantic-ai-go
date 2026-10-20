package openaicodex

import (
	"encoding/json"

	"github.com/Kludex/pydantic-ai-go/ai/internal/oauthcreds"
)

func accountIDFromJWT(token string) string {
	payload, ok := oauthcreds.Claims(token)
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
