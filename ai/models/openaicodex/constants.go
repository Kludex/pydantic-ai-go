package openaicodex

import "time"

const (
	defaultBaseURL   = "https://chatgpt.com/backend-api/codex"
	authorizeURL     = "https://auth.openai.com/oauth/authorize"
	tokenURL         = "https://auth.openai.com/oauth/token"
	publicClientID   = "app_EMoamEEZ73f0CkXaXp7hrann"
	defaultRedirect  = "http://localhost:1455/auth/callback"
	defaultScope     = "openid profile email offline_access"
	credentialBuffer = 30 * time.Second
)
