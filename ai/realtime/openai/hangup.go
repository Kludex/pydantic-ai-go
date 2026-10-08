package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/internal/openaiprotocol"
)

// HangUp ends a browser Realtime API call without relying on the browser to disconnect.
func (model *Model) HangUp(ctx context.Context, session realtime.ProviderSession) error {
	return model.hangUp(ctx, session, "realtime/calls", "call_id_not_found")
}

func (model *Model) hangUp(ctx context.Context, session realtime.ProviderSession, path, missingCode string) error {
	if session == nil || session.ProviderName() != "openai" || session.SessionID() == "" {
		return fmt.Errorf("openai realtime: hangup requires an OpenAI session")
	}
	endpoint, err := httpEndpoint(model.baseURL, path)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	request, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint+"/"+url.PathEscape(session.SessionID())+"/hangup", nil)
	request.Header = openaiprotocol.CloneHeader(model.headers)
	request.Header.Set("Authorization", "Bearer "+model.apiKey)
	response, err := model.client.Do(request)
	if err != nil {
		return fmt.Errorf("openai realtime: hangup: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode >= 200 && response.StatusCode < 300 {
		return nil
	}
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}
	var failure struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	_ = json.Unmarshal(data, &failure)
	if response.StatusCode == http.StatusBadRequest && failure.Error.Code == missingCode {
		return nil
	}
	return fmt.Errorf("openai realtime: hangup returned %s: %s", response.Status, data)
}
