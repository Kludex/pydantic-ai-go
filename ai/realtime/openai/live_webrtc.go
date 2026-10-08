package openai

import (
	"context"
	"fmt"
	"net/url"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

// CreateClientSecret rejects ephemeral credentials, which GPT-Live does not support.
func (*LiveModel) CreateClientSecret(context.Context, string, []ai.ToolDefinition, realtime.Settings, time.Duration) (realtime.ClientSecret, error) {
	return realtime.ClientSecret{}, fmt.Errorf("openai GPT-Live: no client secrets; relay the offer with AnswerWebRTCOffer")
}

// AnswerWebRTCOffer starts a browser media session with server-owned instructions and tools.
func (model *LiveModel) AnswerWebRTCOffer(ctx context.Context, offer, instructions string, tools []ai.ToolDefinition, common realtime.Settings) (realtime.WebRTCAnswer, error) {
	if offer == "" {
		return realtime.WebRTCAnswer{}, fmt.Errorf("openai GPT-Live: SDP offer must not be empty")
	}
	settings, err := model.resolveLiveSettings(common)
	if err != nil {
		return realtime.WebRTCAnswer{}, err
	}
	config, err := model.liveConfig(realtime.ConnectParams{Settings: common, Request: ai.ModelRequestParams{Instructions: instructions, Tools: tools}}, settings, true)
	if err != nil {
		return realtime.WebRTCAnswer{}, err
	}
	var response struct {
		Session struct {
			ID string `json:"id"`
		} `json:"session"`
		Transport struct {
			SDP string `json:"sdp"`
		} `json:"transport"`
	}
	_, err = model.config.postJSON(ctx, "live/sessions", map[string]any{"session": config, "transport": map[string]any{"type": "webrtc", "sdp": offer}}, &response)
	if err != nil {
		return realtime.WebRTCAnswer{}, err
	}
	if response.Session.ID == "" || response.Transport.SDP == "" {
		return realtime.WebRTCAnswer{}, fmt.Errorf("openai GPT-Live: incomplete WebRTC answer")
	}
	return realtime.WebRTCAnswer{SDP: response.Transport.SDP, Session: realtime.WebRTCSession{Provider: "openai", ID: response.Session.ID}}, nil
}

// ConnectWebRTC attaches control to an immutable existing session without replaying history or forwarding audio.
func (model *LiveModel) ConnectWebRTC(ctx context.Context, session realtime.ProviderSession, params realtime.ConnectParams) (realtime.Connection, error) {
	if session == nil || session.ProviderName() != "openai" || session.SessionID() == "" {
		return nil, fmt.Errorf("openai GPT-Live: WebRTC session must be an OpenAI session")
	}
	seed, err := seedLiveItems(params.Messages, false)
	if err != nil {
		return nil, err
	}
	if len(seed) > 0 {
		return nil, fmt.Errorf("openai GPT-Live: a sideband cannot seed history after startup")
	}
	settings, err := model.resolveLiveSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	if _, err = model.liveConfig(params, settings, true); err != nil {
		return nil, err
	}
	socket, started, err := model.openLiveSocket(ctx, "live/sessions/"+url.PathEscape(session.SessionID())+"/attach", nil, params.Settings.HandshakeTimeout)
	if err != nil {
		return nil, err
	}
	return newLiveConnection(socket, started, params.Settings, settings, model.config.profile.AudioOutputSampleRate, model.config.baseURL, true, nil), nil
}

// HangUp ends an OpenAI GPT-Live WebRTC call. Closing a sideband alone leaves media running.
func (model *LiveModel) HangUp(ctx context.Context, session realtime.ProviderSession) error {
	return model.config.hangUp(ctx, session, "live/sessions", "session_id_not_found")
}
