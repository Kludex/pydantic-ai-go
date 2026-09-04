package azure_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/realtime"
	azurert "github.com/Kludex/pydantic-ai-go/realtime/azure"
	openairt "github.com/Kludex/pydantic-ai-go/realtime/openai"
	"github.com/coder/websocket"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (errorReader) Close() error             { return nil }

func TestAzureConfigurationValidation(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	t.Setenv("AZURE_OPENAI_API_KEY", "")
	t.Setenv("AZURE_VOICELIVE_ENDPOINT", "")
	t.Setenv("AZURE_VOICELIVE_API_KEY", "")
	t.Setenv("AZURE_VOICELIVE_API_VERSION", "")
	if _, err := azurert.NewModel("", azurert.Config{}); err == nil {
		t.Fatal("expected empty deployment error")
	}
	if _, err := azurert.NewModel("model", azurert.Config{Endpoint: "https://example.com"}); err == nil {
		t.Fatal("expected missing authentication error")
	}
	if _, err := azurert.NewModel("model", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", TokenProvider: func(context.Context) (string, error) { return "", nil },
	}); err == nil {
		t.Fatal("expected conflicting authentication error")
	}
	if _, err := azurert.NewModel("model", azurert.Config{Endpoint: "://bad", APIKey: "key"}); err == nil {
		t.Fatal("expected endpoint error")
	}

	t.Setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
	t.Setenv("AZURE_OPENAI_API_KEY", "env-key")
	t.Setenv("AZURE_VOICELIVE_ENDPOINT", "https://voice.example.com")
	t.Setenv("AZURE_VOICELIVE_API_KEY", "voice-key")
	t.Setenv("AZURE_VOICELIVE_API_VERSION", "preview")
	model, err := azurert.NewModel("gpt-realtime", azurert.Config{}, azurert.WithSettings(azurert.Settings{VoiceLive: true}))
	if err != nil || model.Profile().SupportsWebRTC {
		t.Fatalf("environment configuration failed: model=%v err=%v", model, err)
	}

	model, err = azurert.NewModel("gpt-realtime", azurert.Config{Endpoint: "https://example.com", APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Provider: map[string]any{"azure_voice_live": true},
	}}); err == nil {
		t.Fatal("expected missing Voice Live endpoint error")
	}

	gaOnly, err := azurert.NewModel("gpt-realtime-2", azurert.Config{Endpoint: "https://example.com", APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gaOnly.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Provider: map[string]any{"azure_voice_live": true},
	}}); err == nil {
		t.Fatal("expected GA-only routing error")
	}
}

func TestAzureSessionSettings(t *testing.T) {
	for _, test := range []struct {
		name     string
		model    string
		settings azurert.Settings
		common   realtime.Settings
	}{
		{
			name: "ga portable", model: "gpt-realtime",
			common: realtime.Settings{
				OutputModality: realtime.OutputModalityAudio, InputTranscriptionModel: ptr(""),
				TurnDetection: &realtime.TurnDetection{
					Enabled: true, Sensitivity: "low", PrefixPadding: time.Millisecond, SilenceDuration: 2 * time.Millisecond,
				},
			},
		},
		{
			name: "ga manual", model: "gpt-realtime",
			common: realtime.Settings{
				OutputModality: realtime.OutputModalityAudio,
				TurnDetection:  &realtime.TurnDetection{Enabled: false},
			},
		},
		{
			name: "voice portable", model: "gpt-realtime",
			common: realtime.Settings{
				Provider: map[string]any{"azure_voice_live": true},
				TurnDetection: &realtime.TurnDetection{
					Enabled: true, Sensitivity: "high", PrefixPadding: time.Millisecond, SilenceDuration: time.Millisecond,
				},
			},
		},
		{
			name: "voice overrides", model: "gpt-realtime",
			settings: azurert.Settings{VoiceLive: true, OpenAI: openairt.Settings{Voice: "alloy"}},
			common: realtime.Settings{
				OutputModality: realtime.OutputModalityText, MaxTokens: 10, ToolChoice: realtime.ToolChoiceRequired,
				InputTranscriptionModel: ptr("auto"),
				Provider: map[string]any{
					"azure_voice_live":                true,
					"azure_voice_live_turn_detection": map[string]any{"type": "semantic_vad"},
					"openai_voice":                    "coral", "openai_input_noise_reduction": "far_field",
					"openai_output_speed": 1.25, "openai_turn_detection": map[string]any{"type": "server_vad"},
					"openai_truncation": "auto",
				},
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := handshakeServer(t, func(*http.Request) {})
			defer server.Close()
			config := azurert.Config{
				Endpoint: server.URL, APIKey: "key", VoiceLiveEndpoint: server.URL,
				VoiceLiveAPIKey: "voice", HTTPClient: server.Client(),
			}
			model, err := azurert.NewModel(test.model, config, azurert.WithSettings(test.settings))
			if err != nil {
				t.Fatal(err)
			}
			connection, err := model.Connect(t.Context(), realtime.ConnectParams{
				Settings: test.common,
				Request:  ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_ = connection.Close(t.Context())
		})
	}

	server := handshakeServer(t, func(*http.Request) {})
	defer server.Close()
	model, _ := azurert.NewModel("gpt-realtime", azurert.Config{
		Endpoint: server.URL, APIKey: "key", HTTPClient: server.Client(),
	}, azurert.WithSettings(azurert.Settings{OpenAI: openairt.Settings{OutputSpeed: 2}}))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected output speed error")
	}
}

func TestAzureInvalidVoiceEndpoint(t *testing.T) {
	t.Setenv("AZURE_VOICELIVE_ENDPOINT", "")
	for _, endpoint := range []string{"", "://bad"} {
		model, err := azurert.NewModel("gpt-5", azurert.Config{
			Endpoint: "https://example.com", APIKey: "key", VoiceLiveEndpoint: endpoint, VoiceLiveAPIKey: "voice",
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
			t.Fatalf("expected Voice Live endpoint error for %q", endpoint)
		}
	}
}

func TestAzureAuthenticationErrors(t *testing.T) {
	for _, provider := range []azurert.TokenProvider{
		func(context.Context) (string, error) { return "", errors.New("token failed") },
		func(context.Context) (string, error) { return "", nil },
	} {
		model, err := azurert.NewModel("model", azurert.Config{
			Endpoint: "https://example.com", TokenProvider: provider,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
			t.Fatal("expected token error")
		}
	}
	model, err := azurert.NewModel("gpt-5", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", VoiceLiveEndpoint: "https://voice.example.com",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected Voice Live key error")
	}
}

func TestAzureHandshakeErrors(t *testing.T) {
	for name, handler := range map[string]http.HandlerFunc{
		"status": func(writer http.ResponseWriter, _ *http.Request) { http.Error(writer, "bad", http.StatusUnauthorized) },
		"closed": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			_ = socket.Close(websocket.StatusInternalError, "closed")
		},
		"malformed": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			defer func() { _ = socket.CloseNow() }()
			_ = socket.Write(request.Context(), websocket.MessageText, []byte(`{`))
		},
		"provider": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			defer func() { _ = socket.CloseNow() }()
			_ = writeFrame(request.Context(), socket, map[string]any{"type": "error", "error": map[string]any{"message": "bad"}})
		},
		"binary": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			defer func() { _ = socket.CloseNow() }()
			_ = socket.Write(request.Context(), websocket.MessageBinary, []byte("ignored"))
			_ = writeFrame(request.Context(), socket, map[string]any{"type": "error", "error": map[string]any{"message": "bad"}})
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(handler)
			defer server.Close()
			model, err := azurert.NewModel("model", azurert.Config{
				Endpoint: server.URL, APIKey: "key", HTTPClient: server.Client(),
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
				t.Fatal("expected handshake error")
			}
		})
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network")
	})}
	model, _ := azurert.NewModel("model", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", HTTPClient: client,
	})
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected network error")
	}
}

func TestAzureSessionUpdateAndHandshakeCompletionErrors(t *testing.T) {
	for _, malformedTool := range []bool{false, true} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			socket, err := websocket.Accept(writer, request, nil)
			if err != nil {
				return
			}
			defer func() { _ = socket.CloseNow() }()
			_ = writeFrame(request.Context(), socket, map[string]any{
				"type": "session.created", "session": map[string]any{},
			})
			if !malformedTool {
				_, _, _ = socket.Read(request.Context())
			}
			_ = socket.Close(websocket.StatusInternalError, "stop")
		}))
		model, _ := azurert.NewModel("model", azurert.Config{
			Endpoint: server.URL, APIKey: "key", HTTPClient: server.Client(),
		})
		params := realtime.ConnectParams{}
		if malformedTool {
			params.Request.Tools = []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{"bad": func() {}}}}
		}
		if _, err := model.Connect(t.Context(), params); err == nil {
			t.Fatal("expected session configuration failure")
		}
		server.Close()
	}
}

func TestAzureClientSecretErrors(t *testing.T) {
	voice, _ := azurert.NewModel("gpt-5", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", VoiceLiveEndpoint: "https://voice.example.com",
		VoiceLiveAPIKey: "voice",
	})
	if _, err := voice.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
		t.Fatal("expected Voice Live WebRTC error")
	}
	gaOnly, _ := azurert.NewModel("gpt-realtime-2", azurert.Config{Endpoint: "https://example.com", APIKey: "key"})
	if _, err := gaOnly.CreateClientSecret(t.Context(), "", nil, realtime.Settings{
		Provider: map[string]any{"azure_voice_live": true},
	}, 0); err == nil {
		t.Fatal("expected route error")
	}
	badSettings := realtime.Settings{Provider: map[string]any{"openai_output_speed": 2.0}}
	if _, err := gaOnly.CreateClientSecret(t.Context(), "", nil, badSettings, 0); err == nil {
		t.Fatal("expected settings error")
	}
	if _, err := gaOnly.AnswerWebRTCOffer(t.Context(), "offer", "", nil, badSettings); err == nil {
		t.Fatal("expected signaling client-secret error")
	}
	invalidPayload := realtime.Settings{Provider: map[string]any{"openai_truncation": func() {}}}
	if _, err := gaOnly.CreateClientSecret(t.Context(), "", nil, invalidPayload, 0); err == nil {
		t.Fatal("expected client-secret serialization error")
	}

	for name, handler := range map[string]http.HandlerFunc{
		"status":     func(writer http.ResponseWriter, _ *http.Request) { http.Error(writer, "bad", http.StatusBadRequest) },
		"json":       func(writer http.ResponseWriter, _ *http.Request) { _, _ = writer.Write([]byte(`{`)) },
		"incomplete": func(writer http.ResponseWriter, _ *http.Request) { _, _ = writer.Write([]byte(`{}`)) },
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(handler)
			defer server.Close()
			model, _ := azurert.NewModel("model", azurert.Config{
				Endpoint: server.URL, APIKey: "key", HTTPClient: server.Client(),
			})
			if _, err := model.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
				t.Fatal("expected client secret error")
			}
		})
	}
}

func TestAzureClientSecretTransportAndAuthErrors(t *testing.T) {
	model, err := azurert.NewModel("model", azurert.Config{
		Endpoint: "https://example.com", TokenProvider: func(context.Context) (string, error) {
			return "", errors.New("token")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
		t.Fatal("expected client-secret token error")
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network")
	})}
	model, err = azurert.NewModel("model", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
		t.Fatal("expected client-secret transport error")
	}
}

func TestAzureWebRTCErrorsAndSideband(t *testing.T) {
	model, _ := azurert.NewModel("model", azurert.Config{Endpoint: "https://example.com", APIKey: "key"})
	if _, err := model.AnswerWebRTCOffer(t.Context(), "", "", nil, realtime.Settings{}); err == nil {
		t.Fatal("expected empty offer error")
	}
	if _, err := model.ConnectWebRTC(t.Context(), providerSession{}, realtime.ConnectParams{}); err == nil {
		t.Fatal("expected invalid sideband session error")
	}
	gaOnly, _ := azurert.NewModel("gpt-realtime-2", azurert.Config{Endpoint: "https://example.com", APIKey: "key"})
	if _, err := gaOnly.ConnectWebRTC(
		t.Context(), providerSession{provider: "azure", id: "call"},
		realtime.ConnectParams{Settings: realtime.Settings{Provider: map[string]any{"azure_voice_live": true}}},
	); err == nil {
		t.Fatal("expected sideband routing error")
	}

	server := handshakeServer(t, func(request *http.Request) {
		if request.URL.Query().Get("call_id") != "call" {
			t.Errorf("missing call ID: %s", request.URL)
		}
	})
	defer server.Close()
	sideband, _ := azurert.NewModel("model", azurert.Config{
		Endpoint: server.URL, APIKey: "key", HTTPClient: server.Client(),
	})
	connection, err := sideband.ConnectWebRTC(t.Context(), providerSession{provider: "azure", id: "call"}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())

	statusServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		http.Error(writer, "bad", http.StatusUnauthorized)
	}))
	defer statusServer.Close()
	statusModel, _ := azurert.NewModel("model", azurert.Config{
		Endpoint: statusServer.URL, APIKey: "key", HTTPClient: statusServer.Client(),
	})
	if _, err := statusModel.ConnectWebRTC(
		t.Context(), providerSession{provider: "azure", id: "call"}, realtime.ConnectParams{},
	); err == nil {
		t.Fatal("expected sideband handshake error")
	}

	voice, _ := azurert.NewModel("gpt-5", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", VoiceLiveEndpoint: "https://voice.example.com", VoiceLiveAPIKey: "voice",
	})
	if _, err := voice.ConnectWebRTC(t.Context(), providerSession{provider: "azure", id: "call"}, realtime.ConnectParams{}); err == nil {
		t.Fatal("expected Voice Live sideband error")
	}
}

func TestAzureWebRTCResponseErrors(t *testing.T) {
	for name, calls := range map[string]http.HandlerFunc{
		"status":   func(writer http.ResponseWriter, _ *http.Request) { http.Error(writer, "bad", http.StatusBadRequest) },
		"location": func(writer http.ResponseWriter, _ *http.Request) { _, _ = writer.Write([]byte("answer")) },
		"query": func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Location", "/calls?call_id=query")
			_, _ = writer.Write([]byte("answer"))
		},
		"malformed": func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Location", "%")
			_, _ = writer.Write([]byte("answer"))
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path == "/openai/v1/realtime/client_secrets" {
					_, _ = writer.Write([]byte(`{"value":"secret","expires_at":2000000000}`))
					return
				}
				calls(writer, request)
			}))
			defer server.Close()
			model, _ := azurert.NewModel("model", azurert.Config{
				Endpoint: server.URL, APIKey: "key", HTTPClient: server.Client(),
			})
			answer, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{})
			if name == "query" {
				if err != nil || answer.Session.ID != "query" {
					t.Fatalf("unexpected query answer: %+v %v", answer, err)
				}
			} else if err == nil {
				t.Fatal("expected WebRTC response error")
			}
		})
	}

	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if strings.HasSuffix(request.URL.Path, "client_secrets") {
			return &http.Response{
				StatusCode: 200, Status: "200 OK", Header: http.Header{},
				Body: io.NopCloser(strings.NewReader(`{"value":"secret","expires_at":2000000000}`)),
			}, nil
		}
		return nil, errors.New("network")
	})}
	model, _ := azurert.NewModel("model", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", HTTPClient: client,
	})
	if _, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{}); err == nil {
		t.Fatal("expected negotiation transport error")
	}

	readClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if strings.HasSuffix(request.URL.Path, "client_secrets") {
			return &http.Response{
				StatusCode: 200, Status: "200 OK", Header: http.Header{},
				Body: io.NopCloser(strings.NewReader(`{"value":"secret","expires_at":2000000000}`)),
			}, nil
		}
		return &http.Response{
			StatusCode: 200, Status: "200 OK", Header: http.Header{"Location": []string{"/calls/id"}}, Body: errorReader{},
		}, nil
	})}
	model, _ = azurert.NewModel("model", azurert.Config{
		Endpoint: "https://example.com", APIKey: "key", HTTPClient: readClient,
	})
	if _, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{}); err == nil {
		t.Fatal("expected answer read error")
	}
}

func ptr[T any](value T) *T { return &value }
