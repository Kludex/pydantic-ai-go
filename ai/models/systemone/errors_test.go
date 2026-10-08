package systemone_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/models/systemone"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }
func (failingBody) Close() error             { return nil }

func TestTransportAndRequestErrors(t *testing.T) {
	model := newModel(t, systemone.WithBaseURL("https://example.com"))
	params := boolParams()
	params.Settings.ExtraBody = map[string]any{"invalid": make(chan int)}
	if _, err := model.Request(t.Context(), prompt(), params); err == nil {
		t.Fatal("expected encoding error")
	}
	failure := errors.New("prepare failed")
	model = newModel(t, systemone.WithProvider(systemone.ProviderConfig{Name: "system-one", BaseURL: "https://example.com", PrepareRequest: func(*http.Request) error { return failure }}))
	if _, err := model.Request(t.Context(), prompt(), boolParams()); !errors.Is(err, failure) {
		t.Fatalf("unexpected preparation error: %v", err)
	}
	for _, transport := range []roundTripFunc{
		func(*http.Request) (*http.Response, error) { return nil, io.ErrUnexpectedEOF },
		func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: failingBody{}}, nil
		},
	} {
		model = newModel(t, systemone.WithBaseURL("https://example.com"), systemone.WithHTTPClient(&http.Client{Transport: transport}))
		_, err := model.Request(t.Context(), prompt(), boolParams())
		var transportError *ai.ModelTransportError
		if !errors.As(err, &transportError) || !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("unexpected transport error: %v", err)
		}
	}
	model = newModel(t, systemone.WithBaseURL("https://example.com"), systemone.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if _, exists := request.Context().Deadline(); !exists {
			t.Error("missing request timeout")
		}
		<-request.Context().Done()
		return nil, request.Context().Err()
	})}))
	params = boolParams()
	params.Settings.RequestTimeout = time.Millisecond
	if _, err := model.Request(t.Context(), prompt(), params); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected timeout: %v", err)
	}
	systemone.WithBaseURL(":")(model)
	if _, err := model.Request(t.Context(), prompt(), boolParams()); err == nil {
		t.Fatal("expected invalid request URL")
	}
}

func TestHTTPErrorAndFallback(t *testing.T) {
	for _, body := range []string{`{"detail":"unknown model"}`, "warming up"} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Retry-After", "2")
			writer.WriteHeader(http.StatusServiceUnavailable)
			_, _ = writer.Write([]byte(body))
		}))
		model := newModel(t, systemone.WithBaseURL(server.URL))
		_, err := model.Request(t.Context(), prompt(), boolParams())
		var apiError *systemone.APIError
		if !errors.As(err, &apiError) || !apiError.IsModelAPIError() || apiError.StatusCode != 503 || apiError.ModelName != "clm-latest" || apiError.Body != body || apiError.Headers.Get("Retry-After") != "2" || !strings.Contains(apiError.Error(), "503") {
			t.Fatalf("unexpected HTTP error: %v", err)
		}
		fallback := ai.NewFallbackModel(model, ai.WithFallbackModels(fakes.NewTestModel()))
		result, err := ai.NewAgent[struct{}, ticket](fallback).Run(t.Context(), "Charged twice.", struct{}{})
		if err != nil || result == nil {
			t.Fatalf("fallback result=%+v error=%v", result, err)
		}
		server.Close()
	}
}

func TestRouteHandOffs(t *testing.T) {
	for _, unsure := range []bool{false, true} {
		model := serverModel(t, `{"model":"clm-1","answers":{"route":{"type":"choice","choice":"write","confidence":0.8,"probabilities":{"write":0.8,"out":0.2}},"out.response":{"type":"noul","noul":0.9}},"usage":{}}`)
		params := boolParams()
		params.Tools = []ai.ToolDefinition{{Name: "write", Description: "Write text.", Schema: map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}}}}
		if unsure {
			var err error
			params.Settings, err = (systemone.Settings{RouteThreshold: floatPointer(.9)}).Build()
			if err != nil {
				t.Fatal(err)
			}
		}
		_, err := model.Request(t.Context(), prompt(), params)
		var handoff systemone.DecisionHandOff
		if !errors.As(err, &handoff) || !handoff.IsModelAPIError() {
			t.Fatalf("expected handoff: %v", err)
		}
		if route, probability := handoff.DecisionRoute(); route != "write" || probability != .8 {
			t.Fatalf("handoff=%s %g", route, probability)
		}
		if unsure {
			var unsureRoute *systemone.UnsureRoute
			if !errors.As(err, &unsureRoute) || unsureRoute.Threshold != .9 || unsureRoute.Probabilities["write"] != .8 || !strings.Contains(unsureRoute.Error(), "below") {
				t.Fatalf("unexpected unsure route: %v", err)
			}
		} else {
			var unfillable *systemone.UnfillableRoute
			if !errors.As(err, &unfillable) || unfillable.ToolName != "write" || !strings.Contains(unfillable.Error(), "cannot fill") {
				t.Fatalf("unexpected unfillable route: %v", err)
			}
		}
	}
}
