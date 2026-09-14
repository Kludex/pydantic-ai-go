package ai_test

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

const testPrices = `[
	{
		"id":"testing","name":"Testing","api_pattern":"https://testing\\.example",
		"models":[{"id":"updated-model","match":{"equals":"updated-model"},"prices":{"input_mtok":2}}]
	},
	{
		"id":"openai","name":"OpenAI","api_pattern":"https://api\\.openai\\.com",
		"models":[{"id":"gpt-5","match":{"starts_with":"gpt-5"},"prices":{"input_mtok":1,"output_mtok":1,"cache_read_mtok":0.1}}]
	}
]`

const replacementTestPrices = `[
	{
		"id":"testing","name":"Testing","api_pattern":"https://testing\\.example",
		"models":[{"id":"updated-model","match":{"equals":"updated-model"},"prices":{"input_mtok":99}}]
	}
]`

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type priceUpdateContextKey struct{}

type failedPriceBody struct{}

func (failedPriceBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (failedPriceBody) Close() error             { return nil }

func TestUpdatePricesInBackground(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
		_, _ = io.WriteString(w, testPrices)
	}))
	defer server.Close()

	updater := ai.UpdatePricesInBackground(t.Context(), ai.PriceUpdateConfig{
		HTTPClient: server.Client(), URL: server.URL,
	})
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("the immediate price download did not start")
	}
	close(release)
	waitForUpdatedPrice(t, 2)
	updater.Stop()
	updater.Stop()
}

func TestPriceUpdatesUseConfiguredInterval(t *testing.T) {
	defer installTestPrices(t)
	firstServed := make(chan struct{})
	secondStarted := make(chan struct{})
	releaseSecond := make(chan struct{})
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		switch requests.Add(1) {
		case 1:
			_, _ = io.WriteString(w, testPrices)
			close(firstServed)
		case 2:
			close(secondStarted)
			<-releaseSecond
			_, _ = io.WriteString(w, replacementTestPrices)
		default:
			_, _ = io.WriteString(w, replacementTestPrices)
		}
	}))
	defer server.Close()

	updater := ai.UpdatePricesInBackground(t.Context(), ai.PriceUpdateConfig{
		HTTPClient: server.Client(), URL: server.URL, Interval: 5 * time.Millisecond,
	})
	<-firstServed
	<-secondStarted
	assertUpdatedPrice(t, 2)
	close(releaseSecond)
	waitForUpdatedPrice(t, 99)
	updater.Stop()
}

func TestPriceUpdateFailuresRetainLastGoodData(t *testing.T) {
	installTestPrices(t)
	statusServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer statusServer.Close()
	largeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "12345")
	}))
	defer largeServer.Close()
	invalidServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "not JSON")
	}))
	defer invalidServer.Close()

	tests := map[string]ai.PriceUpdateConfig{
		"status": {
			HTTPClient: statusServer.Client(), URL: statusServer.URL,
		},
		"body limit": {
			HTTPClient: largeServer.Client(), URL: largeServer.URL, MaxBodyBytes: 4,
		},
		"invalid data": {
			HTTPClient: invalidServer.Client(), URL: invalidServer.URL,
		},
		"invalid URL": {
			URL: "://invalid",
		},
		"read": {
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK, Status: "200 OK", Body: failedPriceBody{}, Header: make(http.Header),
				}, nil
			})},
			URL: "https://testing.example/prices",
		},
		"transport": {
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, errors.New("transport failed")
			})},
			URL: "https://testing.example/prices",
		},
	}
	for name, config := range tests {
		t.Run(name, func(t *testing.T) {
			reported := make(chan error, 1)
			config.OnError = func(ctx context.Context, err error) {
				if ctx.Value(priceUpdateContextKey{}) != name {
					t.Error("error callback did not retain the update context")
				}
				reported <- err
			}
			ctx := context.WithValue(t.Context(), priceUpdateContextKey{}, name)
			updater := ai.UpdatePricesInBackground(ctx, config)
			select {
			case err := <-reported:
				if err == nil || !strings.Contains(err.Error(), "download prices") {
					t.Fatalf("unexpected update error: %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("price update failure was not reported")
			}
			updater.Stop()
			assertUpdatedPrice(t, 2)
		})
	}
}

func TestPriceUpdaterCancellationAndStop(t *testing.T) {
	installTestPrices(t)
	t.Run("stop cancels download", func(t *testing.T) {
		started := make(chan struct{})
		canceled := make(chan struct{})
		reported := make(chan error, 1)
		client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			close(started)
			<-request.Context().Done()
			close(canceled)
			return nil, request.Context().Err()
		})}
		updater := ai.UpdatePricesInBackground(t.Context(), ai.PriceUpdateConfig{
			HTTPClient: client, URL: "https://testing.example/prices",
			OnError: func(_ context.Context, err error) {
				reported <- err
			},
		})
		<-started
		updater.Stop()
		<-canceled
		select {
		case err := <-reported:
			t.Fatalf("cancellation was reported as an update failure: %v", err)
		default:
		}
	})

	t.Run("context cancellation prevents a late swap", func(t *testing.T) {
		started := make(chan context.Context, 1)
		release := make(chan struct{})
		client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			started <- request.Context()
			<-release
			return &http.Response{
				StatusCode: http.StatusOK, Status: "200 OK", Body: io.NopCloser(strings.NewReader(replacementTestPrices)),
				Header: make(http.Header),
			}, nil
		})}
		ctx, cancel := context.WithCancel(t.Context())
		updater := ai.UpdatePricesInBackground(ctx, ai.PriceUpdateConfig{
			HTTPClient: client, URL: "https://testing.example/prices",
		})
		requestContext := <-started
		cancel()
		<-requestContext.Done()
		close(release)
		updater.Stop()
		assertUpdatedPrice(t, 2)
	})

	t.Run("already canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		updater := ai.UpdatePricesInBackground(ctx, ai.PriceUpdateConfig{})
		updater.Stop()
	})
}

func TestDefaultPriceUpdatesAreShared(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	canceled := make(chan struct{}, 1)
	var requests atomic.Int32
	originalTransport := http.DefaultTransport
	http.DefaultTransport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests.Add(1)
		started <- struct{}{}
		select {
		case <-release:
			return &http.Response{
				StatusCode: http.StatusOK, Status: "200 OK", Body: io.NopCloser(strings.NewReader(testPrices)),
				Header: make(http.Header),
			}, nil
		case <-request.Context().Done():
			canceled <- struct{}{}
			return nil, request.Context().Err()
		}
	})
	defer func() { http.DefaultTransport = originalTransport }()

	first := ai.UpdatePricesInBackground(t.Context(), ai.PriceUpdateConfig{})
	<-started
	second := ai.UpdatePricesInBackground(t.Context(), ai.PriceUpdateConfig{})
	select {
	case <-started:
		t.Fatal("default updaters started duplicate downloads")
	case <-time.After(20 * time.Millisecond):
	}
	first.Stop()
	select {
	case <-canceled:
		t.Fatal("stopping one subscriber canceled the shared download")
	default:
	}
	close(release)
	waitForUpdatedPrice(t, 2)
	second.Stop()
	if requests.Load() != 1 {
		t.Fatalf("default updaters made %d requests", requests.Load())
	}
}

func TestPriceUpdaterZeroValue(t *testing.T) {
	var updater ai.PriceUpdater
	updater.Stop()
	updater.Stop()
	var nilUpdater *ai.PriceUpdater
	nilUpdater.Stop()
}

func TestPriceUpdateConfigValidation(t *testing.T) {
	for name, config := range map[string]ai.PriceUpdateConfig{
		"interval":      {Interval: -time.Second},
		"body size":     {MaxBodyBytes: -1},
		"body overflow": {MaxBodyBytes: math.MaxInt64},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected invalid configuration to panic")
				}
			}()
			ai.UpdatePricesInBackground(t.Context(), config)
		})
	}
}

func installTestPrices(t *testing.T) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, testPrices)
	}))
	updater := ai.UpdatePricesInBackground(t.Context(), ai.PriceUpdateConfig{
		HTTPClient: server.Client(), URL: server.URL,
	})
	waitForUpdatedPrice(t, 2)
	updater.Stop()
	server.Close()
}

func waitForUpdatedPrice(t *testing.T, want float64) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		calculation, err := updatedModelResponse().Price()
		if err == nil && calculation.TotalPrice == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	calculation, err := updatedModelResponse().Price()
	t.Fatalf("price was not updated: calculation=%+v error=%v", calculation, err)
}

func assertUpdatedPrice(t *testing.T, want float64) {
	t.Helper()
	calculation, err := updatedModelResponse().Price()
	if err != nil || calculation.TotalPrice != want {
		t.Fatalf("unexpected current price: calculation=%+v error=%v", calculation, err)
	}
}

func updatedModelResponse() ai.ModelResponse {
	return ai.ModelResponse{
		ModelName: "updated-model", ProviderName: "testing", Usage: ai.Usage{InputTokens: 1_000_000},
	}
}
