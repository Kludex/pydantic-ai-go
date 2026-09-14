package ai

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	genaiprices "github.com/pydantic/genai-prices/packages/go"
)

const (
	defaultPriceUpdateInterval = time.Hour
	defaultPriceUpdateMaxBytes = 8 << 20
)

var (
	currentPriceCalculator atomic.Pointer[genaiprices.Calculator]
	defaultPriceHTTPClient = &http.Client{Timeout: 30 * time.Second}
	sharedPriceUpdates     = struct {
		sync.Mutex
		workers map[priceUpdateKey]*priceUpdateWorker
	}{workers: make(map[priceUpdateKey]*priceUpdateWorker)}
)

// PriceUpdateConfig controls background pricing-data downloads.
type PriceUpdateConfig struct {
	// HTTPClient performs downloads. Nil uses a shared client with a 30-second timeout.
	HTTPClient *http.Client
	// URL is the v2 provider-data endpoint. Empty uses genai-prices' RemoteDataURL.
	URL string
	// Interval is the delay between downloads. Zero defaults to one hour.
	Interval time.Duration
	// MaxBodyBytes bounds each response body. Zero defaults to 8 MiB.
	MaxBodyBytes int64
	// OnError receives download and validation failures. Nil discards them.
	OnError func(context.Context, error)
}

// PriceUpdater owns one subscription to background pricing-data updates.
type PriceUpdater struct {
	stopOnce sync.Once
	stopped  chan struct{}
	release  func()
}

// UpdatePricesInBackground downloads current prices now and then at the configured interval.
// The download never blocks the caller. Identical configurations without OnError share one worker.
func UpdatePricesInBackground(ctx context.Context, config PriceUpdateConfig) *PriceUpdater {
	resolved := resolvePriceUpdateConfig(config)
	updater := &PriceUpdater{stopped: make(chan struct{}), release: func() {}}
	if ctx.Err() != nil {
		updater.Stop()
		return updater
	}
	updater.release = acquirePriceUpdateWorker(ctx, resolved)
	go func() {
		select {
		case <-ctx.Done():
			updater.Stop()
		case <-updater.stopped:
		}
	}()
	return updater
}

// Stop releases this updater. It is safe to call Stop more than once.
// Stop waits for an unshared or last shared worker to finish.
func (updater *PriceUpdater) Stop() {
	if updater == nil {
		return
	}
	updater.stopOnce.Do(func() {
		if updater.stopped != nil {
			close(updater.stopped)
		}
		if updater.release != nil {
			updater.release()
		}
	})
}

type resolvedPriceUpdateConfig struct {
	client       *http.Client
	url          string
	interval     time.Duration
	maxBodyBytes int64
	onError      func(context.Context, error)
}

type priceUpdateKey struct {
	client       *http.Client
	url          string
	interval     time.Duration
	maxBodyBytes int64
}

type priceUpdateWorker struct {
	config resolvedPriceUpdateConfig
	cancel context.CancelFunc
	done   chan struct{}
	refs   int
}

func resolvePriceUpdateConfig(config PriceUpdateConfig) resolvedPriceUpdateConfig {
	if config.Interval < 0 {
		panic("ai: price update interval must not be negative")
	}
	if config.MaxBodyBytes < 0 || config.MaxBodyBytes == math.MaxInt64 {
		panic("ai: price update maximum body size is invalid")
	}
	client := config.HTTPClient
	if client == nil {
		client = defaultPriceHTTPClient
	}
	url := config.URL
	if url == "" {
		url = genaiprices.RemoteDataURL
	}
	interval := config.Interval
	if interval == 0 {
		interval = defaultPriceUpdateInterval
	}
	maximumBytes := config.MaxBodyBytes
	if maximumBytes == 0 {
		maximumBytes = defaultPriceUpdateMaxBytes
	}
	return resolvedPriceUpdateConfig{
		client: client, url: url, interval: interval, maxBodyBytes: maximumBytes, onError: config.OnError,
	}
}

func acquirePriceUpdateWorker(ctx context.Context, config resolvedPriceUpdateConfig) func() {
	if config.onError != nil {
		worker := startPriceUpdateWorker(ctx, config)
		return func() {
			worker.cancel()
			<-worker.done
		}
	}
	key := priceUpdateKey{
		client: config.client, url: config.url, interval: config.interval, maxBodyBytes: config.maxBodyBytes,
	}
	sharedPriceUpdates.Lock()
	worker := sharedPriceUpdates.workers[key]
	if worker == nil {
		worker = startPriceUpdateWorker(context.Background(), config)
		sharedPriceUpdates.workers[key] = worker
	}
	worker.refs++
	sharedPriceUpdates.Unlock()
	return func() {
		sharedPriceUpdates.Lock()
		worker.refs--
		last := worker.refs == 0
		if last {
			delete(sharedPriceUpdates.workers, key)
		}
		sharedPriceUpdates.Unlock()
		if last {
			worker.cancel()
			<-worker.done
		}
	}
}

func startPriceUpdateWorker(parent context.Context, config resolvedPriceUpdateConfig) *priceUpdateWorker {
	ctx, cancel := context.WithCancel(parent)
	worker := &priceUpdateWorker{config: config, cancel: cancel, done: make(chan struct{})}
	go worker.run(ctx)
	return worker
}

func (worker *priceUpdateWorker) run(ctx context.Context) {
	defer close(worker.done)
	ticker := time.NewTicker(worker.config.interval)
	defer ticker.Stop()
	for {
		calculator, err := downloadPriceCalculator(ctx, worker.config)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			if worker.config.onError != nil {
				worker.config.onError(ctx, err)
			}
		} else {
			if ctx.Err() != nil {
				return
			}
			currentPriceCalculator.Store(calculator)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func downloadPriceCalculator(
	ctx context.Context, config resolvedPriceUpdateConfig,
) (*genaiprices.Calculator, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, config.url, nil)
	if err != nil {
		return nil, fmt.Errorf("download prices: create request: %w", err)
	}
	response, err := config.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("download prices: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download prices: unexpected HTTP status %s", response.Status)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, config.maxBodyBytes+1))
	if err != nil {
		return nil, fmt.Errorf("download prices: read response: %w", err)
	}
	if int64(len(data)) > config.maxBodyBytes {
		return nil, fmt.Errorf("download prices: response exceeds %d bytes", config.maxBodyBytes)
	}
	calculator, err := genaiprices.NewCalculatorFromJSON(data)
	if err != nil {
		return nil, fmt.Errorf("download prices: %w", err)
	}
	return calculator, nil
}

func calculatePrice(
	calculator *genaiprices.Calculator, request genaiprices.PriceRequest,
) (genaiprices.PriceCalculation, error) {
	if calculator == nil {
		return genaiprices.Calculate(request)
	}
	return calculator.Calculate(request)
}
