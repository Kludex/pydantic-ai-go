// Package oauthcreds owns the credential lifecycle shared by models that
// authenticate with refreshable OAuth tokens instead of an API key: loading
// from caller-owned storage, proactive refresh from the JWT expiry hint,
// single-flight refresh after a 401, and handing rotated tokens back.
package oauthcreds

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"
)

// expiryBuffer refreshes a token this long before its unverified exp claim.
const expiryBuffer = 30 * time.Second

// Source is caller-owned storage for a credential set.
type Source[C any] interface {
	Load(ctx context.Context) (C, error)
	Save(ctx context.Context, credentials C) error
}

// Hooks supply what differs between providers.
type Hooks[C comparable] struct {
	// Refresh exchanges the refresh token in credentials for a new set.
	Refresh func(ctx context.Context, client *http.Client, credentials C) (C, error)
	// Validate rejects credentials the provider cannot use. It runs on every
	// loaded, adopted, and refreshed set before it reaches the API.
	Validate func(credentials C) error
	// AccessToken returns the bearer token, read for its expiry hint.
	AccessToken func(credentials C) string
	// Persistence wraps a failed Save into the provider's error type.
	Persistence func(err error) error
}

// Manager holds one credential set and rotates it.
type Manager[C comparable] struct {
	source Source[C]
	client *http.Client
	hooks  Hooks[C]

	mu            sync.Mutex
	credentials   *C
	revision      uint64
	failureCount  uint64
	lastFailure   refreshFailure
	loadFlight    *flight
	refreshFlight *flight

	// unsavedPredecessor is what the source still held when saving a refresh
	// failed: the set that refresh spent, which must not be adopted back.
	// Only the single refresh flight touches it.
	unsavedPredecessor *C
}

type refreshFailure struct {
	revision uint64
	err      error
}

type flight struct {
	done chan struct{}
	err  error
}

// persistenceError marks a failed Save, which a proactive refresh must still
// surface. The embedded error is the provider's own type.
type persistenceError struct{ error }

// New returns a manager for credentials, or for credentials loaded from
// source on first use. The client sends refresh requests.
func New[C comparable](credentials *C, source Source[C], client *http.Client, hooks Hooks[C]) *Manager[C] {
	return &Manager[C]{credentials: credentials, source: source, client: client, hooks: hooks}
}

// Current returns the credential set held in memory, if any.
func (manager *Manager[C]) Current() (C, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if manager.credentials == nil {
		var zero C
		return zero, false
	}
	return *manager.credentials, true
}

// Prepare returns credentials for an outgoing request, refreshing a stale
// token first, plus the revision and failure count a 401 refresh compares.
func (manager *Manager[C]) Prepare(ctx context.Context) (C, uint64, uint64, error) {
	var zero C
	if err := manager.load(ctx); err != nil {
		return zero, 0, 0, err
	}
	credentials, revision, failures := manager.snapshot()
	if err := manager.hooks.Validate(credentials); err != nil {
		return zero, 0, 0, err
	}
	if expiresAt, ok := ExpiresAt(manager.hooks.AccessToken(credentials)); ok &&
		!time.Now().Before(expiresAt.Add(-expiryBuffer)) {
		// The expiry hint is best-effort: other failures fall through to the 401 path.
		if err := manager.refreshRevision(ctx, revision, failures); err != nil {
			var persistence *persistenceError
			if errors.As(err, &persistence) {
				return zero, 0, 0, persistence.error
			}
		}
		credentials, revision, failures = manager.snapshot()
		if err := manager.hooks.Validate(credentials); err != nil {
			return zero, 0, 0, err
		}
	}
	return credentials, revision, failures, nil
}

// Refresh rotates the credentials after a 401 on revision. A request that saw
// an older revision adopts the newer set instead; a failure of the flight a
// request waited on is shared rather than retried.
func (manager *Manager[C]) Refresh(ctx context.Context, revision, seenFailures uint64) error {
	err := manager.refreshRevision(ctx, revision, seenFailures)
	var persistence *persistenceError
	if errors.As(err, &persistence) {
		return persistence.error
	}
	return err
}

func (manager *Manager[C]) refreshRevision(ctx context.Context, revision, seenFailures uint64) error {
	manager.mu.Lock()
	if manager.revision != revision {
		manager.mu.Unlock()
		return nil
	}
	if manager.lastFailure.err != nil && manager.lastFailure.revision == revision && manager.failureCount != seenFailures {
		err := manager.lastFailure.err
		manager.mu.Unlock()
		return err
	}
	if manager.refreshFlight != nil {
		current := manager.refreshFlight
		manager.mu.Unlock()
		return wait(ctx, current)
	}
	current := &flight{done: make(chan struct{})}
	manager.refreshFlight = current
	rejected := *manager.credentials
	manager.mu.Unlock()

	err := manager.refresh(ctx, rejected)
	manager.mu.Lock()
	if err != nil {
		manager.failureCount++
		manager.lastFailure = refreshFailure{revision: revision, err: err}
	}
	current.err = err
	manager.refreshFlight = nil
	close(current.done)
	manager.mu.Unlock()
	return err
}

func (manager *Manager[C]) refresh(ctx context.Context, rejected C) error {
	var stored C
	if manager.source != nil {
		var err error
		stored, err = manager.source.Load(ctx)
		if err != nil {
			return err
		}
		if err := manager.hooks.Validate(stored); err != nil {
			return err
		}
		// Refresh tokens are single-use: adopt a set another process already rotated.
		if stored != rejected && (manager.unsavedPredecessor == nil || stored != *manager.unsavedPredecessor) {
			manager.replace(stored)
			manager.unsavedPredecessor = nil
			return nil
		}
	}
	credentials, err := manager.hooks.Refresh(ctx, manager.client, rejected)
	if err != nil {
		return err
	}
	manager.replace(credentials)
	if manager.source != nil {
		if err := manager.source.Save(ctx, credentials); err != nil {
			manager.unsavedPredecessor = &stored
			return &persistenceError{manager.hooks.Persistence(err)}
		}
		manager.unsavedPredecessor = nil
	}
	return nil
}

func (manager *Manager[C]) load(ctx context.Context) error {
	manager.mu.Lock()
	if manager.credentials != nil {
		manager.mu.Unlock()
		return nil
	}
	if manager.loadFlight != nil {
		current := manager.loadFlight
		manager.mu.Unlock()
		return wait(ctx, current)
	}
	current := &flight{done: make(chan struct{})}
	manager.loadFlight = current
	manager.mu.Unlock()

	credentials, err := manager.source.Load(ctx)
	if err == nil {
		err = manager.hooks.Validate(credentials)
	}
	manager.mu.Lock()
	if err == nil {
		value := credentials
		manager.credentials = &value
	}
	current.err = err
	manager.loadFlight = nil
	close(current.done)
	manager.mu.Unlock()
	return err
}

func (manager *Manager[C]) snapshot() (C, uint64, uint64) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return *manager.credentials, manager.revision, manager.failureCount
}

func (manager *Manager[C]) replace(credentials C) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	value := credentials
	manager.credentials = &value
	manager.revision++
	manager.lastFailure = refreshFailure{}
}

func wait(ctx context.Context, current *flight) error {
	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-current.done:
		return current.err
	}
}
