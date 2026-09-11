package openaicodex

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"
)

type credentialManager struct {
	source CredentialSource
	client *http.Client

	mu            sync.Mutex
	credentials   *Credentials
	revision      uint64
	failureCount  uint64
	lastFailure   refreshFailure
	loadFlight    *credentialFlight
	refreshFlight *credentialFlight
}

type refreshFailure struct {
	revision uint64
	err      error
}

type credentialFlight struct {
	done chan struct{}
	err  error
}

func newCredentialManager(credentials *Credentials, source CredentialSource, client *http.Client) *credentialManager {
	return &credentialManager{credentials: credentials, source: source, client: client}
}

func (manager *credentialManager) current() (Credentials, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if manager.credentials == nil {
		return Credentials{}, false
	}
	return *manager.credentials, true
}

func (manager *credentialManager) prepare(ctx context.Context) (Credentials, uint64, uint64, error) {
	if err := manager.load(ctx); err != nil {
		return Credentials{}, 0, 0, err
	}
	credentials, revision, failures := manager.snapshot()
	if expiresAt, ok := jwtExpiresAt(credentials.AccessToken); ok && !time.Now().Before(expiresAt.Add(-credentialBuffer)) {
		if err := manager.refresh(ctx, revision, failures); err != nil {
			var persistence *CredentialsPersistenceError
			if errors.As(err, &persistence) {
				return Credentials{}, 0, 0, err
			}
		}
		credentials, revision, failures = manager.snapshot()
	}
	return credentials, revision, failures, nil
}

func (manager *credentialManager) load(ctx context.Context) error {
	manager.mu.Lock()
	if manager.credentials != nil {
		manager.mu.Unlock()
		return nil
	}
	if manager.loadFlight != nil {
		flight := manager.loadFlight
		manager.mu.Unlock()
		return waitCredentialFlight(ctx, flight)
	}
	flight := &credentialFlight{done: make(chan struct{})}
	manager.loadFlight = flight
	manager.mu.Unlock()

	credentials, err := manager.source.Load(ctx)
	if err == nil {
		err = credentials.validate()
	}
	manager.mu.Lock()
	if err == nil {
		value := credentials
		manager.credentials = &value
	}
	flight.err = err
	manager.loadFlight = nil
	close(flight.done)
	manager.mu.Unlock()
	return err
}

func (manager *credentialManager) snapshot() (Credentials, uint64, uint64) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return *manager.credentials, manager.revision, manager.failureCount
}

func waitCredentialFlight(ctx context.Context, flight *credentialFlight) error {
	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-flight.done:
		return flight.err
	}
}
