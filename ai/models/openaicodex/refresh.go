package openaicodex

import "context"

func (manager *credentialManager) refresh(ctx context.Context, revision, seenFailures uint64) error {
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
		flight := manager.refreshFlight
		manager.mu.Unlock()
		return waitCredentialFlight(ctx, flight)
	}
	flight := &credentialFlight{done: make(chan struct{})}
	manager.refreshFlight = flight
	rejected := *manager.credentials
	manager.mu.Unlock()

	err := manager.refreshCredentials(ctx, rejected)
	manager.mu.Lock()
	if err != nil {
		manager.failureCount++
		manager.lastFailure = refreshFailure{revision: revision, err: err}
	}
	flight.err = err
	manager.refreshFlight = nil
	close(flight.done)
	manager.mu.Unlock()
	return err
}

func (manager *credentialManager) refreshCredentials(ctx context.Context, rejected Credentials) error {
	if manager.source != nil {
		stored, err := manager.source.Load(ctx)
		if err != nil {
			return err
		}
		if err := stored.validate(); err != nil {
			return err
		}
		if stored != rejected {
			manager.replace(stored)
			return nil
		}
	}
	credentials, err := RefreshCredentials(ctx, manager.client, rejected)
	if err != nil {
		return err
	}
	manager.replace(credentials)
	if manager.source != nil {
		if err := manager.source.Save(ctx, credentials); err != nil {
			return &CredentialsPersistenceError{err: err}
		}
	}
	return nil
}

func (manager *credentialManager) replace(credentials Credentials) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	value := credentials
	manager.credentials = &value
	manager.revision++
	manager.lastFailure = refreshFailure{}
}
