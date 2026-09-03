// Package dbos configures startup-registered durable operations for DBOS workflows.
package dbos

import "github.com/Kludex/pydantic-ai-go/durable"

// NewBackend creates a backend that rejects operations bound after Freeze.
// Adapt register to the DBOS workflow or step registration API.
func NewBackend(register durable.RegisterFunc) (*durable.RegisteredBackend, error) {
	return durable.NewRegisteredBackend(register, durable.RegistrationStartupOnly)
}
