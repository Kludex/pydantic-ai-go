// Package temporal configures startup-registered durable operations for Temporal workers.
package temporal

import "github.com/Kludex/pydantic-ai-go/durable"

// NewBackend creates a backend that rejects operations bound after Freeze.
// Adapt register to workflow.RegisterActivity or the worker registration API.
func NewBackend(register durable.RegisterFunc) (*durable.RegisteredBackend, error) {
	return durable.NewRegisteredBackend(register, durable.RegistrationStartupOnly)
}
