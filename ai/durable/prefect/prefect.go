// Package prefect configures callable Prefect task operations.
package prefect

import "github.com/Kludex/pydantic-ai-go/ai/durable"

// NewBackend creates a backend that permits explicitly non-executing observers
// after Freeze while rejecting late models, tools, and executing capabilities.
// Adapt register to the Prefect task submission API.
func NewBackend(register durable.RegisterFunc) (*durable.RegisteredBackend, error) {
	return durable.NewRegisteredBackend(register, durable.RegistrationRuntimeObservers)
}
