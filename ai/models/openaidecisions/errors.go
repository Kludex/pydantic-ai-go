package openaidecisions

import (
	"fmt"
	"net/http"
)

// DecisionHandOff is a route the Decisions API could not safely complete.
type DecisionHandOff interface {
	error
	IsModelAPIError() bool
	DecisionRoute() (route string, probability float64)
}

// UnfillableRoute reports a selected route whose fields cannot be expressed.
type UnfillableRoute struct {
	ModelName   string
	ToolName    string
	Route       string
	Probability float64
}

// Error describes the route the model could not fill.
func (err *UnfillableRoute) Error() string {
	return fmt.Sprintf("OpenAI Decisions picked %q (probability %.2f) but cannot fill it", err.Route, err.Probability)
}

// IsModelAPIError allows fallback models to handle this failure.
func (*UnfillableRoute) IsModelAPIError() bool { return true }

// DecisionRoute returns the selected route and its probability.
func (err *UnfillableRoute) DecisionRoute() (string, float64) {
	return err.Route, err.Probability
}

// UnsureRoute reports a selected route below the configured threshold.
type UnsureRoute struct {
	ModelName     string
	Route         string
	Probability   float64
	Probabilities map[string]float64
	Threshold     float64
}

// Error describes the route probability and configured threshold.
func (err *UnsureRoute) Error() string {
	return fmt.Sprintf(
		"OpenAI Decisions picked %q with probability %.2f, below route threshold %.2f",
		err.Route, err.Probability, err.Threshold,
	)
}

// IsModelAPIError allows fallback models to handle this failure.
func (*UnsureRoute) IsModelAPIError() bool { return true }

// DecisionRoute returns the selected route and its probability.
func (err *UnsureRoute) DecisionRoute() (string, float64) {
	return err.Route, err.Probability
}

// APIError reports a non-successful Decisions response.
type APIError struct {
	StatusCode int
	Body       string
	Headers    http.Header
}

// Error describes the HTTP failure.
func (err *APIError) Error() string {
	return fmt.Sprintf("OpenAI Decisions API returned status %d: %s", err.StatusCode, err.Body)
}

// IsModelAPIError allows fallback models to handle this failure.
func (*APIError) IsModelAPIError() bool { return true }
