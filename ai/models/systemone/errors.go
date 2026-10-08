package systemone

import (
	"fmt"
	"net/http"
)

// DecisionHandOff identifies routes a fallback model can handle.
type DecisionHandOff interface {
	error
	IsModelAPIError() bool
	DecisionRoute() (route string, probability float64)
}

// UnfillableRoute reports a selected route with unsupported fields.
type UnfillableRoute struct {
	ModelName   string
	ToolName    string
	Route       string
	Probability float64
}

// Error describes the route the model could not fill.
func (err *UnfillableRoute) Error() string {
	return fmt.Sprintf(
		"System One model %q picked %q (probability %.2f) but cannot fill it", err.ModelName, err.Route, err.Probability,
	)
}

// IsModelAPIError allows fallback models to handle this route.
func (*UnfillableRoute) IsModelAPIError() bool { return true }

// DecisionRoute returns the selected route and its probability.
func (err *UnfillableRoute) DecisionRoute() (string, float64) { return err.Route, err.Probability }

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
		"System One model %q picked %q with probability %.2f, below route threshold %.2f",
		err.ModelName, err.Route, err.Probability, err.Threshold,
	)
}

// IsModelAPIError allows fallback models to handle this route.
func (*UnsureRoute) IsModelAPIError() bool { return true }

// DecisionRoute returns the selected route and its probability.
func (err *UnsureRoute) DecisionRoute() (string, float64) { return err.Route, err.Probability }

// APIError reports a non-successful HTTP response with its original body and headers.
type APIError struct {
	StatusCode int
	ModelName  string
	Body       string
	Headers    http.Header
}

// Error describes the HTTP failure.
func (err *APIError) Error() string {
	return fmt.Sprintf("System One API returned status %d for %q: %s", err.StatusCode, err.ModelName, err.Body)
}

// IsModelAPIError allows fallback models to handle HTTP failures.
func (*APIError) IsModelAPIError() bool { return true }
