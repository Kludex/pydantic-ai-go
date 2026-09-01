package retries

import (
	"context"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"time"
)

// ValidateStatus returns an error for responses outside the 200-299 range.
func ValidateStatus(response *http.Response) error {
	if response == nil {
		return fmt.Errorf("response must not be nil")
	}
	if response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices {
		return nil
	}
	return fmt.Errorf("unexpected HTTP status %s", response.Status)
}

// RetryTransient retries transport errors, HTTP 429, and HTTP 5xx validation errors.
func RetryTransient(err error) bool {
	var responseErr *ResponseError
	if !errors.As(err, &responseErr) {
		return !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)
	}
	if responseErr.Response == nil {
		return false
	}
	status := responseErr.Response.StatusCode
	return status == http.StatusTooManyRequests ||
		status >= http.StatusInternalServerError && status <= 599
}

// ExponentialBackoff returns delays of initial, 2*initial, up to maximum.
func ExponentialBackoff(initial time.Duration, maximum time.Duration) WaitFunc {
	if initial <= 0 || maximum < initial {
		panic("retries: exponential backoff requires 0 < initial <= maximum")
	}
	return func(attempt Attempt) time.Duration {
		delay := initial
		for number := 1; number < attempt.Number && delay < maximum; number++ {
			if delay > maximum/2 {
				return maximum
			}
			delay *= 2
		}
		return delay
	}
}

// WaitRetryAfter honors Retry-After seconds or HTTP dates, capped by maximum.
// It uses fallback when the header is absent or invalid.
func WaitRetryAfter(fallback WaitFunc, maximum time.Duration) WaitFunc {
	if maximum <= 0 {
		panic("retries: Retry-After maximum must be positive")
	}
	if fallback == nil {
		fallback = ExponentialBackoff(time.Second, time.Minute)
	}
	return func(attempt Attempt) time.Duration {
		if attempt.Response != nil {
			value := attempt.Response.Header.Get("Retry-After")
			if seconds, ok := new(big.Int).SetString(value, 10); ok && seconds.Sign() >= 0 {
				if !seconds.IsInt64() || seconds.Int64() > int64(maximum/time.Second) {
					return maximum
				}
				return min(time.Duration(seconds.Int64())*time.Second, maximum)
			}
			if retryAt, err := http.ParseTime(value); err == nil {
				delay := time.Until(retryAt)
				if delay > 0 {
					return min(delay, maximum)
				}
			}
		}
		return min(max(fallback(attempt), 0), maximum)
	}
}
