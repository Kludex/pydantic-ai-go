package download

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// maxRequestBytes is Vertex AI's 500 MiB request payload limit, the largest
// of the providers that receive downloaded files.
const maxRequestBytes = 500 << 20

type budgetKey struct{}

type requestBudget struct {
	mu        sync.Mutex
	total     int64
	remaining int64
}

// WithRequestBudget gives the downloads made with the returned context one
// total, so a model request carrying many file URLs can't download the
// per-file limit once per URL.
func WithRequestBudget(ctx context.Context) context.Context {
	return context.WithValue(ctx, budgetKey{}, &requestBudget{total: maxRequestBytes, remaining: maxRequestBytes})
}

func (budget *requestBudget) fetch(ctx context.Context, rawURL string, options Options) (Result, error) {
	perFile := options.MaxBytes
	budget.mu.Lock()
	// Reserved up front and refunded after, so concurrent downloads can't each claim what is left.
	options.MaxBytes = min(perFile, budget.remaining)
	budget.remaining -= options.MaxBytes
	budget.mu.Unlock()
	result, err := fetch(ctx, rawURL, options)
	budget.mu.Lock()
	budget.remaining += options.MaxBytes - int64(len(result.Data))
	budget.mu.Unlock()
	var tooLarge *tooLargeError
	if options.MaxBytes < perFile && errors.As(err, &tooLarge) {
		return Result{}, fmt.Errorf(
			"download: file URL downloads for one model request exceed the total of %d bytes; "+
				"pass large files as BinaryContent instead", budget.total,
		)
	}
	return result, err
}

type tooLargeError struct{ maximumBytes int64 }

func (err *tooLargeError) Error() string {
	return fmt.Sprintf("download: response exceeds %d bytes", err.maximumBytes)
}
