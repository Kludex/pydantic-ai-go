package vercel

import "errors"

// ErrInvalidInput matches, with errors.Is, every error caused by a malformed client request,
// such as an empty tool name or a file without a URL. Answer it as a client error, not a server failure.
var ErrInvalidInput = errors.New("vercel: invalid input")

type inputError struct{ error }

func (err inputError) Unwrap() error { return err.error }

func (inputError) Is(target error) bool { return target == ErrInvalidInput }
