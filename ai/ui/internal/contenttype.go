// Package internal holds CSRF-shaped helpers shared between UI adapters.
package internal

import (
	"fmt"
	"mime"
	"net/http"
	"slices"
	"strings"
)

// DefaultAllowedContentTypes holds the secure default allowlist applied by
// the AG-UI and Vercel AI handlers when the caller leaves AllowedContentTypes
// unset. application/json is non-CORS-safelisted, so requiring it forces a
// preflight and a cross-origin request cannot otherwise authenticate against
// the endpoint.
var DefaultAllowedContentTypes = []string{"application/json"}

// ContentTypeError rejects an inbound media type. Mirrors Python's
// HTTPException(415); its message names the same allowed set.
type ContentTypeError struct {
	Got      string
	Expected []string
}

func (err *ContentTypeError) Error() string {
	return fmt.Sprintf("ui: expected Content-Type %s, got %s",
		strings.Join(err.Expected, ", "), err.Got)
}

// CheckContentType returns a ContentTypeError when the request's
// Content-Type is not in the allowed set.
//
// Semantics mirror upstream Python:
//
//   - nil slice: apply [DefaultAllowedContentTypes] (application/json)
//   - empty non-nil slice: skip the check entirely
//   - non-empty slice: use the supplied allowlist verbatim
//
// Media types are case-insensitive, and entries are normalized for the
// comparison.
func CheckContentType(request *http.Request, allowedContentTypes []string) error {
	var allowed []string
	switch {
	case allowedContentTypes == nil:
		allowed = append(allowed, DefaultAllowedContentTypes...)
	case len(allowedContentTypes) == 0:
		// An explicit empty allowlist opts out of the CSRF check.
		return nil
	default:
		allowed = make([]string, 0, len(allowedContentTypes))
		for _, mediaType := range allowedContentTypes {
			mediaType = normalizeMediaType(mediaType)
			if mediaType == "" {
				continue
			}
			allowed = append(allowed, mediaType)
		}
		if len(allowed) == 0 {
			return nil
		}
	}
	slices.Sort(allowed)

	got := ""
	if raw := request.Header.Get("Content-Type"); raw != "" {
		mediaType, _, err := mime.ParseMediaType(raw)
		if err == nil {
			got = strings.ToLower(mediaType)
		}
	}
	if slices.Contains(allowed, got) {
		return nil
	}
	return &ContentTypeError{
		Got:      nonempty(got),
		Expected: allowed,
	}
}

func normalizeMediaType(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return ""
	}
	mediaType, _, err := mime.ParseMediaType(value)
	if err != nil {
		return value
	}
	return mediaType
}

func nonempty(value string) string {
	if value == "" {
		return "no content type"
	}
	return value
}
