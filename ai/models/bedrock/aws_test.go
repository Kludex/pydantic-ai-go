package bedrock

import (
	"context"
	"errors"
	"net/http"
	"testing"

	smithyhttp "github.com/aws/smithy-go/transport/http"
)

func TestIsDataRetentionError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "unrelated", err: errors.New("too many requests"), want: false},
		{name: "data retention upper", err: errors.New("Your request failed due to the Data Retention Mode setting."), want: true},
		{name: "data retention lower", err: errors.New("invalid data retention mode for model claude-fable-5"), want: true},
		{name: "data retention mixed", err: errors.New("Rejected for THIS data retention MODE in us-east-1"), want: true},
		{name: "similar phrase no match", err: errors.New("retention policy expired"), want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isDataRetentionError(tc.err); got != tc.want {
				t.Fatalf("isDataRetentionError(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestModelErrorDataRetentionHint(t *testing.T) {
	cases := []struct {
		name        string
		err         error
		wantHint    bool
		wantHasText string
	}{
		{
			name: "data retention hint set",
			err: &smithyhttp.ResponseError{
				Response: &smithyhttp.Response{Response: &http.Response{StatusCode: http.StatusBadRequest}},
				Err:      errors.New("Bedrock rejected the request due to the data retention mode."),
			},
			wantHint:    true,
			wantHasText: "PutAccountDataRetention",
		},
		{
			name: "unrelated error no hint",
			err: &smithyhttp.ResponseError{
				Response: &smithyhttp.Response{Response: &http.Response{StatusCode: http.StatusTooManyRequests}},
				Err:      errors.New("throttled"),
			},
			wantHint: false,
		},
		{
			name:        "non-response error wraps to transport",
			err:         errors.New("connect: connection refused"),
			wantHint:    false,
			wantHasText: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := modelError(context.Background(), nil, "request", tc.err)
			if err == nil {
				t.Fatal("modelError returned nil")
			}
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				if (apiErr.Hint != "") != tc.wantHint {
					t.Fatalf("Hint set = %v, want %v (Hint=%q)", apiErr.Hint != "", tc.wantHint, apiErr.Hint)
				}
				if tc.wantHint && tc.wantHasText != "" {
					if !containsString(apiErr.Error(), tc.wantHasText) {
						t.Fatalf("Error() = %q missing %q", apiErr.Error(), tc.wantHasText)
					}
				}
			} else if tc.wantHint {
				t.Fatalf("expected APIError with hint, got %T: %v", err, err)
			}
		})
	}
}

func containsString(haystack, needle string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
