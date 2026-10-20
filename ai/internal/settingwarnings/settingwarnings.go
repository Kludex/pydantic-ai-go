// Package settingwarnings lets library code silence unsupported-setting
// warnings for settings the user did not choose.
package settingwarnings

import (
	"context"
	"slices"
)

type suppressedKey struct{}

// Suppress returns a context in which warnings about names are not reported.
func Suppress(ctx context.Context, names ...string) context.Context {
	return context.WithValue(ctx, suppressedKey{}, append(Suppressed(ctx), names...))
}

// Suppressed reports the settings whose warnings ctx silences.
func Suppressed(ctx context.Context) []string {
	names, _ := ctx.Value(suppressedKey{}).([]string)
	return slices.Clone(names)
}
