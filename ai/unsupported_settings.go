package ai

import (
	"context"
	"fmt"
	"log/slog"
	"slices"

	"github.com/Kludex/pydantic-ai-go/ai/internal/settingwarnings"
)

// Settings a model warns about ignoring. They change what a request
// guarantees, so dropping one silently would surprise the caller. Other
// settings are tuning knobs a model ignores silently, so the same settings
// work across models.
const (
	SettingStopSequences     = "stop_sequences"
	SettingParallelToolCalls = "parallel_tool_calls"
	SettingExtraHeaders      = "extra_headers"
	SettingExtraBody         = "extra_body"
)

// UnsupportedSettingWarning reports a ModelSettings field that a model ignores
// although the request relies on it.
type UnsupportedSettingWarning struct {
	// ModelName identifies the model that ignores the setting.
	ModelName string
	// Setting is one of the Setting constants.
	Setting string
}

// Error describes the ignored setting.
func (warning UnsupportedSettingWarning) Error() string {
	return fmt.Sprintf("the %s setting is not supported by model %q and will be ignored", warning.Setting, warning.ModelName)
}

// UnsupportedSettingSink receives unsupported-setting warnings.
type UnsupportedSettingSink func(ctx context.Context, warning UnsupportedSettingWarning)

type unsupportedSettingContextKey struct{}

// WithUnsupportedSettingSink returns a context whose model requests report
// ignored settings to sink instead of slog.Warn. Return an error from your own
// code in sink, or fail a test, to make an ignored setting fatal. A nil sink
// silences the warnings.
func WithUnsupportedSettingSink(ctx context.Context, sink UnsupportedSettingSink) context.Context {
	if sink == nil {
		sink = func(context.Context, UnsupportedSettingWarning) {}
	}
	return context.WithValue(ctx, unsupportedSettingContextKey{}, sink)
}

// WarnUnsupportedSettings reports each guarantee setting params sets that model
// ignores. Providers call it before sending a request, listing the Setting
// constants they cannot honor. ParallelToolCalls only counts when it is false
// and the request has function tools, the only case that guarantees anything.
// A provider that keeps its own typed settings in ExtraBody strips them first.
func WarnUnsupportedSettings(ctx context.Context, model Model, params ModelRequestParams, unsupported ...string) {
	settings := params.Settings
	set := map[string]bool{
		SettingStopSequences:     len(settings.StopSequences) > 0,
		SettingParallelToolCalls: settings.ParallelToolCalls != nil && !*settings.ParallelToolCalls && len(params.Tools) > 0,
		SettingExtraHeaders:      len(settings.ExtraHeaders) > 0,
		SettingExtraBody:         len(settings.ExtraBody) > 0,
	}
	sink, ok := ctx.Value(unsupportedSettingContextKey{}).(UnsupportedSettingSink)
	suppressed := settingwarnings.Suppressed(ctx)
	for _, name := range []string{SettingStopSequences, SettingParallelToolCalls, SettingExtraHeaders, SettingExtraBody} {
		if !set[name] || !slices.Contains(unsupported, name) || slices.Contains(suppressed, name) {
			continue
		}
		warning := UnsupportedSettingWarning{ModelName: model.Name(), Setting: name}
		if ok {
			sink(ctx, warning)
		} else {
			slog.WarnContext(ctx, warning.Error())
		}
	}
}
