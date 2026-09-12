package ai

import (
	"fmt"
	"io"
	"sync/atomic"
)

var (
	bannerClaimed atomic.Bool
	bannerEnabled atomic.Bool
)

func init() {
	bannerEnabled.Store(true)
}

type bannerDetails struct {
	name          string
	model         string
	output        string
	tools         *int
	capabilities  int
	observability bool
}

// BannerEnabled reports whether first-run banner display is enabled.
// Environment suppression and the once-per-process claim are checked separately.
func BannerEnabled() bool { return bannerEnabled.Load() }

// SetBannerEnabled controls the first-run banner. Call it before the first agent run.
func SetBannerEnabled(enabled bool) { bannerEnabled.Store(enabled) }

// WriteBanner writes the once-per-process startup banner when writer is interactive
// or a recognized coding agent is running the process. Write failures are ignored.
func (a *Agent[Deps, Output]) WriteBanner(writer io.Writer, opts ...RunOption) {
	if !bannerPending() {
		return
	}
	details, ok := a.startupBannerDetails(buildRunConfig(opts))
	if !ok {
		return
	}
	terminal := writerIsTerminal(writer, true)
	displayBanner(writer, terminal, bannerDestinationAvailable(writer, terminal), details)
}

func displayRunBanner(writer io.Writer, details bannerDetails, instrumented bool) {
	if instrumented || !bannerPending() {
		return
	}
	terminal := writerIsTerminal(writer, true)
	displayBanner(writer, terminal, bannerDestinationAvailable(writer, terminal), details)
}

func displayBanner(writer io.Writer, terminal, available bool, details bannerDetails) {
	if writer == nil {
		return
	}
	claimed := bannerClaimed.CompareAndSwap(false, true)
	if available && claimed {
		color := terminal && !environmentSet("NO_COLOR")
		_, _ = fmt.Fprintln(writer, renderBanner(details, color))
	}
}

func bannerDestinationAvailable(writer io.Writer, terminal bool) bool {
	agent := detectCodingAgent()
	return writer != nil && (terminal || agent != "")
}

func bannerPending() bool {
	return !bannerClaimed.Load() && !bannerSuppressed()
}

func bannerSuppressed() bool {
	return !bannerEnabled.Load() || environmentSet("PYDANTIC_AI_NO_BANNER") || environmentSet("CI") || runningUnderGoTest()
}
