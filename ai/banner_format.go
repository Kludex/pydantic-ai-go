package ai

import (
	"reflect"
	"runtime"
	"runtime/debug"
	"slices"
)

const bannerModulePath = "github.com/Kludex/pydantic-ai-go"

func bannerElideMiddle(value string, width int) string {
	if len(value) <= width {
		return value
	}
	head := (width - 3) / 2
	tail := width - 3 - head
	return value[:head] + "..." + value[len(value)-tail:]
}

func bannerOutputType(output reflect.Type) string {
	if output == reflect.TypeFor[string]() {
		return ""
	}
	name := output.Name()
	if name == "" {
		name = output.String()
	}
	if len(name) > 40 {
		return name[:37] + "..."
	}
	return name
}

func bannerVersionLine() string {
	version := "(devel)"
	if info, ok := debug.ReadBuildInfo(); ok {
		modules := append(slices.Clone(info.Deps), &info.Main)
		for _, module := range modules {
			if module.Path == bannerModulePath && module.Version != "" {
				version = module.Version
			}
		}
	}
	return "pydantic-ai-go " + version + " | Go " + runtime.Version()
}
