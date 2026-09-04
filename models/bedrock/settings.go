package bedrock

import (
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// CacheTTL selects the lifetime of a Bedrock prompt-cache boundary.
type CacheTTL string

const (
	// CacheTTL5Minutes requests five-minute retention.
	CacheTTL5Minutes CacheTTL = "5m"
	// CacheTTL1Hour requests one-hour retention.
	CacheTTL1Hour CacheTTL = "1h"
)

// GuardrailConfig selects one Bedrock guardrail.
type GuardrailConfig struct {
	// Identifier is the guardrail ID or ARN.
	Identifier string
	// Version is a numeric version or DRAFT.
	Version string
	// Trace controls returned guardrail diagnostics.
	Trace types.GuardrailTrace
}

// Settings combines portable settings with Bedrock-specific request options.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// CacheInstructions appends a cache point to system instructions.
	CacheInstructions CacheTTL
	// CacheMessages appends a cache point to the latest user message.
	CacheMessages CacheTTL
	// CacheToolDefinitions appends a cache point after function tools.
	CacheToolDefinitions CacheTTL
	// InferenceProfile overrides the request model ID with a profile ID or ARN.
	InferenceProfile string
	// Guardrail configures Bedrock content moderation.
	Guardrail *GuardrailConfig
	// PerformanceLatency selects standard or optimized model latency.
	PerformanceLatency types.PerformanceConfigLatency
	// RequestMetadata adds key-value invocation-log filters.
	RequestMetadata map[string]string
	// AdditionalModelResponseFieldPaths requests model-specific response fields.
	AdditionalModelResponseFieldPaths []string
	// PromptVariables supplies text values for a prompt-management ARN.
	PromptVariables map[string]string
}

const (
	cacheInstructionsSetting    = "bedrock_cache_instructions"
	cacheMessagesSetting        = "bedrock_cache_messages"
	cacheToolDefinitionsSetting = "bedrock_cache_tool_definitions"
	requestSettingsKey          = "bedrock_request_settings"
)

type cacheSettings struct {
	instructions    CacheTTL
	messages        CacheTTL
	toolDefinitions CacheTTL
}

type requestSettings struct {
	inferenceProfile                  string
	guardrail                         *GuardrailConfig
	performanceLatency                types.PerformanceConfigLatency
	requestMetadata                   map[string]string
	additionalModelResponseFieldPaths []string
	promptVariables                   map[string]string
}

func (settings requestSettings) clone() requestSettings {
	if settings.guardrail != nil {
		guardrail := *settings.guardrail
		settings.guardrail = &guardrail
	}
	settings.requestMetadata = maps.Clone(settings.requestMetadata)
	settings.additionalModelResponseFieldPaths = slices.Clone(settings.additionalModelResponseFieldPaths)
	settings.promptVariables = maps.Clone(settings.promptVariables)
	return settings
}

func (settings requestSettings) isZero() bool {
	return settings.inferenceProfile == "" && settings.guardrail == nil && settings.performanceLatency == "" &&
		len(settings.requestMetadata) == 0 && len(settings.additionalModelResponseFieldPaths) == 0 &&
		len(settings.promptVariables) == 0
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	values := []struct {
		name string
		ttl  CacheTTL
	}{
		{name: cacheInstructionsSetting, ttl: settings.CacheInstructions},
		{name: cacheMessagesSetting, ttl: settings.CacheMessages},
		{name: cacheToolDefinitionsSetting, ttl: settings.CacheToolDefinitions},
	}
	for _, value := range values {
		if _, exists := extra[value.name]; exists {
			return ai.ModelSettings{}, fmt.Errorf("bedrock: setting field %q is reserved", value.name)
		}
		if value.ttl == "" {
			continue
		}
		if err := validateCacheTTL(value.ttl); err != nil {
			return ai.ModelSettings{}, err
		}
		extra[value.name] = value.ttl
	}
	if _, exists := extra[requestSettingsKey]; exists {
		return ai.ModelSettings{}, fmt.Errorf("bedrock: setting field %q is reserved", requestSettingsKey)
	}
	request := requestSettings{
		inferenceProfile: settings.InferenceProfile, guardrail: settings.Guardrail,
		performanceLatency: settings.PerformanceLatency, requestMetadata: settings.RequestMetadata,
		additionalModelResponseFieldPaths: settings.AdditionalModelResponseFieldPaths,
		promptVariables:                   settings.PromptVariables,
	}.clone()
	if request.guardrail != nil && (request.guardrail.Identifier == "" || request.guardrail.Version == "") {
		return ai.ModelSettings{}, fmt.Errorf("bedrock: guardrail identifier and version are required together")
	}
	if !request.isZero() {
		extra[requestSettingsKey] = request
	}
	if len(extra) == 0 {
		extra = nil
	}
	common.ExtraBody = extra
	return common, nil
}

func extractSettings(settings ai.ModelSettings) (ai.ModelSettings, cacheSettings, requestSettings, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	cache := cacheSettings{}
	values := []struct {
		name        string
		destination *CacheTTL
	}{
		{name: cacheInstructionsSetting, destination: &cache.instructions},
		{name: cacheMessagesSetting, destination: &cache.messages},
		{name: cacheToolDefinitionsSetting, destination: &cache.toolDefinitions},
	}
	for _, value := range values {
		raw, exists := extra[value.name]
		if !exists {
			continue
		}
		delete(extra, value.name)
		ttl, ok := raw.(CacheTTL)
		if !ok {
			return ai.ModelSettings{}, cacheSettings{}, requestSettings{}, fmt.Errorf(
				"bedrock: cache setting %q must use CacheTTL", value.name,
			)
		}
		if err := validateCacheTTL(ttl); err != nil {
			return ai.ModelSettings{}, cacheSettings{}, requestSettings{}, err
		}
		*value.destination = ttl
	}
	request := requestSettings{}
	if raw, exists := extra[requestSettingsKey]; exists {
		delete(extra, requestSettingsKey)
		configured, ok := raw.(requestSettings)
		if !ok {
			return ai.ModelSettings{}, cacheSettings{}, requestSettings{}, fmt.Errorf(
				"bedrock: request settings must be built with Settings.Build",
			)
		}
		request = configured.clone()
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, cache, request, nil
}

func validateCacheTTL(ttl CacheTTL) error {
	switch ttl {
	case CacheTTL5Minutes, CacheTTL1Hour:
		return nil
	default:
		return fmt.Errorf("bedrock: invalid cache TTL %q", ttl)
	}
}

func (cache cacheSettings) retention() (time.Duration, bool) {
	for _, ttl := range []CacheTTL{cache.instructions, cache.messages, cache.toolDefinitions} {
		if ttl == CacheTTL1Hour {
			return time.Hour, true
		}
	}
	for _, ttl := range []CacheTTL{cache.instructions, cache.messages, cache.toolDefinitions} {
		if ttl == CacheTTL5Minutes {
			return 5 * time.Minute, true
		}
	}
	return 0, false
}
