package anthropic

import (
	"fmt"
	"maps"

	ai "github.com/Kludex/pydantic-ai-go"
)

// CacheTTL selects the lifetime of an Anthropic prompt-cache breakpoint.
type CacheTTL string

const (
	// CacheTTL5Minutes keeps a cache breakpoint for five minutes.
	CacheTTL5Minutes CacheTTL = "5m"
	// CacheTTL1Hour keeps a cache breakpoint for one hour.
	CacheTTL1Hour CacheTTL = "1h"
)

// Effort selects Anthropic's generation effort.
type Effort string

const (
	// EffortLow minimizes generation effort.
	EffortLow Effort = "low"
	// EffortMedium balances speed and quality.
	EffortMedium Effort = "medium"
	// EffortHigh increases generation effort.
	EffortHigh Effort = "high"
	// EffortXHigh requests the provider's extended high setting.
	EffortXHigh Effort = "xhigh"
	// EffortMax requests the maximum supported effort.
	EffortMax Effort = "max"
)

// CodeExecutionToolVersion selects an Anthropic hosted code execution API version.
type CodeExecutionToolVersion string

const (
	// CodeExecutionToolVersionAuto selects the newest version supported by the model.
	CodeExecutionToolVersionAuto CodeExecutionToolVersion = "auto"
	// CodeExecutionToolVersion20250825 selects the original code execution API.
	CodeExecutionToolVersion20250825 CodeExecutionToolVersion = "20250825"
	// CodeExecutionToolVersion20260120 selects the current code execution API.
	CodeExecutionToolVersion20260120 CodeExecutionToolVersion = "20260120"
)

// ContainerSkill attaches an Anthropic-managed skill to a code execution container.
type ContainerSkill struct {
	// Type identifies the skill provider. Anthropic currently accepts "anthropic".
	Type string `json:"type"`
	// SkillID identifies the managed skill.
	SkillID string `json:"skill_id"`
	// Version selects a skill version, such as "latest".
	Version string `json:"version"`
}

// Container configures an Anthropic code execution container.
type Container struct {
	// ID reuses an existing container when non-empty.
	ID string `json:"id,omitempty"`
	// Skills attaches managed skills to the container.
	Skills []ContainerSkill `json:"skills,omitempty"`
}

// Settings combines portable settings with Anthropic provider settings.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// Cache enables automatic prompt-cache placement with this retention.
	Cache CacheTTL
	// CacheInstructions caches the final stable instruction boundary.
	CacheInstructions CacheTTL
	// CacheMessages caches recent message boundaries.
	CacheMessages CacheTTL
	// CacheToolDefinitions caches the final function-tool definition.
	CacheToolDefinitions CacheTTL
	// Container explicitly reuses or configures a code execution container.
	Container *Container
	// FreshContainer prevents automatic container reuse from message history.
	FreshContainer bool
	// CodeExecutionToolVersion selects the hosted code execution API version.
	CodeExecutionToolVersion CodeExecutionToolVersion
	// Effort overrides the portable thinking level's generation effort.
	Effort Effort
}

const (
	cacheSetting                = "anthropic_cache"
	cacheInstructionsSetting    = "anthropic_cache_instructions"
	cacheMessagesSetting        = "anthropic_cache_messages"
	cacheToolDefinitionsSetting = "anthropic_cache_tool_definitions"
	containerSetting            = "anthropic_container"
	codeExecutionVersionSetting = "anthropic_code_execution_tool_version"
	effortSetting               = "anthropic_effort"
)

type cacheSettings struct {
	Automatic       CacheTTL
	Instructions    CacheTTL
	Messages        CacheTTL
	ToolDefinitions CacheTTL
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	values := []struct {
		name string
		ttl  CacheTTL
	}{
		{name: cacheSetting, ttl: settings.Cache},
		{name: cacheInstructionsSetting, ttl: settings.CacheInstructions},
		{name: cacheMessagesSetting, ttl: settings.CacheMessages},
		{name: cacheToolDefinitionsSetting, ttl: settings.CacheToolDefinitions},
	}
	if settings.Cache != "" && settings.CacheMessages != "" {
		return ai.ModelSettings{}, fmt.Errorf("anthropic: automatic and explicit message caching are mutually exclusive")
	}
	if settings.Container != nil && settings.FreshContainer {
		return ai.ModelSettings{}, fmt.Errorf("anthropic: container reuse and a fresh container are mutually exclusive")
	}
	if settings.Container != nil {
		if err := validateContainer(*settings.Container); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if settings.CodeExecutionToolVersion != "" {
		if err := validateCodeExecutionToolVersion(settings.CodeExecutionToolVersion); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if settings.Effort != "" {
		if err := validateEffort(settings.Effort); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	for _, setting := range values {
		if _, exists := extra[setting.name]; exists {
			return ai.ModelSettings{}, fmt.Errorf("anthropic: setting field %q is reserved", setting.name)
		}
		if setting.ttl == "" {
			continue
		}
		if err := validateCacheTTL(setting.ttl); err != nil {
			return ai.ModelSettings{}, err
		}
		if setting.name == cacheSetting {
			if _, exists := extra["cache_control"]; exists {
				return ai.ModelSettings{}, fmt.Errorf(
					"anthropic: extra body field %q conflicts with typed settings", "cache_control",
				)
			}
		}
		extra[setting.name] = setting.ttl
	}
	if settings.Container != nil || settings.FreshContainer {
		if _, exists := extra[containerSetting]; exists {
			return ai.ModelSettings{}, fmt.Errorf("anthropic: setting field %q is reserved", containerSetting)
		}
		if settings.FreshContainer {
			extra[containerSetting] = false
		} else {
			container := *settings.Container
			container.Skills = append([]ContainerSkill(nil), settings.Container.Skills...)
			extra[containerSetting] = container
		}
	}
	if settings.CodeExecutionToolVersion != "" {
		if _, exists := extra[codeExecutionVersionSetting]; exists {
			return ai.ModelSettings{}, fmt.Errorf("anthropic: setting field %q is reserved", codeExecutionVersionSetting)
		}
		extra[codeExecutionVersionSetting] = settings.CodeExecutionToolVersion
	}
	if settings.Effort != "" {
		if _, exists := extra[effortSetting]; exists {
			return ai.ModelSettings{}, fmt.Errorf("anthropic: setting field %q is reserved", effortSetting)
		}
		extra[effortSetting] = settings.Effort
	}
	if len(extra) == 0 {
		extra = nil
	}
	common.ExtraBody = extra
	return common, nil
}

func extractCacheSettings(settings ai.ModelSettings) (ai.ModelSettings, cacheSettings, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	cache := cacheSettings{}
	values := []struct {
		name        string
		destination *CacheTTL
	}{
		{name: cacheSetting, destination: &cache.Automatic},
		{name: cacheInstructionsSetting, destination: &cache.Instructions},
		{name: cacheMessagesSetting, destination: &cache.Messages},
		{name: cacheToolDefinitionsSetting, destination: &cache.ToolDefinitions},
	}
	for _, setting := range values {
		value, exists := extra[setting.name]
		if !exists {
			continue
		}
		delete(extra, setting.name)
		ttl, ok := value.(CacheTTL)
		if !ok {
			return ai.ModelSettings{}, cacheSettings{}, fmt.Errorf(
				"anthropic: cache setting %q must use CacheTTL", setting.name,
			)
		}
		if err := validateCacheTTL(ttl); err != nil {
			return ai.ModelSettings{}, cacheSettings{}, err
		}
		*setting.destination = ttl
	}
	if cache.Automatic != "" && cache.Messages != "" {
		return ai.ModelSettings{}, cacheSettings{}, fmt.Errorf(
			"anthropic: automatic and explicit message caching are mutually exclusive",
		)
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, cache, nil
}

type providerSettings struct {
	Container                any
	ContainerSet             bool
	CodeExecutionToolVersion CodeExecutionToolVersion
	Effort                   Effort
}

func extractProviderSettings(settings ai.ModelSettings) (ai.ModelSettings, providerSettings, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	provider := providerSettings{CodeExecutionToolVersion: CodeExecutionToolVersionAuto}
	if value, exists := extra[containerSetting]; exists {
		delete(extra, containerSetting)
		provider.ContainerSet = true
		switch value := value.(type) {
		case bool:
			if value {
				return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("anthropic: container setting must not be true")
			}
		case string:
			if value == "" {
				return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("anthropic: container ID must not be empty")
			}
			provider.Container = value
		case Container:
			if err := validateContainer(value); err != nil {
				return ai.ModelSettings{}, providerSettings{}, err
			}
			if value.ID != "" && len(value.Skills) == 0 {
				provider.Container = value.ID
			} else {
				provider.Container = value
			}
		case *Container:
			if value == nil {
				return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("anthropic: container must not be nil")
			}
			if err := validateContainer(*value); err != nil {
				return ai.ModelSettings{}, providerSettings{}, err
			}
			if value.ID != "" && len(value.Skills) == 0 {
				provider.Container = value.ID
			} else {
				provider.Container = *value
			}
		default:
			return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("anthropic: invalid container setting %T", value)
		}
	}
	if value, exists := extra[codeExecutionVersionSetting]; exists {
		delete(extra, codeExecutionVersionSetting)
		version, ok := value.(CodeExecutionToolVersion)
		if !ok {
			return ai.ModelSettings{}, providerSettings{}, fmt.Errorf(
				"anthropic: code execution tool version must use CodeExecutionToolVersion",
			)
		}
		if err := validateCodeExecutionToolVersion(version); err != nil {
			return ai.ModelSettings{}, providerSettings{}, err
		}
		provider.CodeExecutionToolVersion = version
	}
	if value, exists := extra[effortSetting]; exists {
		delete(extra, effortSetting)
		effort, ok := value.(Effort)
		if !ok {
			return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("anthropic: effort must use Effort")
		}
		if err := validateEffort(effort); err != nil {
			return ai.ModelSettings{}, providerSettings{}, err
		}
		provider.Effort = effort
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, provider, nil
}

func validateEffort(effort Effort) error {
	switch effort {
	case EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax:
		return nil
	default:
		return fmt.Errorf("anthropic: invalid effort %q", effort)
	}
}

func validateContainer(container Container) error {
	if container.ID == "" && len(container.Skills) == 0 {
		return fmt.Errorf("anthropic: container requires an ID or skills")
	}
	for _, skill := range container.Skills {
		if skill.Type != "anthropic" || skill.SkillID == "" || skill.Version == "" {
			return fmt.Errorf("anthropic: container skills require type %q, a skill ID, and a version", "anthropic")
		}
	}
	return nil
}

func validateCodeExecutionToolVersion(version CodeExecutionToolVersion) error {
	switch version {
	case CodeExecutionToolVersionAuto, CodeExecutionToolVersion20250825, CodeExecutionToolVersion20260120:
		return nil
	default:
		return fmt.Errorf("anthropic: invalid code execution tool version %q", version)
	}
}

func validateCacheTTL(ttl CacheTTL) error {
	switch ttl {
	case CacheTTL5Minutes, CacheTTL1Hour:
		return nil
	default:
		return fmt.Errorf("anthropic: invalid cache TTL %q", ttl)
	}
}

func promptCacheControl(ttl CacheTTL) *anthropicPromptCacheControl {
	if ttl == "" {
		return nil
	}
	return &anthropicPromptCacheControl{Type: "ephemeral", TTL: ai.CachePointTTL(ttl)}
}
