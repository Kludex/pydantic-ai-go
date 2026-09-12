package ai

import "reflect"

func (a *Agent[Deps, Output]) startupBannerDetails(cfg runConfig) (bannerDetails, bool) {
	selectionModes := 0
	if cfg.model != nil {
		selectionModes++
	}
	if cfg.modelID != "" {
		selectionModes++
	}
	if len(cfg.modelSelectors) > 0 {
		selectionModes++
	}
	if selectionModes > 1 {
		return bannerDetails{}, false
	}

	model := a.model
	modelID := ""
	switch {
	case cfg.model != nil:
		model = cfg.model
	case cfg.modelID != "":
		modelID = cfg.modelID
	case len(cfg.modelSelectors) > 0:
		return bannerDetails{}, false
	}
	if modelID == "" {
		if modelIsNil(model) {
			return bannerDetails{}, false
		}
		modelID = model.Name()
	}

	capabilities, ok := a.startupBannerCapabilities(cfg.capabilities)
	if !ok {
		return bannerDetails{}, false
	}
	var tools *int
	if len(a.toolsets) == 0 && len(cfg.toolsets) == 0 && len(cfg.capabilities) == 0 {
		count := len(a.tools) + len(cfg.tools)
		known := true
		for _, setup := range a.capabilitySetups {
			count += len(setup.tools)
			known = known && len(setup.nativeOrLocal) == 0
		}
		if known {
			tools = &count
		}
	}

	return bannerDetails{
		name: a.name, model: modelID, output: bannerOutputType(reflect.TypeFor[Output]()), tools: tools,
		capabilities:  len(capabilities),
		observability: !hasInstrumentationCapability(capabilities) && !hasInstrumentedModel(model),
	}, true
}

func (a *Agent[Deps, Output]) startupBannerCapabilities(runCapabilities []Capability) ([]Capability, bool) {
	runRoots, err := combineCapabilityLayer(runCapabilities)
	if err != nil {
		return nil, false
	}
	overridden := make(map[string]struct{}, len(runRoots))
	for _, capability := range runRoots {
		if id := capabilityIdentity(capability); id != "" {
			overridden[id] = struct{}{}
		}
	}
	capabilities := make([]Capability, 0, len(a.capabilities)+len(runCapabilities))
	for index, capability := range a.capabilities {
		if _, replaced := overridden[a.capabilityRootIDs[index]]; replaced && a.capabilityRootIDs[index] != "" {
			continue
		}
		capabilities = append(capabilities, capability)
	}
	capabilities = append(capabilities, flattenCapabilities(runRoots)...)
	return capabilities, true
}
