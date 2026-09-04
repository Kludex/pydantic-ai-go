package ai

import "slices"

// NewImageOutputAgent creates an agent whose final output is the first image file
// in a model response. The selected model must advertise image-output support.
func NewImageOutputAgent[Deps any](model Model, opts ...Option) *Agent[Deps, BinaryContent] {
	agent := NewAgent[Deps, BinaryContent](model, opts...)
	agent.outputAllowsImage = true
	agent.outputOverrideErr = ErrOutputTypeOverrideWithImageOutput
	return agent
}

func cloneBinaryContent(content BinaryContent) BinaryContent {
	content.Data = slices.Clone(content.Data)
	content.VendorMetadata = cloneSchemaMap(content.VendorMetadata)
	return content
}
