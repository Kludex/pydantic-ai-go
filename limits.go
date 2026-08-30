package ai

import "context"

// usageLimitsCapability enforces UsageLimits as a model-request wrapper -
// the same mechanism user capabilities use, dogfooding the hook surface.
type usageLimitsCapability struct {
	limits UsageLimits
}

func (usageLimitsCapability) Setup(*CapabilityRegistry) error { return nil }

func (c usageLimitsCapability) WrapModelRequest(ctx context.Context, ri *RunInfo, msgs []ModelMessage, params ModelRequestParams, next ModelRequestFunc) (*ModelResponse, error) {
	resp, err := next(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	projected := ri.Usage()
	projected.Add(resp.Usage)
	if err := c.limits.check(projected); err != nil {
		return nil, err
	}
	return resp, nil
}
