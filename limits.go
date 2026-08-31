package ai

import "context"

// usageLimitsCapability enforces UsageLimits as a model-request wrapper -
// the same mechanism user capabilities use, dogfooding the hook surface.
type usageLimitsCapability struct {
	limits UsageLimits
}

func (usageLimitsCapability) Setup(*CapabilityRegistry) error { return nil }

func (c usageLimitsCapability) WrapModelRequest(
	ctx context.Context, ri *RunInfo, msgs []ModelMessage, params ModelRequestParams, next ModelRequestFunc,
) (*ModelResponse, error) {
	current := ri.Usage()
	resp, err := next(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	fillResponseCost(ctx, resp)
	projected := current
	projected.Add(resp.Usage)
	if err := c.limits.check(projected); err != nil {
		return nil, err
	}
	if err := c.limits.checkResponse(resp.Usage); err != nil {
		return nil, err
	}
	return resp, nil
}

type usagePreRequestLimitsCapability struct {
	limits UsageLimits
}

func (usagePreRequestLimitsCapability) Setup(*CapabilityRegistry) error { return nil }

func (c usagePreRequestLimitsCapability) WrapModelRequest(
	ctx context.Context, ri *RunInfo, msgs []ModelMessage, params ModelRequestParams, next ModelRequestFunc,
) (*ModelResponse, error) {
	current := ri.Usage()
	if err := c.limits.checkBeforeRequest(current); err != nil {
		return nil, err
	}
	if c.limits.CountTokensBeforeRequest {
		model := ri.Model()
		counted, err := CountModelTokens(ctx, model, msgs, params)
		if err != nil {
			return nil, err
		}
		counted = priceProspectiveUsage(ctx, model, counted)
		if err := c.limits.checkCountedRequest(current, counted); err != nil {
			return nil, err
		}
	}
	return next(ctx, msgs, params)
}
