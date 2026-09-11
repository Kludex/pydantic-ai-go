package contextwindow

import genaiprices "github.com/pydantic/genai-prices/packages/go"

// Lookup returns the bundled context window for the best matching provider model.
func Lookup(model, providerID, providerURL string) int {
	if providerURL != "" {
		match, err := genaiprices.Calculate(genaiprices.PriceRequest{Model: model, ProviderAPIURL: providerURL})
		if err == nil {
			return contextWindows[match.ProviderID][match.ModelID]
		}
	}
	if providerID == "" {
		return 0
	}
	match, err := genaiprices.Calculate(genaiprices.PriceRequest{Model: model, ProviderID: providerID})
	if err != nil {
		return 0
	}
	return contextWindows[match.ProviderID][match.ModelID]
}
