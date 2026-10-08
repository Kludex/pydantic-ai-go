package systemone

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
)

func (model *Model) call(
	ctx context.Context, state any, questions map[string]decision.Question, settings ai.ModelSettings,
) (decision.Response, error) {
	body := map[string]any{"state": state, "model": model.name, "questions": questions}
	if settings.Temperature != nil {
		body["temperature"] = *settings.Temperature
	}
	for key, value := range settings.ExtraBody {
		body[key] = value
	}
	data, err := json.Marshal(body)
	if err != nil {
		return decision.Response{}, fmt.Errorf("systemone: encode request: %w", err)
	}
	if _, overridden := settings.ExtraBody["questions"]; overridden {
		var sent struct {
			Questions map[string]decision.Question `json:"questions"`
		}
		if err := json.Unmarshal(data, &sent); err != nil || sent.Questions == nil {
			return decision.Response{}, fmt.Errorf("systemone: invalid extra_body.questions override")
		}
		questions = sent.Questions
		for name, question := range questions {
			valid := false
			switch question.Type {
			case "noul":
				_, valid = question.Criteria.(map[string]any)
				valid = valid || question.Criteria == nil
			case "choice":
				_, valid = question.Criteria.(map[string]any)
			case "score":
				_, valid = question.Criteria.([]any)
			}
			if !valid {
				return decision.Response{}, fmt.Errorf("systemone: invalid extra_body.questions override for %q", name)
			}
		}
	}
	endpoint := model.ProviderURL()
	if !strings.HasSuffix(endpoint, "/v1") {
		endpoint += "/v1"
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint+"/systemone", bytes.NewReader(data))
	if err != nil {
		return decision.Response{}, err
	}
	request.Header = make(http.Header, len(model.provider.Headers))
	for key, values := range model.provider.Headers {
		for _, value := range values {
			request.Header.Add(key, value)
		}
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	if model.provider.APIKey != "" && request.Header.Get("Authorization") == "" {
		request.Header.Set("Authorization", "Bearer "+model.provider.APIKey)
	}
	if model.provider.PrepareRequest != nil {
		if err := model.provider.PrepareRequest(request); err != nil {
			return decision.Response{}, err
		}
	}
	for key, value := range settings.ExtraHeaders {
		request.Header.Set(key, value)
	}
	response, err := model.provider.HTTPClient.Do(request)
	if err != nil {
		return decision.Response{}, ai.NewModelTransportError(ctx, model, "request", err)
	}
	defer func() { _ = response.Body.Close() }()
	data, err = io.ReadAll(response.Body)
	if err != nil {
		return decision.Response{}, ai.NewModelTransportError(ctx, model, "read response", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return decision.Response{}, &APIError{
			StatusCode: response.StatusCode, ModelName: model.name, Body: string(data), Headers: response.Header.Clone(),
		}
	}
	var wire wireResponse
	if err := json.Unmarshal(data, &wire); err != nil {
		return decision.Response{}, &ai.UnexpectedModelBehaviorError{Message: "systemone: invalid response: " + err.Error()}
	}
	if err := wire.validate(questions); err != nil {
		return decision.Response{}, &ai.UnexpectedModelBehaviorError{Message: "systemone: invalid response: " + err.Error()}
	}
	result := decision.Response{Model: *wire.Model, Answers: make(map[string]decision.Answer, len(wire.Answers))}
	result.Usage.InputTokens, result.Usage.OutputTokens = wire.inputTokens, wire.outputTokens
	for name, answer := range wire.Answers {
		decoded := decision.Answer{Type: answer.Type, Noul: answer.Noul, Score: answer.Score}
		if answer.Choice != nil {
			decoded.Choice = *answer.Choice
		}
		if answer.Confidence != nil {
			decoded.Confidence = *answer.Confidence
		}
		if answer.Type == "choice" || answer.Type == "score" {
			decoded.Probabilities = make(map[string]float64, len(answer.Probabilities))
			for label, value := range answer.Probabilities {
				decoded.Probabilities[label] = *value
			}
		}
		result.Answers[name] = decoded
	}
	return result, nil
}
