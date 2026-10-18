package openaidecisions

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
)

type wireQuestion struct {
	Type         string       `json:"type"`
	Name         string       `json:"name"`
	Instructions string       `json:"instructions"`
	Choices      []wireChoice `json:"choices,omitempty"`
	Levels       []wireLevel  `json:"levels,omitempty"`
}
type wireChoice struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}
type wireLevel struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}
type wireInputMessage struct {
	Role    string          `json:"role"`
	Content []wireInputPart `json:"content"`
}
type wireInputPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
}
type wireResponse struct {
	Model   string       `json:"model"`
	Answers []wireAnswer `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		InputDetails struct {
			CachedTokens     int `json:"cached_tokens"`
			CacheWriteTokens int `json:"cache_write_tokens"`
		} `json:"input_tokens_details"`
		OutputDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
}
type wireAnswer struct {
	Type          string            `json:"type"`
	Name          string            `json:"name"`
	Probability   *float64          `json:"probability"`
	Choice        any               `json:"choice"`
	Score         *float64          `json:"score"`
	Confidence    *float64          `json:"confidence"`
	Probabilities []wireProbability `json:"probabilities"`
}
type wireProbability struct {
	Value       any     `json:"value"`
	Probability float64 `json:"probability"`
}

func (model *Model) call(
	ctx context.Context,
	state any,
	questions map[string]decision.Question,
	images []ai.BinaryContent,
	settings ai.ModelSettings,
) (decision.Response, error) {
	body := map[string]any{
		"model": model.name, "input": decisionInput(state, images), "questions": decisionQuestions(questions),
	}
	maps.Copy(body, settings.ExtraBody)
	encoded, err := json.Marshal(body)
	if err != nil {
		return decision.Response{}, fmt.Errorf("openai decisions: encode request: %w", err)
	}
	endpoint := model.provider.BaseURL + "/decisions"
	if len(model.provider.Query) > 0 {
		endpoint += "?" + model.provider.Query.Encode()
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(encoded))
	if err != nil {
		return decision.Response{}, fmt.Errorf("openai decisions: %w", err)
	}
	request.Header = model.provider.Headers.Clone()
	if request.Header == nil {
		request.Header = make(http.Header)
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
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return decision.Response{}, ai.NewModelTransportError(ctx, model, "read response", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return decision.Response{}, &APIError{
			StatusCode: response.StatusCode, Body: string(data), Headers: response.Header.Clone(),
		}
	}
	var wire wireResponse
	if err := json.Unmarshal(data, &wire); err != nil {
		return decision.Response{}, invalidResponse(err)
	}
	answers, refused, err := decisionAnswers(questions, wire.Answers)
	if err != nil {
		return decision.Response{}, invalidResponse(err)
	}
	requestID := response.Header.Get("x-request-id")
	if len(refused) > 0 {
		return decision.Response{}, ai.NewContentFilterError(
			"Content filter triggered. OpenAI Decisions declined questions: "+strings.Join(refused, ", "),
			ai.ModelResponse{
				ModelName: wire.Model, ProviderName: model.ProviderName(), ProviderURL: model.ProviderURL(),
				ProviderResponseID: requestID, FinishReason: ai.FinishReasonContentFilter,
				ProviderDetails: map[string]any{"refused_questions": refused},
			},
		)
	}
	return decision.Response{
		Answers: answers, Model: wire.Model, ProviderResponseID: requestID,
		Usage: ai.Usage{
			Requests: 1, InputTokens: wire.Usage.InputTokens, OutputTokens: wire.Usage.OutputTokens,
			CacheReadTokens:  wire.Usage.InputDetails.CachedTokens,
			CacheWriteTokens: wire.Usage.InputDetails.CacheWriteTokens,
			ReasoningTokens:  wire.Usage.OutputDetails.ReasoningTokens,
		},
	}, nil
}

func invalidResponse(err error) error {
	return &ai.UnexpectedModelBehaviorError{Message: "openai decisions: invalid response: " + err.Error()}
}

func decisionQuestions(questions map[string]decision.Question) []wireQuestion {
	result := make([]wireQuestion, 0, len(questions))
	for _, name := range slices.Sorted(maps.Keys(questions)) {
		question := questions[name]
		mapped := wireQuestion{Type: question.Type, Name: name, Instructions: decisionText(question.Instructions)}
		switch question.Type {
		case "noul":
			mapped.Type = "predicate"
			mapped.Instructions = decisionText(predicateInstructions(question))
		case "choice":
			criteria := question.Criteria.(map[string]any)
			for _, label := range slices.Sorted(maps.Keys(criteria)) {
				mapped.Choices = append(mapped.Choices, wireChoice{Value: label, Description: decisionText(criteria[label])})
			}
		default:
			for index, description := range question.Criteria.([]any) {
				mapped.Levels = append(mapped.Levels, wireLevel{Label: strconv.Itoa(index), Description: decisionText(description)})
			}
		}
		result = append(result, mapped)
	}
	return result
}

func decisionInput(state any, images []ai.BinaryContent) any {
	text := decisionText(state)
	if len(images) == 0 {
		return text
	}
	parts := []wireInputPart{{Type: "input_text", Text: text}}
	for index, image := range images {
		detail, _ := image.VendorMetadata["detail"].(string)
		parts = append(parts,
			wireInputPart{Type: "input_text", Text: fmt.Sprintf("<image %d>:", index+1)},
			wireInputPart{Type: "input_image", ImageURL: "data:" + image.MediaType + ";base64," +
				base64.StdEncoding.EncodeToString(image.Data), Detail: detail},
		)
	}
	return []wireInputMessage{{Role: "user", Content: parts}}
}

func decisionText(value any) string {
	if value == nil {
		return ""
	}
	if text, ok := value.(string); ok {
		return text
	}
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(value)
	return strings.TrimSuffix(encoded.String(), "\n")
}

func predicateInstructions(question decision.Question) any {
	criteria, _ := question.Criteria.(map[string]any)
	meanings := map[string]any{}
	if value, exists := criteria["true"]; exists {
		meanings["yes"] = value
	}
	if value, exists := criteria["false"]; exists {
		meanings["no"] = value
	}
	if len(meanings) == 0 {
		return question.Instructions
	}
	switch instructions := question.Instructions.(type) {
	case nil:
	case map[string]any:
		combined := maps.Clone(instructions)
		maps.Copy(combined, meanings)
		return combined
	default:
		meanings["question"] = instructions
	}
	return meanings
}

func decisionAnswers(
	questions map[string]decision.Question, wire []wireAnswer,
) (map[string]decision.Answer, []string, error) {
	byName := make(map[string]wireAnswer, len(wire))
	for _, answer := range wire {
		if answer.Name == "" {
			return nil, nil, fmt.Errorf("answer omitted name")
		}
		if _, duplicate := byName[answer.Name]; duplicate {
			return nil, nil, fmt.Errorf("duplicate answer %q", answer.Name)
		}
		byName[answer.Name] = answer
	}
	if len(byName) != len(questions) {
		return nil, nil, fmt.Errorf("answer names do not match questions")
	}
	answers := make(map[string]decision.Answer, len(questions))
	var refused []string
	for name, question := range questions {
		wireAnswer, exists := byName[name]
		if !exists {
			return nil, nil, fmt.Errorf("answer names do not match questions")
		}
		if wireAnswer.Type == "refusal" {
			refused = append(refused, name)
			continue
		}
		answer, err := convertAnswer(question, wireAnswer)
		if err != nil {
			return nil, nil, fmt.Errorf("answer %q: %w", name, err)
		}
		answers[name] = answer
	}
	slices.Sort(refused)
	return answers, refused, nil
}

func convertAnswer(question decision.Question, wire wireAnswer) (decision.Answer, error) {
	switch question.Type {
	case "noul":
		if wire.Type != "predicate" || wire.Probability == nil || !probability(*wire.Probability) {
			return decision.Answer{}, fmt.Errorf("invalid predicate")
		}
		return decision.Answer{Type: "noul", Noul: wire.Probability}, nil
	case "choice":
		criteria := question.Criteria.(map[string]any)
		choice, ok := wire.Choice.(string)
		if wire.Type != "choice" || !ok || wire.Confidence == nil || !probability(*wire.Confidence) {
			return decision.Answer{}, fmt.Errorf("invalid choice")
		}
		if _, exists := criteria[choice]; !exists {
			return decision.Answer{}, fmt.Errorf("unknown choice %q", choice)
		}
		probabilities := map[string]float64{}
		for _, item := range wire.Probabilities {
			label, ok := item.Value.(string)
			_, known := criteria[label]
			_, duplicate := probabilities[label]
			if !ok || !known || duplicate || !probability(item.Probability) {
				return decision.Answer{}, fmt.Errorf("invalid choice probabilities")
			}
			probabilities[label] = item.Probability
		}
		if len(probabilities) != len(criteria) || !roundedDistribution(probabilities) {
			return decision.Answer{}, fmt.Errorf("choice probabilities do not match options")
		}
		return decision.Answer{
			Type: "choice", Choice: choice, Confidence: *wire.Confidence, Probabilities: probabilities,
		}, nil
	default:
		levels := len(question.Criteria.([]any))
		if wire.Type != "score" || wire.Score == nil || wire.Confidence == nil || !probability(*wire.Confidence) ||
			*wire.Score < 0 || *wire.Score > float64(levels-1) {
			return decision.Answer{}, fmt.Errorf("invalid score")
		}
		probabilities := map[string]float64{}
		weighted := 0.0
		for _, item := range wire.Probabilities {
			level, ok := item.Value.(float64)
			key := strconv.FormatFloat(level, 'f', -1, 64)
			_, duplicate := probabilities[key]
			if !ok || level != math.Trunc(level) || level < 0 || level >= float64(levels) || duplicate ||
				!probability(item.Probability) {
				return decision.Answer{}, fmt.Errorf("invalid score probabilities")
			}
			probabilities[key] = item.Probability
			weighted += level * item.Probability
		}
		if len(probabilities) != levels || !roundedDistribution(probabilities) || math.Abs(weighted-*wire.Score) > .015 {
			return decision.Answer{}, fmt.Errorf("score probabilities do not match score")
		}
		return decision.Answer{
			Type: "score", Score: wire.Score, Confidence: *wire.Confidence, Probabilities: probabilities,
		}, nil
	}
}

func probability(value float64) bool {
	return !math.IsNaN(value) && value >= 0 && value <= 1
}

func roundedDistribution(values map[string]float64) bool {
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum > 0 && math.Abs(sum-1) <= .005*float64(len(values))+.000001
}
