package googledecisions

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

type wireBlock struct {
	Type       string `json:"type"`
	Text       string `json:"text,omitempty"`
	Data       string `json:"data,omitempty"`
	MimeType   string `json:"mime_type,omitempty"`
	Resolution string `json:"resolution,omitempty"`
}

type wireQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions,omitempty"`
	Criteria     any    `json:"criteria,omitempty"`
}

type wireResponse struct {
	Answers map[string]decision.WireAnswer `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"total_input_tokens"`
		OutputTokens int `json:"total_output_tokens"`
		ByModality   []struct {
			Modality string `json:"modality"`
			Tokens   int    `json:"tokens"`
		} `json:"input_tokens_by_modality"`
	} `json:"usage"`
}

func (model *Model) call(
	ctx context.Context,
	state any,
	questions map[string]decision.Question,
	images []ai.BinaryContent,
	settings ai.ModelSettings,
) (decision.Response, error) {
	// The API takes IDs of [A-Za-z0-9_-] only, which question names such as Ticket.urgent are not.
	names := make(map[string]string, len(questions))
	wireQuestions := make(map[string]wireQuestion, len(questions))
	for index, name := range slices.Sorted(maps.Keys(questions)) {
		key := "q" + strconv.Itoa(index)
		names[key] = name
		wireQuestions[key] = questionWire(questions[name])
	}
	body := map[string]any{"model": model.name, "input": decisionInput(state, images), "questions": wireQuestions}
	maps.Copy(body, settings.ExtraBody)
	encoded, err := json.Marshal(body)
	if err != nil {
		return decision.Response{}, fmt.Errorf("google decisions: encode request: %w", err)
	}
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, model.provider.BaseURL+"/decisions", bytes.NewReader(encoded),
	)
	if err != nil {
		return decision.Response{}, fmt.Errorf("google decisions: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	if model.provider.APIKey != "" {
		request.Header.Set("x-goog-api-key", model.provider.APIKey)
	}
	if model.provider.PrepareRequest != nil {
		if err := model.provider.PrepareRequest(request); err != nil {
			return decision.Response{}, fmt.Errorf("google decisions: prepare request: %w", err)
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
			StatusCode: response.StatusCode, ModelName: model.name, Body: string(data), Headers: response.Header.Clone(),
		}
	}
	var wire wireResponse
	if err := json.Unmarshal(data, &wire); err != nil {
		return decision.Response{}, invalidResponse(err)
	}
	byName := make(map[string]decision.WireAnswer, len(wire.Answers))
	for key, answer := range wire.Answers {
		name, exists := names[key]
		if !exists {
			return decision.Response{}, invalidResponse(fmt.Errorf("answer IDs do not match the questions"))
		}
		byName[name] = answer
	}
	answers, err := decision.DecodeAnswers(questions, byName)
	if err != nil {
		return decision.Response{}, invalidResponse(err)
	}
	for name, answer := range answers {
		if answer.Type != "noul" && !sumsToOne(answer.Probabilities) {
			return decision.Response{}, invalidResponse(fmt.Errorf("answer %q does not match its question", name))
		}
	}
	usage := ai.Usage{Requests: 1, InputTokens: wire.Usage.InputTokens, OutputTokens: wire.Usage.OutputTokens}
	for _, entry := range wire.Usage.ByModality {
		if usage.Details == nil {
			usage.Details = map[string]int{}
		}
		usage.Details[strings.ToLower(entry.Modality)+"_prompt_tokens"] = entry.Tokens
	}
	return decision.Response{Answers: answers, Model: model.name, Usage: usage}, nil
}

func invalidResponse(err error) error {
	return &ai.UnexpectedModelBehaviorError{Message: "google decisions: invalid response: " + err.Error()}
}

// sumsToOne checks unrounded probabilities, which sum to one far more tightly than rounded ones.
func sumsToOne(probabilities map[string]float64) bool {
	total := 0.0
	for _, value := range probabilities {
		total += value
	}
	return math.Abs(total-1) <= 1e-6
}

func questionWire(question decision.Question) wireQuestion {
	wire := wireQuestion{Type: question.Type}
	if question.Instructions != nil {
		wire.Instructions = text(question.Instructions)
	}
	switch question.Type {
	case "noul":
		criteria, _ := question.Criteria.(map[string]any)
		meanings := map[string]string{}
		for _, outcome := range []string{"true", "false"} {
			if meaning, exists := criteria[outcome]; exists && meaning != nil {
				meanings[outcome+"_description"] = text(meaning)
			}
		}
		if len(meanings) > 0 {
			wire.Criteria = meanings
		}
	case "choice":
		criteria := question.Criteria.(map[string]any)
		meanings := make(map[string]*string, len(criteria))
		for label, meaning := range criteria {
			meanings[label] = description(meaning)
		}
		wire.Criteria = meanings
	default:
		levels := question.Criteria.([]any)
		meanings := make([]*string, len(levels))
		for index, meaning := range levels {
			meanings[index] = description(meaning)
		}
		wire.Criteria = meanings
	}
	return wire
}

func description(meaning any) *string {
	if meaning == nil {
		return nil
	}
	value := text(meaning)
	return &value
}

func decisionInput(state any, images []ai.BinaryContent) any {
	stateText := text(state)
	if len(images) == 0 {
		return stateText
	}
	blocks := []wireBlock{{Type: "text", Text: stateText}}
	for index, image := range images {
		blocks = append(blocks,
			wireBlock{Type: "text", Text: fmt.Sprintf("<image %d>:", index+1)},
			wireBlock{
				Type: "image", Data: base64.StdEncoding.EncodeToString(image.Data), MimeType: image.MediaType,
				Resolution: resolution(image.VendorMetadata["media_resolution"]),
			},
		)
	}
	return blocks
}

// resolution reads GoogleModel's media_resolution, either a level or {"level": level}.
func resolution(value any) string {
	if object, ok := value.(map[string]any); ok {
		value = object["level"]
	}
	level, _ := value.(string)
	if level == "" || level == "MEDIA_RESOLUTION_UNSPECIFIED" {
		return ""
	}
	return strings.ToLower(strings.TrimPrefix(level, "MEDIA_RESOLUTION_"))
}

func text(value any) string {
	if value, ok := value.(string); ok {
		return value
	}
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(value)
	return strings.TrimSuffix(encoded.String(), "\n")
}
