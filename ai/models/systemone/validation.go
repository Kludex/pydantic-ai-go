package systemone

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/internal/decision"
)

type wireAnswer struct {
	Type          string              `json:"type"`
	Noul          *float64            `json:"noul"`
	Choice        *string             `json:"choice"`
	Confidence    *float64            `json:"confidence"`
	Probabilities map[string]*float64 `json:"probabilities"`
	Score         *float64            `json:"score"`
	Legend        map[string]any      `json:"legend"`
}

type wireResponse struct {
	Model   *string               `json:"model"`
	Answers map[string]wireAnswer `json:"answers"`
	Usage   *struct {
		InputTokens  json.RawMessage `json:"input_tokens"`
		OutputTokens json.RawMessage `json:"output_tokens"`
	} `json:"usage"`
	inputTokens  int
	outputTokens int
}

func (response *wireResponse) validate(questions map[string]decision.Question) error {
	if response.Model == nil || response.Answers == nil || response.Usage == nil {
		return fmt.Errorf("missing model, answers, or valid usage")
	}
	for _, field := range []struct {
		raw   json.RawMessage
		value *int
	}{
		{response.Usage.InputTokens, &response.inputTokens}, {response.Usage.OutputTokens, &response.outputTokens},
	} {
		if len(field.raw) == 0 {
			continue
		}
		if string(field.raw) == "null" {
			return fmt.Errorf("usage tokens must be non-negative integers")
		}
		if err := json.Unmarshal(field.raw, field.value); err != nil || *field.value < 0 {
			return fmt.Errorf("usage tokens must be non-negative integers")
		}
	}
	if len(response.Answers) != len(questions) {
		return fmt.Errorf("answer names do not match the questions")
	}
	for name, question := range questions {
		answer, exists := response.Answers[name]
		valid := exists && answer.Type == question.Type
		switch question.Type {
		case "noul":
			valid = valid && answer.Noul != nil && probability(*answer.Noul)
		case "choice":
			criteria := question.Criteria.(map[string]any)
			valid = valid && answer.Choice != nil && len(answer.Probabilities) == len(criteria)
			if valid {
				_, valid = criteria[*answer.Choice]
				for label := range criteria {
					valid = valid && answer.Probabilities[label] != nil
				}
			}
		case "score":
			levels := len(question.Criteria.([]any))
			valid = valid && answer.Score != nil && !math.IsNaN(*answer.Score) &&
				*answer.Score >= 0 && *answer.Score <= float64(levels-1) &&
				len(answer.Probabilities) == levels && (len(answer.Legend) == 0 || len(answer.Legend) == levels)
			for level := 0; level < levels; level++ {
				key := strconv.Itoa(level)
				valid = valid && answer.Probabilities[key] != nil
				if len(answer.Legend) > 0 {
					_, exists := answer.Legend[key]
					valid = valid && exists
				}
			}
		}
		if valid && (answer.Type == "choice" || answer.Type == "score") {
			valid = answer.Confidence != nil && probability(*answer.Confidence)
			total := 0.0
			for _, value := range answer.Probabilities {
				if value == nil || !probability(*value) {
					valid = false
				} else {
					total += *value
				}
			}
			valid = valid && math.Abs(total-1) <= 1e-6+float64(len(answer.Probabilities))*.005
			if valid && answer.Type == "score" {
				valid = consistentScore(*answer.Score, answer.Probabilities)
			}
		}
		if !valid {
			return fmt.Errorf("answer %q does not match its question", name)
		}
	}
	return nil
}

func consistentScore(score float64, probabilities map[string]*float64) bool {
	lower := make([]*big.Rat, len(probabilities))
	upper := make([]*big.Rat, len(probabilities))
	remaining, capacity, mean := big.NewRat(1, 1), new(big.Rat), new(big.Rat)
	for level := range lower {
		value := decimal(*probabilities[strconv.Itoa(level)])
		half := halfUnit(*probabilities[strconv.Itoa(level)])
		low, high := new(big.Rat).Sub(value, half), new(big.Rat).Add(value, half)
		if low.Sign() < 0 {
			low.SetInt64(0)
		}
		if high.Cmp(big.NewRat(1, 1)) > 0 {
			high.SetInt64(1)
		}
		lower[level], upper[level] = low, high
		remaining.Sub(remaining, low)
		capacity.Add(capacity, new(big.Rat).Sub(high, low))
		mean.Add(mean, new(big.Rat).Mul(big.NewRat(int64(level), 1), low))
	}
	if remaining.Sign() < 0 || remaining.Cmp(capacity) > 0 {
		return false
	}
	bounds := [2]*big.Rat{new(big.Rat).Set(mean), new(big.Rat).Set(mean)}
	for direction := range bounds {
		rest := new(big.Rat).Set(remaining)
		for index := range lower {
			level := index
			if direction == 1 {
				level = len(lower) - 1 - index
			}
			taken := new(big.Rat).Sub(upper[level], lower[level])
			if taken.Cmp(rest) > 0 {
				taken.Set(rest)
			}
			bounds[direction].Add(bounds[direction], new(big.Rat).Mul(big.NewRat(int64(level), 1), taken))
			rest.Sub(rest, taken)
		}
	}
	value, half := decimal(score), halfUnit(score)
	return new(big.Rat).Add(value, half).Cmp(bounds[0]) >= 0 && new(big.Rat).Sub(value, half).Cmp(bounds[1]) <= 0
}

func decimal(value float64) *big.Rat {
	result, _ := new(big.Rat).SetString(strconv.FormatFloat(value, 'f', -1, 64))
	return result
}

func halfUnit(value float64) *big.Rat {
	text := strconv.FormatFloat(value, 'f', -1, 64)
	decimals := 2
	if dot := strings.IndexByte(text, '.'); dot >= 0 {
		decimals = max(decimals, len(text)-dot-1)
	}
	denominator := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(decimals)), nil)
	denominator.Mul(denominator, big.NewInt(2))
	return new(big.Rat).SetFrac(big.NewInt(1), denominator)
}
