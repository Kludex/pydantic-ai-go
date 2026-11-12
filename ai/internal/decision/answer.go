package decision

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// WireAnswer is one answer as a protocol backend sends it, before validation.
type WireAnswer struct {
	Type          string              `json:"type"`
	Noul          *float64            `json:"noul"`
	Choice        *string             `json:"choice"`
	Confidence    *float64            `json:"confidence"`
	Probabilities map[string]*float64 `json:"probabilities"`
	Score         *float64            `json:"score"`
	Legend        map[string]any      `json:"legend"`
}

// DecodeAnswers checks that answers name each question once and fit it, tolerating
// two-decimal rounding and float error, then converts them.
func DecodeAnswers(questions map[string]Question, answers map[string]WireAnswer) (map[string]Answer, error) {
	if len(answers) != len(questions) {
		return nil, fmt.Errorf("answer names do not match the questions")
	}
	decoded := make(map[string]Answer, len(answers))
	for name, question := range questions {
		answer, exists := answers[name]
		if !exists || !answerFits(question, answer) {
			return nil, fmt.Errorf("answer %q does not match its question", name)
		}
		result := Answer{Type: answer.Type, Noul: answer.Noul, Score: answer.Score}
		if answer.Choice != nil {
			result.Choice = *answer.Choice
		}
		if answer.Confidence != nil {
			result.Confidence = *answer.Confidence
		}
		if answer.Type == "choice" || answer.Type == "score" {
			result.Probabilities = make(map[string]float64, len(answer.Probabilities))
			for label, value := range answer.Probabilities {
				result.Probabilities[label] = *value
			}
		}
		decoded[name] = result
	}
	return decoded, nil
}

func answerFits(question Question, answer WireAnswer) bool {
	valid := answer.Type == question.Type
	switch question.Type {
	case "noul":
		return valid && answer.Noul != nil && probability(*answer.Noul)
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
	if !valid {
		return false
	}
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
	return valid && (answer.Type != "score" || consistentScore(*answer.Score, answer.Probabilities))
}

func probability(value float64) bool {
	return !math.IsNaN(value) && value >= 0 && value <= 1
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
	half := new(big.Rat).SetFrac(big.NewInt(1), denominator)
	// Unrounded floats print every digit, so their interval is never narrower than their arithmetic's error.
	if tolerance := big.NewRat(1, 1_000_000); half.Cmp(tolerance) < 0 {
		return tolerance
	}
	return half
}
