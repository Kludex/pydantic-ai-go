package typesafe

import (
	"fmt"
	"math"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func schemaFields(schema map[string]any) (map[string]map[string]any, error) {
	definitions, _ := schema["$defs"].(map[string]any)
	var resolve func(map[string]any) map[string]any
	resolve = func(value map[string]any) map[string]any {
		if reference, ok := value["$ref"].(string); ok {
			if target, ok := definitions[strings.TrimPrefix(reference, "#/$defs/")].(map[string]any); ok {
				merged := resolve(target)
				for key, item := range value {
					if key != "$ref" {
						merged[key] = item
					}
				}
				value = merged
			}
		}
		return value
	}
	if reference, ok := schema["$ref"].(string); ok {
		if target, ok := definitions[strings.TrimPrefix(reference, "#/$defs/")].(map[string]any); ok {
			schema = target
		}
	}
	properties, _ := schema["properties"].(map[string]any)
	if properties == nil && schema["type"] != "object" {
		return map[string]map[string]any{"response": schema}, nil
	}
	result := map[string]map[string]any{}
	var flatten func(map[string]any, string) error
	flatten = func(values map[string]any, prefix string) error {
		for name, raw := range values {
			if strings.Contains(name, ".") {
				return fmt.Errorf("typesafe: field %q contains a dot", prefix+name)
			}
			property, ok := raw.(map[string]any)
			if !ok {
				return fmt.Errorf("typesafe: field %q has an invalid schema", prefix+name)
			}
			property = resolve(property)
			if nested, ok := property["properties"].(map[string]any); ok {
				if err := flatten(nested, prefix+name+"."); err != nil {
					return err
				}
			} else {
				result[prefix+name] = property
			}
		}
		return nil
	}
	if err := flatten(properties, ""); err != nil {
		return nil, err
	}
	return result, nil
}

func buildQuestions(fields map[string]map[string]any, goal, instructions string) (map[string]question, error) {
	questions := make(map[string]question, len(fields))
	for name, property := range fields {
		ask := map[string]any{"field": name}
		if description, _ := property["description"].(string); description != "" {
			ask["question"] = description
		}
		if goal != "" {
			ask["goal"] = goal
		}
		if instructions != "" {
			ask["instructions"] = instructions
		}
		property, noneKey := optionalProperty(property)
		options := propertyOptions(property)
		if noneKey != "" {
			if options == nil {
				return nil, unsupportedField(name)
			}
			options[noneKey] = "None of these."
		}
		switch property["type"] {
		case "array":
			items, _ := property["items"].(map[string]any)
			labels := propertyOptions(items)
			if len(labels) < 2 {
				return nil, unsupportedField(name)
			}
			for key, description := range labels {
				itemAsk := mapsClone(ask)
				itemAsk["option"] = key
				if description, _ := description.(string); description != "" {
					itemAsk["option"] = key + ": " + description
				}
				questions[name+"."+key] = question{Type: "noul", Instructions: compactInstructions(itemAsk)}
			}
		case "boolean":
			if len(ask) == 1 {
				return nil, fmt.Errorf("typesafe: output field %q asks no question", name)
			}
			questions[name] = question{Type: "noul", Instructions: compactInstructions(ask)}
		case "number":
			if number(property["minimum"]) != 0 || number(property["maximum"]) != 1 || len(ask) == 1 {
				return nil, unsupportedField(name)
			}
			questions[name] = question{Type: "noul", Instructions: compactInstructions(ask)}
		default:
			if rubric := rubricCriteria(property); rubric != nil {
				questions[name] = question{Type: "score", Instructions: compactInstructions(ask), Criteria: rubric}
				continue
			}
			if len(options) < 2 || len(options) > 255 {
				return nil, unsupportedField(name)
			}
			questions[name] = question{Type: "choice", Instructions: compactInstructions(ask), Criteria: options}
		}
	}
	return questions, nil
}

func compactInstructions(values map[string]any) any {
	if len(values) == 1 {
		for _, value := range values {
			return value
		}
	}
	return values
}

func optionalProperty(property map[string]any) (map[string]any, string) {
	variants, ok := property["anyOf"].([]any)
	if !ok || len(variants) != 2 {
		return property, ""
	}
	var inner map[string]any
	for _, raw := range variants {
		candidate, _ := raw.(map[string]any)
		if candidate["type"] != "null" {
			inner = candidate
		}
	}
	if inner == nil {
		return property, ""
	}
	key := "none"
	options := propertyOptions(inner)
	for {
		if _, exists := options[key]; !exists {
			break
		}
		key += "_"
	}
	return inner, key
}

func propertyOptions(property map[string]any) map[string]any {
	if values, ok := property["enum"].([]any); ok {
		result := make(map[string]any, len(values))
		for _, value := range values {
			key, ok := value.(string)
			if !ok {
				return nil
			}
			result[key] = nil
		}
		return result
	}
	if variants, ok := property["anyOf"].([]any); ok {
		result := make(map[string]any, len(variants))
		for _, raw := range variants {
			variant, ok := raw.(map[string]any)
			if !ok {
				return nil
			}
			key, ok := variant["const"].(string)
			if !ok {
				return nil
			}
			result[key] = variant["description"]
		}
		return result
	}
	return nil
}

func rubricCriteria(property map[string]any) []any {
	variants, ok := property["anyOf"].([]any)
	if !ok || len(variants) < 2 {
		return nil
	}
	criteria := make([]any, len(variants))
	for _, raw := range variants {
		variant, ok := raw.(map[string]any)
		if !ok {
			return nil
		}
		level := int(number(variant["const"]))
		description, _ := variant["description"].(string)
		if level < 0 || level >= len(variants) || description == "" || math.IsNaN(number(variant["const"])) {
			return nil
		}
		criteria[level] = description
	}
	for _, item := range criteria {
		if item == nil {
			return nil
		}
	}
	return criteria
}

func decodeAnswers(answers map[string]answer, fields map[string]map[string]any, questions map[string]question, threshold float64) (map[string]any, map[string]any, error) {
	result := map[string]any{}
	confidence := map[string]float64{}
	probabilities := map[string]map[string]float64{}
	scores := map[string]float64{}
	for name, property := range fields {
		property, noneKey := optionalProperty(property)
		if property["type"] == "array" {
			labels := propertyOptions(property["items"].(map[string]any))
			selected := make([]string, 0, len(labels))
			sure := 1.0
			values := map[string]float64{}
			for label := range labels {
				value := answers[name+"."+label]
				if value.Noul == nil {
					return nil, nil, fmt.Errorf("typesafe: invalid answer for %q", name+"."+label)
				}
				chosen, certainty := verdict(*value.Noul, threshold)
				values[label] = *value.Noul
				if chosen {
					selected = append(selected, label)
				}
				sure = min(sure, certainty)
			}
			slices.Sort(selected)
			setNested(result, name, selected)
			confidence[name] = sure
			probabilities[name] = values
			continue
		}
		value := answers[name]
		switch questions[name].Type {
		case "noul":
			if value.Noul == nil {
				return nil, nil, fmt.Errorf("typesafe: invalid answer for %q", name)
			}
			if property["type"] == "number" {
				setNested(result, name, *value.Noul)
			} else {
				chosen, sure := verdict(*value.Noul, threshold)
				setNested(result, name, chosen)
				confidence[name] = sure
			}
		case "choice":
			if value.Choice == "" {
				return nil, nil, fmt.Errorf("typesafe: invalid answer for %q", name)
			}
			var chosen any = value.Choice
			if value.Choice == noneKey {
				chosen = nil
			}
			setNested(result, name, chosen)
			confidence[name] = value.Confidence
			probabilities[name] = value.Probabilities
		case "score":
			if value.Score == nil {
				return nil, nil, fmt.Errorf("typesafe: invalid score for %q", name)
			}
			setNested(result, name, int(math.Floor(*value.Score+.5)))
			confidence[name] = value.Confidence
			probabilities[name] = value.Probabilities
			scores[name] = *value.Score
		}
	}
	return result, map[string]any{"confidence": confidence, "probabilities": probabilities, "scores": scores}, nil
}

func verdict(probability, threshold float64) (bool, float64) {
	if probability < 0 || probability > 1 {
		return false, 0
	}
	if probability >= threshold {
		if threshold == 1 {
			return true, 0
		}
		return true, (probability - threshold) / (1 - threshold)
	}
	if threshold == 0 {
		return false, 0
	}
	return false, (threshold - probability) / threshold
}
func setNested(result map[string]any, name string, value any) {
	parts := strings.Split(name, ".")
	for _, part := range parts[:len(parts)-1] {
		nested, _ := result[part].(map[string]any)
		if nested == nil {
			nested = map[string]any{}
			result[part] = nested
		}
		result = nested
	}
	result[parts[len(parts)-1]] = value
}
func unsupportedField(name string) error {
	return fmt.Errorf("typesafe: output field %q is not supported", name)
}
func number(value any) float64 {
	switch value := value.(type) {
	case float64:
		return value
	case int:
		return float64(value)
	default:
		return math.NaN()
	}
}
func mapsClone(value map[string]any) map[string]any {
	result := make(map[string]any, len(value))
	for key, item := range value {
		result[key] = item
	}
	return result
}

func messageState(messages []ai.ModelMessage) (any, error) {
	history := make([]any, 0)
	latest := ""
	for index, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.SystemPromptPart:
					history = append(history, map[string]any{"system": part.Content})
				case ai.UserPromptPart:
					text, err := promptText(part)
					if err != nil {
						return nil, err
					}
					if index == len(messages)-1 {
						latest = strings.TrimSpace(strings.Join([]string{latest, text}, "\n\n"))
					} else {
						history = append(history, map[string]any{"user": text})
					}
				case ai.ToolReturnPart:
					history = append(history, map[string]any{"tool_return": map[string]any{"name": part.ToolName, "content": fmt.Sprint(part.Content)}})
				case ai.RetryPromptPart:
					history = append(history, map[string]any{"retry": part.ModelResponse()})
				}
			}
		case ai.ModelResponse:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.TextPart:
					history = append(history, map[string]any{"assistant": part.Content})
				case ai.ToolCallPart:
					history = append(history, map[string]any{"tool_call": map[string]any{"name": part.ToolName, "args": string(part.Args)}})
				case ai.FilePart:
					return nil, fmt.Errorf("typesafe: files are not supported")
				}
			}
		}
	}
	if latest == "" && len(history) == 0 {
		return nil, fmt.Errorf("typesafe: request needs text to judge")
	}
	if len(history) == 0 {
		return map[string]any{"prompt": latest}, nil
	}
	state := map[string]any{"history": history}
	if latest != "" {
		state["prompt"] = latest
	}
	return state, nil
}

func promptText(part ai.UserPromptPart) (string, error) {
	if len(part.Contents) == 0 {
		return part.Content, nil
	}
	texts := make([]string, 0, len(part.Contents))
	for _, content := range part.Contents {
		switch content := content.(type) {
		case ai.TextContent:
			texts = append(texts, content.Text)
		case ai.CachePoint:
		default:
			return "", fmt.Errorf("typesafe: files are not supported")
		}
	}
	return strings.Join(texts, "\n\n"), nil
}
