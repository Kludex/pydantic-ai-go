package typesafe

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func schemaFields(schema map[string]any) (map[string]map[string]any, map[string]bool, error) {
	definitions, _ := schema["$defs"].(map[string]any)
	var resolve func(map[string]any) map[string]any
	resolve = func(value map[string]any) map[string]any {
		resolved := mapsClone(value)
		if reference, ok := value["$ref"].(string); ok {
			if target, ok := definitions[strings.TrimPrefix(reference, "#/$defs/")].(map[string]any); ok {
				resolved = resolve(target)
				for key, item := range value {
					if key != "$ref" {
						resolved[key] = item
					}
				}
			}
		}
		for _, key := range []string{"items", "propertyNames"} {
			if nested, ok := resolved[key].(map[string]any); ok {
				resolved[key] = resolve(nested)
			}
		}
		if variants, ok := resolved["anyOf"].([]any); ok {
			cloned := make([]any, len(variants))
			for index, variant := range variants {
				if nested, ok := variant.(map[string]any); ok {
					cloned[index] = resolve(nested)
				} else {
					cloned[index] = variant
				}
			}
			resolved["anyOf"] = cloned
		}
		return resolved
	}
	schema = resolve(schema)
	properties, _ := schema["properties"].(map[string]any)
	if properties == nil && schema["type"] != "object" {
		return map[string]map[string]any{"response": schema}, nil, nil
	}
	result := map[string]map[string]any{}
	defaulted := map[string]bool{}
	var flatten func(map[string]any, string, map[string]bool) error
	flatten = func(values map[string]any, prefix string, seen map[string]bool) error {
		for name, raw := range values {
			if strings.Contains(name, ".") {
				return fmt.Errorf("typesafe: field %q contains a dot", prefix+name)
			}
			property, ok := raw.(map[string]any)
			if !ok {
				return fmt.Errorf("typesafe: field %q has an invalid schema", prefix+name)
			}
			reference, _ := property["$ref"].(string)
			property = resolve(property)
			if nested, ok := property["properties"].(map[string]any); ok {
				if _, ok := property["default"]; ok {
					defaulted[prefix+name] = true
				}
				if reference != "" && seen[reference] {
					return fmt.Errorf("typesafe: output field %q contains itself", prefix+name)
				}
				next := make(map[string]bool, len(seen)+1)
				for item := range seen {
					next[item] = true
				}
				if reference != "" {
					next[reference] = true
				}
				if err := flatten(nested, prefix+name+".", next); err != nil {
					return err
				}
			} else {
				result[prefix+name] = property
			}
		}
		return nil
	}
	if err := flatten(properties, "", map[string]bool{}); err != nil {
		return nil, nil, err
	}
	return result, defaulted, nil
}

func buildQuestions(fields map[string]map[string]any, goal, instructions, chosen string) (map[string]question, error) {
	questions := make(map[string]question, len(fields))
	for name, property := range fields {
		ask := map[string]any{"field": name}
		if description, _ := property["description"].(string); description != "" {
			ask["question"] = description
		}
		if chosen != "" {
			ask["chosen"] = chosen
		}
		if goal != "" {
			ask["goal"] = goal
		}
		if instructions != "" {
			ask["instructions"] = instructions
		}
		property, noneOption := optionalProperty(property)
		options := propertyOptions(property)
		if allBooleanOptions(options) && allOptionsUndescribed(options) {
			options = nil
		}
		if len(noneOption) > 0 {
			if options == nil || rubricCriteria(options) != nil || !pickableOptions(options) {
				return nil, unsupportedField(name)
			}
			options = append(options, noneOption...)
		}
		if labels := mappingOptions(property); labels != nil {
			if len(labels) < 2 {
				return nil, unsupportedField(name)
			}
			fanOut(questions, name, ask, labels)
			continue
		}
		asked := compactInstructions(ask)
		switch {
		case property["type"] == "array":
			items, _ := property["items"].(map[string]any)
			labels := stringOptions(propertyOptions(items))
			if len(labels) < 2 || property["minItems"] != nil || property["maxItems"] != nil {
				return nil, unsupportedField(name)
			}
			fanOut(questions, name, ask, labels)
		case options != nil:
			switch rubric := rubricCriteria(options); {
			case rubric != nil:
				questions[name] = question{Type: "score", Instructions: asked, Criteria: rubric}
			case allBooleanOptions(options) && len(options) == 2:
				criteria := map[string]any{}
				for _, option := range options {
					if option.description != nil {
						criteria[strconv.FormatBool(option.value.(bool))] = option.description
					}
				}
				questions[name] = question{Type: "noul", Instructions: asked, Criteria: criteria}
			case len(options) < 2 || len(options) > 255 || !pickableOptions(options):
				return nil, unsupportedField(name)
			default:
				questions[name] = question{Type: "choice", Instructions: asked, Criteria: choiceCriteria(options)}
			}
		case property["type"] == "boolean":
			if len(ask) == 1 {
				return nil, fmt.Errorf("typesafe: output field %q asks no question", name)
			}
			questions[name] = question{Type: "noul", Instructions: asked}
		case property["type"] == "number":
			if _, ok := boundedNumber(property); !ok || len(ask) == 1 {
				return nil, unsupportedField(name)
			}
			questions[name] = question{Type: "noul", Instructions: asked}
		default:
			return nil, unsupportedField(name)
		}
	}
	return questions, nil
}

func fanOut(questions map[string]question, name string, ask map[string]any, labels map[string]any) {
	for key, description := range labels {
		itemAsk := mapsClone(ask)
		itemAsk["option"] = key
		if description, _ := description.(string); description != "" {
			itemAsk["option"] = key + ": " + description
		}
		questions[name+"."+key] = question{Type: "noul", Instructions: compactInstructions(itemAsk)}
	}
}

func boundedNumber(property map[string]any) (float64, bool) {
	maximum := number(property["maximum"])
	return maximum, property["type"] == "number" && number(property["minimum"]) == 0 && maximum > 0 &&
		!math.IsNaN(maximum) && property["multipleOf"] == nil
}

func mappingOptions(property map[string]any) map[string]any {
	properties, _ := property["properties"].(map[string]any)
	if property["type"] != "object" || len(properties) > 0 || property["minProperties"] != nil ||
		property["maxProperties"] != nil {
		return nil
	}
	values, _ := property["additionalProperties"].(map[string]any)
	if len(values) != 1 || values["type"] != "boolean" {
		return nil
	}
	names, _ := property["propertyNames"].(map[string]any)
	return stringOptions(propertyOptions(names))
}

func compactInstructions(values map[string]any) any {
	if len(values) == 1 {
		for _, value := range values {
			return value
		}
	}
	return values
}

func optionalProperty(property map[string]any) (map[string]any, []schemaOption) {
	variants, ok := property["anyOf"].([]any)
	if !ok || len(variants) != 2 {
		return property, nil
	}
	var inner, null map[string]any
	for _, raw := range variants {
		candidate, _ := raw.(map[string]any)
		if candidate["type"] == "null" {
			if null != nil {
				return property, nil
			}
			null = candidate
		} else {
			inner = candidate
		}
	}
	if inner == nil || null == nil {
		return property, nil
	}
	for key, value := range property {
		if key != "anyOf" && key != "default" {
			inner[key] = value
		}
	}
	taken := map[string]bool{}
	for _, option := range propertyOptions(inner) {
		if value, ok := option.value.(string); ok {
			taken[value] = true
		}
	}
	key := "none"
	for taken[key] {
		key += "_"
	}
	description, _ := null["description"].(string)
	if description == "" {
		description = "None of these."
	}
	return inner, []schemaOption{{value: key, description: description, none: true}}
}

type schemaOption struct {
	value       any
	description any
	none        bool
}

func propertyOptions(property map[string]any) []schemaOption {
	if values, ok := property["enum"].([]any); ok {
		result := make([]schemaOption, len(values))
		for index, value := range values {
			result[index] = schemaOption{value: value}
		}
		return result
	}
	if variants, ok := property["anyOf"].([]any); ok {
		result := make([]schemaOption, 0, len(variants))
		for _, raw := range variants {
			variant, ok := raw.(map[string]any)
			if !ok {
				return nil
			}
			value, ok := variant["const"]
			if !ok {
				return nil
			}
			result = append(result, schemaOption{value: value, description: variant["description"]})
		}
		return result
	}
	return nil
}

func stringOptions(options []schemaOption) map[string]any {
	result := make(map[string]any, len(options))
	for _, option := range options {
		value, ok := option.value.(string)
		if !ok {
			return nil
		}
		result[value] = option.description
	}
	return result
}

func allBooleanOptions(options []schemaOption) bool {
	if len(options) == 0 {
		return false
	}
	for _, option := range options {
		if _, ok := option.value.(bool); !ok {
			return false
		}
	}
	return true
}

func allOptionsUndescribed(options []schemaOption) bool {
	for _, option := range options {
		if description, _ := option.description.(string); description != "" {
			return false
		}
	}
	return true
}

func pickableOptions(options []schemaOption) bool {
	for _, option := range options {
		if _, ok := option.value.(string); ok {
			continue
		}
		if _, ok := wholeNumber(option.value); !ok {
			return false
		}
	}
	return true
}

func labelledOptions(options []schemaOption) map[string]any {
	taken := map[string]bool{}
	for _, option := range options {
		if value, ok := option.value.(string); ok {
			taken[value] = true
		}
	}
	result := make(map[string]any, len(options))
	for _, option := range options {
		label, ok := option.value.(string)
		if !ok {
			value, _ := wholeNumber(option.value)
			label = strconv.FormatInt(value, 10)
			for taken[label] {
				label += " (number)"
			}
			taken[label] = true
		}
		result[label] = option.value
	}
	return result
}

func choiceCriteria(options []schemaOption) map[string]any {
	values := labelledOptions(options)
	criteria := make(map[string]any, len(values))
	for label, value := range values {
		for _, option := range options {
			if reflect.DeepEqual(option.value, value) {
				criteria[label] = option.description
				break
			}
		}
	}
	return criteria
}

func rubricCriteria(options []schemaOption) []any {
	if len(options) < 2 || len(options) > 10 {
		return nil
	}
	criteria := make([]any, len(options))
	for _, option := range options {
		level, ok := wholeNumber(option.value)
		description, described := option.description.(string)
		if !ok || level < 0 || level >= int64(len(options)) || !described || description == "" {
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

func wholeNumber(value any) (int64, bool) {
	if value, ok := value.(json.Number); ok {
		parsed, err := value.Int64()
		return parsed, err == nil
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return reflected.Int(), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if unsigned := reflected.Uint(); unsigned <= math.MaxInt64 {
			return int64(unsigned), true
		}
	case reflect.Float32, reflect.Float64:
		floating := reflected.Float()
		if floating >= math.MinInt64 && floating <= math.MaxInt64 && floating == math.Trunc(floating) {
			return int64(floating), true
		}
	}
	return 0, false
}

func decodeAnswers(
	answers map[string]answer,
	fields map[string]map[string]any,
	questions map[string]question,
	threshold float64,
	defaulted map[string]bool,
) (map[string]any, map[string]any, error) {
	result := map[string]any{}
	confidence := map[string]float64{}
	probabilities := map[string]map[string]float64{}
	scores := map[string]float64{}
	for name, originalProperty := range fields {
		property, noneOption := optionalProperty(originalProperty)
		if labels := mappingOptions(property); labels != nil {
			selected := make(map[string]bool, len(labels))
			values, sure, err := fanIn(answers, name, labels, threshold, func(label string, chosen bool) {
				selected[label] = chosen
			})
			if err != nil {
				return nil, nil, err
			}
			setNested(result, name, selected)
			confidence[name] = sure
			probabilities[name] = values
			continue
		}
		if property["type"] == "array" {
			labels := stringOptions(propertyOptions(property["items"].(map[string]any)))
			selected := make([]string, 0, len(labels))
			values, sure, err := fanIn(answers, name, labels, threshold, func(label string, chosen bool) {
				if chosen {
					selected = append(selected, label)
				}
			})
			if err != nil {
				return nil, nil, err
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
			if maximum, ok := boundedNumber(property); ok {
				setNested(result, name, *value.Noul*maximum)
			} else {
				chosen, sure := verdict(*value.Noul, threshold)
				setNested(result, name, chosen)
				confidence[name] = sure
			}
		case "choice":
			if value.Choice == "" {
				return nil, nil, fmt.Errorf("typesafe: invalid answer for %q", name)
			}
			if isNoneChoice(value.Choice, noneOption) {
				if _, hasDefault := originalProperty["default"]; hasDefault {
					ensureNested(result, name)
				} else {
					setNested(result, name, nil)
				}
			} else {
				chosen, exists := labelledOptions(propertyOptions(property))[value.Choice]
				if !exists {
					chosen = value.Choice
				}
				setNested(result, name, chosen)
			}
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
	leaveOutUnanswered(result, fields, defaulted, "")
	return result, map[string]any{"confidence": confidence, "probabilities": probabilities, "scores": scores}, nil
}

func fanIn(
	answers map[string]answer,
	name string,
	labels map[string]any,
	threshold float64,
	set func(string, bool),
) (map[string]float64, float64, error) {
	values := make(map[string]float64, len(labels))
	sure := 1.0
	for label := range labels {
		value := answers[name+"."+label]
		if value.Noul == nil {
			return nil, 0, fmt.Errorf("typesafe: invalid answer for %q", name+"."+label)
		}
		chosen, certainty := verdict(*value.Noul, threshold)
		values[label] = *value.Noul
		set(label, chosen)
		sure = min(sure, certainty)
	}
	return values, sure, nil
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
	return false, (threshold - probability) / threshold
}
func isNoneChoice(choice string, options []schemaOption) bool {
	for _, option := range options {
		if option.none && option.value == choice {
			return true
		}
	}
	return false
}

func ensureNested(result map[string]any, name string) {
	parts := strings.Split(name, ".")
	for _, part := range parts[:len(parts)-1] {
		nested, _ := result[part].(map[string]any)
		if nested == nil {
			nested = map[string]any{}
			result[part] = nested
		}
		result = nested
	}
}

func setNested(result map[string]any, name string, value any) {
	ensureNested(result, name)
	parts := strings.Split(name, ".")
	for _, part := range parts[:len(parts)-1] {
		result = result[part].(map[string]any)
	}
	result[parts[len(parts)-1]] = value
}

func leaveOutUnanswered(
	result map[string]any, fields map[string]map[string]any, defaulted map[string]bool, prefix string,
) bool {
	answered := false
	for name, value := range result {
		path := prefix + name
		if _, exists := fields[path]; exists {
			answered = true
			continue
		}
		nested, ok := value.(map[string]any)
		if !ok || leaveOutUnanswered(nested, fields, defaulted, path+".") {
			answered = true
		} else if defaulted[path] {
			delete(result, name)
		}
	}
	return answered
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
					var arguments any
					if err := json.Unmarshal(part.Args, &arguments); err != nil {
						arguments = string(part.Args)
					}
					history = append(history, map[string]any{"tool_call": map[string]any{"name": part.ToolName, "args": arguments}})
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
