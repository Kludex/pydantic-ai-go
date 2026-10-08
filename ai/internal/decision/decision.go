package decision

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Config describes one decision request and its provider transport.
type Config struct {
	ModelName            string
	ProviderName         string
	ProviderURL          string
	ToolCallID           string
	BooleanThreshold     float64
	RouteThreshold       float64
	MaxChoiceOptions     int
	MaxScoreLevels       int
	RequiresInstructions bool
	Call                 func(context.Context, any, map[string]Question, ai.ModelSettings) (Response, error)
	HandOff              func(Route) error
}

// Route describes a decision handed off to the provider's public error type.
type Route struct {
	ToolName      string
	Name          string
	Probability   float64
	Probabilities map[string]float64
	Threshold     float64
	Unfillable    bool
}

// Question is one typed protocol question.
type Question struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions,omitempty"`
	Criteria     any    `json:"criteria,omitempty"`
}

// Answer is one typed protocol answer.
type Answer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Score         *float64           `json:"score"`
}

// Response contains protocol answers and token usage.
type Response struct {
	Answers map[string]Answer `json:"answers"`
	Model   string            `json:"model"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// Request maps messages and schemas to decisions and reconstructs tool output.
func Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams, config Config,
) (*ai.ModelResponse, error) {
	if params.AllowText && params.OutputTool == nil {
		return nil, fmt.Errorf("decision: text output is not supported")
	}
	settings := params.Settings
	state, err := messageState(messages)
	if err != nil {
		return nil, err
	}
	type routeShape struct {
		tool      *ai.ToolDefinition
		fields    map[string]map[string]any
		defaults  map[string]bool
		contexts  map[string][]string
		questions map[string]Question
		scoped    map[string]string
		included  bool
		err       error
	}
	available := toolsLeft(messages, params.Tools)
	routes := make(map[string]routeShape, len(available)+1)
	labels := make([]string, 0, len(available)+1)
	addRoute := func(label string, tool *ai.ToolDefinition, purpose string) {
		for routes[label].tool != nil {
			label += " (output)"
		}
		fields, defaults, contexts, shapeErr := schemaFields(tool.Schema)
		var questions map[string]Question
		if shapeErr == nil {
			questions, shapeErr = buildQuestions(fields, contexts, purpose, params.Instructions, label, config, tool.Schema["type"] != "object")
		}
		routes[label] = routeShape{
			tool: tool, fields: fields, defaults: defaults, contexts: contexts, questions: questions, err: shapeErr,
		}
		labels = append(labels, label)
	}
	if params.OutputTool != nil {
		label := outputRouteLabel(*params.OutputTool)
		for slices.ContainsFunc(available, func(tool ai.ToolDefinition) bool { return tool.Name == label }) {
			label += " (output)"
		}
		addRoute(label, params.OutputTool, outputPurpose(*params.OutputTool, nil, params.Instructions))
	}
	for index := range available {
		addRoute(available[index].Name, &available[index], available[index].Description)
	}
	if len(routes) == 0 {
		return nil, fmt.Errorf("decision: request has no typed question")
	}
	selectedLabel := labels[0]
	probability := 1.0
	var routeDetails map[string]any
	questions := map[string]Question{}
	if len(routes) == 1 {
		shape := routes[selectedLabel]
		if shape.err != nil {
			return nil, shape.err
		}
		shape.questions, shape.err = buildQuestions(
			shape.fields, shape.contexts, shape.tool.Description, params.Instructions, "", config,
			shape.tool.Schema["type"] != "object",
		)
		if shape.err != nil {
			return nil, shape.err
		}
		routes[selectedLabel] = shape
		maps.Copy(questions, shape.questions)
	} else {
		criteria := make(map[string]any, len(routes))
		questionBytes := 0
		smallestQuestionBytes := 0
		allFillable := true
		for _, label := range labels {
			shape := routes[label]
			if shape.err != nil {
				allFillable = false
				continue
			}
			encoded, _ := json.Marshal(shape.questions)
			questionBytes += len(encoded)
			if smallestQuestionBytes == 0 || len(encoded) < smallestQuestionBytes {
				smallestQuestionBytes = len(encoded)
			}
		}
		stateJSON, _ := json.Marshal(state)
		unpickedBytes := questionBytes
		if allFillable {
			unpickedBytes -= smallestQuestionBytes
		}
		speculateAll := unpickedBytes/4 <= 260+len(stateJSON)/6 && len(stateJSON)/6+questionBytes/4 <= 16_000
		for _, label := range labels {
			shape := routes[label]
			criteria[label] = outputPurpose(*shape.tool, shape.fields, params.Instructions)
			if shape.tool != params.OutputTool {
				criteria[label] = shape.tool.Description
			}
			if shape.err == nil && (speculateAll || shape.tool == params.OutputTool) {
				shape.included = true
				shape.scoped = make(map[string]string, len(shape.questions))
				for name, value := range shape.questions {
					scoped := freeQuestionName(questions, label+"."+name)
					shape.scoped[name] = scoped
					questions[scoped] = value
				}
				routes[label] = shape
			}
		}
		if config.MaxChoiceOptions > 0 && len(criteria) > config.MaxChoiceOptions {
			return nil, fmt.Errorf("decision: route has too many options (maximum %d)", config.MaxChoiceOptions)
		}
		questions["route"] = Question{
			Type: "choice", Instructions: map[string]any{
				"question": "Which of these does this call for?", "background": params.Instructions,
			}, Criteria: criteria,
		}
	}
	if len(questions) == 0 {
		return nil, fmt.Errorf("decision: request has no typed question")
	}
	response, err := call(ctx, config, state, questions, settings)
	if err != nil {
		return nil, err
	}
	if len(routes) > 1 {
		choice := response.Answers["route"]
		selectedLabel = choice.Choice
		_, ok := routes[selectedLabel]
		var answered bool
		probability, answered = choice.Probabilities[selectedLabel]
		if !ok || !answered || probability < 0 || probability > 1 {
			return nil, fmt.Errorf("decision: invalid route answer %q", selectedLabel)
		}
		if probability < config.RouteThreshold {
			return nil, config.HandOff(Route{
				Name: selectedLabel, Probability: probability,
				Probabilities: maps.Clone(choice.Probabilities), Threshold: config.RouteThreshold,
			})
		}
		routeDetails = map[string]any{
			"choice": selectedLabel, "probabilities": maps.Clone(choice.Probabilities), "offered": labels,
		}
	}
	selected := routes[selectedLabel]
	if selected.err != nil {
		return nil, config.HandOff(Route{
			ToolName: selected.tool.Name, Name: selectedLabel, Probability: probability, Unfillable: true,
		})
	}
	usage := ai.Usage{Requests: 1, InputTokens: response.Usage.InputTokens, OutputTokens: response.Usage.OutputTokens}
	answers := response.Answers
	if len(routes) > 1 && selected.included {
		unscoped := make(map[string]Answer, len(selected.questions))
		for name := range selected.questions {
			unscoped[name] = answers[selected.scoped[name]]
		}
		answers = unscoped
	} else if len(routes) > 1 && len(selected.questions) > 0 {
		response, err = call(ctx, config, state, selected.questions, settings)
		if err != nil {
			return nil, fmt.Errorf("decision: selected %q but failed to fill its fields: %w", selectedLabel, err)
		}
		usage.Requests++
		usage.InputTokens += response.Usage.InputTokens
		usage.OutputTokens += response.Usage.OutputTokens
		answers = response.Answers
	}
	arguments, details, err := decodeAnswers(
		answers, selected.fields, selected.questions, config.BooleanThreshold, selected.defaults,
	)
	if err != nil {
		return nil, err
	}
	if routeDetails != nil {
		details["route"] = routeDetails
	}
	if usage.Requests > 1 {
		details["requests"] = usage.Requests
	}
	var output any = arguments
	if selected.tool.Schema["type"] != "object" && len(arguments) == 1 {
		if scalar, exists := arguments["response"]; exists {
			output = scalar
		}
	}
	encoded, _ := json.Marshal(output)
	return &ai.ModelResponse{
		Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: selected.tool.Name, ToolCallID: config.ToolCallID,
			Args: encoded, ProviderName: config.ProviderName,
		}},
		Usage:     usage,
		ModelName: response.Model, ProviderName: config.ProviderName, ProviderURL: config.ProviderURL,
		ProviderDetails: details, FinishReason: ai.FinishReasonToolCall,
		State: ai.ModelResponseStateComplete, Timestamp: time.Now().UTC(),
	}, nil
}

func toolsLeft(messages []ai.ModelMessage, tools []ai.ToolDefinition) []ai.ToolDefinition {
	returned := map[string]bool{}
	for _, message := range messages {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			switch part := part.(type) {
			case ai.UserPromptPart:
				clear(returned)
			case ai.ToolReturnPart:
				returned[part.ToolName] = true
			}
		}
	}
	available := make([]ai.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if !returned[tool.Name] {
			available = append(available, tool)
		}
	}
	return available
}

func outputRouteLabel(tool ai.ToolDefinition) string {
	if title, _ := tool.Schema["title"].(string); title != "" {
		return title
	}
	if tool.Name != "" && tool.Name != "final_result" {
		return strings.TrimPrefix(tool.Name, "final_result_")
	}
	return "output"
}

func outputPurpose(tool ai.ToolDefinition, fields map[string]map[string]any, instructions string) string {
	if tool.Description != "" && tool.Description != "The final result of the run." {
		return tool.Description
	}
	if field := fields["response"]; field != nil {
		if description, _ := field["description"].(string); description != "" {
			return description
		}
	}
	return firstNonEmpty(instructions, tool.Description)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return "Structured output"
}
func freeQuestionName(questions map[string]Question, name string) string {
	for {
		if _, exists := questions[name]; !exists {
			return name
		}
		name += "_"
	}
}
