package google

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

type embedRequest struct {
	Requests   []geminiEmbedRequest `json:"requests,omitempty"`
	Instances  []vertexInstance     `json:"instances,omitempty"`
	Parameters *vertexParameters    `json:"parameters,omitempty"`
}

type geminiEmbedRequest struct {
	Model                string        `json:"model"`
	Content              geminiContent `json:"content"`
	TaskType             string        `json:"taskType,omitempty"`
	Title                string        `json:"title,omitempty"`
	OutputDimensionality *int          `json:"outputDimensionality,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type vertexInstance struct {
	Content  string `json:"content"`
	TaskType string `json:"task_type,omitempty"`
	Title    string `json:"title,omitempty"`
}

type vertexParameters struct {
	OutputDimensionality int `json:"outputDimensionality"`
}

// Embed creates one embedding for each input.
func (model *Model) Embed(
	ctx context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings,
) (*embeddings.Result, error) {
	settings = mergeSettings(model.settings, settings)
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	settings, local, err := extractSettings(settings)
	if err != nil {
		return nil, err
	}
	texts, taskType := model.prepareInputs(inputs, inputType, local)
	warnings := model.settingWarnings(local)
	title := local.title
	if model.name == "gemini-embedding-2" {
		title = ""
	}
	payload := embedRequest{}
	var endpoint string
	if model.transport == modelgoogle.TransportVertexAI {
		endpoint = fmt.Sprintf("%s/models/%s:predict", model.baseURL, model.name)
		payload.Instances = make([]vertexInstance, len(texts))
		for index, text := range texts {
			payload.Instances[index] = vertexInstance{Content: text, TaskType: taskType, Title: title}
		}
		if settings.Dimensions != nil {
			payload.Parameters = &vertexParameters{OutputDimensionality: *settings.Dimensions}
		}
	} else {
		endpoint = fmt.Sprintf("%s/models/%s:batchEmbedContents", model.baseURL, model.name)
		payload.Requests = make([]geminiEmbedRequest, len(texts))
		wireModel := "models/" + strings.TrimPrefix(model.name, "models/")
		for index, text := range texts {
			payload.Requests[index] = geminiEmbedRequest{
				Model: wireModel, Content: geminiContent{Parts: []geminiPart{{Text: text}}},
				TaskType: taskType, Title: title, OutputDimensionality: clonePointer(settings.Dimensions),
			}
		}
	}
	body, err := requestBody(payload, settings.ExtraBody)
	if err != nil {
		return nil, err
	}
	responseBody, err := model.doRequest(ctx, endpoint, body, settings.ExtraHeaders)
	if err != nil {
		return nil, err
	}
	result, err := model.parseResponse(responseBody, inputs, inputType)
	if err != nil {
		return nil, err
	}
	result.Warnings = warnings
	return result, nil
}

func requestBody(payload embedRequest, extraBody map[string]any) ([]byte, error) {
	body := maps.Clone(extraBody)
	if body == nil {
		body = map[string]any{}
	}
	for _, key := range []string{"requests", "instances", "parameters"} {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("google embeddings: extra body field %q conflicts with typed settings", key)
		}
	}
	if payload.Requests != nil {
		body["requests"] = payload.Requests
	}
	if payload.Instances != nil {
		body["instances"] = payload.Instances
	}
	if payload.Parameters != nil {
		body["parameters"] = payload.Parameters
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("google embeddings: encode request: %w", err)
	}
	return encoded, nil
}
