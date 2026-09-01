package google

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	modelgoogle "github.com/Kludex/pydantic-ai-go/models/google"
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
	settings = embeddings.MergeSettings(model.settings, settings)
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
	body, _ := json.Marshal(payload)
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
