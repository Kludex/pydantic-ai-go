package google

import (
	"fmt"
	"slices"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

func mergeSettings(base, override embeddings.Settings) embeddings.Settings {
	merged := embeddings.MergeSettings(base, override)
	if override.ExtraBody == nil || base.ExtraBody == nil {
		return merged
	}
	for _, key := range []string{taskKey, taskTypeKey, titleKey} {
		if _, overridden := override.ExtraBody[key]; overridden {
			continue
		}
		if value, exists := base.ExtraBody[key]; exists {
			merged.ExtraBody[key] = value
		}
	}
	return merged
}

func (model *Model) prepareInputs(
	inputs []string, inputType embeddings.InputType, settings localSettings,
) ([]string, string) {
	if model.name == "gemini-embedding-2" {
		task := settings.task
		if task == "" {
			task = TaskSearchResult
		}
		if task == TaskRaw {
			return slices.Clone(inputs), ""
		}
		texts := make([]string, len(inputs))
		if inputType == embeddings.InputTypeDocument && !isSymmetricTask(task) {
			title := settings.title
			if title == "" {
				title = "none"
			}
			for index, input := range inputs {
				texts[index] = fmt.Sprintf("title: %s | text: %s", title, input)
			}
			return texts, ""
		}
		for index, input := range inputs {
			texts[index] = fmt.Sprintf("task: %s | query: %s", task, input)
		}
		return texts, ""
	}
	taskType := settings.taskType
	if taskType == "" {
		taskType = "RETRIEVAL_QUERY"
		if inputType == embeddings.InputTypeDocument {
			taskType = "RETRIEVAL_DOCUMENT"
		}
	}
	return slices.Clone(inputs), taskType
}

func (model *Model) settingWarnings(settings localSettings) []string {
	if model.name == "gemini-embedding-2" && settings.taskType != "" {
		return []string{"google embeddings: TaskType is not supported by gemini-embedding-2 and was ignored"}
	}
	if model.name != "gemini-embedding-2" && settings.task != "" {
		return []string{"google embeddings: Task is only supported by gemini-embedding-2 and was ignored"}
	}
	return nil
}

func isSymmetricTask(task Task) bool {
	return task == TaskClassification || task == TaskClustering || task == TaskSentenceSimilarity
}
