package google

import (
	"fmt"
	"maps"

	"github.com/Kludex/pydantic-ai-go/embeddings"
)

// Task identifies the retrieval or similarity task used by gemini-embedding-2.
type Task string

const (
	// TaskSearchResult retrieves documents relevant to a search query.
	TaskSearchResult Task = "search result"
	// TaskQuestionAnswering retrieves passages that answer a question.
	TaskQuestionAnswering Task = "question answering"
	// TaskFactChecking retrieves evidence for a claim.
	TaskFactChecking Task = "fact checking"
	// TaskCodeRetrieval retrieves code for natural-language queries.
	TaskCodeRetrieval Task = "code retrieval"
	// TaskClassification optimizes vectors for classification.
	TaskClassification Task = "classification"
	// TaskClustering optimizes vectors for clustering.
	TaskClustering Task = "clustering"
	// TaskSentenceSimilarity optimizes vectors for symmetric similarity.
	TaskSentenceSimilarity Task = "sentence similarity"
	// TaskRaw disables the gemini-embedding-2 task prefix.
	TaskRaw Task = "raw"
)

const (
	taskKey     = "google_embedding_task"
	taskTypeKey = "google_embedding_task_type"
	titleKey    = "google_embedding_title"
)

// Settings combines portable settings with Google embedding controls.
type Settings struct {
	// Common contains portable embedding settings.
	Common embeddings.Settings
	// Task selects gemini-embedding-2 task-prefix behavior.
	Task Task
	// TaskType sends a provider task type for older embedding models.
	TaskType string
	// Title identifies a retrieval document when the model supports it.
	Title string
}

// Build returns detached settings accepted by embeddings.Embedder and Model.Embed.
func (settings Settings) Build() (embeddings.Settings, error) {
	common := settings.Common.Clone()
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	set := func(name string, value any) error {
		if _, exists := extra[name]; exists {
			return fmt.Errorf("google embeddings: extra body field %q conflicts with typed settings", name)
		}
		extra[name] = value
		return nil
	}
	if settings.Task != "" {
		if err := validateTask(settings.Task); err != nil {
			return embeddings.Settings{}, err
		}
		if err := set(taskKey, settings.Task); err != nil {
			return embeddings.Settings{}, err
		}
	}
	if settings.TaskType != "" {
		if err := set(taskTypeKey, settings.TaskType); err != nil {
			return embeddings.Settings{}, err
		}
	}
	if settings.Title != "" {
		if err := set(titleKey, settings.Title); err != nil {
			return embeddings.Settings{}, err
		}
	}
	if len(extra) == 0 {
		extra = nil
	}
	common.ExtraBody = extra
	return common, nil
}

type localSettings struct {
	task     Task
	taskType string
	title    string
}

func extractSettings(settings embeddings.Settings) (embeddings.Settings, localSettings, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	local := localSettings{}
	for _, item := range []struct {
		name string
		set  func(string)
	}{
		{name: taskKey, set: func(value string) { local.task = Task(value) }},
		{name: taskTypeKey, set: func(value string) { local.taskType = value }},
		{name: titleKey, set: func(value string) { local.title = value }},
	} {
		value, exists := extra[item.name]
		if !exists {
			continue
		}
		delete(extra, item.name)
		text, ok := value.(string)
		if !ok {
			if task, taskOK := value.(Task); taskOK {
				text = string(task)
			} else {
				return embeddings.Settings{}, localSettings{}, fmt.Errorf(
					"google embeddings: setting %q must be a string", item.name,
				)
			}
		}
		item.set(text)
	}
	if err := validateTask(local.task); err != nil {
		return embeddings.Settings{}, localSettings{}, err
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, local, nil
}

func validateTask(task Task) error {
	switch task {
	case "", TaskSearchResult, TaskQuestionAnswering, TaskFactChecking, TaskCodeRetrieval,
		TaskClassification, TaskClustering, TaskSentenceSimilarity, TaskRaw:
		return nil
	default:
		return fmt.Errorf("google embeddings: invalid task %q", task)
	}
}
