package webchat

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"unicode"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type frontendConfiguration struct {
	Models       []modelInfo       `json:"models"`
	BuiltinTools []builtinToolInfo `json:"builtinTools"`
	models       map[string]ai.Model
	tools        map[string]ai.NativeTool
}

type modelInfo struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	BuiltinTools []string `json:"builtinTools"`
}

type builtinToolInfo struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

func newFrontendConfiguration(
	defaultModel ai.Model, defaultID, defaultName string, models []ModelOption, nativeTools []ai.NativeTool,
) (*frontendConfiguration, error) {
	if modelIsNil(defaultModel) && (strings.TrimSpace(defaultID) != "" || strings.TrimSpace(defaultName) != "") {
		return nil, fmt.Errorf("webchat: default model labels require an agent model")
	}
	options := make([]ModelOption, 0, len(models)+1)
	if !modelIsNil(defaultModel) {
		options = append(options, ModelOption{ID: defaultID, Name: defaultName, Model: defaultModel})
	}
	options = append(options, models...)
	configuration := &frontendConfiguration{
		Models: make([]modelInfo, 0, len(options)), BuiltinTools: make([]builtinToolInfo, 0, len(nativeTools)),
		models: make(map[string]ai.Model, len(options)), tools: make(map[string]ai.NativeTool, len(nativeTools)),
	}
	for _, tool := range nativeTools {
		if err := ai.ValidateNativeTools([]ai.NativeTool{tool}); err != nil {
			return nil, fmt.Errorf("webchat: invalid native tool: %w", err)
		}
		id := tool.UniqueID()
		if _, exists := configuration.tools[id]; exists {
			return nil, fmt.Errorf("webchat: duplicate native tool ID %q", id)
		}
		configuration.tools[id] = tool.CloneNativeTool()
		configuration.BuiltinTools = append(configuration.BuiltinTools, builtinToolInfo{ID: id, Name: toolLabel(id)})
	}
	for _, option := range options {
		if modelIsNil(option.Model) {
			return nil, fmt.Errorf("webchat: model option must not be nil")
		}
		id := strings.TrimSpace(option.ID)
		if id == "" {
			id = option.Model.Name()
		}
		if id == "" {
			return nil, fmt.Errorf("webchat: model option ID must not be empty")
		}
		if _, exists := configuration.models[id]; exists {
			return nil, fmt.Errorf("webchat: duplicate model ID %q", id)
		}
		name := strings.TrimSpace(option.Name)
		if name == "" {
			name = id
		}
		info := modelInfo{ID: id, Name: name, BuiltinTools: []string{}}
		for _, tool := range configuration.BuiltinTools {
			if modelSupportsNativeTool(option.Model, configuration.tools[tool.ID]) {
				info.BuiltinTools = append(info.BuiltinTools, tool.ID)
			}
		}
		configuration.models[id] = option.Model
		configuration.Models = append(configuration.Models, info)
	}
	return configuration, nil
}

func (configuration *frontendConfiguration) runOptions(modelID string, toolIDs []string) ([]ai.RunOption, error) {
	var options []ai.RunOption
	if modelID != "" {
		model, exists := configuration.models[modelID]
		if !exists {
			return nil, fmt.Errorf("model %q is not in the allowed models list", modelID)
		}
		options = append(options, ai.WithRunModel(model))
	}
	seen := make(map[string]struct{}, len(toolIDs))
	for _, id := range toolIDs {
		if _, exists := seen[id]; exists {
			continue
		}
		seen[id] = struct{}{}
		tool, exists := configuration.tools[id]
		if !exists {
			return nil, fmt.Errorf("native tool %q is not in the allowed tools list", id)
		}
		options = append(options, ai.WithRunNativeTools(tool.CloneNativeTool()))
	}
	return options, nil
}

func modelSupportsNativeTool(model ai.Model, tool ai.NativeTool) bool {
	support, known := model.(ai.NativeToolSupportModel)
	if !known {
		support, known = ai.UnwrapModel(model).(ai.NativeToolSupportModel)
	}
	return !known || support.SupportsNativeTool(tool.CloneNativeTool())
}

func modelIsNil(model ai.Model) bool {
	if model == nil {
		return true
	}
	value := reflect.ValueOf(model)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func toolLabel(id string) string {
	words := strings.Fields(strings.ReplaceAll(id, "_", " "))
	for index, word := range words {
		runes := []rune(word)
		runes[0] = unicode.ToUpper(runes[0])
		words[index] = string(runes)
	}
	return strings.Join(words, " ")
}

func serveConfiguration(response http.ResponseWriter, request *http.Request, configuration *frontendConfiguration) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeJSONError(response, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(response, http.StatusOK, configuration)
}

func serveHealth(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeJSONError(response, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(response, http.StatusOK, map[string]bool{"ok": true})
}

func writeJSONError(response http.ResponseWriter, status int, message string) {
	writeJSON(response, status, map[string]string{"error": message})
}

func writeJSON(response http.ResponseWriter, status int, value any) {
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(status)
	_ = json.NewEncoder(response).Encode(value)
}
