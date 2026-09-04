package a2a

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"

	protocol "github.com/a2aproject/a2a-go/a2a"
)

func cloneSendConfig(config *protocol.MessageSendConfig) *protocol.MessageSendConfig {
	if config == nil {
		return nil
	}
	cloned := *config
	cloned.AcceptedOutputModes = slices.Clone(config.AcceptedOutputModes)
	if config.Blocking != nil {
		value := *config.Blocking
		cloned.Blocking = &value
	}
	if config.HistoryLength != nil {
		value := *config.HistoryLength
		cloned.HistoryLength = &value
	}
	if config.PushConfig != nil {
		push := *config.PushConfig
		if config.PushConfig.Auth != nil {
			auth := *config.PushConfig.Auth
			auth.Schemes = slices.Clone(auth.Schemes)
			push.Auth = &auth
		}
		cloned.PushConfig = &push
	}
	return &cloned
}

func cloneModelMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return nil
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		panic(fmt.Sprintf("ai/a2a: model metadata must be JSON-compatible: %v", err))
	}
	var cloned map[string]any
	_ = json.Unmarshal(encoded, &cloned)
	return cloned
}

func nilModelClient(client ModelClient) bool {
	if client == nil {
		return true
	}
	value := reflect.ValueOf(client)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
