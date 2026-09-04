package ai_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestMetadataClonesTypedAndCyclicCollections(t *testing.T) {
	typedMap := map[string]string{"key": "value"}
	typedSlice := []map[string]any{{"key": "value"}}
	cycleMap := map[string]any{}
	cycleMap["self"] = cycleMap
	cycleSlice := make([]any, 1)
	cycleSlice[0] = cycleSlice
	var nilMap map[string]string
	var nilSlice []string
	array := [1]map[string]any{{"key": "value"}}
	metadata := map[string]any{
		"nil": nil, "nil_map": nilMap, "nil_slice": nilSlice,
		"map": typedMap, "slice": typedSlice, "cycle_map": cycleMap,
		"cycle_slice": cycleSlice, "array": array,
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithMetadata(metadata))
	result, err := agent.Run(t.Context(), "hello", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	cloned := result.Metadata()
	clonedMap := cloned["map"].(map[string]string)
	clonedSlice := cloned["slice"].([]map[string]any)
	clonedArray := cloned["array"].([1]map[string]any)
	clonedMap["key"] = "changed"
	clonedSlice[0]["key"] = "changed"
	clonedArray[0]["key"] = "changed"
	if typedMap["key"] != "value" || typedSlice[0]["key"] != "value" || array[0]["key"] != "value" {
		t.Fatal("typed metadata collections were not detached")
	}
	clonedCycleMap := cloned["cycle_map"].(map[string]any)
	clonedCycleMap["self"].(map[string]any)["marker"] = true
	if clonedCycleMap["marker"] != true {
		t.Fatal("cyclic metadata map was not preserved")
	}
	clonedCycleSlice := cloned["cycle_slice"].([]any)
	clonedCycleSlice[0].([]any)[0] = "changed"
	if clonedCycleSlice[0] != "changed" {
		t.Fatal("cyclic metadata slice was not preserved")
	}
	if cloned["nil"] != nil || cloned["nil_map"].(map[string]string) != nil ||
		cloned["nil_slice"].([]string) != nil {
		t.Fatal("nil metadata collections changed")
	}
}
