package openai

import (
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestResponsesSettingsBuildAndExtract(t *testing.T) {
	enabled := true
	settings, err := (Settings{
		Common:                ai.ModelSettings{ExtraBody: map[string]any{"custom": true}},
		IncludeRawAnnotations: &enabled,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	enabled = false
	cleaned, extracted, err := extractResponsesSettings(settings)
	if err != nil {
		t.Fatal(err)
	}
	if !extracted.IncludeRawAnnotations || cleaned.ExtraBody["custom"] != true {
		t.Fatalf("unexpected Responses settings: cleaned=%#v extracted=%#v", cleaned, extracted)
	}

	disabled := false
	settings, err = (Settings{IncludeRawAnnotations: &disabled}).Build()
	if err != nil {
		t.Fatal(err)
	}
	cleaned, extracted, err = extractResponsesSettings(settings)
	if err != nil || extracted.IncludeRawAnnotations || cleaned.ExtraBody != nil {
		t.Fatalf("unexpected disabled settings: cleaned=%#v extracted=%#v err=%v", cleaned, extracted, err)
	}
}

func TestResponsesSettingsValidation(t *testing.T) {
	_, _, err := extractResponsesSettings(ai.ModelSettings{ExtraBody: map[string]any{
		includeRawAnnotationsSetting: "yes",
	}})
	if err == nil || !strings.Contains(err.Error(), "must be a boolean") {
		t.Fatalf("unexpected extraction error: %v", err)
	}

	malformed := ai.ModelSettings{ExtraBody: map[string]any{includeRawAnnotationsSetting: "yes"}}
	history := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true},
	}}
	chat := NewModel("model")
	_, chatErr := chat.Request(t.Context(), nil, ai.ModelRequestParams{Settings: malformed})
	if chatErr == nil || !strings.Contains(chatErr.Error(), "must be a boolean") {
		t.Fatalf("unexpected Chat request error: %v", chatErr)
	}

	model := NewResponsesModel("model")
	_, requestErr := model.Request(t.Context(), history, ai.ModelRequestParams{Settings: malformed})
	if requestErr == nil || !strings.Contains(requestErr.Error(), "must be a boolean") {
		t.Fatalf("unexpected suspended request error: %v", requestErr)
	}
	_, streamErr := model.StreamRequest(t.Context(), history, ai.ModelRequestParams{Settings: malformed})
	if streamErr == nil || !strings.Contains(streamErr.Error(), "must be a boolean") {
		t.Fatalf("unexpected suspended stream error: %v", streamErr)
	}

	enabled := true
	chatSettings, buildErr := (Settings{IncludeRawAnnotations: &enabled}).Build()
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	_, chatErr = chat.Request(t.Context(), nil, ai.ModelRequestParams{Settings: chatSettings})
	if chatErr == nil || !strings.Contains(chatErr.Error(), "only supported by Responses") {
		t.Fatalf("unexpected Chat setting error: %v", chatErr)
	}

	_, err = (Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{
		includeRawAnnotationsSetting: true,
	}}}).Build()
	if err == nil || !strings.Contains(err.Error(), "is reserved") {
		t.Fatalf("unexpected reserved setting error: %v", err)
	}
}
