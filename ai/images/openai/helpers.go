package openai

import (
	"encoding/json"
	"fmt"
	"net/textproto"
	"reflect"
	"strconv"
	"strings"
)

func moderationBlocked(body []byte) bool {
	var response struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
		Code string `json:"code"`
	}
	return json.Unmarshal(body, &response) == nil &&
		(response.Code == "moderation_blocked" || response.Error.Code == "moderation_blocked")
}

func conflictWarnings(conflicts []string) []string {
	warnings := make([]string, len(conflicts))
	for index, conflict := range conflicts {
		warnings[index] = "openai images: used provider-specific size instead of portable " + conflict
	}
	return warnings
}

func formValue(value any) (string, error) {
	switch value := value.(type) {
	case string:
		return value, nil
	case fmt.Stringer:
		return value.String(), nil
	case int:
		return strconv.Itoa(value), nil
	case bool:
		return strconv.FormatBool(value), nil
	}
	reflected := reflect.ValueOf(value)
	if reflected.IsValid() && reflected.Kind() == reflect.String {
		return reflected.String(), nil
	}
	encoded, err := json.Marshal(value)
	return string(encoded), err
}

func mediaExtension(mediaType string) string {
	switch strings.ToLower(mediaType) {
	case "image/png":
		return "png"
	case "image/jpeg":
		return "jpg"
	case "image/webp":
		return "webp"
	default:
		return ""
	}
}

func fileHeader(name, filename, mediaType string) textproto.MIMEHeader {
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name=%q; filename=%q`, name, filename))
	header.Set("Content-Type", mediaType)
	return header
}
