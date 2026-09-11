package images

import "strings"

// MediaTypeFromBytes returns the recognized image media type for data.
func MediaTypeFromBytes(data []byte) string {
	switch {
	case len(data) >= 4 && string(data[:4]) == "\x89PNG":
		return "image/png"
	case len(data) >= 3 && string(data[:3]) == "\xff\xd8\xff":
		return "image/jpeg"
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		return "image/webp"
	default:
		return ""
	}
}

// OutputFormat returns the subtype of an image media type.
func OutputFormat(mediaType string) string {
	if !strings.HasPrefix(mediaType, "image/") {
		return ""
	}
	return strings.TrimPrefix(mediaType, "image/")
}
