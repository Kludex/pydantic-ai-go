package ai

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"net/url"
	"path"
	"strings"
)

var mediaTypesByExtension = map[string]string{
	".3gp":  "video/3gpp",
	".aac":  "audio/aac",
	".avi":  "video/x-msvideo",
	".bmp":  "image/bmp",
	".csv":  "text/csv",
	".doc":  "application/msword",
	".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	".flac": "audio/flac",
	".flv":  "video/x-flv",
	".gif":  "image/gif",
	".htm":  "text/html",
	".html": "text/html",
	".jpeg": "image/jpeg",
	".jpg":  "image/jpeg",
	".json": "application/json",
	".m4a":  "audio/mp4a-latm",
	".md":   "text/markdown",
	".mkv":  "video/x-matroska",
	".mov":  "video/quicktime",
	".mp3":  "audio/mpeg",
	".mp4":  "video/mp4",
	".mpeg": "video/mpeg",
	".mpg":  "video/mpeg",
	".oga":  "audio/ogg",
	".ogg":  "audio/ogg",
	".opus": "audio/ogg",
	".pdf":  "application/pdf",
	".png":  "image/png",
	".ppt":  "application/vnd.ms-powerpoint",
	".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
	".rtf":  "application/rtf",
	".svg":  "image/svg+xml",
	".tif":  "image/tiff",
	".tiff": "image/tiff",
	".txt":  "text/plain",
	".wav":  "audio/wav",
	".webm": "video/webm",
	".webp": "image/webp",
	".wmv":  "video/x-ms-wmv",
	".xls":  "application/vnd.ms-excel",
	".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
	".xml":  "application/xml",
}

func resolveURLMediaType(rawURL, explicit, kind string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", fmt.Errorf("ai: parse %s URL: %w", kind, err)
	}
	mediaType := mediaTypesByExtension[strings.ToLower(path.Ext(parsed.Path))]
	if mediaType == "" {
		return "", fmt.Errorf("ai: cannot infer media type from %s URL %q", kind, rawURL)
	}
	return mediaType, nil
}

func resolveContentIdentifier(explicit string, content []byte) string {
	if explicit != "" {
		return explicit
	}
	digest := sha1.Sum(content)
	return hex.EncodeToString(digest[:])[:6]
}
