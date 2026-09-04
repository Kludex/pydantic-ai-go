package ai

import (
	"fmt"
	"net/url"
	"slices"
	"strings"
)

func (state *messageSanitizer) userContents(contents []UserContent) ([]UserContent, error) {
	filtered := make([]UserContent, 0, len(contents))
	for _, content := range contents {
		sanitized, keep, err := state.userContent(content)
		if err != nil {
			return nil, err
		}
		if keep {
			filtered = append(filtered, sanitized)
		}
	}
	return filtered, nil
}

func (state *messageSanitizer) userContent(content UserContent) (UserContent, bool, error) {
	switch content := content.(type) {
	case TextContent:
		content.Metadata = cloneSchemaValue(content.Metadata)
		return content, true, nil
	case ImageURL:
		value, keep, err := state.fileURL(content.URL, content.ForceDownload)
		content.ForceDownload = value
		content.VendorMetadata = cloneSchemaMap(content.VendorMetadata)
		return content, keep, err
	case VideoURL:
		value, keep, err := state.fileURL(content.URL, content.ForceDownload)
		content.ForceDownload = value
		content.VendorMetadata = cloneSchemaMap(content.VendorMetadata)
		return content, keep, err
	case AudioURL:
		value, keep, err := state.fileURL(content.URL, content.ForceDownload)
		content.ForceDownload = value
		content.VendorMetadata = cloneSchemaMap(content.VendorMetadata)
		return content, keep, err
	case DocumentURL:
		value, keep, err := state.fileURL(content.URL, content.ForceDownload)
		content.ForceDownload = value
		content.VendorMetadata = cloneSchemaMap(content.VendorMetadata)
		return content, keep, err
	case UploadedFile:
		content.VendorMetadata = cloneSchemaMap(content.VendorMetadata)
		if state.allowUploadedFiles {
			return content, true, nil
		}
		state.droppedProviders[content.ProviderName] = struct{}{}
		return content, false, nil
	case BinaryContent:
		content.Data = slices.Clone(content.Data)
		content.VendorMetadata = cloneSchemaMap(content.VendorMetadata)
		return content, true, nil
	default:
		return content, true, nil
	}
}

func (state *messageSanitizer) fileURL(rawURL string, mode FileDownloadMode) (FileDownloadMode, bool, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return FileDownloadNever, false, fmt.Errorf("ai: sanitize file URL %q: %w", rawURL, err)
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "" {
		if _, allowed := state.allowedSchemes[scheme]; !allowed {
			state.droppedSchemes[scheme] = struct{}{}
			return FileDownloadNever, false, nil
		}
	}
	if mode != FileDownloadNever {
		if _, allowed := state.allowedModes[mode]; !allowed {
			state.resetModes[mode] = struct{}{}
			mode = FileDownloadNever
		}
	}
	return mode, true, nil
}

func (state *messageSanitizer) toolReturnContent(content any) (any, bool, error) {
	switch content := content.(type) {
	case ImageURL:
		return state.fileContentValue(content)
	case VideoURL:
		return state.fileContentValue(content)
	case AudioURL:
		return state.fileContentValue(content)
	case DocumentURL:
		return state.fileContentValue(content)
	case UploadedFile:
		value, keep, err := state.userContent(content)
		return value, keep, err
	case BinaryContent:
		value, _, _ := state.userContent(content)
		return value, true, nil
	case map[string]any:
		result := make(map[string]any, len(content))
		for key, item := range content {
			sanitized, keep, err := state.toolReturnContent(item)
			if err != nil {
				return nil, false, err
			}
			if keep {
				result[key] = sanitized
			}
		}
		return result, true, nil
	case []any:
		result := make([]any, 0, len(content))
		for _, item := range content {
			sanitized, keep, err := state.toolReturnContent(item)
			if err != nil {
				return nil, false, err
			}
			if keep {
				result = append(result, sanitized)
			}
		}
		return result, true, nil
	default:
		return cloneSchemaValue(content), true, nil
	}
}

func (state *messageSanitizer) fileContentValue(content UserContent) (any, bool, error) {
	value, keep, err := state.userContent(content)
	return value, keep, err
}
