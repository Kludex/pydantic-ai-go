package ai

import (
	"fmt"
	"net/url"
	"strings"
)

// FileDownloadMode controls whether a URL is downloaded by this process.
type FileDownloadMode string

const (
	// FileDownloadNever allows direct provider fetching and safely downloads when required.
	FileDownloadNever FileDownloadMode = ""
	// FileDownloadSafe requests a download while blocking private networks and cloud metadata.
	FileDownloadSafe FileDownloadMode = "safe"
	// FileDownloadAllowLocal permits private networks but still blocks cloud metadata.
	FileDownloadAllowLocal FileDownloadMode = "allow-local"
)

// Validate checks whether the download mode is supported.
func (mode FileDownloadMode) Validate() error {
	switch mode {
	case FileDownloadNever, FileDownloadSafe, FileDownloadAllowLocal:
		return nil
	default:
		return fmt.Errorf("ai: invalid file download mode %q", mode)
	}
}

// ImageURL references an image by URL.
type ImageURL struct {
	URL            string
	MediaType      string
	Identifier     string
	ForceDownload  FileDownloadMode
	VendorMetadata map[string]any
}

// ResolvedMediaType returns the explicit media type or infers it from the URL.
func (image ImageURL) ResolvedMediaType() (string, error) {
	return resolveURLMediaType(image.URL, image.MediaType, "image")
}

// ResolvedIdentifier returns the caller-provided identifier or a stable URL digest.
func (image ImageURL) ResolvedIdentifier() string {
	return resolveContentIdentifier(image.Identifier, []byte(image.URL))
}

func (ImageURL) userContentKind() string { return "image-url" }
func (ImageURL) enqueueItemKind() string { return "user-content" }

// VideoURL references a video by URL.
type VideoURL struct {
	URL            string
	MediaType      string
	Identifier     string
	ForceDownload  FileDownloadMode
	VendorMetadata map[string]any
}

// ResolvedMediaType returns the explicit media type or infers it from the URL.
func (video VideoURL) ResolvedMediaType() (string, error) {
	if video.MediaType == "" && video.IsYouTube() {
		return "video/mp4", nil
	}
	return resolveURLMediaType(video.URL, video.MediaType, "video")
}

// ResolvedIdentifier returns the caller-provided identifier or a stable URL digest.
func (video VideoURL) ResolvedIdentifier() string {
	return resolveContentIdentifier(video.Identifier, []byte(video.URL))
}

// IsYouTube reports whether Google models can consume the URL directly as a YouTube video.
func (video VideoURL) IsYouTube() bool {
	parsed, err := url.Parse(video.URL)
	if err != nil {
		return false
	}
	switch strings.ToLower(parsed.Hostname()) {
	case "youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com":
		return true
	default:
		return false
	}
}

func (VideoURL) userContentKind() string { return "video-url" }
func (VideoURL) enqueueItemKind() string { return "user-content" }

// AudioURL references an audio file by URL.
type AudioURL struct {
	URL            string
	MediaType      string
	Identifier     string
	ForceDownload  FileDownloadMode
	VendorMetadata map[string]any
}

// ResolvedMediaType returns the explicit media type or infers it from the URL.
func (audio AudioURL) ResolvedMediaType() (string, error) {
	return resolveURLMediaType(audio.URL, audio.MediaType, "audio")
}

// ResolvedIdentifier returns the caller-provided identifier or a stable URL digest.
func (audio AudioURL) ResolvedIdentifier() string {
	return resolveContentIdentifier(audio.Identifier, []byte(audio.URL))
}

func (AudioURL) userContentKind() string { return "audio-url" }
func (AudioURL) enqueueItemKind() string { return "user-content" }

// DocumentURL references a document by URL.
type DocumentURL struct {
	URL            string
	MediaType      string
	Identifier     string
	ForceDownload  FileDownloadMode
	VendorMetadata map[string]any
}

// ResolvedMediaType returns the explicit media type or infers it from the URL.
func (document DocumentURL) ResolvedMediaType() (string, error) {
	return resolveURLMediaType(document.URL, document.MediaType, "document")
}

// ResolvedIdentifier returns the caller-provided identifier or a stable URL digest.
func (document DocumentURL) ResolvedIdentifier() string {
	return resolveContentIdentifier(document.Identifier, []byte(document.URL))
}

func (DocumentURL) userContentKind() string { return "document-url" }
func (DocumentURL) enqueueItemKind() string { return "user-content" }

// BinaryContent carries inline file data.
type BinaryContent struct {
	Data           []byte
	MediaType      string
	Identifier     string
	VendorMetadata map[string]any
}

// ResolvedIdentifier returns the caller-provided identifier or a stable content digest.
func (content BinaryContent) ResolvedIdentifier() string {
	return resolveContentIdentifier(content.Identifier, content.Data)
}

func (BinaryContent) userContentKind() string { return "binary" }
func (BinaryContent) enqueueItemKind() string { return "user-content" }
