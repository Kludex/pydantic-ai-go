package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"

	"github.com/JohannesKaufmann/html-to-markdown/v2/converter"
	"github.com/JohannesKaufmann/html-to-markdown/v2/plugin/base"
	"github.com/JohannesKaufmann/html-to-markdown/v2/plugin/commonmark"
	"github.com/Kludex/pydantic-ai-go/internal/download"
	"golang.org/x/net/html"
	"golang.org/x/net/html/charset"
)

const (
	defaultWebFetchContentLength = 50_000
	defaultWebFetchDownloadBytes = 50 << 20
)

var excessiveWebFetchNewlines = regexp.MustCompile(`\n{3,}`)

// LocalWebFetchConfig controls the SSRF-protected local web-fetch tool.
type LocalWebFetchConfig struct {
	// MaxContentLength limits returned text by Unicode code points. Zero defaults to 50,000.
	MaxContentLength int
	// DisableContentLimit returns complete text after the bounded download.
	DisableContentLimit bool
	// AllowLocalURLs permits loopback and private network targets.
	AllowLocalURLs bool
	// Timeout bounds the complete fetch. Zero uses the downloader default.
	Timeout time.Duration
	// MaxDownloadBytes bounds compressed and decompressed response data. Zero defaults to 50 MiB.
	MaxDownloadBytes int64
	// AllowedDomains restricts destinations when non-empty.
	AllowedDomains []string
	// BlockedDomains rejects matching destinations in addition to SSRF defaults.
	BlockedDomains []string
	// Headers adds detached request headers.
	Headers map[string]string
}

// WebFetchArgs is the model-generated input for a local web fetch.
type WebFetchArgs struct {
	// URL is the HTTP or HTTPS destination to fetch.
	URL string `json:"url" jsonschema_description:"The HTTP or HTTPS URL to fetch."`
}

// WebFetchResult is textual URL content returned to the model.
type WebFetchResult struct {
	// URL is the final destination after redirects.
	URL string `json:"url"`
	// Title is the HTML document title when available.
	Title string `json:"title"`
	// Content is normalized Markdown, JSON, or plain text.
	Content string `json:"content"`
}

// NewLocalWebFetchTool creates an SSRF-protected tool that returns Markdown, text, or binary content.
func NewLocalWebFetchTool[Deps any](config LocalWebFetchConfig) Tool[Deps] {
	if config.MaxContentLength < 0 {
		panic("ai: local web-fetch maximum content length must not be negative")
	}
	if config.Timeout < 0 {
		panic("ai: local web-fetch timeout must not be negative")
	}
	if config.MaxDownloadBytes < 0 {
		panic("ai: local web-fetch maximum download bytes must not be negative")
	}
	contentLimit := config.MaxContentLength
	if contentLimit == 0 {
		contentLimit = defaultWebFetchContentLength
	}
	maximumBytes := config.MaxDownloadBytes
	if maximumBytes == 0 {
		maximumBytes = defaultWebFetchDownloadBytes
	}
	allowedDomains := append([]string(nil), config.AllowedDomains...)
	blockedDomains := append([]string(nil), config.BlockedDomains...)
	headers := make(map[string]string, len(config.Headers)+1)
	headers["Accept"] = "text/markdown, text/html;q=0.9, */*;q=0.8"
	for name, value := range config.Headers {
		headers[name] = value
	}
	return NewTool[Deps](
		"web_fetch",
		func(ctx context.Context, _ *RunContext[Deps], args WebFetchArgs) (any, error) {
			result, err := download.FetchWithOptions(ctx, args.URL, download.Options{
				AllowLocal: config.AllowLocalURLs, Timeout: config.Timeout, MaxBytes: maximumBytes,
				Headers: headers, PreserveOctetStream: true,
				AllowedDomains: allowedDomains, BlockedDomains: blockedDomains,
			})
			if err != nil {
				return nil, Retryf("Failed to fetch %s: %v", args.URL, err)
			}
			mediaType := strings.ToLower(strings.TrimSpace(result.MediaType))
			if mediaType != "" && !webFetchTextMediaType(mediaType) {
				return ToolReturn{
					ReturnValue: fmt.Sprintf("Fetched binary content from %s", result.URL),
					Content: []UserContent{BinaryContent{
						Data: result.Data, MediaType: mediaType,
					}},
				}, nil
			}
			text := string(result.Data)
			decoded, decodingErr := charset.NewReader(bytes.NewReader(result.Data), result.ContentType)
			if decodingErr == nil {
				if decodedText, readErr := io.ReadAll(decoded); readErr == nil {
					text = string(decodedText)
				}
			}
			title, content := "", text
			switch {
			case mediaType == "text/markdown" || mediaType == "text/x-markdown":
			case mediaType == "application/json" || strings.HasSuffix(mediaType, "+json"):
				var value any
				if json.Unmarshal(result.Data, &value) == nil {
					formatted, _ := json.MarshalIndent(value, "", "  ")
					content = "```json\n" + string(formatted) + "\n```"
				}
			case mediaType == "" || mediaType == "text/html" || mediaType == "application/xhtml+xml":
				title = webFetchHTMLTitle(text)
				markdownConverter := converter.NewConverter(converter.WithPlugins(
					base.NewBasePlugin(), commonmark.NewCommonmarkPlugin(),
				))
				markdownConverter.Register.TagType("img", converter.TagTypeRemove, converter.PriorityEarly)
				converted, conversionErr := markdownConverter.ConvertString(text, converter.WithDomain(result.URL))
				if conversionErr == nil {
					content = converted
				}
			}
			content = strings.TrimSpace(excessiveWebFetchNewlines.ReplaceAllString(content, "\n\n"))
			if !config.DisableContentLimit {
				runes := []rune(content)
				if len(runes) > contentLimit {
					content = string(runes[:contentLimit]) + "\n\n[Content truncated]"
				}
			}
			return WebFetchResult{URL: result.URL, Title: title, Content: content}, nil
		},
		WithDescription("Fetch the content of a web page and return Markdown, text, JSON, or binary content."),
	)
}

func webFetchTextMediaType(mediaType string) bool {
	return strings.HasPrefix(mediaType, "text/") || mediaType == "application/json" ||
		mediaType == "application/xml" || mediaType == "application/xhtml+xml" ||
		mediaType == "application/javascript" || strings.HasSuffix(mediaType, "+json") ||
		strings.HasSuffix(mediaType, "+xml")
}

func webFetchHTMLTitle(document string) string {
	root, _ := html.Parse(strings.NewReader(document))
	for node := range root.Descendants() {
		if node.Type == html.ElementNode && strings.EqualFold(node.Data, "title") && node.FirstChild != nil {
			return strings.TrimSpace(node.FirstChild.Data)
		}
	}
	return ""
}
