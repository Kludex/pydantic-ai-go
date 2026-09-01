package ai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

const (
	defaultDuckDuckGoEndpoint  = "https://html.duckduckgo.com/html/"
	maxDuckDuckGoResponseBytes = 5 << 20
)

// LocalWebSearchConfig controls the DuckDuckGo local web-search tool.
type LocalWebSearchConfig struct {
	HTTPClient *http.Client
	Endpoint   string
	Timeout    time.Duration
	MaxResults int
}

// WebSearchArgs is the model-generated input for local web search.
type WebSearchArgs struct {
	Query string `json:"query" jsonschema_description:"The web search query."`
}

// WebSearchResult is one DuckDuckGo result returned to the model.
type WebSearchResult struct {
	Title string `json:"title"`
	URL   string `json:"href"`
	Body  string `json:"body"`
}

// NewLocalWebSearchTool creates a DuckDuckGo HTML-search fallback.
func NewLocalWebSearchTool[Deps any](config LocalWebSearchConfig) Tool[Deps] {
	if config.Timeout < 0 {
		panic("ai: local web-search timeout must not be negative")
	}
	if config.MaxResults < 0 {
		panic("ai: local web-search maximum results must not be negative")
	}
	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = defaultDuckDuckGoEndpoint
	}
	parsedEndpoint, err := url.Parse(endpoint)
	if err != nil || parsedEndpoint.Hostname() == "" ||
		parsedEndpoint.Scheme != "http" && parsedEndpoint.Scheme != "https" {
		panic(fmt.Sprintf("ai: local web-search endpoint %q must be an HTTP or HTTPS URL", endpoint))
	}
	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	return NewTool[Deps](
		"duckduckgo_search",
		func(ctx context.Context, _ *RunContext[Deps], args WebSearchArgs) ([]WebSearchResult, error) {
			if strings.TrimSpace(args.Query) == "" {
				return nil, Retryf("Search query must not be empty")
			}
			requestContext, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			form := url.Values{"q": {args.Query}}
			request, _ := http.NewRequestWithContext(
				requestContext, http.MethodPost, parsedEndpoint.String(), strings.NewReader(form.Encode()),
			)
			request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			request.Header.Set("User-Agent", "pydantic-ai-go web search")
			response, err := client.Do(request)
			if err != nil {
				return nil, Retryf("DuckDuckGo search failed: %v", err)
			}
			defer func() { _ = response.Body.Close() }()
			if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
				return nil, Retryf("DuckDuckGo search returned %s", response.Status)
			}
			limited := io.LimitReader(response.Body, maxDuckDuckGoResponseBytes+1)
			data, err := io.ReadAll(limited)
			if err != nil {
				return nil, Retryf("DuckDuckGo search response could not be read: %v", err)
			}
			if len(data) > maxDuckDuckGoResponseBytes {
				return nil, Retryf("DuckDuckGo search response exceeds %d bytes", maxDuckDuckGoResponseBytes)
			}
			document, _ := goquery.NewDocumentFromReader(strings.NewReader(string(data)))
			results := make([]WebSearchResult, 0)
			document.Find(".result").EachWithBreak(func(_ int, selection *goquery.Selection) bool {
				link := selection.Find("a.result__a").First()
				href, ok := link.Attr("href")
				if !ok || strings.TrimSpace(link.Text()) == "" {
					return true
				}
				results = append(results, WebSearchResult{
					Title: strings.Join(strings.Fields(link.Text()), " "),
					URL:   duckDuckGoResultURL(parsedEndpoint, href),
					Body: strings.Join(strings.Fields(
						selection.Find(".result__snippet").First().Text(),
					), " "),
				})
				return config.MaxResults == 0 || len(results) < config.MaxResults
			})
			return results, nil
		},
		WithDescription("Search DuckDuckGo and return result titles, URLs, and excerpts."),
	)
}

func duckDuckGoResultURL(endpoint *url.URL, href string) string {
	parsed, err := url.Parse(strings.TrimSpace(href))
	if err != nil {
		return href
	}
	if destination := parsed.Query().Get("uddg"); destination != "" {
		return destination
	}
	return endpoint.ResolveReference(parsed).String()
}
