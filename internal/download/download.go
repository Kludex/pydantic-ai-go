package download

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"strings"
	"time"
)

const maxResponseBytes = 50 << 20

// Options controls a validated download.
type Options struct {
	AllowLocal          bool
	Timeout             time.Duration
	MaxBytes            int64
	Headers             map[string]string
	PreserveOctetStream bool
	AllowedDomains      []string
	BlockedDomains      []string
}

// Result contains bytes downloaded from a validated URL.
type Result struct {
	URL         string
	Data        []byte
	MediaType   string
	ContentType string
}

// Fetch downloads an HTTP resource with DNS-rebinding and SSRF protection.
func Fetch(ctx context.Context, rawURL string, allowLocal bool) (Result, error) {
	return FetchWithOptions(ctx, rawURL, Options{AllowLocal: allowLocal})
}

// FetchWithOptions downloads an HTTP resource under explicit security and resource limits.
func FetchWithOptions(ctx context.Context, rawURL string, options Options) (Result, error) {
	if options.Timeout < 0 {
		return Result{}, fmt.Errorf("download: timeout must not be negative")
	}
	if options.MaxBytes < 0 {
		return Result{}, fmt.Errorf("download: maximum bytes must not be negative")
	}
	if options.MaxBytes == 1<<63-1 {
		return Result{}, fmt.Errorf("download: maximum bytes is too large")
	}
	timeout := options.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	maximumBytes := options.MaxBytes
	if maximumBytes == 0 {
		maximumBytes = maxResponseBytes
	}
	parsed, err := validatedDestination(rawURL, options)
	if err != nil {
		return Result{}, err
	}
	if _, err := resolveHost(ctx, strings.TrimSuffix(parsed.Hostname(), "."), options.AllowLocal); err != nil {
		return Result{}, err
	}
	dialer := &net.Dialer{Timeout: timeout, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		Proxy:              nil,
		DisableCompression: true,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, _ := net.SplitHostPort(address)
			addresses, err := resolveHost(ctx, strings.TrimSuffix(host, "."), options.AllowLocal)
			if err != nil {
				return nil, err
			}
			var dialErr error
			for _, address := range addresses {
				connection, err := dialer.DialContext(ctx, network, net.JoinHostPort(address.String(), port))
				if err == nil {
					return connection, nil
				}
				dialErr = err
			}
			return nil, dialErr
		},
	}
	defer transport.CloseIdleConnections()
	jar, _ := cookiejar.New(nil)
	client := &http.Client{
		Transport: transport,
		Timeout:   timeout,
		Jar:       jar,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if len(via) > 10 {
				return fmt.Errorf("download: too many redirects")
			}
			if _, err := validatedDestination(request.URL.String(), options); err != nil {
				return err
			}
			if len(via) > 0 && !mayForwardSensitiveHeaders(via[len(via)-1].URL, request.URL) {
				request.Header.Del("Authorization")
				request.Header.Del("Cookie")
				request.Header.Del("Proxy-Authorization")
			}
			return nil
		},
	}
	request, _ := http.NewRequestWithContext(ctx, http.MethodGet, parsed.String(), nil)
	request.Header.Set("Accept-Encoding", "identity, gzip")
	for name, value := range options.Headers {
		request.Header.Set(name, value)
	}
	response, err := client.Do(request)
	if err != nil {
		return Result{}, fmt.Errorf("download: request %q: %w", rawURL, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return Result{}, fmt.Errorf("download: request %q returned %s", rawURL, response.Status)
	}
	encoded, err := readLimited(response.Body, maximumBytes)
	if err != nil {
		return Result{}, err
	}
	data := encoded
	switch encoding := strings.TrimSpace(strings.ToLower(response.Header.Get("Content-Encoding"))); encoding {
	case "", "identity":
	case "gzip":
		reader, err := gzip.NewReader(bytes.NewReader(encoded))
		if err != nil {
			return Result{}, fmt.Errorf("download: decode gzip response: %w", err)
		}
		data, err = readLimited(reader, maximumBytes)
		_ = reader.Close()
		if err != nil {
			return Result{}, err
		}
	default:
		return Result{}, fmt.Errorf("download: unsupported content encoding %q", encoding)
	}
	contentType := response.Header.Get("Content-Type")
	mediaType := contentType
	if parsedType, _, err := mime.ParseMediaType(mediaType); err == nil {
		mediaType = parsedType
	}
	if mediaType == "application/octet-stream" && !options.PreserveOctetStream {
		mediaType = ""
	}
	return Result{
		URL: response.Request.URL.String(), Data: data, MediaType: mediaType, ContentType: contentType,
	}, nil
}

func validatedDestination(rawURL string, options Options) (*url.URL, error) {
	parsed, err := validateURL(rawURL)
	if err != nil {
		return nil, err
	}
	hostname := strings.ToLower(strings.TrimSuffix(parsed.Hostname(), "."))
	for _, blocked := range options.BlockedDomains {
		if hostname == normalizeDomain(blocked) {
			return nil, fmt.Errorf("download: domain %q is blocked", hostname)
		}
	}
	if options.AllowedDomains != nil {
		for _, allowed := range options.AllowedDomains {
			if hostname == normalizeDomain(allowed) {
				return parsed, nil
			}
		}
		return nil, fmt.Errorf("download: domain %q is not allowed", hostname)
	}
	return parsed, nil
}

func validateURL(rawURL string) (*url.URL, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("download: parse URL: %w", err)
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("download: URL scheme %q is not allowed", parsed.Scheme)
	}
	if parsed.Hostname() == "" {
		return nil, fmt.Errorf("download: URL has no hostname")
	}
	if parsed.User != nil {
		return nil, fmt.Errorf("download: URL must not contain credentials")
	}
	return parsed, nil
}

func normalizeDomain(domain string) string {
	return strings.ToLower(strings.TrimSuffix(strings.TrimSpace(domain), "."))
}

func mayForwardSensitiveHeaders(previous *url.URL, next *url.URL) bool {
	if !strings.EqualFold(previous.Hostname(), next.Hostname()) {
		return false
	}
	previousPort, nextPort := previous.Port(), next.Port()
	if previous.Scheme == next.Scheme && previousPort == nextPort {
		return true
	}
	return previous.Scheme == "http" && next.Scheme == "https" &&
		(previousPort == "" || previousPort == "80") && (nextPort == "" || nextPort == "443")
}

func readLimited(reader io.Reader, maximumBytes int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(reader, maximumBytes+1))
	if err != nil {
		return nil, fmt.Errorf("download: read response: %w", err)
	}
	if int64(len(data)) > maximumBytes {
		return nil, fmt.Errorf("download: response exceeds %d bytes", maximumBytes)
	}
	return data, nil
}
