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

// Result contains bytes downloaded from a validated URL.
type Result struct {
	Data      []byte
	MediaType string
}

// Fetch downloads an HTTP resource with DNS-rebinding and SSRF protection.
func Fetch(ctx context.Context, rawURL string, allowLocal bool) (Result, error) {
	parsed, err := validateURL(rawURL)
	if err != nil {
		return Result{}, err
	}
	if _, err := resolveHost(ctx, strings.TrimSuffix(parsed.Hostname(), "."), allowLocal); err != nil {
		return Result{}, err
	}
	dialer := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		Proxy:              nil,
		DisableCompression: true,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, _ := net.SplitHostPort(address)
			addresses, err := resolveHost(ctx, strings.TrimSuffix(host, "."), allowLocal)
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
		Timeout:   30 * time.Second,
		Jar:       jar,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if len(via) > 10 {
				return fmt.Errorf("download: too many redirects")
			}
			_, err := validateURL(request.URL.String())
			return err
		},
	}
	request, _ := http.NewRequestWithContext(ctx, http.MethodGet, parsed.String(), nil)
	request.Header.Set("Accept-Encoding", "identity, gzip")
	response, err := client.Do(request)
	if err != nil {
		return Result{}, fmt.Errorf("download: request %q: %w", rawURL, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return Result{}, fmt.Errorf("download: request %q returned %s", rawURL, response.Status)
	}
	encoded, err := readLimited(response.Body)
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
		data, err = readLimited(reader)
		_ = reader.Close()
		if err != nil {
			return Result{}, err
		}
	default:
		return Result{}, fmt.Errorf("download: unsupported content encoding %q", encoding)
	}
	mediaType := response.Header.Get("Content-Type")
	if parsedType, _, err := mime.ParseMediaType(mediaType); err == nil {
		mediaType = parsedType
	}
	if mediaType == "application/octet-stream" {
		mediaType = ""
	}
	return Result{Data: data, MediaType: mediaType}, nil
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

func readLimited(reader io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(reader, maxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("download: read response: %w", err)
	}
	if len(data) > maxResponseBytes {
		return nil, fmt.Errorf("download: response exceeds %d bytes", maxResponseBytes)
	}
	return data, nil
}
