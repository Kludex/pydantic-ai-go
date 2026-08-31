package download

import (
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

func TestFetchDownloadsValidatedResponses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/redirect":
			http.SetCookie(writer, &http.Cookie{Name: "download", Value: "allowed", Path: "/"})
			http.Redirect(writer, request, "/gzip", http.StatusFound)
		case "/gzip":
			if cookie, err := request.Cookie("download"); err != nil || cookie.Value != "allowed" {
				http.Error(writer, "missing cookie", http.StatusForbidden)
				return
			}
			writer.Header().Set("Content-Encoding", "gzip")
			writer.Header().Set("Content-Type", "video/mp4; charset=binary")
			compressed := gzip.NewWriter(writer)
			_, _ = compressed.Write([]byte("video"))
			_ = compressed.Close()
		case "/octet":
			writer.Header().Set("Content-Type", "application/octet-stream")
			_, _ = writer.Write([]byte("bytes"))
		case "/invalid-type":
			writer.Header().Set("Content-Type", "not a type;")
			_, _ = writer.Write([]byte("bytes"))
		case "/bad-gzip":
			writer.Header().Set("Content-Encoding", "gzip")
			_, _ = writer.Write([]byte("bad"))
		case "/truncated-gzip":
			var encoded bytes.Buffer
			compressed := gzip.NewWriter(&encoded)
			_, _ = compressed.Write([]byte("video"))
			_ = compressed.Close()
			writer.Header().Set("Content-Encoding", "gzip")
			_, _ = writer.Write(encoded.Bytes()[:encoded.Len()-4])
		case "/truncated":
			connection, _, _ := writer.(http.Hijacker).Hijack()
			_, _ = io.WriteString(connection, "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nx")
			_ = connection.Close()
		case "/metadata":
			http.Redirect(writer, request, "http://169.254.169.254/video.mp4", http.StatusFound)
		case "/br":
			writer.Header().Set("Content-Encoding", "br")
			_, _ = writer.Write([]byte("bytes"))
		case "/error":
			http.Error(writer, "no", http.StatusBadRequest)
		case "/loop":
			http.Redirect(writer, request, "/loop", http.StatusFound)
		}
	}))
	t.Cleanup(server.Close)

	result, err := Fetch(t.Context(), strings.Replace(server.URL, "127.0.0.1", "localhost", 1)+"/redirect", true)
	if err != nil || string(result.Data) != "video" || result.MediaType != "video/mp4" {
		t.Fatalf("unexpected download result=%+v err=%v", result, err)
	}
	result, err = Fetch(t.Context(), strings.Replace(server.URL, "http://", "HTTP://", 1)+"/octet", true)
	if err != nil || string(result.Data) != "bytes" || result.MediaType != "" {
		t.Fatalf("unexpected octet result=%+v err=%v", result, err)
	}
	result, err = Fetch(t.Context(), server.URL+"/invalid-type", true)
	if err != nil || result.MediaType != "not a type;" {
		t.Fatalf("unexpected invalid content type result=%+v err=%v", result, err)
	}
	for path, contains := range map[string]string{
		"/bad-gzip": "decode gzip", "/truncated-gzip": "read response", "/truncated": "read response",
		"/metadata": "cloud metadata", "/br": "unsupported content encoding", "/error": "400 Bad Request",
		"/loop": "too many redirects",
	} {
		if _, err := Fetch(t.Context(), server.URL+path, true); err == nil || !strings.Contains(err.Error(), contains) {
			t.Fatalf("fetch %s error = %v, want %q", path, err, contains)
		}
	}
}

func TestFetchRejectsUnsafeURLs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(server.Close)
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	closedURL := "http://" + listener.Addr().String() + "/video.mp4"
	_ = listener.Close()
	for name, test := range map[string]struct {
		url        string
		allowLocal bool
		contains   string
	}{
		"invalid":          {url: "http://%", contains: "parse URL"},
		"scheme":           {url: "file:///tmp/video.mp4", contains: "scheme"},
		"host":             {url: "https:///video.mp4", contains: "no hostname"},
		"credentials":      {url: "https://user:pass@example.com/video.mp4", contains: "credentials"},
		"private":          {url: server.URL, contains: "private or internal"},
		"private hostname": {url: strings.Replace(server.URL, "127.0.0.1", "localhost", 1), contains: "private or internal"},
		"cloud metadata":   {url: "http://169.254.169.254/video.mp4", allowLocal: true, contains: "cloud metadata"},
		"metadata FQDN":    {url: "http://169.254.169.254./video.mp4", allowLocal: true, contains: "cloud metadata"},
		"azure metadata":   {url: "http://168.63.129.16/video.mp4", allowLocal: true, contains: "cloud metadata"},
		"resolution error": {url: "http://does-not-exist.invalid/video.mp4", contains: "resolve host"},
		"connection error": {url: closedURL, allowLocal: true, contains: "request"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Fetch(t.Context(), test.url, test.allowLocal); err == nil ||
				!strings.Contains(err.Error(), test.contains) {
				t.Fatalf("Fetch(%q) error = %v, want %q", test.url, err, test.contains)
			}
		})
	}
}

func TestReadLimitedErrors(t *testing.T) {
	if _, err := readLimited(failingReader{}); err == nil || !strings.Contains(err.Error(), "read failed") {
		t.Fatalf("unexpected read error: %v", err)
	}
	if _, err := readLimited(io.LimitReader(bytes.NewReader(make([]byte, maxResponseBytes+1)), maxResponseBytes+1)); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("unexpected size error: %v", err)
	}
}

func TestResolveHostHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := resolveHost(ctx, "does-not-resolve.invalid", false); err == nil {
		t.Fatal("cancelled resolution succeeded")
	}
}
