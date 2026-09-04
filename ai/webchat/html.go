package webchat

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	atomicfile "github.com/natefinch/atomic"
)

const (
	// ChatUIVersion is the bundled @pydantic/ai-chat-ui release.
	ChatUIVersion = "2.1.0"
	// DefaultHTMLURL is the default split browser build served by NewHandler.
	DefaultHTMLURL = "https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@" + ChatUIVersion + "/dist/index.html"
	// OfflineHTMLURL is the self-contained browser build for offline deployments.
	OfflineHTMLURL = "https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@" + ChatUIVersion + "/offline/index.html"
)

var htmlCacheLock sync.Mutex

type htmlLoader struct {
	source    string
	cacheFile string
	client    *http.Client
}

func newHTMLLoader(config Config) (*htmlLoader, error) {
	source := config.HTMLSource
	if source == "" {
		source = DefaultHTMLURL
	}
	loader := &htmlLoader{source: source, client: config.HTTPClient}
	if loader.client == nil {
		loader.client = http.DefaultClient
	}
	if !remoteHTMLSource(source) {
		return loader, nil
	}
	cacheDir := config.CacheDir
	if cacheDir == "" {
		root, err := os.UserCacheDir()
		if err != nil {
			return nil, fmt.Errorf("webchat: locate user cache: %w", err)
		}
		cacheDir = filepath.Join(root, "pydantic-ai", "web-ui")
	}
	cacheName := ChatUIVersion + ".html"
	if source != DefaultHTMLURL {
		digest := fmt.Sprintf("%x", sha256.Sum256([]byte(source)))
		cacheName = "url_" + digest[:16] + ".html"
	}
	loader.cacheFile = filepath.Join(cacheDir, cacheName)
	return loader, nil
}

func (loader *htmlLoader) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	content, err := loader.load(request.Context())
	if err != nil {
		http.Error(response, err.Error(), http.StatusInternalServerError)
		return
	}
	response.Header().Set("Content-Type", "text/html; charset=utf-8")
	response.Header().Set("Cache-Control", "public, max-age=3600")
	_, _ = response.Write(content)
}

func (loader *htmlLoader) load(ctx context.Context) ([]byte, error) {
	if !remoteHTMLSource(loader.source) {
		path, err := expandHome(loader.source)
		if err != nil {
			return nil, err
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("webchat: read UI HTML: %w", err)
		}
		if len(content) == 0 {
			return nil, fmt.Errorf("webchat: UI HTML is empty")
		}
		return content, nil
	}
	if err := os.MkdirAll(filepath.Dir(loader.cacheFile), 0o700); err != nil {
		return nil, fmt.Errorf("webchat: create UI cache: %w", err)
	}
	content, found, err := readCachedHTML(loader.cacheFile)
	if err != nil {
		return nil, err
	}
	if found {
		return content, nil
	}
	content, err = loader.fetch(ctx)
	if err != nil {
		return nil, err
	}
	if err := writeCachedHTML(loader.cacheFile, content); err != nil {
		return nil, err
	}
	return content, nil
}

func (loader *htmlLoader) fetch(ctx context.Context) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, loader.source, nil)
	if err != nil {
		return nil, fmt.Errorf("webchat: prepare UI request: %w", err)
	}
	response, err := loader.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("webchat: fetch UI HTML: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("webchat: UI HTML returned %s", response.Status)
	}
	content, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("webchat: read UI response: %w", err)
	}
	if len(content) == 0 {
		return nil, fmt.Errorf("webchat: UI HTML response is empty")
	}
	return content, nil
}

func readCachedHTML(path string) ([]byte, bool, error) {
	htmlCacheLock.Lock()
	defer htmlCacheLock.Unlock()
	content, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("webchat: read UI cache: %w", err)
	}
	if len(content) == 0 {
		_ = os.Remove(path)
		return nil, false, nil
	}
	return content, true, nil
}

func writeCachedHTML(path string, content []byte) error {
	htmlCacheLock.Lock()
	defer htmlCacheLock.Unlock()
	if err := atomicfile.WriteFile(path, bytes.NewReader(content)); err != nil {
		return fmt.Errorf("webchat: write UI cache: %w", err)
	}
	return nil
}

func remoteHTMLSource(source string) bool {
	return strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://")
}

func expandHome(path string) (string, error) {
	if path != "~" && !strings.HasPrefix(path, "~/") {
		return path, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("webchat: locate home directory: %w", err)
	}
	if path == "~" {
		return home, nil
	}
	return filepath.Join(home, strings.TrimPrefix(path, "~/")), nil
}
