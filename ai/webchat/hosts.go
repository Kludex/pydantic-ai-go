package webchat

import (
	"fmt"
	"net"
	"net/netip"
	"strings"

	"golang.org/x/net/idna"
)

func normalizeAllowedHosts(hosts []string) ([]string, error) {
	normalized := make([]string, 0, len(hosts))
	for _, value := range hosts {
		pattern := strings.ToLower(strings.TrimSpace(value))
		if pattern == "*" {
			normalized = append(normalized, pattern)
			continue
		}
		wildcard := strings.HasPrefix(pattern, "*.")
		if wildcard {
			pattern = strings.TrimPrefix(pattern, "*.")
		}
		if pattern == "" || strings.Contains(pattern, "*") {
			return nil, fmt.Errorf("webchat: invalid allowed host %q", value)
		}
		host, ok := normalizeHostname(pattern)
		if !ok || net.ParseIP(host) != nil {
			return nil, fmt.Errorf("webchat: invalid allowed host %q", value)
		}
		if wildcard {
			host = "*." + host
		}
		normalized = append(normalized, host)
	}
	return normalized, nil
}

func hostAllowed(value string, allowed []string) bool {
	host, ip, ok := requestHost(value)
	if !ok {
		return false
	}
	if ip || host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	for _, pattern := range allowed {
		if pattern == "*" || pattern == host {
			return true
		}
		if strings.HasPrefix(pattern, "*.") && strings.HasSuffix(host, strings.TrimPrefix(pattern, "*")) {
			return true
		}
	}
	return false
}

func requestHost(value string) (string, bool, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", false, false
	}
	host := value
	if parsed, _, err := net.SplitHostPort(value); err == nil {
		host = parsed
	} else if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
		host = strings.TrimSuffix(strings.TrimPrefix(value, "["), "]")
	} else if strings.Contains(value, ":") {
		address, err := netip.ParseAddr(value)
		if err != nil {
			return "", false, false
		}
		return address.String(), true, true
	}
	host = strings.TrimSuffix(strings.Trim(host, "[]"), ".")
	if address, err := netip.ParseAddr(host); err == nil {
		return address.String(), true, true
	}
	host, ok := normalizeHostname(host)
	return host, false, ok
}

func normalizeHostname(value string) (string, bool) {
	ascii, err := idna.Lookup.ToASCII(strings.TrimSuffix(strings.ToLower(value), "."))
	if err != nil || len(ascii) == 0 || len(ascii) > 253 {
		return "", false
	}
	return ascii, true
}
