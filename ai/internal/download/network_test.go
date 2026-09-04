package download

import (
	"net/netip"
	"strings"
	"testing"
)

func TestValidateAddress(t *testing.T) {
	for name, test := range map[string]struct {
		address    string
		allowLocal bool
		wantError  string
	}{
		"public IPv4":       {address: "8.8.8.8"},
		"public IPv6":       {address: "2606:4700:4700::1111"},
		"private":           {address: "10.0.0.1", wantError: "private or internal"},
		"allowed private":   {address: "10.0.0.1", allowLocal: true},
		"metadata":          {address: "169.254.169.254", allowLocal: true, wantError: "cloud metadata"},
		"mapped private":    {address: "::ffff:10.0.0.1", wantError: "private or internal"},
		"6to4 private":      {address: "2002:0a00:0001::", wantError: "private or internal"},
		"NAT64 private":     {address: "64:ff9b::a00:1", wantError: "private or internal"},
		"local NAT64":       {address: "64:ff9b:1:0:a00:100::", wantError: "private or internal"},
		"IPv4 compatible":   {address: "::10.0.0.1", wantError: "private or internal"},
		"ISATAP private":    {address: "2001:4860::5efe:a00:1", wantError: "private or internal"},
		"ISATAP variant":    {address: "2001:4860::200:5efe:a00:1", wantError: "private or internal"},
		"embedded metadata": {address: "2001:4860::a9fe:a9fe", allowLocal: true, wantError: "cloud metadata"},
		"Teredo metadata":   {address: "2001::5601:5601", allowLocal: true, wantError: "cloud metadata"},
	} {
		t.Run(name, func(t *testing.T) {
			err := validateAddress(netip.MustParseAddr(test.address), test.allowLocal)
			if test.wantError == "" && err != nil {
				t.Fatal(err)
			}
			if test.wantError != "" && (err == nil || !strings.Contains(err.Error(), test.wantError)) {
				t.Fatalf("validation error = %v, want %q", err, test.wantError)
			}
		})
	}
}

func TestEmbeddedIPv4AddressModes(t *testing.T) {
	if values := embeddedIPv4Addresses(netip.MustParseAddr("8.8.8.8"), false); values != nil {
		t.Fatalf("IPv4 produced embedded values: %v", values)
	}
	exhaustive := embeddedIPv4Addresses(netip.MustParseAddr("2001:4860::a9fe:a9fe"), true)
	if len(exhaustive) != 7 {
		t.Fatalf("unexpected exhaustive embedded values: %v", exhaustive)
	}
	teredo := embeddedIPv4Addresses(netip.MustParseAddr("2001::5601:5601"), true)
	if len(teredo) != 8 || teredo[7] != netip.MustParseAddr("169.254.169.254") {
		t.Fatalf("unexpected Teredo values: %v", teredo)
	}
	if values := embeddedIPv4Addresses(netip.MustParseAddr("2001:4860::1"), false); len(values) != 0 {
		t.Fatalf("ordinary IPv6 produced embedded values: %v", values)
	}
}
