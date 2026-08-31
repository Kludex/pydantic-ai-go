package download

import (
	"context"
	"fmt"
	"net"
	"net/netip"
)

func resolveHost(ctx context.Context, host string, allowLocal bool) ([]netip.Addr, error) {
	if address, err := netip.ParseAddr(host); err == nil {
		if err := validateAddress(address, allowLocal); err != nil {
			return nil, err
		}
		return []netip.Addr{address}, nil
	}
	addresses, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
	if err != nil {
		return nil, fmt.Errorf("download: resolve host %q: %w", host, err)
	}
	for _, address := range addresses {
		if err := validateAddress(address, allowLocal); err != nil {
			return nil, err
		}
	}
	return addresses, nil
}

func validateAddress(address netip.Addr, allowLocal bool) error {
	if isCloudMetadataAddress(address) {
		return fmt.Errorf("download: access to cloud metadata address %s is blocked", address)
	}
	if !allowLocal && isPrivateAddress(address) {
		return fmt.Errorf("download: access to private or internal address %s is blocked", address)
	}
	return nil
}

func isPrivateAddress(address netip.Addr) bool {
	address = address.Unmap()
	for _, network := range privateNetworks {
		if network.Contains(address) {
			return true
		}
	}
	for _, embedded := range embeddedIPv4Addresses(address, false) {
		for _, network := range privateNetworks {
			if network.Contains(embedded) {
				return true
			}
		}
	}
	return false
}

func isCloudMetadataAddress(address netip.Addr) bool {
	address = address.Unmap()
	if _, blocked := cloudMetadataAddresses[address]; blocked {
		return true
	}
	for _, embedded := range embeddedIPv4Addresses(address, true) {
		if _, blocked := cloudMetadataAddresses[embedded]; blocked {
			return true
		}
	}
	return false
}

// IPv6 transition formats can hide blocked IPv4 destinations at several byte offsets.
func embeddedIPv4Addresses(address netip.Addr, exhaustive bool) []netip.Addr {
	if !address.Is6() {
		return nil
	}
	bytes := address.As16()
	offsets := [][4]int{{4, 5, 6, 7}, {5, 6, 7, 9}, {6, 7, 9, 10}, {7, 9, 10, 11}, {9, 10, 11, 12}, {12, 13, 14, 15}, {2, 3, 4, 5}}
	if exhaustive {
		result := make([]netip.Addr, 0, len(offsets)+1)
		for _, offset := range offsets {
			result = append(result, netip.AddrFrom4([4]byte{
				bytes[offset[0]], bytes[offset[1]], bytes[offset[2]], bytes[offset[3]],
			}))
		}
		if netip.MustParsePrefix("2001::/32").Contains(address) {
			result = append(result, netip.AddrFrom4([4]byte{
				^bytes[12], ^bytes[13], ^bytes[14], ^bytes[15],
			}))
		}
		return result
	}
	var result []netip.Addr
	if netip.MustParsePrefix("2002::/16").Contains(address) {
		result = append(result, netip.AddrFrom4([4]byte{bytes[2], bytes[3], bytes[4], bytes[5]}))
	}
	if netip.MustParsePrefix("64:ff9b::/96").Contains(address) {
		result = append(result, netip.AddrFrom4([4]byte{bytes[12], bytes[13], bytes[14], bytes[15]}))
	}
	if netip.MustParsePrefix("64:ff9b:1::/48").Contains(address) {
		for _, offset := range [][4]int{{6, 7, 9, 10}, {7, 9, 10, 11}, {9, 10, 11, 12}, {12, 13, 14, 15}} {
			result = append(result, netip.AddrFrom4([4]byte{
				bytes[offset[0]], bytes[offset[1]], bytes[offset[2]], bytes[offset[3]],
			}))
		}
	}
	if bytes[0] == 0 && bytes[1] == 0 && bytes[2] == 0 && bytes[3] == 0 &&
		bytes[4] == 0 && bytes[5] == 0 && bytes[6] == 0 && bytes[7] == 0 &&
		bytes[8] == 0 && bytes[9] == 0 && bytes[10] == 0 && bytes[11] == 0 &&
		!address.IsUnspecified() && !address.IsLoopback() {
		result = append(result, netip.AddrFrom4([4]byte{bytes[12], bytes[13], bytes[14], bytes[15]}))
	}
	isISATAP := (bytes[8] == 0 || bytes[8] == 2) && bytes[9] == 0 && bytes[10] == 0x5e && bytes[11] == 0xfe
	if isISATAP {
		result = append(result, netip.AddrFrom4([4]byte{bytes[12], bytes[13], bytes[14], bytes[15]}))
	}
	return result
}
