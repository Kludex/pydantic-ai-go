package download

import "net/netip"

var privateNetworks = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("10.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("127.0.0.0/8"),
	netip.MustParsePrefix("169.254.0.0/16"),
	netip.MustParsePrefix("172.16.0.0/12"),
	netip.MustParsePrefix("192.168.0.0/16"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("192.88.99.2/32"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("198.51.100.0/24"),
	netip.MustParsePrefix("203.0.113.0/24"),
	netip.MustParsePrefix("224.0.0.0/4"),
	netip.MustParsePrefix("240.0.0.0/4"),
	netip.MustParsePrefix("::/128"),
	netip.MustParsePrefix("::1/128"),
	netip.MustParsePrefix("fe80::/10"),
	netip.MustParsePrefix("fc00::/7"),
	netip.MustParsePrefix("fec0::/10"),
	netip.MustParsePrefix("::ffff:0:0:0/96"),
	netip.MustParsePrefix("100::/64"),
	netip.MustParsePrefix("100:0:0:1::/64"),
	netip.MustParsePrefix("2001::/23"),
	netip.MustParsePrefix("2001::/32"),
	netip.MustParsePrefix("2001:2::/48"),
	netip.MustParsePrefix("2001:db8::/32"),
	netip.MustParsePrefix("3fff::/20"),
	netip.MustParsePrefix("5f00::/16"),
	netip.MustParsePrefix("ff00::/8"),
}

// privateNetworkExceptions are the globally reachable assignments inside 2001::/23.
// An IPv4 address they embed is still checked.
var privateNetworkExceptions = []netip.Prefix{
	netip.MustParsePrefix("2001:1::1/128"),
	netip.MustParsePrefix("2001:1::2/128"),
	netip.MustParsePrefix("2001:1::3/128"),
	netip.MustParsePrefix("2001:3::/32"),
	netip.MustParsePrefix("2001:4:112::/48"),
	netip.MustParsePrefix("2001:20::/28"),
	netip.MustParsePrefix("2001:30::/28"),
}

var cloudMetadataAddresses = map[netip.Addr]struct{}{
	netip.MustParseAddr("169.254.169.254"): {},
	netip.MustParseAddr("169.254.170.2"):   {},
	netip.MustParseAddr("169.254.170.23"):  {},
	netip.MustParseAddr("168.63.129.16"):   {},
	netip.MustParseAddr("100.100.100.200"): {},
	netip.MustParseAddr("192.0.0.192"):     {},
	netip.MustParseAddr("169.254.42.42"):   {},
	netip.MustParseAddr("fd00:ec2::254"):   {},
	netip.MustParseAddr("fd00:ec2::23"):    {},
	netip.MustParseAddr("fd20:ce::254"):    {},
	netip.MustParseAddr("fd00:42::42"):     {},
}
