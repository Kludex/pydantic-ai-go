package thinking

import "regexp"

// Hosts spell V4.1 as deepseek-v4.1-flash, deepseek-v4p1-flash, or deepseek-v4-1-flash.
var deepSeekV4 = regexp.MustCompile(`^deepseek-v4(?:[.p-]\d+)?-`)

// DeepSeekV4 reports whether name is a DeepSeek V4 model or point release, which thinks by default and can turn it off.
func DeepSeekV4(name string) bool {
	return deepSeekV4.MatchString(name)
}
