package ai

import (
	"fmt"
	"strings"
)

const (
	bannerGutter     = 2
	bannerWidth      = 100
	bannerTextIndent = "  "
	bannerSeparator  = " | "
)

var bannerLogoLines = strings.Split(`      / \
     /   \
   /___.___\
  /    |    \
/      |      \
`+"`---.._|_..---'", "\n")

type bannerInfo struct {
	label       string
	value       string
	highlighted bool
}

func renderBanner(details bannerDetails, color bool) string {
	info := make([]bannerInfo, 0, 5)
	if details.name != "" {
		info = append(info, bannerInfo{label: "agent", value: details.name, highlighted: true})
	}
	info = append(info, bannerInfo{label: "model", value: details.model, highlighted: true})
	if details.output != "" {
		info = append(info, bannerInfo{label: "output", value: details.output})
	}
	if details.tools != nil {
		info = append(info, bannerInfo{label: "tools", value: fmt.Sprint(*details.tools)})
	}
	info = append(info, bannerInfo{label: "capabilities", value: fmt.Sprint(details.capabilities)})

	textWidth := bannerWidth - bannerLogoWidth() - bannerGutter
	lines := append(wrapBannerText(bannerVersionLine(), textWidth, ""), "")
	lines = append(lines, bannerInfoLines(info, textWidth, color)...)
	if details.observability {
		lines = append(lines, "")
		lines = append(lines, wrapBannerText(
			"observability: off - see every model and tool call live, with cost", textWidth, "",
		)...)
		lines = append(lines, wrapBannerText(
			"set it up with OpenTelemetry: https://github.com/Kludex/pydantic-ai-go/blob/main/docs/observability.md",
			textWidth, bannerTextIndent,
		)...)
		lines = append(lines, "")
		lines = append(lines, wrapBannerText(
			"goes away once observability is on - or PYDANTIC_AI_NO_BANNER=1", textWidth, "",
		)...)
	}
	return bannerBesideLogo(lines, color)
}

func bannerInfoLines(info []bannerInfo, width int, color bool) []string {
	var lines []string
	lineWidth := 0
	for _, item := range info {
		value := bannerElideMiddle(item.value, width-len(bannerTextIndent)-len(item.label)-2)
		plain := item.label + ": " + value
		styled := plain
		if color && item.highlighted {
			styled = item.label + ": \x1b[32m" + value + "\x1b[0m"
		}
		if len(lines) > 0 && lineWidth+len(bannerSeparator)+len(plain) <= width {
			lines[len(lines)-1] += bannerSeparator + styled
			lineWidth += len(bannerSeparator) + len(plain)
			continue
		}
		indent := ""
		if len(lines) > 0 {
			indent = bannerTextIndent
		}
		lines = append(lines, indent+styled)
		lineWidth = len(indent) + len(plain)
	}
	return lines
}

func wrapBannerText(text string, width int, indent string) []string {
	words := strings.Fields(text)
	lines := []string{indent}
	for _, word := range words {
		line := len(lines) - 1
		separator := ""
		if lines[line] != indent {
			separator = " "
		}
		if len(lines[line])+len(separator)+len(word) <= width {
			lines[line] += separator + word
			continue
		}
		lines = append(lines, bannerTextIndent+word)
	}
	return lines
}

func bannerLogoWidth() int {
	width := 0
	for _, line := range bannerLogoLines {
		width = max(width, len(line))
	}
	return width
}

func bannerBesideLogo(lines []string, color bool) string {
	logoPadding := max(0, (len(lines)-len(bannerLogoLines))/2)
	textPadding := max(0, (len(bannerLogoLines)-len(lines))/2)
	rows := max(logoPadding+len(bannerLogoLines), textPadding+len(lines))
	output := make([]string, rows)
	logoWidth := bannerLogoWidth()
	for row := range rows {
		logo := ""
		if index := row - logoPadding; index >= 0 && index < len(bannerLogoLines) {
			logo = bannerLogoLines[index]
		}
		text := ""
		if index := row - textPadding; index >= 0 && index < len(lines) {
			text = lines[index]
		}
		visibleLogo := logo
		if color && logo != "" {
			logo = "\x1b[35m" + logo + "\x1b[0m"
		}
		output[row] = strings.TrimRight(logo+strings.Repeat(" ", logoWidth-len(visibleLogo)+bannerGutter)+text, " ")
	}
	return strings.Join(output, "\n")
}
