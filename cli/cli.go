// Package cli runs typed agents in terminal chat sessions.
package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/mcp"
)

// Config controls terminal input, output, history, and MCP tools.
type Config struct {
	// Input supplies newline-delimited prompts. Nil uses os.Stdin.
	Input io.Reader
	// Output receives assistant text and tool progress. Nil uses os.Stdout.
	Output io.Writer
	// ErrorOutput receives run errors. Nil uses os.Stderr.
	ErrorOutput io.Writer
	// Prompt is written before each interactive input. Empty uses "> ".
	Prompt string
	// MCPConfigPath loads common mcpServers JSON once before the session.
	MCPConfigPath string
	// RunOptions are copied and reused for each turn.
	RunOptions []ai.RunOption
	// History seeds the first turn and is detached before use.
	History []ai.ModelMessage
}

// Run starts a newline-delimited chat session. Commands are /clear, /usage,
// and /exit. EOF ends the session successfully.
func Run[Deps, Output any](
	ctx context.Context, agent *ai.Agent[Deps, Output], deps Deps, config Config,
) error {
	if agent == nil {
		return fmt.Errorf("cli: agent must not be nil")
	}
	input := config.Input
	if input == nil {
		input = os.Stdin
	}
	output := config.Output
	if output == nil {
		output = os.Stdout
	}
	errorOutput := config.ErrorOutput
	if errorOutput == nil {
		errorOutput = os.Stderr
	}
	promptLabel := config.Prompt
	if promptLabel == "" {
		promptLabel = "> "
	}
	options := append([]ai.RunOption(nil), config.RunOptions...)
	if config.MCPConfigPath != "" {
		toolsets, err := mcp.LoadToolsets[Deps](config.MCPConfigPath)
		if err != nil {
			return fmt.Errorf("cli: load MCP configuration: %w", err)
		}
		options = append(options, ai.WithRunToolsets(toolsets...))
	}
	history := append([]ai.ModelMessage(nil), config.History...)
	usage := ai.Usage{}
	scanner := bufio.NewScanner(input)
	for {
		if _, err := io.WriteString(output, promptLabel); err != nil {
			return fmt.Errorf("cli: write prompt: %w", err)
		}
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return fmt.Errorf("cli: read prompt: %w", err)
			}
			return nil
		}
		prompt := strings.TrimSpace(scanner.Text())
		switch prompt {
		case "":
			continue
		case "/exit", "/quit":
			return nil
		case "/clear":
			history = nil
			usage = ai.Usage{}
			if _, err := io.WriteString(output, "History cleared.\n"); err != nil {
				return fmt.Errorf("cli: write output: %w", err)
			}
			continue
		case "/usage":
			if _, err := fmt.Fprintf(
				output, "requests=%d input_tokens=%d output_tokens=%d tool_calls=%d\n",
				usage.Requests, usage.InputTokens, usage.OutputTokens, usage.ToolCalls,
			); err != nil {
				return fmt.Errorf("cli: write output: %w", err)
			}
			continue
		}
		runOptions := append([]ai.RunOption(nil), options...)
		if len(history) > 0 {
			runOptions = append(runOptions, ai.WithMessageHistory(history))
		}
		stream := agent.RunStream(ctx, prompt, deps, runOptions...)
		wroteText := false
		var streamWriteErr error
	events:
		for event, eventErr := range stream.Events() {
			if eventErr != nil {
				if _, err := fmt.Fprintf(errorOutput, "Error: %v\n", eventErr); err != nil {
					return fmt.Errorf("cli: write error: %w", err)
				}
				break
			}
			switch value := event.(type) {
			case ai.PartStartEvent:
				if text, ok := value.Part.(ai.TextPart); ok && text.Content != "" {
					_, streamWriteErr = io.WriteString(output, text.Content)
					wroteText = true
				}
			case ai.PartDeltaEvent:
				if delta, ok := value.Delta.(ai.TextPartDelta); ok && delta.ContentDelta != "" {
					_, streamWriteErr = io.WriteString(output, delta.ContentDelta)
					wroteText = true
				}
			case ai.FunctionToolCallEvent:
				_, streamWriteErr = fmt.Fprintf(output, "\n[tool] %s %s\n", value.Part.ToolName, value.Part.Args)
			case ai.FunctionToolResultEvent:
				_, streamWriteErr = fmt.Fprintf(output, "[tool result] %s\n", toolResult(value.Part))
			}
			if streamWriteErr != nil {
				break events
			}
		}
		if streamWriteErr != nil {
			return fmt.Errorf("cli: write stream: %w", streamWriteErr)
		}
		result := stream.Result()
		if result == nil {
			continue
		}
		if !wroteText {
			encoded, err := json.Marshal(result.Output)
			if err != nil {
				return fmt.Errorf("cli: encode output: %w", err)
			}
			if _, err := output.Write(encoded); err != nil {
				return fmt.Errorf("cli: write output: %w", err)
			}
		}
		if _, err := io.WriteString(output, "\n"); err != nil {
			return fmt.Errorf("cli: write output: %w", err)
		}
		history = result.Messages()
		usage.Add(result.Usage())
	}
}

func toolResult(part ai.RequestPart) string {
	if value, ok := part.(ai.ToolReturnPart); ok {
		encoded, err := json.Marshal(value.Content)
		if err == nil {
			return value.ToolName + " " + string(encoded)
		}
		return value.ToolName + " <unprintable>"
	}
	value := part.(ai.RetryPromptPart)
	return value.ToolName + " retry: " + value.ModelResponse()
}
