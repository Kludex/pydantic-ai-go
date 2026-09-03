// Command pydantic-ai-go runs terminal and browser chat interfaces.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/cli"
	"github.com/Kludex/pydantic-ai-go/models/infer"
	"github.com/Kludex/pydantic-ai-go/webchat"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(arguments []string) error {
	web := len(arguments) > 0 && arguments[0] == "web"
	if web {
		arguments = arguments[1:]
	}
	flags := flag.NewFlagSet("pydantic-ai-go", flag.ContinueOnError)
	modelName := flags.String("model", "openai:gpt-5-mini", "provider:model name")
	instructions := flags.String("instructions", "", "agent instructions")
	mcpConfig := flags.String("mcp-config", "", "path to mcpServers JSON")
	listen := flags.String("listen", ":8080", "web listen address")
	allowedHosts := flags.String("allowed-hosts", "", "comma-separated web Host allowlist")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	model, err := infer.Model(*modelName)
	if err != nil {
		return err
	}
	var options []ai.Option
	if *instructions != "" {
		options = append(options, ai.WithInstructions(*instructions))
	}
	agent := ai.NewAgent[struct{}, string](model, options...)
	if !web {
		return cli.Run(context.Background(), agent, struct{}{}, cli.Config{MCPConfigPath: *mcpConfig})
	}
	var hosts []string
	if *allowedHosts != "" {
		hosts = strings.Split(*allowedHosts, ",")
	}
	handler, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{
		AllowedHosts: hosts, MCPConfigPath: *mcpConfig,
	})
	if err != nil {
		return err
	}
	server := &http.Server{Addr: *listen, Handler: handler}
	return server.ListenAndServe()
}
