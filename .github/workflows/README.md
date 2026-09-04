# Agentic workflows

`agentic-ai-sync.md` checks upstream `pydantic_ai` every day. It reviews the
diff since `.upstream-sync.json`, ports applicable behavior, advances the pin,
and opens one draft pull request.

The Markdown source compiles to `agentic-ai-sync.lock.yml` with `gh aw compile`.
Commit both files after changing the workflow.

## Enable the workflow

1. Set the repository variable `AGENTIC_WORKFLOWS_ENABLED` to `true`.
2. Add the repository secret `FIREWORKS_API_KEY`.
3. Add the `agentic-workflows` and `upstream-sync` labels.

Set `AGENTIC_WORKFLOWS_ENABLED` to `false` to stop scheduled runs without
editing the workflow.

## Update the workflow

```bash
gh aw compile agentic-ai-sync --strict --approve --schedule-seed Kludex/pydantic-ai-go
```

The generated lock file pins actions and container images. Do not edit it by
hand.
