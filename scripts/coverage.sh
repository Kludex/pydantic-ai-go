#!/usr/bin/env bash
# Enforce 100% statement coverage per package, reached honestly.
set -euo pipefail

fail=0
packages=(
    ./ai
    ./ai/a2a
    ./ai/cli
    ./ai/durable
    ./ai/durable/dbos
    ./ai/durable/prefect
    ./ai/durable/temporal
    ./ai/internal/download
    ./ai/internal/schema
    ./ai/retries
    ./ai/evals
    ./ai/embeddings
    ./ai/embeddings/bedrock
    ./ai/embeddings/cohere
    ./ai/embeddings/fakes
    ./ai/embeddings/google
    ./ai/embeddings/infer
    ./ai/embeddings/ollama
    ./ai/embeddings/openai
    ./ai/embeddings/voyageai
    ./ai/models/fakes
    ./ai/models/openai
    ./ai/models/openrouter
    ./ai/models/anthropic
    ./ai/models/azure
    ./ai/models/bedrock
    ./ai/models/bedrockmantle
    ./ai/models/cerebras
    ./ai/models/cohere
    ./ai/models/crusoe
    ./ai/models/githubcopilot
    ./ai/models/google
    ./ai/models/groq
    ./ai/models/huggingface
    ./ai/models/infer
    ./ai/models/mistral
    ./ai/models/ollama
    ./ai/models/snowflake
    ./ai/models/vllm
    ./ai/models/xai
    ./ai/models/zai
    ./ai/mcp
    ./ai/realtime
    ./ai/realtime/azure
    ./ai/realtime/google
    ./ai/realtime/infer
    ./ai/realtime/internal/openaiprotocol
    ./ai/realtime/openai
    ./ai/realtime/xai
    ./ai/ui/agui
    ./ai/ui/vercel
    ./ai/webchat
)
for pkg in "${packages[@]}"; do
    profile=$(mktemp)
    go test "$pkg" -coverprofile="$profile" > /dev/null
    total=$(go tool cover -func="$profile" | tail -1 | awk '{print $3}')
    if [ "$total" != "100.0%" ]; then
        echo "coverage for $pkg is $total, expected 100.0%"
        go tool cover -func="$profile" | awk '$3+0 < 100 && $1 != "total:"'
        fail=1
    fi
    rm -f "$profile"
done
exit $fail
