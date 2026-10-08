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
    ./ai/internal/contextwindow
    ./ai/internal/download
    ./ai/internal/schema
    ./ai/internal/promptcache
    ./ai/images
    ./ai/images/fakes
    ./ai/images/google
    ./ai/images/infer
    ./ai/images/openai
    ./ai/images/xai
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
    ./ai/models/openaicodex
    ./ai/models/openrouter
    ./ai/models/anthropic
    ./ai/models/azure
    ./ai/models/bedrock
    ./ai/models/bedrockmantle
    ./ai/models/cerebras
    ./ai/models/cohere
    ./ai/models/crusoe
    ./ai/models/deepseek
    ./ai/models/githubcopilot
    ./ai/models/google
    ./ai/models/groq
    ./ai/models/huggingface
    ./ai/models/infer
    ./ai/models/mistral
    ./ai/models/ollama
    ./ai/models/snowflake
    ./ai/models/systemone
    ./ai/models/typesafe
    ./ai/internal/decision
    ./ai/models/together
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
    if [ "$pkg" = "./ai/internal/decision" ]; then
        go test ./ai/models/typesafe ./ai/models/systemone -coverpkg="$pkg" -coverprofile="$profile" > /dev/null
    elif [ "$pkg" = "./ai/internal/promptcache" ]; then
        go test ./ai/models/anthropic ./ai/models/bedrock ./ai/models/openai ./ai/models/openrouter \
            -coverpkg="$pkg" -coverprofile="$profile" > /dev/null
    else
        go test "$pkg" -coverprofile="$profile" > /dev/null
    fi
    filtered=$(mktemp)
    awk -v prefix="$(go list -m)/" '
        NR == 1 { print; next }
        {
            split($1, location, ":")
            file = substr(location[1], length(prefix) + 1)
            split(location[2], position, ".")
            if (!(file in loaded)) {
                line = 0
                while ((getline text < file) > 0) source[file, ++line] = text
                close(file)
                loaded[file] = 1
            }
            if (source[file, position[1]] ~ /\/\/ pragma: no cover - [[:alnum:]]/) next
            print
        }
    ' "$profile" > "$filtered"
    mv "$filtered" "$profile"
    total=$(go tool cover -func="$profile" | tail -1 | awk '{print $3}')
    if [ "$total" != "100.0%" ]; then
        echo "coverage for $pkg is $total, expected 100.0%"
        go tool cover -func="$profile" | awk '$3+0 < 100 && $1 != "total:"'
        fail=1
    fi
    rm -f "$profile"
done
exit $fail
