#!/usr/bin/env bash
# Enforce 100% statement coverage per package, reached honestly.
set -euo pipefail

fail=0
for pkg in . ./internal/schema ./models/fakes ./models/openai ./models/anthropic ./models/azure ./models/google ./mcp; do
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
