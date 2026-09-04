# Benchmarks

Run every benchmark from the repository root:

```console
go test -run '^$' -bench . -benchmem ./ai
```

The benchmark suite covers four hot paths:

| Benchmark | Measured operation |
| --- | --- |
| `BenchmarkAgentLoop` | One complete text run against an in-process model |
| `BenchmarkStreaming` | One streamed text run consumed through the public event API |
| `BenchmarkSchemaReflection` | Agent construction and typed tool schema reflection |
| `BenchmarkParallelTools` | One run that validates and executes eight independent tool calls |

The models do not perform network I/O. This keeps the results focused on library overhead. Provider latency would otherwise hide regressions in allocation count, schema compilation, stream accumulation, or tool scheduling.

## Compare a change

Record the same benchmarks before and after a change:

```console
go install golang.org/x/perf/cmd/benchstat@latest
git switch main
go test -run '^$' -bench . -benchmem -count 10 . > /tmp/before.txt
git switch my-branch
go test -run '^$' -bench . -benchmem -count 10 . > /tmp/after.txt
benchstat /tmp/before.txt /tmp/after.txt
```

Use at least ten samples. `benchstat` reports whether timing or allocation differences are statistically significant. Compare results on the same machine without other CPU-heavy work.

The parallel-tool benchmark reports `tools/op` in addition to time and allocations. It intentionally uses fast local tools. This makes scheduler and validation overhead visible instead of measuring artificial sleeps.
