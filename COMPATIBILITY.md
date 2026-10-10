# Compatibility policy

This project follows [Semantic Versioning](https://semver.org/).

## Version guarantees

The module is currently in the `v0` development series. A minor release may change a public API when the change is required for correctness or PydanticAI parity. Each breaking change must appear in `CHANGELOG.md` with a migration path. Patch releases do not intentionally break documented public APIs.

After `v1.0.0`, incompatible public API changes require a major release. Additive APIs and backward-compatible behavior changes may ship in a minor release. Compatible bug fixes may ship in a patch release.

Deprecated APIs remain available for at least one minor release when a compatibility shim is practical. A security issue, data-corruption bug, or upstream provider removal may require immediate removal. The changelog will state the reason.

## Public Go API

The compatibility promise covers exported identifiers in the `ai` package and its provider, embedding, MCP, retry, workspace, and evaluation packages. It also covers documented option precedence, lifecycle ownership, concurrency behavior, and errors intended for `errors.Is` or `errors.As`.

The following details are not stable contracts:

- Identifiers below an `internal/` directory.
- Exact error strings.
- Examples and test helpers not exposed from a library package.
- Undocumented provider metadata fields.
- Private OpenTelemetry attributes. Use documented `gen_ai.*`, `pydantic_ai.*`, and versioned instrumentation fields.

Adding a method to a required Go interface is a breaking change. New provider features should use optional interfaces discovered by type assertion. Adding a field to a public struct is compatible for keyed literals, but consumers should not use unkeyed literals for library structs.

## Go versions

The `go` directive in `go.mod` is the minimum supported Go version. CI tests that version and the next Go release.

Raising the minimum Go version may happen in a minor release before `v1.0.0`. After `v1.0.0`, it requires a major release unless the old Go version no longer receives upstream security fixes.

## Persisted messages

`MarshalMessages`, `UnmarshalMessages`, and `Conversation` JSON are the stable persistence boundary. The project preserves documented discriminators and continues to decode supported legacy aliases. New optional fields may appear in serialized messages without a major release.

Compatibility claims apply to the PydanticAI baseline recorded in `CHECKLIST.md` and `.upstream-sync.json`. A checklist item marked partial does not promise compatibility for the missing upstream variants.

Application-defined values in metadata or tool returns must remain valid for the application's own decoder. Provider-specific opaque values may stop replaying when a provider removes the corresponding API.

## Providers

Provider APIs change independently of this module. A compatible release may add model names, settings, metadata, finish reasons, or native tools. Removing a provider package or a documented setting follows the version guarantees above.

A provider can reject a model or feature that its remote API no longer supports. This is not a Go API compatibility break. The library should return an inspectable error instead of silently changing the request.

## Audited upstream additions

The port includes these applicable changes through the baseline recorded in `.upstream-sync.json`:

| Commit | Go behavior |
| --- | --- |
| `e6eee68add2178c45d0e5603ae57f52b248d0d7d` | Generic System One questions when options provide the meaning but instructions are absent; `systemone.Profile.RequiresInstructions` can explicitly disable the fallback |
| `5855737715205bafee933f842b413f7cb483ffa1` | Indexed keyword tool-search corpus, invalidated by names, descriptions, membership, and order; concurrency-safe ranking preserves undiscovered-first and corpus-order ties |
| `5274216031eeceef1523d78e835c34410e13b376` | Portable `ModelSettings.Cache`, `CacheConfig`, and `Caching`; provider-local precedence, retention snapping, stable-prefix-only caching, wide-turn boundaries, honored-tier outlook, and missing-configuration telemetry |
| `4e9d555f7` | Anthropic and Bedrock raise each catchment-breakpoint TTL to the longest TTL of a breakpoint after it (`promptcache.RaiseEarlierCacheTTLs`) |
| `be23a9774` | Bedrock `APIError.Hint` appended to the error message when the AWS response references the account's data retention mode |

Go keeps its existing tool-search limit contract: zero selects ten results and negative values are rejected. It does not adopt Python's negative slice behavior. Portable caching uses a typed configuration instead of Python's boolean/string/dictionary union. Nil is unset, an empty `CacheConfig` enables caching, and `CacheRetentionDisabled` disables it.

These harness-only commits are permanently excluded. No Go code is added for either:

- `d9a8a4bbee3f37d990a7a9e9bfdc71c8ddba4d33`: canonical session events in the gh-aw clai2 runner. This port does not own a harness runner.
- `2132206fb5d6b7cbe0abaef6e17a4825bddde409`: relocate `MCPReadOnlyNoToolsWarning` to keep Python harness imports light. There is no corresponding library behavior or Go import-time warning mechanism.
- `f55bfd2d7b190c68ac7f78371f5da89fa9b11164`: change Harness spend accrual around durable continuation retries. The core change only documents a private Python observer consumed by that version-pinned Harness package.

Upstream patches are reference data, not executable inputs. These additions do not introduce harness capability loading or self-extension.

## Reporting compatibility problems

Open an issue with the module version, Go version, provider, model name, and a minimal reproducer. Include serialized messages only after removing credentials and private content.
