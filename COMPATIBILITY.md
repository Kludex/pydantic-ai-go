# Compatibility policy

This project follows [Semantic Versioning](https://semver.org/).

## Version guarantees

The module is currently in the `v0` development series. A minor release may change a public API when the change is required for correctness or PydanticAI parity. Each breaking change must appear in `CHANGELOG.md` with a migration path. Patch releases do not intentionally break documented public APIs.

After `v1.0.0`, incompatible public API changes require a major release. Additive APIs and backward-compatible behavior changes may ship in a minor release. Compatible bug fixes may ship in a patch release.

Deprecated APIs remain available for at least one minor release when a compatibility shim is practical. A security issue, data-corruption bug, or upstream provider removal may require immediate removal. The changelog will state the reason.

## Public Go API

The compatibility promise covers exported identifiers in the `ai` package and its provider, embedding, MCP, retry, and evaluation packages. It also covers documented option precedence, lifecycle ownership, concurrency behavior, and errors intended for `errors.Is` or `errors.As`.

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

`MarshalMessages` and `UnmarshalMessages` are the stable persistence boundary. The project preserves documented discriminators and continues to decode supported legacy aliases. New optional fields may appear in serialized messages without a major release.

Compatibility claims apply to the PydanticAI baseline recorded in `CHECKLIST.md` and `.upstream-sync.json`. A checklist item marked partial does not promise compatibility for the missing upstream variants.

Application-defined values in metadata or tool returns must remain valid for the application's own decoder. Provider-specific opaque values may stop replaying when a provider removes the corresponding API.

## Providers

Provider APIs change independently of this module. A compatible release may add model names, settings, metadata, finish reasons, or native tools. Removing a provider package or a documented setting follows the version guarantees above.

A provider can reject a model or feature that its remote API no longer supports. This is not a Go API compatibility break. The library should return an inspectable error instead of silently changing the request.

## Reporting compatibility problems

Open an issue with the module version, Go version, provider, model name, and a minimal reproducer. Include serialized messages only after removing credentials and private content.
