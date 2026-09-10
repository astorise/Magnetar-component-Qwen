# magnetar-component-Qwen

## Purpose

A WebAssembly Component implementing the Qwen decoder-only model
architecture for the [Magnetar](https://github.com/astorise/Magnetar) local
AI Runtime. It supplies prefill/decode graph semantics for Qwen-compatible
Model Artifacts through the Runtime's Model Component contract, without
ever selecting a Provider, Device, or receiving native Runtime handles --
per Magnetar's architectural invariants (Components request Capabilities,
not Providers or Devices).

## Status

**Real, configurable, production implementation.** `src/lib.rs` builds
prefill and decode Execution Graphs through
`magnetar:model-component-graph@1.1.0`'s `graph-builder` and `model-config`
Capabilities (vendored under `wit/`), using `wit-bindgen`-generated
bindings -- this crate never depends on `magnetar-runtime`'s Rust types,
only the WIT contract. Compiles to a real Component via
`cargo build --target wasm32-unknown-unknown --release` followed by
`wasm-tools component new`, and has been verified end to end against a
real `GraphBuilderCapability`/`ModelConfigCapability` (a real,
correctly-shaped `ExecutionGraph`, correct tied-embeddings weight
aliasing, correct KV cache metadata) -- see `magnetar-runtime`'s
`component_wasmtime::tests::wasmtime_engine_builds_a_real_qwen_prefill_graph_through_the_graph_builder_capability`
and `first_native_runtime::tests::real_qwen_component_produces_a_config_correct_graph_for_a_genuinely_gqa_shaped_config`
(the latter proving a materially different, grouped-query-shaped
configuration -- distinct attention/KV head counts -- produces a correct
graph through this same compiled binary, not only the canonical tiny E2E
fixture dimensions), plus real CUDA hardware evidence in the main
repository's `integration-tests/cuda-first-native`.

`1.1.0` (`implement-production-qwen-model-loading` task group 8) made this
Component checkpoint-configurable: architecture dimensions (hidden size,
layer count, attention/KV head counts, head dimension, intermediate size,
vocabulary size, RMSNorm epsilon, RoPE parameters, tied embeddings) are no
longer compiled-in Rust constants -- every `build-prefill-graph`/
`build-decode-graph` call queries them fresh from `model-config`, so the
same compiled Component binary produces a configuration-correct graph for
any compatible Qwen checkpoint the Runtime has bound configuration for,
including a decoder stack whose layer count is read from the bound
configuration rather than fixed at compile time. It is wired into
`magnetar-runtime`'s production generation path (task 11.5 in the earlier
`reach-architecture-freeze-1` change) as the exclusive strict first-native
graph source.

## Governing contract

[`qwen-model-component`](https://github.com/astorise/Magnetar/blob/main/openspec/specs/qwen-model-component/spec.md)
in the main Magnetar repository's OpenSpec capability set defines this
Component's baseline requirements: architecture metadata, graph
requirements, operator scope, and tokenizer/cache compatibility.
[`model-component-graph-contract`](https://github.com/astorise/Magnetar/blob/main/openspec/changes/reach-architecture-freeze-1/specs/model-component-graph-contract/spec.md)
defines the `graph-builder` Capability this Component implements.

## Relationship to magnetar-runtime

This Component is loaded and validated by the Runtime's Component Engine
(Wasmtime-backed on native targets, browser-hosted on `wasm32`) and never
runs with direct hardware, filesystem, or network access. It is pinned into
the main [Magnetar](https://github.com/astorise/Magnetar) repository as a
git submodule at `components/qwen`.
