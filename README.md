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

**Real, minimal implementation.** `src/lib.rs` builds prefill and decode
Execution Graphs through `magnetar:model-component-graph@1.0.0`'s
`graph-builder` Capability (vendored under `wit/`), using
`wit-bindgen`-generated bindings -- this crate never depends on
`magnetar-runtime`'s Rust types, only the WIT contract. Compiles to a real
Component via `cargo build --target wasm32-unknown-unknown --release`
followed by `wasm-tools component new`, and has been verified end to end
against a real `GraphBuilderCapability` (a real, correctly-shaped
`ExecutionGraph`, correct tied-embeddings weight aliasing, correct KV
cache metadata) -- see `magnetar-runtime`'s
`component_wasmtime::tests::wasmtime_engine_builds_a_real_qwen_prefill_graph_through_the_graph_builder_capability`.

Deliberately minimal, not general: the architecture dimensions (hidden
size, layer count, head counts, vocabulary size, tied embeddings, ...) are
hard-coded to match `magnetar-runtime`'s own E2E first-native fixture
exactly, since this contract has no architecture-configuration mechanism
yet. A Component for a real, configurable checkpoint -- reading weight
shapes and hyperparameters from an actual Model Artifact rather than
Rust constants -- is real follow-up work, tracked as
`reach-architecture-freeze-1` task group 12 (moving Qwen weight-name and
KV-name mappings, and eventually all Qwen-specific semantics, out of
`magnetar-runtime`'s generic Core). This Component is also not yet wired
into `magnetar-runtime`'s production generation path (task 11.5) -- today
it is proven correct in isolation, not yet load-bearing for real
inference.

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
