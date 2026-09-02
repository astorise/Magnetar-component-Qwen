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

**Empty template.** This crate is currently a bare `cargo new --lib`
scaffold: no Qwen graph logic, no WIT bindings, no Component Model exports
exist here yet. The current, real Qwen graph-building logic lives
temporarily inside `magnetar-runtime` itself (`qwen_model_component.rs` and
`first_native_runtime.rs`'s `qwen_prefill_graph`/`qwen_decode_graph`), as a
deliberate, tracked architectural debt -- not the target state. Migrating
that logic into this Component is task group 12 of Magnetar's
`reach-architecture-freeze-1` OpenSpec change, itself gated on task group
11 (`model-component-graph-contract`), neither of which is implemented yet.

## Governing contract

[`qwen-model-component`](https://github.com/astorise/Magnetar/blob/main/openspec/specs/qwen-model-component/spec.md)
in the main Magnetar repository's OpenSpec capability set defines this
Component's baseline requirements: architecture metadata, graph
requirements, operator scope, and tokenizer/cache compatibility. Once
`model-component-graph-contract` lands, this Component will also implement
that contract's WIT interface directly, rather than the Runtime
reconstructing Qwen's graph topology in Rust from attested Component
output as it does today.

## Relationship to magnetar-runtime

This Component is loaded and validated by the Runtime's Component Engine
(Wasmtime-backed on native targets, browser-hosted on `wasm32`) and never
runs with direct hardware, filesystem, or network access. It is pinned into
the main [Magnetar](https://github.com/astorise/Magnetar) repository as a
git submodule at `components/qwen`.
