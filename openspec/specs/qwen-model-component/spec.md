# qwen-model-component Specification

## Purpose
This specification defines the Qwen model Component baseline, metadata validation, graph requirements, operator scope, tokenizer/cache compatibility, and adapter targets.
## Requirements
### Requirement: Qwen Model Component Baseline

Magnetar SHALL define a Qwen-like Model Component baseline for decoder-only
inference architecture behavior.

#### Scenario: Resolve Qwen component

Given a Model Artifact declares Qwen-compatible architecture metadata

When Model Loading resolves architecture support

Then Runtime resolves a compatible Qwen Model Component or native
implementation.

---

### Requirement: Qwen Component Is Not Provider

Qwen Model Component SHALL NOT introduce or require a QwenProvider.

#### Scenario: Execute Qwen graph

Given Qwen Component produces a decode graph

When Runtime executes it

Then execution uses portable Operators and Runtime-selected Providers.

---

### Requirement: Qwen Component Is Not Kernel

Qwen Model Component SHALL NOT execute Kernels directly or expose native Kernel
requirements.

#### Scenario: Attention execution

Given Qwen graph contains attention Operator

When Runtime plans execution

Then Kernel Registry selects a compatible attention Kernel.

---

### Requirement: Qwen Baseline Uses First Operator Scope

Qwen baseline SHALL use required-now Operators for the first executable path.

#### Scenario: Unsupported flash attention

Given Qwen Component requires flash-attention

When first baseline validates it

Then Runtime rejects the path or requires a non-flash graph alternative.

---

### Requirement: Qwen Architecture Metadata Validation

Qwen Model Component SHALL validate decoder-only architecture metadata: every dimension non-zero, and a KV head count that divides the attention head count. The hidden size SHALL NOT be required to equal the attention head count times the head dimension, since a checkpoint may declare a head width apart from it; the query and output projections' expected shapes carry both widths.

#### Scenario: Invalid head configuration
- **WHEN** Qwen config validation runs on a zero head dimension, or a KV head count that does not divide the attention head count
- **THEN** Runtime returns qwen-config-invalid or qwen-architecture-unsupported

#### Scenario: A declared head width apart from the hidden size
- **WHEN** Qwen config validation runs on a hidden size that is not the attention head count times the head dimension
- **THEN** the configuration is accepted

### Requirement: Qwen Config Path Access Denied

Qwen Model Component SHALL not read arbitrary config file paths.

#### Scenario: Arbitrary config path

Given Qwen Component attempts to read `/models/config.json`

When Runtime checks authority

Then filesystem access is denied.

---

### Requirement: Qwen Model Artifact Compatibility

Qwen Model Component SHALL validate compatible Model Artifact metadata supplied
by Runtime.

#### Scenario: Untrusted artifact

Given Model Artifact is untrusted

When Qwen Component is compatible

Then Model Loading still fails before ready instance creation.

---

### Requirement: Qwen Tensor Inventory

Qwen Model Component SHALL define expected logical tensors for baseline
execution.

#### Scenario: Missing q projection

Given layer 0 `q_proj` tensor is missing

When tensor inventory validation runs

Then Runtime returns qwen-tensor-inventory-missing.

---

### Requirement: Qwen Tensor Shape Validation

Qwen Model Component SHALL validate required tensor shapes.

#### Scenario: Invalid lm_head shape

Given `lm_head` shape does not match vocabulary and hidden size metadata

When validation runs

Then Runtime returns qwen-tensor-shape-mismatch.

---

### Requirement: Qwen Target Modules

Qwen Model Component SHALL expose target modules for adapters where supported.

#### Scenario: LoRA target

Given LoRA adapter targets `q_proj`

When Adapter Loading validates the adapter

Then Runtime uses Qwen target module metadata.

---

### Requirement: Qwen Tokenizer Compatibility

Qwen Model Component SHALL declare tokenizer compatibility requirements.

#### Scenario: Vocabulary mismatch

Given tokenizer vocabulary size differs from model config

When compatibility validation runs

Then Runtime returns qwen-tokenizer-incompatible.

---

### Requirement: Qwen Generation Metadata

Qwen Model Component SHALL validate generation metadata where relevant without
owning Generation semantics.

#### Scenario: Invalid EOS reference

Given generation defaults reference token ID outside vocabulary

When validation runs

Then Runtime returns qwen-generation-metadata-invalid.

---

### Requirement: Qwen Prefill Graph

Qwen Model Component SHALL produce or describe a prefill graph for baseline
execution.

#### Scenario: Produce prefill graph

Given a ready Qwen Model Instance

When Generation requests prefill graph

Then Qwen Component returns graph metadata using portable Operators.

---

### Requirement: Qwen Decode Graph

Qwen Model Component SHALL produce or describe a decode graph for baseline
execution.

#### Scenario: Produce decode graph

Given decode operation has current token and optional KV cache

When Generation requests decode graph

Then Qwen Component returns graph metadata producing logits.

---

### Requirement: Qwen Decoder Layer Graph

Qwen decoder layer graph SHALL be expressible with RMSNorm, matmul, RoPE,
attention, residual-add, SiLU, mul, and matmul.

#### Scenario: Unfused layer graph

Given fused kernels are unavailable

When graph production runs

Then Qwen Component emits unfused required-now Operators.

---

### Requirement: Qwen Attention Metadata

Qwen attention metadata SHALL be explicit.

#### Scenario: Unsupported GQA

Given Qwen config requires grouped-query attention

And first baseline CPU attention cannot represent it correctly

When graph validation runs

Then Runtime rejects with qwen-attention-variant-unsupported.

---

### Requirement: Qwen RoPE Metadata

Qwen RoPE metadata SHALL be explicit and unsupported variants SHALL fail.

#### Scenario: Unsupported dynamic scaling

Given config requires unsupported RoPE scaling

When Qwen validation runs

Then Runtime returns qwen-RoPE-unsupported.

---

### Requirement: Qwen MLP Baseline

Qwen MLP SHALL be representable as matmul, SiLU, mul, and matmul in the first
baseline.

#### Scenario: Gated MLP

Given Qwen layer requires gated MLP

When graph production runs

Then graph contains gate projection, SiLU, up projection, mul, and down
projection.

---

### Requirement: Qwen RMSNorm Baseline

Qwen baseline SHALL use RMSNorm metadata and SHALL not require LayerNorm.

#### Scenario: LayerNorm required

Given model metadata requires LayerNorm

When Qwen baseline validates it

Then Runtime rejects as unsupported for this baseline.

---

### Requirement: Qwen Logits Projection

Qwen baseline SHALL produce logits using `lm_head` or tied embedding metadata.

#### Scenario: Logits output

Given hidden state after final norm

When logits projection runs

Then graph uses matmul to produce vocabulary logits.

---

### Requirement: Qwen KV Cache Metadata

Qwen Model Component SHALL declare KV cache metadata.

#### Scenario: KV metadata requested

Given Runtime prepares decode cache

When Qwen metadata is queried

Then layer count, KV head count, head dimension, dtype, layout preference, and
append behavior are available.

---

### Requirement: Qwen Prefix Cache Metadata

Qwen Model Component SHALL expose metadata needed for Prefix Cache
compatibility.

#### Scenario: Prefix fingerprint

Given Prefix Cache fingerprint is computed

When Qwen model context is active

Then Qwen architecture and component metadata may participate in compatibility.

---

### Requirement: Qwen Adapter Compatibility

Qwen Model Component SHALL expose adapter compatibility metadata where adapters
are supported.

#### Scenario: Unsupported adapter graph

Given LoRA activation requires graph overlay

And Qwen baseline does not support overlay graph

When activation is requested

Then Runtime rejects activation with qwen-adapter-unsupported or
activation-denied.

---

### Requirement: Qwen Quantization Rejection

Quantization metadata for a Qwen artifact SHALL be validated; the baseline MAY
reject quantized artifacts unless explicit dequantization or quantized
execution is implemented.

#### Scenario: Quantized artifact

Given Qwen artifact uses unsupported quantization

When loading validates it

Then Runtime returns qwen-quantization-unsupported.

---

### Requirement: Qwen Tensor Layout And DType Scope

Unsupported tensor layout or dtype SHALL fail or require explicit conversion;
the Qwen baseline SHOULD target contiguous layout and f32 compute through
Reference CPU.

#### Scenario: BF16 artifact

Given artifact stores BF16 tensors

And policy does not allow explicit conversion

When Qwen loading runs

Then Runtime returns qwen-dtype-unsupported.

---

### Requirement: Qwen Model Instance Metadata

Qwen Model Instance metadata SHALL be permitted to reference Qwen Component
identity and config compatibility metadata.

#### Scenario: Instance ready

Given Qwen Model Instance becomes ready

When Runtime reports redacted metadata

Then Qwen Component identity may be included.

---

### Requirement: Qwen Generation Boundary

Qwen Model Component SHALL not own Generation request lifecycle, Sampling, stop
conditions, streaming, or cancellation.

#### Scenario: Decode logits

Given Qwen decode graph produces logits

When next token is needed

Then Sampling Contract selects the token.

---

### Requirement: Qwen Reference CPU Coverage

Qwen baseline execution on Reference CPU SHALL require compatible CPU kernels
for all required operators.

#### Scenario: Missing attention kernel

Given Qwen graph contains attention

And Reference CPU lacks compatible attention kernel

When graph planning runs

Then Runtime returns qwen-Reference-CPU-coverage-missing or kernel missing.

---

### Requirement: Qwen Component Authority

Qwen Component authority SHALL be inference-scoped only.

#### Scenario: Network denied

Given Qwen Component attempts network access

When Runtime authorizes it

Then access is denied.

---

### Requirement: Qwen Versioning

Qwen Component baseline compatibility SHALL be versioned.

#### Scenario: Tensor contract mismatch

Given Qwen Component requires unsupported Tensor contract version

When Runtime validates it

Then Runtime rejects with qwen-component-unsupported-version.

---

### Requirement: Qwen Conformance

Qwen Model Component baseline SHALL have conformance fixtures for config,
tensor inventory, graph production, operator scope, tokenizer, KV metadata,
adapter metadata, unsupported quantization, authority, and raw handle safety.

#### Scenario: Invalid graph conformance

Given Qwen Component emits invalid graph

When conformance runs

Then conformance fails.

---

### Requirement: Qwen Error Categories

Qwen Component failures SHALL use structured error categories.

#### Scenario: Unsupported operator

Given Qwen graph requires unsupported operator

When planning runs

Then Runtime returns qwen-operator-unsupported.

---

### Requirement: Qwen Observability

Qwen Component observations SHALL NOT expose raw prompts, weights, adapter
tensors, KV cache contents, handles, or memory pointers; Runtime SHOULD emit
such observations in redacted form.

#### Scenario: Qwen graph produced

Given Qwen Component produces decode graph

When observability records it

Then Runtime emits redacted graph production metadata.

---

### Requirement: E2E Uses Qwen Baseline Component

E2E conformance SHALL use the Qwen-like Model Component baseline or compatible
native architecture implementation for the first fixture path.

#### Scenario: Qwen fixture

Given fixture declares Qwen-like architecture

When Model Loading resolves architecture

Then Qwen baseline component validates it.

---

### Requirement: E2E Validates Qwen Graph Production

E2E conformance SHALL validate Qwen prefill and decode graph production.

#### Scenario: Decode graph produced

Given fixture Model Instance is ready

When decode is requested

Then Qwen baseline component produces a portable decode graph that Runtime
validates.

---

### Requirement: Qwen Baseline Implemented After Operator Scope

Qwen baseline implementation SHALL depend on first operator scope metadata.

#### Scenario: Produce graph

Given Qwen baseline produces decode graph

When graph is validated

Then all operator requirements are checked against first scope.

---

### Requirement: Qwen Baseline Does Not Introduce QwenProvider

Implementation SHALL not add QwenProvider.

#### Scenario: Provider registry

Given Qwen baseline is implemented

When Provider registry is inspected

Then no QwenProvider appears.

### Requirement: Qwen Component Uses Normalized Config Metadata

Qwen Model Component SHALL use normalized config metadata from Model Artifact.

#### Scenario: Qwen hidden size

Given config normalization extracts hidden size

When Qwen validation runs

Then Qwen Component validates hidden size through normalized metadata.

---

### Requirement: Qwen Component Does Not Parse Raw Format Files

Qwen Model Component SHALL not parse raw safetensors, GGUF, tokenizer.json, or
local file structures directly.

#### Scenario: Raw file parse requested

Given Qwen Component attempts to parse raw GGUF file

When Runtime validates authority

Then access is denied or rejected as boundary violation.

### Requirement: Qwen WASM Component Is First-Native Authority
The first-native Qwen E2E path SHALL instantiate an executable Qwen WASM Component Artifact and use its validated output as the source of graph semantics.

#### Scenario: Component supplies graph
- **WHEN** first-native Qwen E2E builds its execution graph
- **THEN** the graph originates from the instantiated Qwen WASM Component and passes Runtime validation before planning.

#### Scenario: Missing component fails
- **WHEN** the Qwen Component Artifact is unavailable, invalid, untrusted, or cannot be instantiated
- **THEN** first-native Qwen E2E fails with a structured component or trust error and does not fall back to a Rust fixture graph.

### Requirement: Qwen Component Has No Provider Authority
The Qwen Component SHALL NOT receive Provider, Device, Kernel, stream, queue, memory pointer, or native resource handles.

#### Scenario: Provider selection stays runtime-owned
- **WHEN** Qwen Component describes model operations
- **THEN** Runtime remains responsible for Provider, Device, Kernel, memory, and KV resource decisions.

### Requirement: Production Qwen Component Is Configuration-Driven

The production Qwen Model Component SHALL derive all load-bearing architecture dimensions and graph structure from Runtime-authorized normalized model configuration rather than compile-time fixture constants.

This SHALL include hidden size, intermediate size, layer count, attention head count, KV head count, head dimension, vocabulary size, RMSNorm epsilon, RoPE parameters, and tied-embedding behavior where applicable.

#### Scenario: Two different compatible Qwen configs
- **WHEN** Model Instance A and Model Instance B declare different valid layer/head/intermediate dimensions
- **THEN** the same production Qwen Component implementation can produce configuration-correct graphs for each
- **AND** it does not require rebuilding the Component with different Rust constants.

---

### Requirement: Production Qwen Graph Shape Follows Loaded Artifact Configuration

The Qwen Component SHALL derive logical tensor names and expected tensor shapes from normalized configuration and SHALL validate them against the Runtime-recognized weight inventory.

#### Scenario: Layer tensor missing
- **WHEN** `num_hidden_layers` requires layer N
- **AND** a required layer-N tensor is absent from the validated Model Artifact
- **THEN** the model cannot become a ready executable Qwen Model Instance.

#### Scenario: Tensor shape mismatches config
- **WHEN** a required Qwen weight's parsed shape disagrees with the shape implied by normalized architecture config
- **THEN** compatibility/loading fails before Kernel dispatch.

---

### Requirement: Production Qwen Component Remains Provider And Source Agnostic

Making the Qwen Component configurable SHALL NOT grant it model-source, filesystem/network, Provider, or Device authority.

#### Scenario: Production graph on CUDA
- **WHEN** Runtime selects a CUDA Provider for a loaded Qwen Model Instance
- **THEN** the Qwen Component produces the same portable architecture graph semantics without receiving CUDA or Device identity.

### Requirement: Layered artifacts are validated per layer

When the architecture configuration declares a layer layout, tensor inventory and shape validation SHALL use each layer's own geometry:
- a layer reusing an earlier layer's K/V SHALL NOT require K or V projections;
- a layer deriving V from K SHALL NOT require a V projection;
- each layer's projections and MLP SHALL be sized with that layer's head width, KV head count and feed-forward width.

A layer computing its own K/V SHALL still require both projections.

#### Scenario: Shared-KV layer without K/V projections
- **WHEN** a layer's layout entry names a K/V source layer and the artifact has no K or V projection for it
- **THEN** inventory validation accepts the artifact

#### Scenario: Wider full-attention heads
- **WHEN** a layer's layout entry declares a wider head than the architecture-wide one
- **THEN** that layer's `q_proj` is expected with the wider width, and the architecture-wide width is refused
