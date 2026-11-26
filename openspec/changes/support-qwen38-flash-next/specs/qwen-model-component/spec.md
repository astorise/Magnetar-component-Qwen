## ADDED Requirements

### Requirement: Qwen3.5 hybrid graph

The Qwen Component SHALL recognize the Qwen3.5 text configuration and emit prefill and decode graphs with the following layers:
- each `linear_attention` layer is a `gated-delta-mixer` node with two recurrent slots, `conv` and `delta`, declared through `recurrent-resource`, `recurrent-input` and `recurrent-output`;
- each `full_attention` layer is the gated attention block with KV.

The layer kinds and geometry SHALL follow the configuration. The Component SHALL reject a configuration it cannot express, naming the field.

#### Scenario: Layer kinds follow the configuration

- **WHEN** the Component builds the graphs for a Qwen3.5 configuration whose `layer_types` alternate three `linear_attention` layers and one `full_attention` layer
- **THEN** every linear layer has exactly one mixer node with `conv` and `delta` recurrent bindings, every full layer has KV bindings, and the prefill and decode graphs have the same recurrent layouts

#### Scenario: An unsupported field

- **WHEN** a Qwen3.5 configuration requests a feature the Component does not implement
- **THEN** graph construction fails with an error naming that field, and no graph is returned

### Requirement: Qwen3.5 checkpoint qualification

Before Qwen3.5 is claimed as supported on a Provider, the project SHALL qualify Qwen3.5-0.8B-Base at a pinned revision and digest against the pinned Transformers source on that Provider. The last-position logits SHALL agree within the stated tolerance, and the first 16 greedy tokens SHALL be equal. The evidence SHALL be recorded in the change.

#### Scenario: Reference CPU qualification

- **WHEN** the qualification runs the pinned checkpoint on Reference CPU for the recorded prompt
- **THEN** the logits agree with Transformers within tolerance and the 16 greedy tokens are equal

### Requirement: Qwen3.8-Flash-Next is refused until qualified

The Qwen Component SHALL recognize `qwen4_exp` configurations. It SHALL refuse to build their graphs, with an explicit unsupported-architecture error, until its blocks are implemented with fixtures from a pinned upstream source: sparse attention, the gated residual, the n-gram embedding and routing. Qualification SHALL compare decode tokens per second, time to first token and 4K-token prefill against the recorded Strata reference on the same runner and protocol.

#### Scenario: Before the blocks are qualified

- **WHEN** a `qwen4_exp` configuration is given to the Component
- **THEN** it returns an unsupported-architecture error naming `qwen4_exp`, without emitting a partial graph
