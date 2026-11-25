//! Production, configurable Qwen Model Component
//! (`magnetar:model-component-graph@1.1.0`,
//! `implement-production-qwen-model-loading` task group 8).
//!
//! Builds prefill and decode Execution Graphs for whatever Qwen-compatible
//! architecture the active Model Instance was loaded with, via the
//! Runtime-owned `graph-builder` and `model-config` Capabilities -- this
//! Component never touches `magnetar-runtime`'s Rust types at all, only the
//! WIT contract.
//!
//! Architecture dimensions (hidden size, layer count, head counts, head
//! dimension, vocabulary size, RMSNorm epsilon, RoPE parameters, tied
//! embeddings) are queried from `model-config` at the start of each
//! `build-prefill-graph`/`build-decode-graph` call, not compiled-in
//! constants: the same Component binary produces a configuration-correct
//! graph for any compatible Qwen checkpoint the Runtime has bound
//! configuration for (Decision 7 -- "Qwen weight names and shapes become
//! configuration-derived").

/// Standalone CPU reference for Qwen3.5's gated-delta recurrence.
pub mod qwen35_delta_reference;
/// Standalone CPU reference for one Qwen3.5 linear-attention mixer.
pub mod qwen35_mixer_reference;

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, KvCacheRole, KvInvolvement, TensorShape, add_node,
    add_node_outputs, alias_weight_edge, begin_graph, declare_input, finish_graph, kv_resource,
    recurrent_input, recurrent_output, recurrent_resource, weight_edge,
};
use magnetar::model_component_graph::model_config::{
    self, ArchitectureConfig, GateActivation, HyperConnectionConfig, LinearAttentionConfig,
    MoeConfig, NgramEmbeddingConfig, TokenSelectionConfig,
};

/// Qwen3.5's decoder forms (`model-config` 1.13.0, `support-qwen38-flash-next`),
/// read once per graph: all neutral for a Qwen2/2.5 checkpoint.
struct Variants {
    /// Every decoder RMSNorm scales by `norm_offset + weight`.
    norm_offset: f64,
    /// Each attention head's query projection is followed by its gate, and
    /// the query and key are normalized per head (Qwen3.5's attention).
    gated_attention: bool,
    /// The width of each head that rotates, when only part of it does.
    rotary_dimension: Option<u64>,
    /// The gated-delta linear-attention layers, if the decoder is hybrid.
    linear: Option<LinearAttentionConfig>,
    /// Qwen4-Exp's residual streams (`model-config` 1.14.0): each block
    /// reads a gated mix of them and writes back to every stream.
    hyper: Option<HyperConnectionConfig>,
    /// Qwen4-Exp's n-gram embeddings: the layers that add one before their
    /// first block, and its geometry.
    ngram: Option<NgramEmbeddingConfig>,
    /// The linear layers' output gate is `sigmoid` instead of `silu`.
    sigmoid_gate: bool,
    /// Qwen4-Exp's token indexer: every full-attention layer attends the
    /// tokens it selects (`model-config` 1.14.0).
    selection: Option<TokenSelectionConfig>,
}

impl Variants {
    fn read(config: &ArchitectureConfig) -> Self {
        let variants = model_config::decoder_variants();
        let linear = model_config::linear_attention();
        // Only Qwen3.5's decoder reads the layer layout, for its partial
        // rotary: a Qwen2 graph leaves a declared layout unread, so the
        // Runtime refuses it instead of running it without its windows.
        let hybrid = variants.is_some() || linear.is_some();
        // A Qwen3.5 layout: uniform full-attention layers whose rotary may
        // be partial, and nothing else.
        let layout = if hybrid {
            model_config::layer_layout()
        } else {
            None
        };
        let rotary_dimension = layout.and_then(|layout| {
            assert_eq!(
                layout.len(),
                config.num_hidden_layers as usize,
                "the layer layout must have one entry per layer"
            );
            let first = &layout[0];
            for layer in &layout {
                assert!(
                    layer.sliding_window.is_none()
                        && layer.kv_source_layer.is_none()
                        && !layer.value_from_key
                        && layer.head_dimension == config.head_dimension
                        && layer.num_key_value_heads == config.num_key_value_heads
                        && layer.intermediate_size == config.intermediate_size
                        && layer.rope_theta == config.rope_theta
                        && layer.partial_rotary_factor == first.partial_rotary_factor,
                    "the Qwen Component implements uniform full-attention layers only"
                );
            }
            first.partial_rotary_factor.map(|factor| {
                let dimension = (config.head_dimension as f64 * factor) as u64;
                assert!(
                    dimension > 0 && dimension.is_multiple_of(2),
                    "the rotary dimension must be positive and even"
                );
                dimension
            })
        });
        if let Some(linear) = &linear {
            assert_eq!(
                linear.layers.len(),
                config.num_hidden_layers as usize,
                "linear attention must name every layer"
            );
        }
        // Qwen4-Exp's forms: every declared one is read; the ones this
        // graph does not build yet are refused rather than left out.
        let hyper = model_config::hyper_connections();
        let ngram = model_config::ngram_embeddings();
        assert!(
            ngram.is_none() || hyper.is_some(),
            "n-gram embeddings are added to residual streams"
        );
        let selection = model_config::token_selection();
        let sigmoid_gate = matches!(
            model_config::linear_attention_gate(),
            Some(GateActivation::Sigmoid)
        );
        Self {
            norm_offset: variants.as_ref().map_or(0.0, |v| v.norm_weight_offset),
            gated_attention: variants.is_some_and(|v| v.attention_output_gate),
            rotary_dimension,
            linear,
            hyper,
            ngram,
            sigmoid_gate,
            selection,
        }
    }

    /// The KV layer holding `layer`'s index keys: the stores follow the
    /// decoder layers, one per full-attention layer in order (the
    /// Runtime's `kv_layer_count`).
    fn index_store(&self, config: &ArchitectureConfig, layer: u64) -> u64 {
        let earlier = (0..layer).filter(|&other| !self.is_linear(other)).count() as u64;
        u64::from(config.num_hidden_layers) + earlier
    }

    fn is_linear(&self, layer: u64) -> bool {
        self.linear
            .as_ref()
            .is_some_and(|linear| linear.layers[layer as usize])
    }

    /// An RMSNorm node's attributes.
    fn norm(&self, epsilon: f64) -> Vec<Attribute> {
        let mut attributes = vec![attr("epsilon", AttributeValue::FloatValue(epsilon))];
        if self.norm_offset != 0.0 {
            attributes.push(attr(
                "weight_offset",
                AttributeValue::FloatValue(self.norm_offset),
            ));
        }
        attributes
    }
}

const ROPE_POSITION_MODE: &str = "sequential";
const ROPE_ROTATION_PAIRING: &str = "half_split";

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
}

/// How this node may be split across Devices (`tensor_parallel`, see
/// the Runtime's `operator::TENSOR_PARALLEL_ATTRIBUTE`): `column` and
/// `row` matmuls, `local` elementwise nodes between them.
fn tensor_parallel(kind: &str) -> Attribute {
    attr("tensor_parallel", AttributeValue::Text(kind.to_string()))
}

fn attr(name: &str, value: AttributeValue) -> Attribute {
    Attribute {
        name: name.to_string(),
        value,
    }
}

/// Maps a per-layer weight to the canonical Model Artifact tensor name
/// `weight_edge` is called with. The Runtime's `graph-builder` Capability
/// only ever wraps whatever logical name it is given into an edge id
/// (`weight.{logical_name}`, generically) -- it has no Qwen-specific
/// knowledge of `self_attn`/`mlp` naming. Supplying the canonical Model
/// Artifact name directly here, rather than this Component's own
/// `layerN.suffix` shorthand, is what lets `magnetar-runtime`'s Model
/// Loading resolve weight edges by a single, generic prefix strip instead
/// of carrying a Qwen-specific suffix-mapping table itself.
fn weight_tensor_name(layer: u64, suffix: &str) -> String {
    match suffix {
        "input_norm" | "post_attn_norm" => format!("layers.{layer}.{suffix}"),
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_bias" | "k_bias" | "v_bias" | "q_norm"
        | "k_norm" | "index_q_proj" | "index_k_proj" | "index_q_norm" | "index_k_norm" => {
            format!("layers.{layer}.self_attn.{suffix}")
        }
        // Qwen3.5's gated-delta mixer, in checkpoint `[out, in]` order.
        "in_qkv" | "in_z" | "in_a" | "in_b" | "conv1d" | "a_log" | "dt_bias" | "norm" | "out" => {
            format!("layers.{layer}.linear_attn.{suffix}")
        }
        // Qwen4-Exp's gated residuals, in checkpoint `[out, in]` order.
        "attn_hyper.norm" | "attn_hyper.down" | "attn_hyper.up" | "attn_hyper.inject"
        | "mlp_hyper.norm" | "mlp_hyper.down" | "mlp_hyper.up" | "mlp_hyper.inject" => {
            format!("layers.{layer}.{suffix}")
        }
        // Qwen4-Exp's n-gram embedding: the hashed table and the weights
        // that gate its value into the streams.
        "ple.table" | "ple.key" | "ple.value" | "ple.norm_key" | "ple.norm_query"
        | "ple.norm_conv" | "ple.conv1d" => format!("layers.{layer}.{suffix}"),
        "gate_proj" | "up_proj" | "down_proj" => format!("layers.{layer}.mlp.{suffix}"),
        "router_proj" | "experts_gate_proj" | "experts_up_proj" | "experts_down_proj"
        | "shared_gate_proj" | "shared_up_proj" | "shared_down_proj" | "shared_expert_gate" => {
            format!("layers.{layer}.moe.{suffix}")
        }
        other => unreachable!("weight_tensor_name: unknown weight suffix '{other}'"),
    }
}

/// Builds one pre-norm decoder layer, matching
/// `magnetar_runtime::qwen_model_component::qwen_build_graph`'s per-layer
/// node sequence exactly (RMSNorm, QKV matmul, RoPE, attention, output
/// projection, residual-add, RMSNorm, gated MLP, residual-add), through
/// `graph-builder` calls instead of direct `ExecutionGraph` construction.
/// Every dimension comes from `config`, queried once per
/// `build-prefill-graph`/`build-decode-graph` call -- not a compiled-in
/// constant.
fn build_layer(
    config: &ArchitectureConfig,
    variants: &Variants,
    moe: Option<&MoeConfig>,
    layer: u64,
    residual_in: &str,
    token_ids: &str,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    if let Some(hyper) = &variants.hyper {
        return build_hyper_layer(
            config,
            variants,
            moe,
            hyper,
            layer,
            residual_in,
            token_ids,
            sequence_length,
            kv_append,
        );
    }
    let prefix = format!("layer{layer}");
    let hidden_size = config.hidden_size;

    let input_norm_weight = weight_edge(&weight_tensor_name(layer, "input_norm"));
    let normed = add_node(
        &format!("{prefix}.input_norm"),
        "rmsnorm",
        "Normalization",
        &variants.norm(config.rms_norm_eps),
        &[residual_in.to_string(), input_norm_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );
    let attn_proj = if variants.is_linear(layer) {
        build_linear_mixer(config, variants, layer, normed, sequence_length)
    } else {
        build_attention_block(config, variants, layer, normed, sequence_length, kv_append)
    };
    finish_layer(
        config,
        variants,
        moe,
        layer,
        residual_in,
        attn_proj,
        sequence_length,
    )
}

/// A full-attention layer's token mixer: the query, key and value
/// projections (their biases and Qwen3.5's per-head norms and gate), RoPE,
/// attention over the KV history and the output projection. `normed` is
/// the block's input, `[sequence_length, hidden_size]`.
fn build_attention_block(
    config: &ArchitectureConfig,
    variants: &Variants,
    layer: u64,
    normed: String,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    let prefix = format!("layer{layer}");
    let hidden_size = config.hidden_size;
    let attention_head_count = config.num_attention_heads as u64;
    let kv_head_count = config.num_key_value_heads as u64;
    let head_dimension = config.head_dimension;
    let q_dim = attention_head_count * head_dimension;
    let kv_dim = kv_head_count * head_dimension;

    // Real Qwen2/2.5 checkpoints declare an additive bias term for
    // q/k/v_proj (an architectural default, not `config.json`-driven --
    // see `ArchitectureConfig::attention-bias`'s WIT doc comment); o_proj
    // and every MLP/lm_head projection never carry one in this baseline.
    // Bias, when present, is added to the raw projection output *before*
    // RoPE/KV caching -- the cached/rotated value is the biased one, the
    // real Qwen2 semantics -- so a bias-add node's output (not the
    // matmul's) becomes `q`/`k`/`v` below, and the v-projection's KV
    // involvement attaches to whichever node is actually last.
    let q_weight = weight_edge(&weight_tensor_name(layer, "q_proj"));
    // Qwen3.5's query projection holds, per head, the query then its gate.
    let q_width = if variants.gated_attention {
        2 * q_dim
    } else {
        q_dim
    };
    let q = add_node(
        &format!("{prefix}.q_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed.clone(), q_weight],
        &shape(vec![sequence_length, q_width]),
        None,
    );
    let (q, gate) = if variants.gated_attention {
        let halves = add_node_outputs(
            &format!("{prefix}.q_gate_split"),
            "split",
            "Tensor",
            &[attr(
                "head_count",
                AttributeValue::Integer(attention_head_count as i64),
            )],
            &[q],
            &[
                shape(vec![sequence_length, q_dim]),
                shape(vec![sequence_length, q_dim]),
            ],
        );
        let q = per_head_norm(
            config,
            variants,
            &format!("{prefix}.q_norm"),
            halves[0].clone(),
            &weight_tensor_name(layer, "q_norm"),
            attention_head_count,
            sequence_length,
        );
        (q, Some(halves[1].clone()))
    } else {
        (q, None)
    };
    let q = if config.attention_bias {
        let q_bias = weight_edge(&weight_tensor_name(layer, "q_bias"));
        add_node(
            &format!("{prefix}.q_bias"),
            "add",
            "Tensor",
            &[],
            &[q, q_bias],
            &shape(vec![sequence_length, q_dim]),
            None,
        )
    } else {
        q
    };
    let k_weight = weight_edge(&weight_tensor_name(layer, "k_proj"));
    let k = add_node(
        &format!("{prefix}.k_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed.clone(), k_weight],
        &shape(vec![sequence_length, kv_dim]),
        None,
    );
    let k = if config.attention_bias {
        let k_bias = weight_edge(&weight_tensor_name(layer, "k_bias"));
        add_node(
            &format!("{prefix}.k_bias"),
            "add",
            "Tensor",
            &[],
            &[k, k_bias],
            &shape(vec![sequence_length, kv_dim]),
            None,
        )
    } else {
        k
    };
    let k = if variants.gated_attention {
        per_head_norm(
            config,
            variants,
            &format!("{prefix}.k_norm"),
            k,
            &weight_tensor_name(layer, "k_norm"),
            kv_head_count,
            sequence_length,
        )
    } else {
        k
    };
    let selected = variants.selection.as_ref().map(|selection| {
        build_token_selection(
            config,
            variants,
            selection,
            layer,
            normed.clone(),
            sequence_length,
            kv_append,
        )
    });
    let v_weight = weight_edge(&weight_tensor_name(layer, "v_proj"));
    let v_kv = kv_resource(layer, KvCacheRole::V);
    let v_kv_involvement = KvInvolvement {
        resource_id: v_kv,
        role: KvCacheRole::V,
        append: kv_append,
    };
    let v = add_node(
        &format!("{prefix}.v_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed, v_weight],
        &shape(vec![sequence_length, kv_dim]),
        if config.attention_bias {
            None
        } else {
            Some(&v_kv_involvement)
        },
    );
    let v = if config.attention_bias {
        let v_bias = weight_edge(&weight_tensor_name(layer, "v_bias"));
        add_node(
            &format!("{prefix}.v_bias"),
            "add",
            "Tensor",
            &[],
            &[v, v_bias],
            &shape(vec![sequence_length, kv_dim]),
            Some(&v_kv_involvement),
        )
    } else {
        v
    };

    // `head_count`: the number of independently-rotated head blocks this
    // row's columns divide into (`make-first-native-cuda-hot-path-device-
    // resident`'s `operator`/`operator-scope` deltas in the main Magnetar
    // repository) -- explicit graph data, matching
    // `magnetar_runtime::qwen_model_component::qwen_build_graph`'s own Q =
    // `attention_head_count` / K = `kv_head_count` split exactly, so a
    // grouped-query/multi-query configuration (`kv_head_count <
    // attention_head_count`) rotates each projection's own head width
    // correctly instead of the Runtime having to infer it from `rope_q`/
    // `rope_k`'s node id suffix.
    let rope_attrs = |offset: i64, head_count: u64| {
        vec![
            attr("base", AttributeValue::FloatValue(config.rope_theta)),
            // A partial rotary rotates the first `rotary_dimension` values
            // of each head, in half-split pairs within them.
            attr(
                "dimension",
                AttributeValue::Integer(variants.rotary_dimension.unwrap_or(head_dimension) as i64),
            ),
            attr(
                "position_mode",
                AttributeValue::Text(ROPE_POSITION_MODE.to_string()),
            ),
            // Hugging Face checkpoints rotate (i, i + dimension/2)
            // (`rotate_half`), not adjacent pairs; declared explicitly so
            // the Runtime never infers it (`fix-rope-rotation-pairing`).
            attr(
                "rotation_pairing",
                AttributeValue::Text(ROPE_ROTATION_PAIRING.to_string()),
            ),
            attr("position_offset", AttributeValue::Integer(offset)),
            attr("head_count", AttributeValue::Integer(head_count as i64)),
        ]
    };
    let q_rope = add_node(
        &format!("{prefix}.rope_q"),
        "rope",
        "PositionEncoding",
        &rope_attrs(0, attention_head_count),
        &[q],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    let k_kv = kv_resource(layer, KvCacheRole::K);
    let k_rope = add_node(
        &format!("{prefix}.rope_k"),
        "rope",
        "PositionEncoding",
        &rope_attrs(0, kv_head_count),
        &[k],
        &shape(vec![sequence_length, kv_dim]),
        Some(&KvInvolvement {
            resource_id: k_kv,
            role: KvCacheRole::K,
            append: kv_append,
        }),
    );

    let mut attention_attributes = vec![
        attr("causal", AttributeValue::Boolean(true)),
        attr(
            "head_count",
            AttributeValue::Integer(attention_head_count as i64),
        ),
        attr(
            "kv_head_count",
            AttributeValue::Integer(kv_head_count as i64),
        ),
        attr(
            "head_dimension",
            AttributeValue::Integer(head_dimension as i64),
        ),
        attr(
            "attention_mask_kind",
            AttributeValue::Text("causal".to_string()),
        ),
    ];
    let mut attention_inputs = vec![q_rope, k_rope, v];
    // The indexer's selection is the attention's fourth input.
    if let Some(selected) = selected {
        attention_attributes.push(attr("token_selection", AttributeValue::Boolean(true)));
        attention_inputs.push(selected);
    }
    let attn_out = add_node(
        &format!("{prefix}.attention"),
        "attention",
        "Attention",
        &attention_attributes,
        &attention_inputs,
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    // Qwen3.5 multiplies the attention output by its gate's sigmoid.
    let attn_out = match gate {
        Some(gate) => {
            let gate = add_node(
                &format!("{prefix}.gate_sigmoid"),
                "activation",
                "Activation",
                &[attr("kind", AttributeValue::Text("sigmoid".to_string()))],
                &[gate],
                &shape(vec![sequence_length, q_dim]),
                None,
            );
            add_node(
                &format!("{prefix}.gated"),
                "mul",
                "Tensor",
                &[],
                &[attn_out, gate],
                &shape(vec![sequence_length, q_dim]),
                None,
            )
        }
        None => attn_out,
    };

    let o_weight = weight_edge(&weight_tensor_name(layer, "o_proj"));
    add_node(
        &format!("{prefix}.o_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[attn_out, o_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    )
}

/// One gated residual over the streams `residual` (`[T, streams *
/// hidden]`): the block's input `[T, hidden]` and, with `inject`, the
/// weights the block's output is written back with `[T, streams]`.
#[allow(clippy::too_many_arguments)]
fn gated_residual(
    config: &ArchitectureConfig,
    variants: &Variants,
    hyper: &HyperConnectionConfig,
    node_id: &str,
    weights: &str,
    residual: &str,
    sequence_length: u64,
    inject: bool,
) -> (String, Option<String>) {
    let weight = |part: &str| weight_edge(&format!("{weights}.{part}"));
    let mut inputs = vec![
        residual.to_string(),
        weight("norm"),
        weight("down"),
        weight("up"),
    ];
    let mut outputs = vec![shape(vec![sequence_length, config.hidden_size])];
    if inject {
        inputs.push(weight("inject"));
        outputs.push(shape(vec![sequence_length, u64::from(hyper.streams)]));
    }
    let attributes = [
        attr(
            "hc_count",
            AttributeValue::Integer(i64::from(hyper.streams)),
        ),
        attr("epsilon", AttributeValue::FloatValue(config.rms_norm_eps)),
        attr(
            "weight_offset",
            AttributeValue::FloatValue(variants.norm_offset),
        ),
    ];
    let operator = if inject {
        "gated-residual"
    } else {
        "gated-residual-final"
    };
    let mut produced = add_node_outputs(
        node_id,
        operator,
        "LinearAlgebra",
        &attributes,
        &inputs,
        &outputs,
    )
    .into_iter();
    (produced.next().expect("a mixed input"), produced.next())
}

/// What a layer's n-gram embedding adds to the residual streams: the hashed
/// embedding of each token's n-grams (its two earlier tokens are a recurrent
/// state, held relative to the end token so that a new sequence's zeros are
/// the end-token fill), gated by the streams and followed by a dilated
/// depthwise convolution (its history is a recurrent state too). Returns
/// the sum with `residual`.
#[allow(clippy::too_many_arguments)]
fn build_ngram_embedding(
    config: &ArchitectureConfig,
    variants: &Variants,
    hyper: &HyperConnectionConfig,
    ngram: &NgramEmbeddingConfig,
    layer: u64,
    residual: &str,
    token_ids: &str,
    sequence_length: u64,
) -> String {
    let prefix = format!("layer{layer}");
    let integer = |name: &str, value: u64| attr(name, AttributeValue::Integer(value as i64));
    let width = u64::from(hyper.streams) * config.hidden_size;
    let context_len = u64::from(ngram.ngram_size) - 1;
    let state_len = u64::from(ngram.conv_kernel - 1) * u64::from(ngram.ngram_size);
    // Which n-gram embedding this is among the layers that have one.
    let layer_index = ngram.layers[..layer as usize]
        .iter()
        .filter(|has| **has)
        .count() as u64;
    let context = recurrent_resource(layer, "ngram_context", &shape(vec![context_len]));
    let history = recurrent_resource(layer, "ngram_history", &shape(vec![width, state_len]));
    let embedded = add_node_outputs(
        &format!("{prefix}.ngram_embedding"),
        "ngram-embedding",
        "Tensor",
        &[
            integer("ngram_size", u64::from(ngram.ngram_size)),
            integer("heads_per_ngram", u64::from(ngram.heads_per_ngram)),
            integer("vocab_size", config.vocabulary_size),
            integer("vocab_base", ngram.vocabulary_base),
            integer("layer_index", layer_index),
            integer("seed", ngram.seed),
            integer("eos_token_id", u64::from(ngram.eos_token)),
        ],
        &[
            token_ids.to_string(),
            recurrent_input(&context),
            weight_edge(&weight_tensor_name(layer, "ple.table")),
        ],
        &[
            shape(vec![sequence_length, ngram.embedding_dim]),
            shape(vec![context_len]),
        ],
    );
    recurrent_output(&context, &embedded[1]);
    let mut inputs = vec![residual.to_string(), embedded[0].clone()];
    for weight in [
        "ple.key",
        "ple.value",
        "ple.norm_key",
        "ple.norm_query",
        "ple.norm_conv",
        "ple.conv1d",
    ] {
        inputs.push(weight_edge(&weight_tensor_name(layer, weight)));
    }
    inputs.push(recurrent_input(&history));
    let added = add_node_outputs(
        &format!("{prefix}.ngram_ple"),
        "ngram-ple",
        "LinearAlgebra",
        &[
            integer("hc_count", u64::from(hyper.streams)),
            attr("epsilon", AttributeValue::FloatValue(config.rms_norm_eps)),
            attr(
                "weight_offset",
                AttributeValue::FloatValue(variants.norm_offset),
            ),
            integer("conv_kernel", u64::from(ngram.conv_kernel)),
            integer("conv_dilation", u64::from(ngram.ngram_size)),
        ],
        &inputs,
        &[
            shape(vec![sequence_length, width]),
            shape(vec![width, state_len]),
        ],
    );
    recurrent_output(&history, &added[1]);
    add_node(
        &format!("{prefix}.ngram_add"),
        "residual-add",
        "Tensor",
        &[],
        &[residual.to_string(), added[0].clone()],
        &shape(vec![sequence_length, width]),
        None,
    )
}

/// One Qwen4-Exp decoder layer: the token mixer (gated-delta or attention)
/// and the mixture of experts, each reading a gated mix of the residual
/// streams and written back into every stream. No decoder norms.
#[allow(clippy::too_many_arguments)]
fn build_hyper_layer(
    config: &ArchitectureConfig,
    variants: &Variants,
    moe: Option<&MoeConfig>,
    hyper: &HyperConnectionConfig,
    layer: u64,
    residual_in: &str,
    token_ids: &str,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    let prefix = format!("layer{layer}");
    let moe = moe.expect("a Qwen4-Exp layer is a mixture of experts");
    // An n-gram embedding is added to every stream before the first block.
    let embedded;
    let residual_in = match variants
        .ngram
        .as_ref()
        .filter(|ngram| ngram.layers[layer as usize])
    {
        Some(ngram) => {
            embedded = build_ngram_embedding(
                config,
                variants,
                hyper,
                ngram,
                layer,
                residual_in,
                token_ids,
                sequence_length,
            );
            embedded.as_str()
        }
        None => residual_in,
    };
    let streams_shape = shape(vec![
        sequence_length,
        u64::from(hyper.streams) * config.hidden_size,
    ]);
    let write_back = |name: &str, residual: &str, block: String, write: Option<String>| {
        add_node(
            &format!("{prefix}.{name}_inject"),
            "gated-residual-inject",
            "Tensor",
            &[],
            &[
                residual.to_string(),
                block,
                write.expect("a block's gated residual writes back"),
            ],
            &streams_shape,
            None,
        )
    };
    let (mixed, write) = gated_residual(
        config,
        variants,
        hyper,
        &format!("{prefix}.attn_hyper"),
        &format!("layers.{layer}.attn_hyper"),
        residual_in,
        sequence_length,
        true,
    );
    let mixer = if variants.is_linear(layer) {
        build_linear_mixer(config, variants, layer, mixed, sequence_length)
    } else {
        build_attention_block(config, variants, layer, mixed, sequence_length, kv_append)
    };
    let residual = write_back("attn", residual_in, mixer, write);
    let (mixed, write) = gated_residual(
        config,
        variants,
        hyper,
        &format!("{prefix}.mlp_hyper"),
        &format!("layers.{layer}.mlp_hyper"),
        &residual,
        sequence_length,
        true,
    );
    let experts = moe_mlp(
        &prefix,
        layer,
        moe,
        mixed,
        sequence_length,
        config.hidden_size,
    );
    write_back("mlp", &residual, experts, write)
}

/// The rest of a decoder layer after its token mixer (attention or the
/// linear mixer): residual, post-mixer norm, MLP, residual.
fn finish_layer(
    config: &ArchitectureConfig,
    variants: &Variants,
    moe: Option<&MoeConfig>,
    layer: u64,
    residual_in: &str,
    attn_proj: String,
    sequence_length: u64,
) -> String {
    let prefix = format!("layer{layer}");
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let post_attn = add_node(
        &format!("{prefix}.residual1"),
        "residual-add",
        "Tensor",
        &[],
        &[residual_in.to_string(), attn_proj],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let post_attn_norm_weight = weight_edge(&weight_tensor_name(layer, "post_attn_norm"));
    let mlp_normed = add_node(
        &format!("{prefix}.post_attn_norm"),
        "rmsnorm",
        "Normalization",
        &variants.norm(config.rms_norm_eps),
        &[post_attn.clone(), post_attn_norm_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let mlp_out = match moe {
        None => dense_mlp(
            &prefix,
            layer,
            mlp_normed,
            sequence_length,
            hidden_size,
            intermediate_size,
        ),
        Some(moe) => moe_mlp(
            &prefix,
            layer,
            moe,
            mlp_normed,
            sequence_length,
            hidden_size,
        ),
    };

    add_node(
        &format!("{prefix}.residual2"),
        "residual-add",
        "Tensor",
        &[],
        &[post_attn, mlp_out],
        &shape(vec![sequence_length, hidden_size]),
        None,
    )
}

/// Qwen3.5's per-head query/key RMSNorm (zero-centered when the decoder's
/// norms are).
fn per_head_norm(
    config: &ArchitectureConfig,
    variants: &Variants,
    node_id: &str,
    input: String,
    weight_name: &str,
    head_count: u64,
    sequence_length: u64,
) -> String {
    head_norm(
        config,
        variants,
        node_id,
        input,
        weight_name,
        head_count,
        config.head_dimension,
        sequence_length,
    )
}

/// An RMSNorm over each of `head_count` heads of `head_dimension` values.
#[allow(clippy::too_many_arguments)]
fn head_norm(
    config: &ArchitectureConfig,
    variants: &Variants,
    node_id: &str,
    input: String,
    weight_name: &str,
    head_count: u64,
    head_dimension: u64,
    sequence_length: u64,
) -> String {
    let mut attributes = variants.norm(config.rms_norm_eps);
    attributes.push(attr(
        "head_count",
        AttributeValue::Integer(head_count as i64),
    ));
    attributes.push(attr(
        "head_dimension",
        AttributeValue::Integer(head_dimension as i64),
    ));
    add_node(
        node_id,
        "rmsnorm",
        "Normalization",
        &attributes,
        &[input, weight_edge(weight_name)],
        &shape(vec![sequence_length, head_count * head_dimension]),
        None,
    )
}

/// Qwen4-Exp's token indexer for one full-attention layer
/// (`Qwen4ExpTextQSAIndexer`): queries from the layer's input are normalized
/// per head and rotated, the raw key of each token joins the layer's index
/// store, and `qsa-select` returns, per query, the positions it attends.
fn build_token_selection(
    config: &ArchitectureConfig,
    variants: &Variants,
    selection: &TokenSelectionConfig,
    layer: u64,
    normed: String,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    assert_eq!(selection.key_heads, 1, "the indexer has one key head");
    let prefix = format!("layer{layer}.indexer");
    let heads = u64::from(selection.heads);
    let dim = selection.head_dimension;
    let rotary = variants.rotary_dimension.unwrap_or(config.head_dimension);
    assert!(
        rotary <= dim && rotary.is_multiple_of(2),
        "the indexer rotates the first {rotary} values of a {dim}-wide head"
    );
    let q = add_node(
        &format!("{prefix}.q_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[
            normed.clone(),
            weight_edge(&weight_tensor_name(layer, "index_q_proj")),
        ],
        &shape(vec![sequence_length, heads * dim]),
        None,
    );
    let q = head_norm(
        config,
        variants,
        &format!("{prefix}.q_norm"),
        q,
        &weight_tensor_name(layer, "index_q_norm"),
        heads,
        dim,
        sequence_length,
    );
    let q = add_node(
        &format!("{prefix}.rope_q"),
        "rope",
        "PositionEncoding",
        &[
            attr("base", AttributeValue::FloatValue(config.rope_theta)),
            attr("dimension", AttributeValue::Integer(rotary as i64)),
            attr(
                "position_mode",
                AttributeValue::Text(ROPE_POSITION_MODE.to_string()),
            ),
            attr(
                "rotation_pairing",
                AttributeValue::Text(ROPE_ROTATION_PAIRING.to_string()),
            ),
            attr("position_offset", AttributeValue::Integer(0)),
            attr("head_count", AttributeValue::Integer(heads as i64)),
        ],
        &[q],
        &shape(vec![sequence_length, heads * dim]),
        None,
    );
    // The raw keys of every token so far: the cache holds the history, and
    // every consumer of this edge reads all of it.
    let keys = add_node(
        &format!("{prefix}.k_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[
            normed,
            weight_edge(&weight_tensor_name(layer, "index_k_proj")),
        ],
        &shape(vec![sequence_length, dim]),
        Some(&KvInvolvement {
            resource_id: kv_resource(variants.index_store(config, layer), KvCacheRole::K),
            role: KvCacheRole::K,
            append: kv_append,
        }),
    );
    add_node(
        &format!("{prefix}.select"),
        "qsa-select",
        "Tensor",
        &[
            attr("head_count", AttributeValue::Integer(heads as i64)),
            attr("head_dim", AttributeValue::Integer(dim as i64)),
            attr(
                "compress_ratio",
                AttributeValue::Integer(i64::from(selection.compress_ratio)),
            ),
            attr(
                "budget",
                AttributeValue::Integer(i64::from(selection.budget)),
            ),
            attr("rotary_dim", AttributeValue::Integer(rotary as i64)),
            attr("epsilon", AttributeValue::FloatValue(config.rms_norm_eps)),
            attr("rope_base", AttributeValue::FloatValue(config.rope_theta)),
            attr(
                "weight_offset",
                AttributeValue::FloatValue(variants.norm_offset),
            ),
        ],
        &[
            q,
            keys,
            weight_edge(&weight_tensor_name(layer, "index_k_norm")),
        ],
        &shape(vec![
            sequence_length,
            u64::from(selection.budget) + u64::from(selection.compress_ratio) - 1,
        ]),
        None,
    )
}

/// One Qwen3.5 gated-delta linear-attention mixer: a `gated-delta-mixer`
/// node whose convolution (`conv`) and recurrent (`delta`) states are
/// recurrent resources the Runtime keeps between steps. It has no KV.
fn build_linear_mixer(
    config: &ArchitectureConfig,
    variants: &Variants,
    layer: u64,
    normed: String,
    sequence_length: u64,
) -> String {
    let linear = variants
        .linear
        .as_ref()
        .expect("a linear layer has a config");
    let keys = u64::from(linear.key_heads) * linear.key_head_dimension;
    let value_heads = u64::from(linear.value_heads);
    let values = value_heads * linear.value_head_dimension;
    let channels = 2 * keys + values;
    let conv_shape = shape(vec![channels, u64::from(linear.conv_kernel)]);
    let delta_shape = shape(vec![
        value_heads,
        linear.key_head_dimension,
        linear.value_head_dimension,
    ]);
    let conv = recurrent_resource(layer, "conv", &conv_shape);
    let delta = recurrent_resource(layer, "delta", &delta_shape);
    let mut inputs = vec![normed, recurrent_input(&conv), recurrent_input(&delta)];
    for weight in [
        "in_qkv", "in_z", "in_a", "in_b", "conv1d", "a_log", "dt_bias", "norm", "out",
    ] {
        inputs.push(weight_edge(&weight_tensor_name(layer, weight)));
    }
    let integer = |name: &str, value: u64| attr(name, AttributeValue::Integer(value as i64));
    let mut attributes = vec![
        integer("key_heads", u64::from(linear.key_heads)),
        integer("value_heads", value_heads),
        integer("key_dim", linear.key_head_dimension),
        integer("value_dim", linear.value_head_dimension),
        integer("conv_kernel", u64::from(linear.conv_kernel)),
        attr("epsilon", AttributeValue::FloatValue(config.rms_norm_eps)),
    ];
    // Qwen4-Exp's sigmoid output gate; `silu` is the operator's default.
    if variants.sigmoid_gate {
        attributes.push(attr(
            "output_gate_type",
            AttributeValue::Text("sigmoid".to_string()),
        ));
    }
    let outputs = add_node_outputs(
        &format!("layer{layer}.linear_attn"),
        "gated-delta-mixer",
        "LinearAlgebra",
        &attributes,
        &inputs,
        &[
            shape(vec![sequence_length, config.hidden_size]),
            conv_shape,
            delta_shape,
        ],
    );
    recurrent_output(&conv, &outputs[1]);
    recurrent_output(&delta, &outputs[2]);
    outputs[0].clone()
}

/// The dense gated MLP: `down(silu(gate(h)) * up(h))`, declared
/// tensor-parallel: `gate`/`up` by output columns, `silu`/`mul` on each
/// share, `down` by input rows (partial sums the Runtime's Provider reduces).
fn dense_mlp(
    prefix: &str,
    layer: u64,
    mlp_normed: String,
    sequence_length: u64,
    hidden_size: u64,
    intermediate_size: u64,
) -> String {
    let gate_weight = weight_edge(&weight_tensor_name(layer, "gate_proj"));
    let gate = add_node(
        &format!("{prefix}.gate_proj"),
        "matmul",
        "LinearAlgebra",
        &[tensor_parallel("column")],
        &[mlp_normed.clone(), gate_weight],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let up_weight = weight_edge(&weight_tensor_name(layer, "up_proj"));
    let up = add_node(
        &format!("{prefix}.up_proj"),
        "matmul",
        "LinearAlgebra",
        &[tensor_parallel("column")],
        &[mlp_normed, up_weight],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let activated = add_node(
        &format!("{prefix}.silu"),
        "silu",
        "Activation",
        &[tensor_parallel("local")],
        &[gate],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let mlp_hidden = add_node(
        &format!("{prefix}.mul"),
        "mul",
        "Tensor",
        &[tensor_parallel("local")],
        &[activated, up],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let down_weight = weight_edge(&weight_tensor_name(layer, "down_proj"));
    add_node(
        &format!("{prefix}.down_proj"),
        "matmul",
        "LinearAlgebra",
        &[tensor_parallel("row")],
        &[mlp_hidden, down_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    )
}

/// A routed mixture of experts (`execute-and-cache-moe-experts` F02):
/// `moe-experts(h, matmul(h, router), experts)`, plus, when the checkpoint
/// has one, the always-active shared expert scaled by
/// `sigmoid(matmul(h, shared_expert_gate))`.
fn moe_mlp(
    prefix: &str,
    layer: u64,
    moe: &MoeConfig,
    mlp_normed: String,
    sequence_length: u64,
    hidden_size: u64,
) -> String {
    let rows = |cols: u64| shape(vec![sequence_length, cols]);
    let matmul = |name: &str, input: &str, weight: &str, cols: u64, parallel: Option<&str>| {
        let attributes: Vec<Attribute> = parallel.map(tensor_parallel).into_iter().collect();
        add_node(
            &format!("{prefix}.{name}"),
            "matmul",
            "LinearAlgebra",
            &attributes,
            &[
                input.to_string(),
                weight_edge(&weight_tensor_name(layer, weight)),
            ],
            &rows(cols),
            None,
        )
    };
    // Tensor parallel: the router is replicated (every Device routes over
    // every expert), the routed experts split whole across Devices, the
    // shared expert like a dense MLP.
    let logits = matmul(
        "router",
        &mlp_normed,
        "router_proj",
        u64::from(moe.experts),
        None,
    );
    let routed = add_node(
        &format!("{prefix}.experts"),
        "moe-experts",
        "LinearAlgebra",
        &[
            attr("experts", AttributeValue::Integer(i64::from(moe.experts))),
            attr(
                "top_k",
                AttributeValue::Integer(i64::from(moe.experts_per_token)),
            ),
            attr(
                "normalize_top_k",
                AttributeValue::Boolean(moe.normalize_top_k),
            ),
            tensor_parallel("experts"),
        ],
        &[
            mlp_normed.clone(),
            logits,
            weight_edge(&weight_tensor_name(layer, "experts_gate_proj")),
            weight_edge(&weight_tensor_name(layer, "experts_up_proj")),
            weight_edge(&weight_tensor_name(layer, "experts_down_proj")),
        ],
        &rows(hidden_size),
        None,
    );
    let Some(shared_size) = moe.shared_expert_intermediate_size else {
        return routed;
    };
    let gate = matmul(
        "shared_gate_proj",
        &mlp_normed,
        "shared_gate_proj",
        shared_size,
        Some("column"),
    );
    let up = matmul(
        "shared_up_proj",
        &mlp_normed,
        "shared_up_proj",
        shared_size,
        Some("column"),
    );
    let activated = add_node(
        &format!("{prefix}.shared_silu"),
        "silu",
        "Activation",
        &[tensor_parallel("local")],
        &[gate],
        &rows(shared_size),
        None,
    );
    let hidden = add_node(
        &format!("{prefix}.shared_mul"),
        "mul",
        "Tensor",
        &[tensor_parallel("local")],
        &[activated, up],
        &rows(shared_size),
        None,
    );
    let shared = matmul(
        "shared_down_proj",
        &hidden,
        "shared_down_proj",
        hidden_size,
        Some("row"),
    );
    let gate_logit = matmul(
        "shared_expert_gate",
        &mlp_normed,
        "shared_expert_gate",
        1,
        None,
    );
    let scale = add_node(
        &format!("{prefix}.shared_expert_scale"),
        "activation",
        "Activation",
        &[attr("kind", AttributeValue::Text("sigmoid".to_string()))],
        &[gate_logit],
        &rows(1),
        None,
    );
    let scaled = add_node(
        &format!("{prefix}.shared_expert_scaled"),
        "row-scale",
        "Tensor",
        &[],
        &[shared, scale],
        &rows(hidden_size),
        None,
    );
    add_node(
        &format!("{prefix}.moe_out"),
        "residual-add",
        "Tensor",
        &[],
        &[routed, scaled],
        &rows(hidden_size),
        None,
    )
}

/// Builds one *segment* of the full decoder stack's graph -- real decoder
/// layers `[start_layer, end_layer)` only (`add-real-multi-device-model-
/// instance-placement`). `build_graph` (below) is now exactly
/// `build_graph_segment(..., 0, config.num_hidden_layers)` -- source-
/// identical, not merely output-equivalent, so the existing, already-
/// verified full-graph behavior cannot silently drift from this segment
/// path.
///
/// `start_layer == 0` begins from a declared `token_ids` input with the
/// real token embedding lookup, exactly as the full graph always has.
/// `start_layer > 0` begins from a declared `hidden_states_in` input of
/// shape `[sequence_length, hidden_size]` instead -- a prior segment's
/// own real output, for a caller to have moved here explicitly (this
/// Component never assumes how; that is Runtime-side, real
/// `multi-device-placement` cross-Device movement).
///
/// `end_layer == config.num_hidden_layers` ends with the real
/// final-norm/lm-head projection, exactly as the full graph always has.
/// `end_layer < config.num_hidden_layers` ends with the raw post-layer
/// hidden state as this graph's own declared output instead, for a
/// subsequent segment to consume as its own `hidden_states_in`.
fn build_graph_segment(
    phase: GraphPhase,
    sequence_length: u64,
    position_offset: u64,
    kv_append: bool,
    start_layer: u32,
    end_layer: u32,
) -> String {
    match model_config::hidden_activation().as_deref() {
        Some("silu") => {}
        Some(other) => panic!("the Qwen Component does not support activation {other:?}"),
        None => panic!("the Qwen Component requires a known checkpoint activation"),
    }
    let config = model_config::model_config();
    // A scheme this Component does not apply must not be ignored.
    assert!(
        model_config::rope_scaling().is_none(),
        "the Qwen Component does not apply a per-frequency RoPE scaling"
    );
    let moe = model_config::mixture_of_experts();
    let variants = Variants::read(&config);
    begin_graph(phase, sequence_length, position_offset);

    // The residual's width: one stream, or the streams side by side.
    let streams = variants
        .hyper
        .as_ref()
        .map_or(1, |hyper| u64::from(hyper.streams));
    if variants.hyper.is_some() {
        // A segment would hand the streams to the next one; not built yet.
        assert!(
            start_layer == 0 && end_layer == config.num_hidden_layers,
            "the Qwen Component builds Qwen4-Exp's residual streams only as a whole graph"
        );
    }
    let mut token_ids = String::new();
    let mut hidden = if start_layer == 0 {
        token_ids = declare_input("token_ids", &shape(vec![sequence_length]));
        let token_embedding = weight_edge("token_embedding");
        let embedded = add_node(
            "embedding",
            "embedding",
            "Tensor",
            &[],
            &[token_ids.clone(), token_embedding],
            &shape(vec![sequence_length, config.hidden_size]),
            None,
        );
        match &variants.hyper {
            Some(hyper) => add_node(
                "repeat_streams",
                "repeat-features",
                "Tensor",
                &[attr(
                    "count",
                    AttributeValue::Integer(i64::from(hyper.streams)),
                )],
                &[embedded],
                &shape(vec![sequence_length, streams * config.hidden_size]),
                None,
            ),
            None => embedded,
        }
    } else {
        declare_input(
            "hidden_states_in",
            &shape(vec![sequence_length, config.hidden_size]),
        )
    };

    for layer in start_layer as u64..end_layer as u64 {
        hidden = build_layer(
            &config,
            &variants,
            moe.as_ref(),
            layer,
            &hidden,
            &token_ids,
            sequence_length,
            kv_append,
        );
    }

    if end_layer == config.num_hidden_layers {
        let hidden_final = match &variants.hyper {
            // The final mix of the streams feeds the LM head directly.
            Some(hyper) => {
                gated_residual(
                    &config,
                    &variants,
                    hyper,
                    "final_hyper",
                    "final_hyper",
                    &hidden,
                    sequence_length,
                    false,
                )
                .0
            }
            None => {
                let final_norm_weight = weight_edge("final_norm");
                add_node(
                    "final_norm",
                    "rmsnorm",
                    "Normalization",
                    &variants.norm(config.rms_norm_eps),
                    &[hidden, final_norm_weight],
                    &shape(vec![sequence_length, config.hidden_size]),
                    None,
                )
            }
        };

        let lm_head_weight = weight_edge("lm_head");
        if config.tie_word_embeddings {
            // `alias-weight-edge`'s host implementation requires its target
            // edge to already exist in this graph. A segment ending at
            // `num_hidden_layers` but NOT starting at layer 0 (multi-device
            // pipeline placement) never otherwise calls
            // `weight-edge("token_embedding")` -- that embedding lookup
            // lives in an earlier segment's own graph -- so register it
            // here too before aliasing. Idempotent with the embedding
            // branch's own call above when both run in the same
            // (unsegmented) graph.
            weight_edge("token_embedding");
            alias_weight_edge(&lm_head_weight, "token_embedding");
        }
        let logits = add_node(
            "lm_head",
            "matmul",
            "LinearAlgebra",
            &[],
            &[hidden_final, lm_head_weight],
            &shape(vec![sequence_length, config.vocabulary_size]),
            None,
        );

        finish_graph(&logits)
    } else {
        finish_graph(&hidden)
    }
}

fn build_graph(
    phase: GraphPhase,
    sequence_length: u64,
    position_offset: u64,
    kv_append: bool,
) -> String {
    let num_hidden_layers = model_config::model_config().num_hidden_layers;
    build_graph_segment(
        phase,
        sequence_length,
        position_offset,
        kv_append,
        0,
        num_hidden_layers,
    )
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        build_graph(GraphPhase::Prefill, sequence_length, 0, false)
    }

    fn build_decode_graph(cached_token_count: u64) -> String {
        build_graph(GraphPhase::Decode, 1, cached_token_count, true)
    }

    fn build_prefill_graph_segment(
        sequence_length: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build_graph_segment(
            GraphPhase::Prefill,
            sequence_length,
            0,
            false,
            start_layer,
            end_layer,
        )
    }

    fn build_decode_graph_segment(
        cached_token_count: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build_graph_segment(
            GraphPhase::Decode,
            1,
            cached_token_count,
            true,
            start_layer,
            end_layer,
        )
    }
}

export!(Component);
