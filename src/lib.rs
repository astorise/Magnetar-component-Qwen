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

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, KvCacheRole, KvInvolvement, TensorShape, add_node,
    alias_weight_edge, begin_graph, declare_input, finish_graph, kv_resource, weight_edge,
};
use magnetar::model_component_graph::model_config::{self, ArchitectureConfig};

const ROPE_POSITION_MODE: &str = "sequential";

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
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
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_bias" | "k_bias" | "v_bias" => {
            format!("layers.{layer}.self_attn.{suffix}")
        }
        "gate_proj" | "up_proj" | "down_proj" => format!("layers.{layer}.mlp.{suffix}"),
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
    layer: u64,
    residual_in: &str,
    sequence_length: u64,
    kv_append: bool,
) -> String {
    let prefix = format!("layer{layer}");
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let attention_head_count = config.num_attention_heads as u64;
    let kv_head_count = config.num_key_value_heads as u64;
    let head_dimension = config.head_dimension;
    let q_dim = attention_head_count * head_dimension;
    let kv_dim = kv_head_count * head_dimension;

    let input_norm_weight = weight_edge(&weight_tensor_name(layer, "input_norm"));
    let normed = add_node(
        &format!("{prefix}.input_norm"),
        "rmsnorm",
        "Normalization",
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(config.rms_norm_eps),
        )],
        &[residual_in.to_string(), input_norm_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

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
    let q = add_node(
        &format!("{prefix}.q_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed.clone(), q_weight],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
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
            attr("dimension", AttributeValue::Integer(head_dimension as i64)),
            attr(
                "position_mode",
                AttributeValue::Text(ROPE_POSITION_MODE.to_string()),
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

    let attn_out = add_node(
        &format!("{prefix}.attention"),
        "attention",
        "Attention",
        &[
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
        ],
        &[q_rope, k_rope, v],
        &shape(vec![sequence_length, q_dim]),
        None,
    );

    let o_weight = weight_edge(&weight_tensor_name(layer, "o_proj"));
    let attn_proj = add_node(
        &format!("{prefix}.o_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[attn_out, o_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

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
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(config.rms_norm_eps),
        )],
        &[post_attn.clone(), post_attn_norm_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

    let gate_weight = weight_edge(&weight_tensor_name(layer, "gate_proj"));
    let gate = add_node(
        &format!("{prefix}.gate_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_normed.clone(), gate_weight],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let up_weight = weight_edge(&weight_tensor_name(layer, "up_proj"));
    let up = add_node(
        &format!("{prefix}.up_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_normed, up_weight],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let activated = add_node(
        &format!("{prefix}.silu"),
        "silu",
        "Activation",
        &[],
        &[gate],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let mlp_hidden = add_node(
        &format!("{prefix}.mul"),
        "mul",
        "Tensor",
        &[],
        &[activated, up],
        &shape(vec![sequence_length, intermediate_size]),
        None,
    );
    let down_weight = weight_edge(&weight_tensor_name(layer, "down_proj"));
    let mlp_out = add_node(
        &format!("{prefix}.down_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_hidden, down_weight],
        &shape(vec![sequence_length, hidden_size]),
        None,
    );

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
    let config = model_config::model_config();
    begin_graph(phase, sequence_length, position_offset);

    let mut hidden = if start_layer == 0 {
        let token_ids = declare_input("token_ids", &shape(vec![sequence_length]));
        let token_embedding = weight_edge("token_embedding");
        add_node(
            "embedding",
            "embedding",
            "Tensor",
            &[],
            &[token_ids, token_embedding],
            &shape(vec![sequence_length, config.hidden_size]),
            None,
        )
    } else {
        declare_input(
            "hidden_states_in",
            &shape(vec![sequence_length, config.hidden_size]),
        )
    };

    for layer in start_layer as u64..end_layer as u64 {
        hidden = build_layer(&config, layer, &hidden, sequence_length, kv_append);
    }

    if end_layer == config.num_hidden_layers {
        let final_norm_weight = weight_edge("final_norm");
        let hidden_final = add_node(
            "final_norm",
            "rmsnorm",
            "Normalization",
            &[attr(
                "epsilon",
                AttributeValue::FloatValue(config.rms_norm_eps),
            )],
            &[hidden, final_norm_weight],
            &shape(vec![sequence_length, config.hidden_size]),
            None,
        );

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
