//! First real, minimal Qwen Model Component
//! (`magnetar:model-component-graph@1.0.0`, task 11.4).
//!
//! Builds prefill and decode Execution Graphs for the exact fixed
//! architecture Magnetar's own E2E first-native fixture uses
//! (`magnetar-runtime::first_native_runtime::e2e_fixture_config`), via the
//! Runtime-owned `graph-builder` Capability rather than by constructing an
//! `ExecutionGraph` directly -- this Component never touches
//! `magnetar-runtime`'s Rust types at all, only the WIT contract.
//!
//! Deliberately minimal, not general: the architecture dimensions below are
//! this one fixture's exact values, hard-coded, not read from any
//! configuration input (this contract has no such input yet). A Component
//! for a real, configurable Qwen checkpoint is real follow-up work; this is
//! the first proof that a Component can produce a real graph through this
//! contract at all.

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    add_node, alias_weight_edge, begin_graph, declare_input, finish_graph, kv_resource,
    weight_edge, Attribute, AttributeValue, GraphPhase, KvCacheRole, KvInvolvement, TensorShape,
};

const HIDDEN_SIZE: u64 = 4;
const LAYER_COUNT: u64 = 1;
const ATTENTION_HEAD_COUNT: u64 = 2;
const KV_HEAD_COUNT: u64 = 2;
const HEAD_DIMENSION: u64 = 2;
const INTERMEDIATE_SIZE: u64 = 8;
const VOCABULARY_SIZE: u64 = 258;
const RMSNORM_EPSILON: f64 = 1e-6;
const ROPE_BASE: f64 = 10_000.0;
const ROPE_DIMENSION: i64 = HEAD_DIMENSION as i64;
const ROPE_POSITION_MODE: &str = "sequential";
const TIED_EMBEDDINGS: bool = true;

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
}

fn attr(name: &str, value: AttributeValue) -> Attribute {
    Attribute {
        name: name.to_string(),
        value,
    }
}

/// Builds one pre-norm decoder layer, matching
/// `magnetar_runtime::qwen_model_component::qwen_build_graph`'s per-layer
/// node sequence exactly (RMSNorm, QKV matmul, RoPE, attention, output
/// projection, residual-add, RMSNorm, gated MLP, residual-add), through
/// `graph-builder` calls instead of direct `ExecutionGraph` construction.
fn build_layer(layer: u64, residual_in: &str, sequence_length: u64, kv_append: bool) -> String {
    let prefix = format!("layer{layer}");
    let q_dim = ATTENTION_HEAD_COUNT * HEAD_DIMENSION;
    let kv_dim = KV_HEAD_COUNT * HEAD_DIMENSION;

    let input_norm_weight = weight_edge(&format!("{prefix}.input_norm"));
    let normed = add_node(
        &format!("{prefix}.input_norm"),
        "rmsnorm",
        "Normalization",
        &[attr("epsilon", AttributeValue::FloatValue(RMSNORM_EPSILON))],
        &[residual_in.to_string(), input_norm_weight],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    );

    let q_weight = weight_edge(&format!("{prefix}.q_proj"));
    let q = add_node(
        &format!("{prefix}.q_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed.clone(), q_weight],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    let k_weight = weight_edge(&format!("{prefix}.k_proj"));
    let k = add_node(
        &format!("{prefix}.k_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed.clone(), k_weight],
        &shape(vec![sequence_length, kv_dim]),
        None,
    );
    let v_weight = weight_edge(&format!("{prefix}.v_proj"));
    let v_kv = kv_resource(layer, KvCacheRole::V);
    let v = add_node(
        &format!("{prefix}.v_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[normed, v_weight],
        &shape(vec![sequence_length, kv_dim]),
        Some(&KvInvolvement {
            resource_id: v_kv,
            role: KvCacheRole::V,
            append: kv_append,
        }),
    );

    let rope_attrs = |offset: i64| {
        vec![
            attr("base", AttributeValue::FloatValue(ROPE_BASE)),
            attr("dimension", AttributeValue::Integer(ROPE_DIMENSION)),
            attr(
                "position_mode",
                AttributeValue::Text(ROPE_POSITION_MODE.to_string()),
            ),
            attr("position_offset", AttributeValue::Integer(offset)),
        ]
    };
    let q_rope = add_node(
        &format!("{prefix}.rope_q"),
        "rope",
        "PositionEncoding",
        &rope_attrs(0),
        &[q],
        &shape(vec![sequence_length, q_dim]),
        None,
    );
    let k_kv = kv_resource(layer, KvCacheRole::K);
    let k_rope = add_node(
        &format!("{prefix}.rope_k"),
        "rope",
        "PositionEncoding",
        &rope_attrs(0),
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
                AttributeValue::Integer(ATTENTION_HEAD_COUNT as i64),
            ),
            attr(
                "kv_head_count",
                AttributeValue::Integer(KV_HEAD_COUNT as i64),
            ),
            attr(
                "head_dimension",
                AttributeValue::Integer(HEAD_DIMENSION as i64),
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

    let o_weight = weight_edge(&format!("{prefix}.o_proj"));
    let attn_proj = add_node(
        &format!("{prefix}.o_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[attn_out, o_weight],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    );

    let post_attn = add_node(
        &format!("{prefix}.residual1"),
        "residual-add",
        "Tensor",
        &[],
        &[residual_in.to_string(), attn_proj],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    );

    let post_attn_norm_weight = weight_edge(&format!("{prefix}.post_attn_norm"));
    let mlp_normed = add_node(
        &format!("{prefix}.post_attn_norm"),
        "rmsnorm",
        "Normalization",
        &[attr("epsilon", AttributeValue::FloatValue(RMSNORM_EPSILON))],
        &[post_attn.clone(), post_attn_norm_weight],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    );

    let gate_weight = weight_edge(&format!("{prefix}.gate_proj"));
    let gate = add_node(
        &format!("{prefix}.gate_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_normed.clone(), gate_weight],
        &shape(vec![sequence_length, INTERMEDIATE_SIZE]),
        None,
    );
    let up_weight = weight_edge(&format!("{prefix}.up_proj"));
    let up = add_node(
        &format!("{prefix}.up_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_normed, up_weight],
        &shape(vec![sequence_length, INTERMEDIATE_SIZE]),
        None,
    );
    let activated = add_node(
        &format!("{prefix}.silu"),
        "silu",
        "Activation",
        &[],
        &[gate],
        &shape(vec![sequence_length, INTERMEDIATE_SIZE]),
        None,
    );
    let mlp_hidden = add_node(
        &format!("{prefix}.mul"),
        "mul",
        "Tensor",
        &[],
        &[activated, up],
        &shape(vec![sequence_length, INTERMEDIATE_SIZE]),
        None,
    );
    let down_weight = weight_edge(&format!("{prefix}.down_proj"));
    let mlp_out = add_node(
        &format!("{prefix}.down_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[mlp_hidden, down_weight],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    );

    add_node(
        &format!("{prefix}.residual2"),
        "residual-add",
        "Tensor",
        &[],
        &[post_attn, mlp_out],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    )
}

fn build_graph(phase: GraphPhase, sequence_length: u64, position_offset: u64, kv_append: bool) -> String {
    begin_graph(phase, sequence_length, position_offset);

    let token_ids = declare_input("token_ids", &shape(vec![sequence_length]));
    let token_embedding = weight_edge("token_embedding");
    let mut hidden = add_node(
        "embedding",
        "embedding",
        "Tensor",
        &[],
        &[token_ids, token_embedding],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    );

    for layer in 0..LAYER_COUNT {
        hidden = build_layer(layer, &hidden, sequence_length, kv_append);
    }

    let final_norm_weight = weight_edge("final_norm");
    let hidden_final = add_node(
        "final_norm",
        "rmsnorm",
        "Normalization",
        &[attr("epsilon", AttributeValue::FloatValue(RMSNORM_EPSILON))],
        &[hidden, final_norm_weight],
        &shape(vec![sequence_length, HIDDEN_SIZE]),
        None,
    );

    let lm_head_weight = weight_edge("lm_head");
    if TIED_EMBEDDINGS {
        alias_weight_edge(&lm_head_weight, "token_embedding");
    }
    let logits = add_node(
        "lm_head",
        "matmul",
        "LinearAlgebra",
        &[],
        &[hidden_final, lm_head_weight],
        &shape(vec![sequence_length, VOCABULARY_SIZE]),
        None,
    );

    finish_graph(&logits)
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        build_graph(GraphPhase::Prefill, sequence_length, 0, false)
    }

    fn build_decode_graph(cached_token_count: u64) -> String {
        build_graph(GraphPhase::Decode, 1, cached_token_count, true)
    }
}

export!(Component);

