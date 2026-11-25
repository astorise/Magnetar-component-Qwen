//! Single-sequence f32 reference for Qwen3.5's gated-delta rule.
//!
//! This is only the recurrent attention core. The input projections, causal
//! convolution, gate projection, output normalization, model graph, and Runtime
//! management of recurrent state are separate and are not implemented here.

use std::{error::Error, fmt};

const NORMALIZATION_EPSILON: f32 = 1e-6;
const MAX_ELEMENTS: usize = 67_108_864;

/// Dimensions for a single sequence. Value heads are grouped over key heads,
/// matching the repeat-interleave mapping in the upstream implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaDimensions {
    pub tokens: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

/// All values use contiguous, row-major f32 storage.
pub struct DeltaInputs<'a> {
    /// Query/key shapes: [tokens, key_heads, key_dim].
    pub query: &'a [f32],
    pub key: &'a [f32],
    /// Value shape: [tokens, value_heads, value_dim].
    pub value: &'a [f32],
    /// Log decay and beta shapes: [tokens, value_heads].
    pub log_decay: &'a [f32],
    pub beta: &'a [f32],
    /// Optional initial state: [value_heads, key_dim, value_dim].
    pub initial_state: Option<&'a [f32]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeltaOutput {
    /// [tokens, value_heads, value_dim].
    pub values: Vec<f32>,
    /// [value_heads, key_dim, value_dim], suitable for a later chunk.
    pub state: Vec<f32>,
}

/// Execute one nonempty chunk without mutating the caller's initial state.
/// Beta must be in [0, 1] and log decay must be nonpositive.
pub fn gated_delta_f32(
    dims: DeltaDimensions,
    inputs: DeltaInputs<'_>,
) -> Result<DeltaOutput, DeltaError> {
    fn count(parts: &[usize]) -> Option<usize> {
        parts.iter().try_fold(1usize, |total, &part| {
            total
                .checked_mul(part)
                .filter(|&size| part > 0 && size <= MAX_ELEMENTS)
        })
    }
    let qk_len = count(&[dims.tokens, dims.key_heads, dims.key_dim]);
    let value_len = count(&[dims.tokens, dims.value_heads, dims.value_dim]);
    let scalar_len = count(&[dims.tokens, dims.value_heads]);
    let state_len = count(&[dims.value_heads, dims.key_dim, dims.value_dim]);
    if dims.key_heads == 0
        || !dims.value_heads.is_multiple_of(dims.key_heads)
        || qk_len != Some(inputs.query.len())
        || qk_len != Some(inputs.key.len())
        || value_len != Some(inputs.value.len())
        || scalar_len != Some(inputs.log_decay.len())
        || scalar_len != Some(inputs.beta.len())
        || state_len.is_none()
        || inputs
            .initial_state
            .is_some_and(|state| state_len != Some(state.len()))
    {
        return Err(DeltaError::InvalidDimensions);
    }
    if inputs
        .query
        .iter()
        .chain(inputs.key)
        .chain(inputs.value)
        .chain(inputs.log_decay)
        .chain(inputs.beta)
        .chain(inputs.initial_state.into_iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(DeltaError::NonFiniteValue);
    }
    if inputs.log_decay.iter().any(|&value| value > 0.0)
        || inputs
            .beta
            .iter()
            .any(|&value| !(0.0..=1.0).contains(&value))
    {
        return Err(DeltaError::InvalidGate);
    }
    let mut state = match inputs.initial_state {
        Some(initial) => initial.to_vec(),
        None => vec![0.0; state_len.expect("checked above")],
    };
    let mut output = vec![0.0; value_len.expect("checked above")];
    let group = dims.value_heads / dims.key_heads;
    let q_scale = (dims.key_dim as f32).sqrt().recip();
    for token in 0..dims.tokens {
        for key_head in 0..dims.key_heads {
            let qk_start = (token * dims.key_heads + key_head) * dims.key_dim;
            let query = &inputs.query[qk_start..qk_start + dims.key_dim];
            let key = &inputs.key[qk_start..qk_start + dims.key_dim];
            let norm = |row: &[f32]| {
                let sum: f32 = row.iter().map(|value| value * value).sum();
                (sum + NORMALIZATION_EPSILON).sqrt().recip()
            };
            let query_norm = norm(query) * q_scale;
            let key_norm = norm(key);
            if !query_norm.is_finite()
                || !key_norm.is_finite()
                || query_norm == 0.0
                || key_norm == 0.0
            {
                return Err(DeltaError::NonFiniteValue);
            }
            for value_head in (key_head * group)..((key_head + 1) * group) {
                let gate_index = token * dims.value_heads + value_head;
                let decay = inputs.log_decay[gate_index].exp();
                let beta = inputs.beta[gate_index];
                let value_start = gate_index * dims.value_dim;
                let state_start = value_head * dims.key_dim * dims.value_dim;
                let head_state =
                    &mut state[state_start..state_start + dims.key_dim * dims.value_dim];
                for cell in head_state.iter_mut() {
                    *cell *= decay;
                }
                for column in 0..dims.value_dim {
                    let mut memory = 0.0;
                    for row in 0..dims.key_dim {
                        memory += head_state[row * dims.value_dim + column] * (key[row] * key_norm);
                    }
                    let delta = (inputs.value[value_start + column] - memory) * beta;
                    let mut result = 0.0;
                    for row in 0..dims.key_dim {
                        let cell = &mut head_state[row * dims.value_dim + column];
                        *cell += (key[row] * key_norm) * delta;
                        result += *cell * (query[row] * query_norm);
                    }
                    output[value_start + column] = result;
                }
            }
        }
    }
    if output.iter().chain(&state).any(|value| !value.is_finite()) {
        return Err(DeltaError::NonFiniteValue);
    }
    Ok(DeltaOutput {
        values: output,
        state,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaError {
    InvalidDimensions,
    InvalidGate,
    NonFiniteValue,
}

impl fmt::Display for DeltaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDimensions => "invalid gated-delta dimensions",
            Self::InvalidGate => "invalid gated-delta beta or log decay",
            Self::NonFiniteValue => "non-finite gated-delta input or result",
        })
    }
}

impl Error for DeltaError {}
