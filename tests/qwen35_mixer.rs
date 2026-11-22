use std::collections::BTreeMap;

use magnetar_component_Qwen::qwen35_mixer_reference::{
    MixerConfig, MixerError, MixerOutput, MixerState, MixerWeights, mixer_f32,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    hidden_size: usize,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel: usize,
    epsilon: f32,
    hidden: Vec<f32>,
    mask: Vec<u8>,
    weights: BTreeMap<String, Vec<f32>>,
    output: Vec<f32>,
    conv_state: Vec<f32>,
    state: Vec<f32>,
    prefix_output: Vec<f32>,
    prefix_conv_state: Vec<f32>,
    prefix_state: Vec<f32>,
}

impl Case {
    fn config(&self) -> MixerConfig {
        MixerConfig {
            hidden: self.hidden_size,
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            conv_kernel: self.kernel,
            norm_epsilon: self.epsilon,
        }
    }

    fn weights(&self) -> MixerWeights<'_> {
        let weight = |name: &str| self.weights[name].as_slice();
        MixerWeights {
            in_qkv: weight("in_proj_qkv.weight"),
            in_z: weight("in_proj_z.weight"),
            in_a: weight("in_proj_a.weight"),
            in_b: weight("in_proj_b.weight"),
            convolution: weight("conv1d.weight"),
            a_log: weight("A_log"),
            dt_bias: weight("dt_bias"),
            norm: weight("norm.weight"),
            out: weight("out_proj.weight"),
        }
    }
}

fn run(case: &Case, start: usize, end: usize, initial: Option<MixerState<'_>>) -> MixerOutput {
    mixer_f32(
        case.config(),
        end - start,
        &case.hidden[start * case.hidden_size..end * case.hidden_size],
        Some(&case.mask[start..end]),
        case.weights(),
        initial,
    )
    .unwrap()
}

fn close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 4e-6 + 1e-6 * expected.abs(),
            "{what}[{index}]: {actual} != {expected}"
        );
    }
}

#[test]
fn complete_mixer_matches_pinned_pytorch_and_its_actual_cache_layer() {
    let cases: Vec<Case> =
        serde_json::from_slice(include_bytes!("fixtures/qwen35-mixer-reference.json")).unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let full = run(&case, 0, case.tokens, None);
        close(&full.values, &case.output, "full output");
        close(
            &full.convolution_state,
            &case.conv_state,
            "full convolution",
        );
        close(&full.recurrent_state, &case.state, "full recurrence");

        let prefix = run(&case, 0, 2, None);
        close(&prefix.values, &case.prefix_output, "prefix output");
        close(
            &prefix.convolution_state,
            &case.prefix_conv_state,
            "prefix convolution",
        );
        close(
            &prefix.recurrent_state,
            &case.prefix_state,
            "prefix recurrence",
        );
        let saved = prefix.clone();
        let suffix = run(
            &case,
            2,
            case.tokens,
            Some(MixerState {
                convolution: &prefix.convolution_state,
                recurrent: &prefix.recurrent_state,
            }),
        );
        assert_eq!(prefix, saved);
        let joined: Vec<_> = prefix
            .values
            .iter()
            .chain(&suffix.values)
            .copied()
            .collect();
        close(&joined, &case.output, "chunked output");
        close(
            &suffix.convolution_state,
            &case.conv_state,
            "chunked convolution",
        );
        close(&suffix.recurrent_state, &case.state, "chunked recurrence");

        let mut state: Option<MixerOutput> = None;
        let mut values = Vec::new();
        for token in 0..case.tokens {
            let initial = state.as_ref().map(|state| MixerState {
                convolution: &state.convolution_state,
                recurrent: &state.recurrent_state,
            });
            let next = run(&case, token, token + 1, initial);
            values.extend_from_slice(&next.values);
            state = Some(next);
        }
        close(&values, &case.output, "single-token output");
        let final_state = state.unwrap();
        close(
            &final_state.convolution_state,
            &case.conv_state,
            "single-token convolution",
        );
        close(
            &final_state.recurrent_state,
            &case.state,
            "single-token recurrence",
        );
        let reset = run(&case, 0, case.tokens, None);
        close(&reset.values, &case.output, "reset output");
        close(
            &reset.convolution_state,
            &case.conv_state,
            "reset convolution",
        );
        close(&reset.recurrent_state, &case.state, "reset recurrence");
    }
}

#[test]
fn mixer_rejects_invalid_shapes_masks_states_and_non_finite_weights() {
    let cases: Vec<Case> =
        serde_json::from_slice(include_bytes!("fixtures/qwen35-mixer-reference.json")).unwrap();
    let case = &cases[0];
    let test = |weights, mask, initial| {
        mixer_f32(
            case.config(),
            case.tokens,
            &case.hidden,
            mask,
            weights,
            initial,
        )
    };
    let mut weights = case.weights();
    weights.in_qkv = &weights.in_qkv[1..];
    assert_eq!(
        test(weights, Some(&case.mask), None),
        Err(MixerError::InvalidDimensions)
    );
    let mut mask = case.mask.clone();
    mask[0] = 2;
    assert_eq!(
        test(case.weights(), Some(&mask), None),
        Err(MixerError::InvalidDimensions)
    );
    assert_eq!(
        test(
            case.weights(),
            Some(&case.mask),
            Some(MixerState {
                convolution: &[0.0],
                recurrent: &case.state,
            })
        ),
        Err(MixerError::InvalidDimensions)
    );
    let mut bad_norm = case.weights["norm.weight"].clone();
    bad_norm[0] = f32::NAN;
    let mut weights = case.weights();
    weights.norm = &bad_norm;
    assert_eq!(
        test(weights, Some(&case.mask), None),
        Err(MixerError::NonFiniteValue)
    );
}
