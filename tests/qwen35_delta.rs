use magnetar_component_Qwen::qwen35_delta_reference::{
    DeltaDimensions, DeltaError, DeltaInputs, DeltaOutput, gated_delta_f32,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    initial_state: Option<Vec<f32>>,
    output: Vec<f32>,
    state: Vec<f32>,
    prefix_output: Vec<f32>,
    prefix_state: Vec<f32>,
    reset_output: Vec<f32>,
    reset_state: Vec<f32>,
}

fn run(case: &Case, start: usize, end: usize, initial: Option<&[f32]>) -> DeltaOutput {
    let q_stride = case.key_heads * case.key_dim;
    let v_stride = case.value_heads * case.value_dim;
    let gate_stride = case.value_heads;
    gated_delta_f32(
        DeltaDimensions {
            tokens: end - start,
            key_heads: case.key_heads,
            value_heads: case.value_heads,
            key_dim: case.key_dim,
            value_dim: case.value_dim,
        },
        DeltaInputs {
            query: &case.query[start * q_stride..end * q_stride],
            key: &case.key[start * q_stride..end * q_stride],
            value: &case.value[start * v_stride..end * v_stride],
            log_decay: &case.log_decay[start * gate_stride..end * gate_stride],
            beta: &case.beta[start * gate_stride..end * gate_stride],
            initial_state: initial,
        },
    )
    .unwrap()
}

fn close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6 + 1e-6 * expected.abs(),
            "{what}[{index}]: {actual} != {expected}"
        );
    }
}

#[test]
fn qwen35_recurrence_matches_pinned_pytorch_across_chunks_and_reset() {
    let cases: Vec<Case> =
        serde_json::from_slice(include_bytes!("fixtures/qwen35-delta-reference.json")).unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let original_state = case.initial_state.clone();
        let full = run(&case, 0, case.tokens, case.initial_state.as_deref());
        close(&full.values, &case.output, "full output");
        close(&full.state, &case.state, "full state");
        assert_eq!(case.initial_state, original_state);

        let prefix = run(&case, 0, 2, case.initial_state.as_deref());
        close(&prefix.values, &case.prefix_output, "prefix output");
        close(&prefix.state, &case.prefix_state, "prefix state");
        let suffix = run(&case, 2, case.tokens, Some(&prefix.state));
        let joined: Vec<_> = prefix
            .values
            .iter()
            .chain(&suffix.values)
            .copied()
            .collect();
        close(&joined, &case.output, "chunked output");
        close(&suffix.state, &case.state, "chunked state");

        let mut state = case.initial_state.clone();
        let mut decoded = Vec::new();
        for token in 0..case.tokens {
            let next = run(&case, token, token + 1, state.as_deref());
            decoded.extend(next.values);
            state = Some(next.state);
        }
        close(&decoded, &case.output, "single-token output");
        close(state.as_deref().unwrap(), &case.state, "single-token state");

        let reset = run(&case, 0, case.tokens, None);
        close(&reset.values, &case.reset_output, "reset output");
        close(&reset.state, &case.reset_state, "reset state");
    }
}

#[test]
fn invalid_dimensions_gates_and_values_are_rejected() {
    let dims = DeltaDimensions {
        tokens: 1,
        key_heads: 1,
        value_heads: 1,
        key_dim: 2,
        value_dim: 2,
    };
    let q = [0.2, 0.3];
    let k = [0.4, -0.5];
    let v = [0.1, 0.7];
    let decay = [-0.2];
    let beta = [0.5];
    let state = [0.2, 0.3, -0.1, 0.5];
    let inputs = || DeltaInputs {
        query: &q,
        key: &k,
        value: &v,
        log_decay: &decay,
        beta: &beta,
        initial_state: Some(&state),
    };
    assert!(gated_delta_f32(dims, inputs()).is_ok());
    let mut invalid = dims;
    invalid.value_heads = 0;
    assert_eq!(
        gated_delta_f32(invalid, inputs()),
        Err(DeltaError::InvalidDimensions)
    );
    invalid = dims;
    invalid.tokens = usize::MAX;
    assert_eq!(
        gated_delta_f32(invalid, inputs()),
        Err(DeltaError::InvalidDimensions)
    );
    invalid = dims;
    invalid.key_heads = 2;
    invalid.value_heads = 3;
    assert_eq!(
        gated_delta_f32(invalid, inputs()),
        Err(DeltaError::InvalidDimensions)
    );

    let mut invalid = inputs();
    invalid.initial_state = Some(&state[..3]);
    assert_eq!(
        gated_delta_f32(dims, invalid),
        Err(DeltaError::InvalidDimensions)
    );
    let mut invalid = inputs();
    invalid.log_decay = &[0.01];
    assert_eq!(gated_delta_f32(dims, invalid), Err(DeltaError::InvalidGate));
    let mut invalid = inputs();
    invalid.beta = &[1.01];
    assert_eq!(gated_delta_f32(dims, invalid), Err(DeltaError::InvalidGate));
    let mut invalid = inputs();
    invalid.query = &[f32::NAN, 0.2];
    assert_eq!(
        gated_delta_f32(dims, invalid),
        Err(DeltaError::NonFiniteValue)
    );
    let mut invalid = inputs();
    invalid.query = &[f32::MAX, 0.2];
    assert_eq!(
        gated_delta_f32(dims, invalid),
        Err(DeltaError::NonFiniteValue)
    );
    assert_eq!(state, [0.2, 0.3, -0.1, 0.5]);
}
