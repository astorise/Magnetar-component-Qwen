//! Optional real-weight f32 comparison; checkpoint data stays outside Git.

use magnetar_component_Qwen::qwen35_mixer_reference::{
    MixerConfig, MixerState, MixerWeights, mixer_f32,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

fn checked_bytes(path: &std::path::Path, expected: &str) -> Vec<u8> {
    let data = std::fs::read(path).unwrap();
    let digest: String = Sha256::digest(&data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, expected, "unexpected digest for {}", path.display());
    data
}

struct RealWeights {
    qkv: Vec<f32>,
    z: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    convolution: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    out: Vec<f32>,
}

impl RealWeights {
    fn borrowed(&self) -> MixerWeights<'_> {
        MixerWeights {
            in_qkv: &self.qkv,
            in_z: &self.z,
            in_a: &self.a,
            in_b: &self.b,
            convolution: &self.convolution,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
            out: &self.out,
        }
    }
}

#[derive(Deserialize)]
struct Reference {
    hidden: Vec<f32>,
    output: Vec<f32>,
    conv_state: Vec<f32>,
    state: Vec<f32>,
    prefix_output: Vec<f32>,
    prefix_conv_state: Vec<f32>,
    prefix_state: Vec<f32>,
}

fn close(actual: &[f32], expected: &[f32], what: &str) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        assert!(
            error <= 2e-5 + 2e-5 * expected.abs(),
            "{what}[{index}]: {actual} != {expected}"
        );
    }
    maximum
}

#[test]
#[ignore = "requires verified Qwen3.5 base layer segments and PyTorch reference"]
fn real_qwen35_first_mixer_matches_pytorch_in_f32() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("MAGNETAR_QWEN35_CHECKPOINT_DIR")
            .expect("set MAGNETAR_QWEN35_CHECKPOINT_DIR"),
    );
    let header = checked_bytes(
        &dir.join("kev-qwen35-base-header.json"),
        "c894b6055113f9d5a9c5c68e703662de8ba27df5a9631b47f10b75ac3fe0b083",
    );
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    let scalars = checked_bytes(
        &dir.join("qwen35-base-layer0-scalars.bin"),
        "274bf2ffc2f29f3381a158434c3028694c8859a2c1bafe83df567058fa2fafc7",
    );
    let block = checked_bytes(
        &dir.join("qwen35-base-layer0-block.bin"),
        "6c68593c6ecdc8ac34a85216d1db652e3fc02899fa6f3bce4a8ecfe227d39eb8",
    );
    let read = |name: &str, dtype: &str, shape: &[usize]| -> Vec<f32> {
        let entry = &header[format!("model.language_model.layers.0.linear_attn.{name}")];
        assert_eq!(entry["dtype"], dtype);
        assert_eq!(entry["shape"], serde_json::json!(shape));
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let raw = if end <= scalars.len() {
            &scalars[start..end]
        } else {
            const BLOCK_START: usize = 508_571_776;
            &block[start - BLOCK_START..end - BLOCK_START]
        };
        let count = shape.iter().product::<usize>();
        match dtype {
            "BF16" => {
                assert_eq!(raw.len(), count * 2);
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16))
                    .collect()
            }
            "F32" => {
                assert_eq!(raw.len(), count * 4);
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect()
            }
            _ => panic!("unexpected dtype"),
        }
    };
    let weights = RealWeights {
        qkv: read("in_proj_qkv.weight", "BF16", &[6144, 1024]),
        z: read("in_proj_z.weight", "BF16", &[2048, 1024]),
        a: read("in_proj_a.weight", "BF16", &[16, 1024]),
        b: read("in_proj_b.weight", "BF16", &[16, 1024]),
        convolution: read("conv1d.weight", "BF16", &[6144, 1, 4]),
        a_log: read("A_log", "F32", &[16]),
        dt_bias: read("dt_bias", "BF16", &[16]),
        norm: read("norm.weight", "F32", &[128]),
        out: read("out_proj.weight", "BF16", &[1024, 2048]),
    };
    let reference: Reference = serde_json::from_slice(
        &std::fs::read(dir.join("qwen35-real-mixer-reference.json")).unwrap(),
    )
    .unwrap();
    let config = MixerConfig {
        hidden: 1024,
        key_heads: 16,
        value_heads: 16,
        key_dim: 128,
        value_dim: 128,
        conv_kernel: 4,
        norm_epsilon: 1e-6,
    };
    assert_eq!(reference.hidden.len(), 3 * config.hidden);
    let full = mixer_f32(config, 3, &reference.hidden, None, weights.borrowed(), None).unwrap();
    let max_output = close(&full.values, &reference.output, "full output");
    let max_conv = close(
        &full.convolution_state,
        &reference.conv_state,
        "full convolution",
    );
    let max_state = close(&full.recurrent_state, &reference.state, "full recurrence");

    let prefix = mixer_f32(
        config,
        2,
        &reference.hidden[..2 * config.hidden],
        None,
        weights.borrowed(),
        None,
    )
    .unwrap();
    close(&prefix.values, &reference.prefix_output, "prefix output");
    close(
        &prefix.convolution_state,
        &reference.prefix_conv_state,
        "prefix convolution",
    );
    close(
        &prefix.recurrent_state,
        &reference.prefix_state,
        "prefix recurrence",
    );
    let suffix = mixer_f32(
        config,
        1,
        &reference.hidden[2 * config.hidden..],
        None,
        weights.borrowed(),
        Some(MixerState {
            convolution: &prefix.convolution_state,
            recurrent: &prefix.recurrent_state,
        }),
    )
    .unwrap();
    close(
        &suffix.values,
        &reference.output[2 * config.hidden..],
        "decode output",
    );
    close(
        &suffix.convolution_state,
        &reference.conv_state,
        "decode convolution",
    );
    close(
        &suffix.recurrent_state,
        &reference.state,
        "decode recurrence",
    );
    eprintln!(
        "maximum absolute errors: output {max_output}, convolution {max_conv}, state {max_state}"
    );
}
