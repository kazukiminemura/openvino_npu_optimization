import argparse
import time

import numpy as np
import openvino as ov
from openvino import opset10 as ops


def _randn(shape, scale=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape).astype(np.float32) * scale)


def _layer_norm(x, gamma, beta, eps=1e-5):
    # x: [B, T, C]
    axes = ops.constant(np.array([-1], dtype=np.int64))
    mean = ops.reduce_mean(x, axes, True)
    centered = ops.subtract(x, mean)
    var = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
    inv_std = ops.sqrt(ops.add(var, np.array([eps], dtype=np.float32)))
    normed = ops.divide(centered, inv_std)
    return ops.add(ops.multiply(normed, gamma), beta)


def build_samba_block_model(
    d_model=128,
    expansion=2,
    conv_kernel=3,
    seed=42,
    name="samba_block",
):
    """
    Build a Samba-like sequence block as an OpenVINO model.

    Input shape:  [B, T, C] where C=d_model
    Output shape: [B, T, C]
    """
    inner = d_model * expansion

    x_param = ops.parameter([-1, -1, d_model], dtype=np.float32, name="x")

    gamma = ops.constant(np.ones((1, 1, d_model), dtype=np.float32), name="ln_gamma")
    beta = ops.constant(np.zeros((1, 1, d_model), dtype=np.float32), name="ln_beta")
    x_norm = _layer_norm(x_param, gamma, beta)

    w_in = ops.constant(_randn((d_model, inner), seed=seed + 1), name="w_in")
    b_in = ops.constant(np.zeros((inner,), dtype=np.float32), name="b_in")
    w_gate = ops.constant(_randn((d_model, inner), seed=seed + 2), name="w_gate")
    b_gate = ops.constant(np.zeros((inner,), dtype=np.float32), name="b_gate")

    x_proj = ops.add(ops.matmul(x_norm, w_in, False, False), b_in)
    gate_proj = ops.add(ops.matmul(x_norm, w_gate, False, False), b_gate)

    # Depthwise Conv1D over sequence dimension T:
    # [B, T, C] -> [B, C, T] -> conv -> [B, C, T] -> [B, T, C]
    x_t = ops.transpose(x_proj, ops.constant(np.array([0, 2, 1], dtype=np.int64)))
    x_4d = ops.unsqueeze(x_t, axes=np.array([2], dtype=np.int64))  # [B, C, 1, T]

    # Grouped (depthwise) conv: groups=inner, each group has in_ch=1,out_ch=1
    dw_kernel = _randn((inner, 1, 1, 1, conv_kernel), seed=seed + 3)
    dw_w = ops.constant(dw_kernel, name="dw_w")
    dw_b = ops.constant(np.zeros((inner,), dtype=np.float32), name="dw_b")

    pads = conv_kernel // 2
    conv = ops.group_convolution(
        x_4d,
        dw_w,
        strides=np.array([1, 1], dtype=np.int64),
        pads_begin=np.array([0, pads], dtype=np.int64),
        pads_end=np.array([0, pads], dtype=np.int64),
        dilations=np.array([1, 1], dtype=np.int64),
        auto_pad="explicit",
    )
    conv = ops.add(conv, ops.reshape(dw_b, ops.constant(np.array([1, inner, 1, 1], dtype=np.int64)), False))

    conv_3d = ops.squeeze(conv, axes=np.array([2], dtype=np.int64))
    conv_bt = ops.transpose(conv_3d, ops.constant(np.array([0, 2, 1], dtype=np.int64)))

    # Non-linearity + gating
    act = ops.swish(conv_bt)
    gate = ops.sigmoid(gate_proj)
    mixed = ops.multiply(act, gate)

    w_out = ops.constant(_randn((inner, d_model), seed=seed + 4), name="w_out")
    b_out = ops.constant(np.zeros((d_model,), dtype=np.float32), name="b_out")
    y = ops.add(ops.matmul(mixed, w_out, False, False), b_out)

    out = ops.add(y, x_param, name="residual_out")
    out = ops.result(out, name="output")

    return ov.Model([out], [x_param], name)


def run_prefill_decode_demo(device="CPU", batch=2, seq=16, d_model=128, decode_steps=32, warmup=3):
    model = build_samba_block_model(d_model=d_model)
    core = ov.Core()
    compiled = core.compile_model(model, device)

    prefill_x = np.random.randn(batch, seq, d_model).astype(np.float32)
    decode_x = np.random.randn(batch, 1, d_model).astype(np.float32)

    for _ in range(warmup):
        compiled([prefill_x])
        compiled([decode_x])

    t0 = time.perf_counter()
    prefill_y = compiled([prefill_x])[0]
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000.0

    t2 = time.perf_counter()
    for _ in range(decode_steps):
        decode_y = compiled([decode_x])[0]
    t3 = time.perf_counter()
    decode_total_ms = (t3 - t2) * 1000.0
    decode_per_token_ms = decode_total_ms / decode_steps

    print(f"device: {device}")
    print("prefill: one-shot inference on full prompt sequence")
    print(f"  input shape: {prefill_x.shape}")
    print(f"  output shape: {prefill_y.shape}")
    print(f"  latency: {prefill_ms:.3f} ms")
    print("decode: iterative inference for one token at a time")
    print(f"  input shape per step: {decode_x.shape}")
    print(f"  output shape per step: {decode_y.shape}")
    print(f"  steps: {decode_steps}")
    print(f"  total latency: {decode_total_ms:.3f} ms")
    print(f"  latency/token: {decode_per_token_ms:.3f} ms")
    print("note: this demo block is stateless; decode here emulates token-by-token execution.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Samba-like OpenVINO block prefill/decode demo")
    parser.add_argument("--device", default="CPU", help="OpenVINO device name, e.g. CPU, GPU, NPU")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seq", type=int, default=16, help="Prompt sequence length for prefill")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden size")
    parser.add_argument("--decode-steps", type=int, default=32, help="Number of decode token steps")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before timing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prefill_decode_demo(
        device=args.device,
        batch=args.batch,
        seq=args.seq,
        d_model=args.d_model,
        decode_steps=args.decode_steps,
        warmup=args.warmup,
    )
