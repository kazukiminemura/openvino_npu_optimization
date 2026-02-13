import argparse
import time

import numpy as np
import openvino as ov
from openvino import opset10 as ops


def _randn(shape, scale=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32) * scale


def _layer_norm(x, gamma, beta, eps=1e-5):
    # x: [B, T, C]
    axes = ops.constant(np.array([-1], dtype=np.int64))
    mean = ops.reduce_mean(x, axes, True)
    centered = ops.subtract(x, mean)
    var = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
    inv_std = ops.sqrt(ops.add(var, np.array([eps], dtype=np.float32)))
    normed = ops.divide(centered, inv_std)
    return ops.add(ops.multiply(normed, gamma), beta)


def _bytes_to_mib(n_bytes):
    return n_bytes / (1024.0 * 1024.0)


def estimate_sram_mib(
    batch,
    seq,
    d_model,
    num_layers,
    expansion,
    conv_kernel,
    dtype_bytes=4,
):
    """
    Estimate on-chip SRAM working-set requirement (MiB).

    Conservative approximation for one-layer execution window:
    - Per-layer weights resident on fast memory
    - Peak major activations resident during layer forward
    """
    inner = d_model * expansion
    ffn_inner = inner * 2

    block_weight_elems = (
        d_model * inner            # w_in
        + d_model * inner          # w_gate
        + inner * conv_kernel      # depthwise conv kernel
        + inner * ffn_inner        # w_mid
        + ffn_inner * d_model      # w_out
        + inner + inner            # b_in, b_gate
        + inner                    # dw_b
        + ffn_inner + d_model      # b_mid, b_out
    )

    # Conservative peak activation estimate for one block.
    # x/y: 2*[B, T, C], projected/gated path: 5*[B, T, inner], mid: [B, T, 2*inner]
    def activation_elems(tokens):
        return batch * tokens * (2 * d_model + 5 * inner + ffn_inner)

    per_block_weight_bytes = block_weight_elems * dtype_bytes
    total_weight_bytes = per_block_weight_bytes * num_layers

    prefill_act_bytes = activation_elems(seq) * dtype_bytes
    decode_act_bytes = activation_elems(1) * dtype_bytes

    prefill_sram_req_bytes = per_block_weight_bytes + prefill_act_bytes
    decode_sram_req_bytes = per_block_weight_bytes + decode_act_bytes

    return {
        "per_block_weight_mib": _bytes_to_mib(per_block_weight_bytes),
        "total_weight_mib": _bytes_to_mib(total_weight_bytes),
        "prefill_activation_mib": _bytes_to_mib(prefill_act_bytes),
        "decode_activation_mib": _bytes_to_mib(decode_act_bytes),
        "prefill_sram_req_mib": _bytes_to_mib(prefill_sram_req_bytes),
        "decode_sram_req_mib": _bytes_to_mib(decode_sram_req_bytes),
    }


def _print_sram_fit(device_name, capacity_mib, prefill_req_mib, decode_req_mib):
    prefill_fit = "FIT" if prefill_req_mib <= capacity_mib else "OVER"
    decode_fit = "FIT" if decode_req_mib <= capacity_mib else "OVER"
    print(f"{device_name} SRAM capacity (assumed): {capacity_mib:.2f} MiB")
    print(f"  prefill working set: {prefill_req_mib:.2f} MiB -> {prefill_fit}")
    print(f"  decode working set: {decode_req_mib:.2f} MiB -> {decode_fit}")


def _samba_block(x, d_model, expansion, conv_kernel, seed, layer_idx):
    inner = d_model * expansion

    gamma = ops.constant(np.ones((1, 1, d_model), dtype=np.float32), name=f"ln_gamma_{layer_idx}")
    beta = ops.constant(np.zeros((1, 1, d_model), dtype=np.float32), name=f"ln_beta_{layer_idx}")
    x_norm = _layer_norm(x, gamma, beta)

    w_in = ops.constant(_randn((d_model, inner), seed=seed + 1), name=f"w_in_{layer_idx}")
    b_in = ops.constant(np.zeros((inner,), dtype=np.float32), name=f"b_in_{layer_idx}")
    w_gate = ops.constant(_randn((d_model, inner), seed=seed + 2), name=f"w_gate_{layer_idx}")
    b_gate = ops.constant(np.zeros((inner,), dtype=np.float32), name=f"b_gate_{layer_idx}")

    x_proj = ops.add(ops.matmul(x_norm, w_in, False, False), b_in)
    gate_proj = ops.add(ops.matmul(x_norm, w_gate, False, False), b_gate)

    # [B, T, C] -> [B, C, 1, T] for depthwise 1D conv over T.
    x_t = ops.transpose(x_proj, ops.constant(np.array([0, 2, 1], dtype=np.int64)))
    x_4d = ops.unsqueeze(x_t, axes=np.array([2], dtype=np.int64))

    dw_kernel = _randn((inner, 1, 1, 1, conv_kernel), seed=seed + 3)
    dw_w = ops.constant(dw_kernel, name=f"dw_w_{layer_idx}")
    dw_b = ops.constant(np.zeros((inner,), dtype=np.float32), name=f"dw_b_{layer_idx}")

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

    act = ops.swish(conv_bt)
    gate = ops.sigmoid(gate_proj)
    mixed = ops.multiply(act, gate)

    # Wider FFN-style projection to increase compute intensity.
    ffn_inner = inner * 2
    w_mid = ops.constant(_randn((inner, ffn_inner), seed=seed + 4), name=f"w_mid_{layer_idx}")
    b_mid = ops.constant(np.zeros((ffn_inner,), dtype=np.float32), name=f"b_mid_{layer_idx}")
    w_out = ops.constant(_randn((ffn_inner, d_model), seed=seed + 5), name=f"w_out_{layer_idx}")
    b_out = ops.constant(np.zeros((d_model,), dtype=np.float32), name=f"b_out_{layer_idx}")

    mid = ops.add(ops.matmul(mixed, w_mid, False, False), b_mid)
    mid = ops.gelu(mid, "tanh")
    y = ops.add(ops.matmul(mid, w_out, False, False), b_out)

    return ops.add(y, x, name=f"residual_out_{layer_idx}")


def build_samba_stack_model(
    d_model=256,
    expansion=4,
    conv_kernel=5,
    num_layers=6,
    seed=42,
    name="samba_stack",
):
    """
    Build a heavier Samba-like stacked model.

    Input shape:  [B, T, C] where C=d_model
    Output shape: [B, T, C]
    """
    x_param = ops.parameter([-1, -1, d_model], dtype=np.float32, name="x")

    x = x_param
    for i in range(num_layers):
        x = _samba_block(
            x=x,
            d_model=d_model,
            expansion=expansion,
            conv_kernel=conv_kernel,
            seed=seed + i * 100,
            layer_idx=i,
        )

    out = ops.result(x, name="output")
    return ov.Model([out], [x_param], name)


def run_prefill_decode_demo(
    device="CPU",
    batch=1,
    seq=128,
    d_model=256,
    num_layers=6,
    expansion=4,
    conv_kernel=5,
    decode_steps=64,
    warmup=3,
    sram_dtype="fp32",
    gpu_sram_mib=None,
    npu_sram_mib=None,
):
    model = build_samba_stack_model(
        d_model=d_model,
        expansion=expansion,
        conv_kernel=conv_kernel,
        num_layers=num_layers,
    )
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

    dtype_bytes_map = {"fp32": 4, "fp16": 2, "int8": 1}
    sram = estimate_sram_mib(
        batch=batch,
        seq=seq,
        d_model=d_model,
        num_layers=num_layers,
        expansion=expansion,
        conv_kernel=conv_kernel,
        dtype_bytes=dtype_bytes_map[sram_dtype],
    )

    print(f"device: {device}")
    print("model scale:")
    print(f"  d_model={d_model}, layers={num_layers}, expansion={expansion}, conv_kernel={conv_kernel}")
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
    print("note: this demo stack is stateless; decode here emulates token-by-token execution.")

    print("sram estimate (approx.):")
    print(f"  estimate dtype: {sram_dtype}")
    print(f"  per-block weight: {sram['per_block_weight_mib']:.2f} MiB")
    print(f"  total model weight: {sram['total_weight_mib']:.2f} MiB")
    print(f"  prefill activation: {sram['prefill_activation_mib']:.2f} MiB")
    print(f"  decode activation: {sram['decode_activation_mib']:.4f} MiB")
    print(f"  prefill SRAM working set: {sram['prefill_sram_req_mib']:.2f} MiB")
    print(f"  decode SRAM working set: {sram['decode_sram_req_mib']:.2f} MiB")

    if gpu_sram_mib is not None:
        _print_sram_fit("GPU", gpu_sram_mib, sram["prefill_sram_req_mib"], sram["decode_sram_req_mib"])
    if npu_sram_mib is not None:
        _print_sram_fit("NPU", npu_sram_mib, sram["prefill_sram_req_mib"], sram["decode_sram_req_mib"])
    if gpu_sram_mib is None and npu_sram_mib is None:
        print("hint: pass --gpu-sram-mib / --npu-sram-mib to check fit against assumed capacities.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run heavier Samba-like OpenVINO prefill/decode demo")
    parser.add_argument("--device", default="CPU", help="OpenVINO device name, e.g. CPU, GPU, NPU")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq", type=int, default=128, help="Prompt sequence length for prefill")
    parser.add_argument("--d-model", type=int, default=256, help="Model hidden size")
    parser.add_argument("--layers", type=int, default=6, help="Number of stacked Samba blocks")
    parser.add_argument("--expansion", type=int, default=4, help="Block expansion ratio")
    parser.add_argument("--conv-kernel", type=int, default=5, help="Depthwise conv kernel size")
    parser.add_argument("--decode-steps", type=int, default=64, help="Number of decode token steps")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before timing")
    parser.add_argument(
        "--sram-dtype",
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Precision assumed for SRAM estimate",
    )
    parser.add_argument(
        "--gpu-sram-mib",
        type=float,
        default=None,
        help="Assumed GPU SRAM capacity (MiB) for fit check",
    )
    parser.add_argument(
        "--npu-sram-mib",
        type=float,
        default=None,
        help="Assumed NPU SRAM capacity (MiB) for fit check",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prefill_decode_demo(
        device=args.device,
        batch=args.batch,
        seq=args.seq,
        d_model=args.d_model,
        num_layers=args.layers,
        expansion=args.expansion,
        conv_kernel=args.conv_kernel,
        decode_steps=args.decode_steps,
        warmup=args.warmup,
        sram_dtype=args.sram_dtype,
        gpu_sram_mib=args.gpu_sram_mib,
        npu_sram_mib=args.npu_sram_mib,
    )
