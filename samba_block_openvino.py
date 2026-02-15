import argparse
import time

import numpy as np
import openvino as ov
from openvino import opset10 as ops


def _randn(shape, scale=0.02, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(dtype) * scale


def _layer_norm(x, gamma, beta, eps=1e-5, data_dtype=np.float32):
    # x: [B, T, C]
    axes = ops.constant(np.array([-1], dtype=np.int64))
    mean = ops.reduce_mean(x, axes, True)
    centered = ops.subtract(x, mean)
    var = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
    inv_std = ops.sqrt(ops.add(var, np.array([eps], dtype=data_dtype)))
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


def _normalize_exec_devices(exec_devices):
    if isinstance(exec_devices, str):
        return [exec_devices]
    return list(exec_devices)


def _samba_block(x, d_model, expansion, conv_kernel, seed, layer_idx):
    inner = d_model * expansion

    # Note: layer norm and weights always use fp32 for numerical stability.
    # The data_dtype specified by the user affects input/output tensors and SRAM estimates.
    gamma = ops.constant(np.ones((1, 1, d_model), dtype=np.float32), name=f"ln_gamma_{layer_idx}")
    beta = ops.constant(np.zeros((1, 1, d_model), dtype=np.float32), name=f"ln_beta_{layer_idx}")
    x_norm = _layer_norm(x, gamma, beta, data_dtype=np.float32)

    w_in = ops.constant(_randn((d_model, inner), seed=seed + 1, dtype=np.float32), name=f"w_in_{layer_idx}")
    b_in = ops.constant(np.zeros((inner,), dtype=np.float32), name=f"b_in_{layer_idx}")
    w_gate = ops.constant(_randn((d_model, inner), seed=seed + 2, dtype=np.float32), name=f"w_gate_{layer_idx}")
    b_gate = ops.constant(np.zeros((inner,), dtype=np.float32), name=f"b_gate_{layer_idx}")

    x_proj = ops.add(ops.matmul(x_norm, w_in, False, False), b_in)
    gate_proj = ops.add(ops.matmul(x_norm, w_gate, False, False), b_gate)

    # [B, T, C] -> [B, C, 1, T] for depthwise 1D conv over T.
    x_t = ops.transpose(x_proj, ops.constant(np.array([0, 2, 1], dtype=np.int64)))
    x_4d = ops.unsqueeze(x_t, axes=np.array([2], dtype=np.int64))

    dw_kernel = _randn((inner, 1, 1, 1, conv_kernel), seed=seed + 3, dtype=np.float32)
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
    w_mid = ops.constant(_randn((inner, ffn_inner), seed=seed + 4, dtype=np.float32), name=f"w_mid_{layer_idx}")
    b_mid = ops.constant(np.zeros((ffn_inner,), dtype=np.float32), name=f"b_mid_{layer_idx}")
    w_out = ops.constant(_randn((ffn_inner, d_model), seed=seed + 5, dtype=np.float32), name=f"w_out_{layer_idx}")
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
    batch_size=None,
    seq_length=None,
):
    """
    Build a heavier Samba-like stacked model.

    Input shape:  [B, T, C] where C=d_model
    Output shape: [B, T, C]
    If batch_size and seq_length are provided, creates static shapes (required for NPU).
    Note: All internal weights use fp32 for numerical stability.
    """
    if batch_size is not None and seq_length is not None:
        # Static shapes for NPU
        x_param = ops.parameter([batch_size, seq_length, d_model], dtype=np.float32, name="x")
    else:
        # Dynamic shapes for CPU/GPU
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
    seq=256,
    d_model=512,
    num_layers=12,
    expansion=8,
    conv_kernel=5,
    decode_steps=128,
    warmup=3,
    data_dtype="fp32",
    sram_dtype="fp32",
    gpu_sram_mib=None,
    npu_sram_mib=None,
    strict_device=False,
    verbose=True,
):
    # Map data type string to numpy dtype
    dtype_map = {"fp32": np.float32, "fp16": np.float16, "int8": np.int8}
    data_np_dtype = dtype_map[data_dtype]
    
    # For NPU, use static shapes; for others, use dynamic shapes
    requested_device_root = device.split(":")[0].split(".")[0].upper()

    if device.upper() == "NPU":
        # NPU requires separate models for prefill and decode with their respective static shapes
        prefill_model = build_samba_stack_model(
            d_model=d_model,
            expansion=expansion,
            conv_kernel=conv_kernel,
            num_layers=num_layers,
            batch_size=batch,
            seq_length=seq,
        )
        decode_model = build_samba_stack_model(
            d_model=d_model,
            expansion=expansion,
            conv_kernel=conv_kernel,
            num_layers=num_layers,
            batch_size=batch,
            seq_length=1,
        )
        core = ov.Core()
        prefill_compiled = core.compile_model(prefill_model, device)
        decode_compiled = core.compile_model(decode_model, device)
    else:
        model = build_samba_stack_model(
            d_model=d_model,
            expansion=expansion,
            conv_kernel=conv_kernel,
            num_layers=num_layers,
        )
        core = ov.Core()
        prefill_compiled = core.compile_model(model, device)
        decode_compiled = prefill_compiled

    prefill_exec_devices = _normalize_exec_devices(prefill_compiled.get_property("EXECUTION_DEVICES"))
    decode_exec_devices = _normalize_exec_devices(decode_compiled.get_property("EXECUTION_DEVICES"))

    if strict_device:
        prefill_roots = {d.split(".")[0].upper() for d in prefill_exec_devices}
        decode_roots = {d.split(".")[0].upper() for d in decode_exec_devices}
        if requested_device_root not in prefill_roots:
            raise RuntimeError(
                f"Prefill model did not execute on requested device '{requested_device_root}'. "
                f"Actual execution devices: {prefill_exec_devices}"
            )
        if requested_device_root not in decode_roots:
            raise RuntimeError(
                f"Decode model did not execute on requested device '{requested_device_root}'. "
                f"Actual execution devices: {decode_exec_devices}"
            )

    # Generate input tensors with specified data type
    prefill_x = np.random.randn(batch, seq, d_model).astype(data_np_dtype)
    decode_x = np.random.randn(batch, 1, d_model).astype(data_np_dtype)

    # =================================================================
    # WARMUP PHASE
    # =================================================================
    if verbose:
        print("\n[warmup] running both prefill and decode models...")
    for _ in range(warmup):
        prefill_compiled([prefill_x])
        decode_compiled([decode_x])

    # =================================================================
    # PREFILL PHASE: process full prompt sequence in one shot
    # =================================================================
    if verbose:
        print(f"\n[prefill] processing full prompt in one shot")
        print(f"  input shape: {prefill_x.shape} (batch={batch}, seq={seq}, d_model={d_model})")
    t0 = time.perf_counter()
    prefill_y = prefill_compiled([prefill_x])[0]
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000.0
    if verbose:
        print(f"  output shape: {prefill_y.shape}")
        print(f"  latency: {prefill_ms:.3f} ms")

    # =================================================================
    # DECODE PHASE: iterative inference for one token at a time
    # =================================================================
    if verbose:
        print(f"\n[decode] running {decode_steps} token-by-token steps")
        print(f"  input shape per step: {decode_x.shape} (batch={batch}, seq=1, d_model={d_model})")
    t2 = time.perf_counter()
    decode_y = decode_compiled([decode_x])[0]
    t_first = time.perf_counter()
    for _ in range(decode_steps - 1):
        decode_y = decode_compiled([decode_x])[0]
    t3 = time.perf_counter()
    decode_total_ms = (t3 - t2) * 1000.0
    first_token_ms = (t_first - t2) * 1000.0
    ttft_ms = prefill_ms + first_token_ms
    decode_per_token_ms = decode_total_ms / decode_steps
    if verbose:
        print(f"  output shape per step: {decode_y.shape}")
        print(f"  total latency: {decode_total_ms:.3f} ms")
        print(f"  first-token latency: {first_token_ms:.3f} ms")
        print(f"  time to first token (prefill + first token): {ttft_ms:.3f} ms")
        print(f"  per-token latency: {decode_per_token_ms:.3f} ms")

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

    # =================================================================
    # RESULTS SUMMARY
    # =================================================================
    if verbose:
        print("\n" + "="*65)
        print("INFERENCE RESULTS")
        print("="*65)
        print(f"\ndevice: {device}")
        print(f"actual prefill execution devices: {prefill_exec_devices}")
        print(f"actual decode execution devices: {decode_exec_devices}")
        print(f"specified data dtype: {data_dtype}")
        print(f"  (note: internal computation uses fp32, specified dtype affects SRAM estimates)")
        print(f"model config: d_model={d_model}, layers={num_layers}, expansion={expansion}, conv_kernel={conv_kernel}")
        
        print("\n[prefill] Full prompt processing:")
        print(f"  input shape: {prefill_x.shape}")
        print(f"  output shape: {prefill_y.shape}")
        print(f"  latency: {prefill_ms:.3f} ms")
        
        print("\n[decode] Token-by-token generation:")
        print(f"  input shape per step: {decode_x.shape}")
        print(f"  output shape per step: {decode_y.shape}")
        print(f"  steps: {decode_steps}")
        print(f"  total latency: {decode_total_ms:.3f} ms")
        print(f"  first-token latency: {first_token_ms:.3f} ms")
        print(f"  time to first token (prefill + first token): {ttft_ms:.3f} ms")
        print(f"  per-token latency: {decode_per_token_ms:.3f} ms")
        
        print("\nsram estimate (approx.):")
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

    return {
        "device": device,
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_total_ms,
        "first_token_ms": first_token_ms,
        "ttft_ms": ttft_ms,
        "decode_per_token_ms": decode_per_token_ms,
        "prefill_exec_devices": prefill_exec_devices,
        "decode_exec_devices": decode_exec_devices,
    }


def run_multi_device_summary(
    devices,
    batch,
    seq,
    d_model,
    num_layers,
    expansion,
    conv_kernel,
    decode_steps,
    warmup,
    data_dtype,
    sram_dtype,
    gpu_sram_mib,
    npu_sram_mib,
    strict_device,
):
    print("\n" + "="*85)
    print("MULTI-DEVICE BENCHMARK (CPU/GPU/NPU)")
    print("="*85)

    results = []
    for device in devices:
        print(f"\n[run] device={device}")
        try:
            res = run_prefill_decode_demo(
                device=device,
                batch=batch,
                seq=seq,
                d_model=d_model,
                num_layers=num_layers,
                expansion=expansion,
                conv_kernel=conv_kernel,
                decode_steps=decode_steps,
                warmup=warmup,
                data_dtype=data_dtype,
                sram_dtype=sram_dtype,
                gpu_sram_mib=gpu_sram_mib,
                npu_sram_mib=npu_sram_mib,
                strict_device=strict_device,
                verbose=False,
            )
            results.append(res)
            print(
                f"  prefill={res['prefill_ms']:.3f} ms, "
                f"decode_total={res['decode_total_ms']:.3f} ms, "
                f"first_token={res['first_token_ms']:.3f} ms, "
                f"ttft={res['ttft_ms']:.3f} ms, "
                f"decode_per_token={res['decode_per_token_ms']:.3f} ms"
            )
        except Exception as e:
            results.append({"device": device, "error": str(e)})
            print(f"  failed: {e}")

    print("\n" + "-"*85)
    print(
        f"{'device':<10} {'prefill(ms)':>12} {'decode_total(ms)':>16} "
        f"{'first_token(ms)':>16} {'ttft(ms)':>10} {'decode/token(ms)':>17}  execution_devices"
    )
    print("-"*85)
    for res in results:
        if "error" in res:
            print(f"{res['device']:<10} {'-':>12} {'-':>16} {'-':>17}  ERROR: {res['error']}")
            continue
        exec_devices = ",".join(res["prefill_exec_devices"])
        print(
            f"{res['device']:<10} "
            f"{res['prefill_ms']:>12.3f} "
            f"{res['decode_total_ms']:>16.3f} "
            f"{res['first_token_ms']:>16.3f} "
            f"{res['ttft_ms']:>10.3f} "
            f"{res['decode_per_token_ms']:>17.3f}  "
            f"{exec_devices}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run heavier Samba-like OpenVINO prefill/decode demo")
    parser.add_argument("--device", default="CPU", help="OpenVINO device name, e.g. CPU, GPU, NPU")
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run benchmark on CPU,GPU,NPU and print one summary table",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq", type=int, default=256, help="Prompt sequence length for prefill")
    parser.add_argument("--d-model", type=int, default=512, help="Model hidden size")
    parser.add_argument("--layers", type=int, default=12, help="Number of stacked Samba blocks")
    parser.add_argument("--expansion", type=int, default=8, help="Block expansion ratio")
    parser.add_argument("--conv-kernel", type=int, default=5, help="Depthwise conv kernel size")
    parser.add_argument("--decode-steps", type=int, default=128, help="Number of decode token steps")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before timing")
    parser.add_argument(
        "--data-dtype",
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Data type for model weights and inputs",
    )
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
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Fail if the compiled model does not execute on the requested device root (e.g. CPU/GPU/NPU)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.compare_all:
        run_multi_device_summary(
            devices=["CPU", "GPU", "NPU"],
            batch=args.batch,
            seq=args.seq,
            d_model=args.d_model,
            num_layers=args.layers,
            expansion=args.expansion,
            conv_kernel=args.conv_kernel,
            decode_steps=args.decode_steps,
            warmup=args.warmup,
            data_dtype=args.data_dtype,
            sram_dtype=args.sram_dtype,
            gpu_sram_mib=args.gpu_sram_mib,
            npu_sram_mib=args.npu_sram_mib,
            strict_device=args.strict_device,
        )
    else:
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
            data_dtype=args.data_dtype,
            sram_dtype=args.sram_dtype,
            gpu_sram_mib=args.gpu_sram_mib,
            npu_sram_mib=args.npu_sram_mib,
            strict_device=args.strict_device,
        )
