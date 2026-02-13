# OpenVINO NPU Optimization

OpenVINO Runtimeで動作する**Sambaアーキテクチャ風スタック**の実装です。
GPU/NPUの恩恵が出やすいように、複数層+広い内部次元の重め構成を使えるようにしています。

## 追加ファイル
- `samba_block_openvino.py`

## 実装内容
入力 `[B, T, C]` に対して、各層で以下を実行し、`--layers`回スタックします。

1. LayerNorm
2. 入力投影 + ゲート投影
3. depthwise 1D convolution（系列方向）
4. Swish活性化 + Sigmoidゲーティング
5. FFN風の中間拡張 (`inner -> 2*inner`) + GELU
6. 出力投影
7. 残差接続

## prefill / decode の違い
- `prefill`: プロンプト全体（`[B, T, C]`）を一括で1回推論
- `decode`: 1トークン（`[B, 1, C]`）を`--decode-steps`回繰り返し推論

実行時に、それぞれの入力shapeとレイテンシを別々に表示します。

## GPU/NPU SRAM推定
推論条件に対して、以下を近似で表示します。
- 1ブロック分の重みサイズ
- モデル総重みサイズ
- prefill/decodeのアクティベーションサイズ
- prefill/decodeの推定SRAMワーキングセット

さらに、GPU/NPUの想定SRAM容量を与えるとFIT/OVER判定できます。

```bash
python samba_block_openvino.py --device GPU --seq 256 --d-model 512 --layers 8 --expansion 4 --decode-steps 128 --sram-dtype fp16 --gpu-sram-mib 96 --npu-sram-mib 64
```

## 使い方
```bash
python samba_block_openvino.py --device CPU
```

GPU向けに重くする例:
```bash
python samba_block_openvino.py --device GPU --batch 1 --seq 256 --d-model 512 --layers 8 --expansion 4 --decode-steps 128
```

NPU向け例:
```bash
python samba_block_openvino.py --device NPU --batch 1 --seq 128 --d-model 256 --layers 6 --expansion 4 --decode-steps 64
```

## 主なCLI引数
- `--device`: 実行デバイス (`CPU`, `GPU`, `NPU` など)
- `--batch`: バッチサイズ
- `--seq`: prefillの系列長
- `--d-model`: 隠れ次元
- `--layers`: スタック層数
- `--expansion`: ブロック展開率
- `--conv-kernel`: depthwise convカーネルサイズ
- `--decode-steps`: decodeの繰り返し回数（トークン数）
- `--warmup`: 計測前ウォームアップ回数
- `--sram-dtype`: SRAM推定時の精度 (`fp32`, `fp16`, `int8`)
- `--gpu-sram-mib`: GPUの想定SRAM容量(MiB)
- `--npu-sram-mib`: NPUの想定SRAM容量(MiB)
