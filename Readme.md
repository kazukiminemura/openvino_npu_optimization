# OpenVINO NPU Optimization

OpenVINO Runtimeで動作する**Sambaアーキテクチャ風ブロック**の最小実装を追加しています。

## 追加ファイル
- `samba_block_openvino.py`

## 実装内容
入力 `[B, T, C]` に対して、以下の流れをOpenVINOグラフで構築します。

1. LayerNorm
2. 入力投影 + ゲート投影
3. depthwise 1D convolution（系列方向）
4. Swish活性化
5. Sigmoidゲーティング
6. 出力投影
7. 残差接続

## 使い方
```bash
python samba_block_openvino.py
```

デバイス指定例:
```bash
python samba_block_openvino.py --device NPU
python samba_block_openvino.py --device GPU --batch 1 --seq 64 --decode-steps 64 --d-model 256
```

このスクリプトは`prefill`と`decode`を分けて表示します。

- `prefill`: プロンプト全体（`[B, T, C]`）を一括で1回推論
- `decode`: 1トークン（`[B, 1, C]`）を`--decode-steps`回繰り返し推論

実行すると、各モードの入力shapeとレイテンシが表示されるため、違いを比較できます。

## カスタマイズ
`build_samba_block_model()` の引数で調整できます。
- `d_model`: 埋め込み次元
- `expansion`: 内部拡張率
- `conv_kernel`: depthwise convカーネルサイズ
- `seed`: 初期重み生成用シード

CLI引数:
- `--device`: 実行デバイス (`CPU`, `GPU`, `NPU` など)
- `--batch`: バッチサイズ
- `--seq`: prefillの系列長
- `--decode-steps`: decodeの繰り返し回数（トークン数）
- `--d-model`: 隠れ次元
- `--warmup`: 計測前ウォームアップ回数
