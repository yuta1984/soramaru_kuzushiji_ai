<p align="center">
  <img src="https://yuta1984.github.io/soramaru_kuzushiji_ai/soramaru/12_smile.png" alt="そらまる" height="300">
</p>

# そらまる くずし字認識 (soramaru_kuzushiji_ai)

ブラウザ完結で動作するくずし字（一文字）認識デモサイト。[みんなで翻刻](https://app.honkoku.org) のマスコット「そらまる」がアシスタント役を務めます。

- 🌐 **デモサイト**: https://yuta1984.github.io/soramaru_kuzushiji_ai/
- 🤖 **モデル (Hugging Face)**: https://huggingface.co/yuta1984/soramaru_kuzushiji_ai

## 概要

ユーザーがクリップボードからペーストまたはファイル選択した画像に対し、ConvNeXt-tiny ベースの分類モデルがくずし字を一文字認識し、上位 20 候補と確率を表示します。すべての推論は [ONNX Runtime Web](https://onnxruntime.ai/) を介してブラウザ内で完結し、サーバーへの画像送信は発生しません。

## 技術スタック

- **フロントエンド**: 素の HTML / CSS / JavaScript（ビルドなし）
- **推論ランタイム**: ONNX Runtime Web 1.19.2 (WebAssembly + SIMD, 1 thread)
- **モデル**: ConvNeXt-tiny (ImageNet-22k → 1k 事前学習版を fine-tune)
  - 入力: 384×384 RGB
  - 出力: **3673 クラス**（softmax）
  - PyTorch / timm で訓練したのち ONNX 形式にエクスポート
  - 学習設定: cutoff=3（出現 3 件未満の文字クラスを除外）+ WeightedSampler でクラス不均衡を緩和
- **前処理**: 中心正方形クロップ → 384×384 リサイズ → ImageNet 標準の mean/std で正規化

## 学習データ

本モデルは以下のデータセットを用いて学習されました：

- [Kaggle Kuzushiji Recognition](https://www.kaggle.com/competitions/kuzushiji-recognition) コンペティションデータセット
- [東京大学史料編纂所くずし字データセット](https://lab.hi.u-tokyo.ac.jp/datasets/kuzushiji)

ページ画像からの一文字切り出しを行ったうえで分類モデルを訓練しています。

### サンプル数 / クラス数

| 区分 | 件数 |
|---|---|
| 学習サンプル | 887,133 |
| 検証サンプル | 68,481 |
| 出力クラス数 | 3,673 |

学習サンプルの分布は文字種ごとに大きく偏るため（出現頻度上位の仮名と、低頻度の漢字で 3 桁のオーダー差）、`WeightedRandomSampler` で 1 epoch あたりの各クラスの期待出現回数を平準化しています。出現件数が 3 件未満のクラスは学習対象から除外。

## 性能評価

検証セット **val_kr**（Kaggle Kuzushiji Recognition の hold-out 68,481 件、3,673 クラス出力空間で argmax）：

| 指標 | top-1 | top-5 | top-20 |
|---|---|---|---|
| micro 平均 | **96.5%** | 99.5% | 99.6% |
| macro 平均（val に出現する 1,268 クラス） | 96.9% | 99.4% | 99.5% |

`val_kr + val_extra` を併せた **3,673 全クラスでの評価**（極稀少クラスを含む厳しめの条件）：

| 指標 | top-1 | top-5 | top-20 |
|---|---|---|---|
| micro | 95.3% | 99.0% | 99.4% |
| macro（出現 3,673 クラス） | 71.9% | 91.3% | 95.7% |

### 訓練サンプル数別の per-class recall（macro 平均）

| 訓練サンプル数 | クラス数 | 検証件数 | top-1 | top-5 | top-20 |
|---|---|---|---|---|---|
| 3 〜 10 | 1,690 | 1,690 | 52.5% | 85.2% | 93.5% |
| 10 〜 30 | 431 | 431 | 66.6% | 90.0% | 93.7% |
| 30 〜 100 | 622 | 2,168 | 92.5% | 97.9% | 98.4% |
| 100 〜 500 | 632 | 7,890 | 96.1% | 98.4% | 98.6% |
| 500 〜 2,000 | 213 | 11,096 | 95.7% | 98.9% | 99.1% |
| 2,000 以上 | 85 | 47,611 | 95.2% | 99.3% | 99.5% |

頻度の高い文字では top-1 が 95% 以上に達します。出現 10 件未満の極稀少クラスでは top-1 を外しがちですが、**top-20 候補に含まれる確率は 93%** あり、候補列挙ベースの翻刻支援用途では有効です。

## ファイル構成

```
.
├── index.html                    # 単一ページ UI（埋め込み CSS）
├── app.js                        # ONNX 推論ロジック
├── soramaru/                     # マスコット画像
└── model/
    ├── convnext_v4.meta.json     # クラスラベル / 入力サイズ / 正規化パラメータ
    └── unicode_translation.csv   # Unicode コード → グリフ変換テーブル
```

`convnext_v4.onnx` 本体（約 117 MB）は GitHub のファイルサイズ制限を超えるため、別途 [Hugging Face Hub](https://huggingface.co/yuta1984/soramaru_kuzushiji_ai) にホストし、`app.js` から HTTPS で直接ロードしています。ローカルで動かす場合も追加ダウンロードは不要（初回アクセス時にブラウザがフェッチ）。

## ローカル実行

ONNX Runtime の WebAssembly フェッチが `file://` では失敗するため、HTTP サーバー越しに開く必要があります。

```bash
# 依存ゼロの方法
python -m http.server 8000
# http://localhost:8000/ を開く
```

ブラウザは WebAssembly + SIMD 対応のもの（最近の Chrome / Edge / Firefox / Safari）が必要です。

## クレジット

- マスコットキャラクター「そらまる」: [みんなで翻刻](https://app.honkoku.org)
- 学習データ: [Kaggle Kuzushiji Recognition](https://www.kaggle.com/competitions/kuzushiji-recognition) / [東京大学史料編纂所](https://lab.hi.u-tokyo.ac.jp/datasets/kuzushiji)
