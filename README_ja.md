# Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering (sfi_mwe_2sc)

[English README](https://github.com/Kosuke-Yamada/sfi-mwe-2sc/blob/main/README.md)

これは、ACL-IJCNLP 2021 で採択された [Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering](https://aclanthology.org/2021.acl-short.102/) のリポジトリです。

FrameNet の用例文中の動詞を、それが喚起する意味フレームごとにクラスタリングする（意味フレーム推定）タスクを扱います。本リポジトリには以下が含まれます。

- **マスクされた単語埋め込み**: 事前学習済みマスク言語モデル（BERT、RoBERTa など）から得る対象動詞の文脈化埋め込み。以下の 3 種類があります。
  - `word`: 対象動詞そのものの埋め込み
  - `mask`: 対象動詞を `[MASK]` に置き換えたときの `[MASK]` の埋め込み
  - `wm`: 両者の重み付き平均 `(1 - α) · word + α · mask`
- **2 段階クラスタリング**: まず動詞ごとに用例をクラスタリングし（X-means / 群平均法 / 1 動詞 1 クラスタ）、次に得られたクラスタを動詞横断でクラスタリングします（群平均法 / Ward 法）。
- **比較用のベースラインと先行手法**: 1 動詞 1 クラスタ（`1cpv`）、Arefyev et al. (2019)、Anwar et al. (2019)、Ribeiro et al. (2019)（SemEval-2019 Task 2）。

## 動作環境

- Python 3.10 以上
- [uv](https://docs.astral.sh/uv/)
- 埋め込みの獲得には CUDA 対応 GPU を推奨します（`--device cpu` で CPU でも動作します）。

## インストール

```sh
# 仮想環境 (.venv) を作成し、依存パッケージをすべてインストール
$ uv sync

# 仮想環境を有効化（script/ 内のシェルスクリプトは `python` を直接呼び出すため）
$ source .venv/bin/activate
```

NLTK で使用する FrameNet 1.7 のデータをダウンロードします。

```sh
$ python -c "import nltk; nltk.download('framenet_v17')"
```

Stanza の英語モデルは、`apply_stanza.py` の初回実行時に自動でダウンロードされます。

先行手法を再現する場合は、以下の事前学習済みモデルを `data/raw/` 以下に配置してください。

| 手法 | ファイル |
|---|---|
| Anwar et al. | `data/raw/word2vec/GoogleNews-vectors-negative300.bin`（[word2vec](https://code.google.com/archive/p/word2vec/)） |
| Ribeiro et al. | `data/raw/elmo/elmo_2x4096_512_2048cnn_2xhighway_{options.json,weights.hdf5}`（[ELMo](https://allenai.org/allennlp/software/elmo)） |

ELMo は AllenNLP の実装を推論用に移植した軽量版（`sfimwe2sc/f_induc/elmo.py`）で読み込むため、AllenNLP 本体は不要です。

## ディレクトリ構成

```
.
├── script/              # 各処理を実行するシェルスクリプト（実行の入口）
│   ├── preprocessing/
│   └── verb_clustering/
├── source/              # シェルスクリプトから呼ばれる Python スクリプト
│   ├── preprocessing/
│   └── verb_clustering/
├── sfimwe2sc/           # ライブラリ（モデル、埋め込み、クラスタリング、評価）
└── data/                # 出力先（自動作成。git 管理外）
```

## 使用方法

`source/` の各 Python スクリプトには、対応するシェルスクリプトが `script/(ディレクトリ名)/(ファイル名).sh` として用意されています。シェルスクリプトは相対パスを使用しているため、**スクリプトがあるディレクトリに移動してから実行してください**。出力結果は、初期設定では `data/` に格納されます。事前学習済みモデル、層、GPU デバイスなどの設定は各スクリプト冒頭の変数で定義されているので、必要に応じて変更してください。

### 1. 前処理 (`script/preprocessing/`)

```sh
$ cd script/preprocessing
$ bash make_exemplars.sh   # FrameNet 1.7 から用例文を抽出
$ bash apply_stanza.sh     # Stanza で単語分割し、文字単位のラベルを単語単位に変換
```

### 2. 意味フレーム推定実験 (`script/verb_clustering/`)

```sh
$ cd script/verb_clustering
```

1. **データセットの作成** — `make_dataset.sh`

   用例文が 20 件以上存在する動詞（LU）を対象とし、LU ごとに最大 100 件の用例を無作為に選択します（論文と同じ設定 `20-100`）。動詞は開発セット (20%) とテストセット (80%) に分割され、その際に単義動詞と多義動詞の比率が保たれるようにしています。

2. **埋め込みの獲得** — `get_embedding.sh`

   事前学習済みモデルの各層から、対象動詞の `word` / `mask` 埋め込みを獲得します。

3. **クラスタリング**

   | 手法 | スクリプト |
   |---|---|
   | 1 段階クラスタリング | `run_onestep_clustering.sh`（= `find_best_params_onestep_clustering.sh` → `perform_onestep_clustering.sh` → 評価） |
   | 2 段階クラスタリング（提案手法） | `run_twostep_clustering.sh`（= `find_best_params_twostep_clustering.sh` → `perform_twostep_clustering.sh` → 評価） |
   | ベースライン（1 動詞 1 クラスタ） | `perform_baseline_clustering.sh` |
   | 先行手法 | `perform_previous_clustering-{arefyev,anwar,ribeiro}.sh` |

   ハイパーパラメータ（層、α、クラスタリングの距離閾値）は開発セットで決定し、その値をテストセットに適用します。
   なお、論文 Table 3 の 1 段階クラスタリングは、テストセットでも正解フレーム数でクラスタリングを打ち切った結果です。再現する場合は `perform_onestep_clustering.py` に `--oracle` を指定してください。

4. **評価** — `evaluate_clustering-{ours,baseline,previous}.sh`

   B-cubed の適合率・再現率・F 値（`bcp` / `bcr` / `bcf`）と、Purity・Inverse Purity・その F 値（`pu` / `ipu` / `puf`）を算出します。

## 引用

Please cite our paper if this source code is helpful in your work.

```bibtex
@inproceedings{yamada-etal-2021-semantic,
    title = "Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering",
    author = "Yamada, Kosuke  and
      Sasano, Ryohei  and
      Takeda, Koichi",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-short.102",
    pages = "811--816",
}
```
