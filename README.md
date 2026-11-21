# Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering (sfi_mwe_2sc)

[日本語のREADME](https://github.com/Kosuke-Yamada/sfi-mwe-2sc/blob/main/README_ja.md)

This is the official repository of [Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering](https://aclanthology.org/2021.acl-short.102/) (ACL-IJCNLP 2021, short paper).

The task is to group verb usages in FrameNet example sentences according to the semantic frame they evoke. This repository provides:

- **Masked word embeddings**: contextualized embeddings of a target verb obtained from a pretrained masked language model (BERT, RoBERTa, ...), with three variants
  - `word`: embedding of the target verb itself
  - `mask`: embedding of the `[MASK]` token that replaces the target verb
  - `wm`: weighted average of the two, `(1 - α) · word + α · mask`
- **Two-step clustering**: verb usages are first clustered per verb (X-means / group average / one-cluster-per-verb), and the resulting clusters are then clustered across verbs (group average / Ward).
- **Baselines and previous methods** for comparison: one-cluster-per-verb (`1cpv`), Arefyev et al. (2019), Anwar et al. (2019), and Ribeiro et al. (2019) (SemEval-2019 Task 2).

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- A CUDA-capable GPU is recommended for extracting embeddings (CPU also works with `--device cpu`).

## Installation

```sh
# Create a virtual environment (.venv) and install all dependencies
$ uv sync

# Activate the environment (the shell scripts in script/ call `python` directly)
$ source .venv/bin/activate
```

Download the FrameNet 1.7 data used by NLTK:

```sh
$ python -c "import nltk; nltk.download('framenet_v17')"
```

The Stanza English model is downloaded automatically on the first run of `apply_stanza.py`.

To reproduce the previous methods, also place the following pretrained models under `data/raw/`:

| Method | File |
|---|---|
| Anwar et al. | `data/raw/word2vec/GoogleNews-vectors-negative300.bin` ([word2vec](https://code.google.com/archive/p/word2vec/)) |
| Ribeiro et al. | `data/raw/elmo/elmo_2x4096_512_2048cnn_2xhighway_{options.json,weights.hdf5}` ([ELMo](https://allenai.org/allennlp/software/elmo)) |

ELMo is loaded by a lightweight inference-only port of AllenNLP's implementation (`sfimwe2sc/f_induc/elmo.py`), so AllenNLP itself is not required.

## Repository structure

```
.
├── script/              # Shell scripts to run each step (entry points)
│   ├── preprocessing/
│   └── verb_clustering/
├── source/              # Python scripts called by the shell scripts
│   ├── preprocessing/
│   └── verb_clustering/
├── sfimwe2sc/           # Library code (models, embeddings, clustering, evaluation)
└── data/                # Outputs (created automatically; git-ignored)
```

## Usage

Each Python script in `source/` has a corresponding shell script in `script/` with the same name (`script/(directory name)/(file name).sh`). The shell scripts use relative paths, so **run them from the directory where they are located**. Outputs are written to `data/` by default. Settings such as the pretrained model, layers, and GPU device are defined as variables at the top of each script; edit them as needed.

### 1. Preprocessing (`script/preprocessing/`)

```sh
$ cd script/preprocessing
$ bash make_exemplars.sh   # Extract example sentences from FrameNet 1.7
$ bash apply_stanza.sh     # Tokenize with Stanza and convert character-level labels to word-level
```

### 2. Semantic frame induction (`script/verb_clustering/`)

```sh
$ cd script/verb_clustering
```

1. **Create the dataset** — `make_dataset.sh`

   Uses verbs (lexical units) with at least 20 examples, randomly sampling up to 100 examples per lexical unit (setting `20-100`, as in the paper). The verbs are split into a development set (20%) and a test set (80%) so that the ratio of monosemous and polysemous verbs is preserved.

2. **Extract embeddings** — `get_embedding.sh`

   Extracts `word` / `mask` embeddings of the target verbs from every layer of the pretrained model.

3. **Clustering**

   | Method | Script |
   |---|---|
   | One-step clustering | `run_onestep_clustering.sh` (= `find_best_params_onestep_clustering.sh` → `perform_onestep_clustering.sh` → evaluation) |
   | Two-step clustering (proposed) | `run_twostep_clustering.sh` (= `find_best_params_twostep_clustering.sh` → `perform_twostep_clustering.sh` → evaluation) |
   | Baseline (one cluster per verb) | `perform_baseline_clustering.sh` |
   | Previous methods | `perform_previous_clustering-{arefyev,anwar,ribeiro}.sh` |

   The hyperparameters (layer, α, and the distance threshold for clustering) are tuned on the development set and then applied to the test set.
   Note that the one-step clustering results in Table 3 of the paper stop clustering at the gold number of frames on the test set as well. To reproduce them, pass `--oracle` to `perform_onestep_clustering.py`.

4. **Evaluate** — `evaluate_clustering-{ours,baseline,previous}.sh`

   Reports B-cubed precision/recall/F-score (`bcp` / `bcr` / `bcf`) and purity / inverse purity / their F-score (`pu` / `ipu` / `puf`).

## Citation

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
