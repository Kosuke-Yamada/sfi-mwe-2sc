from tqdm import tqdm

from sfimwe2sc.f_induc.embedding import read_embedding
from sfimwe2sc.modules.score_clustering import bcubed_f1

# 埋め込みの種類ごとの alpha の候補（word: 0, mask: 1, wm: 0.0, 0.1, ..., 1.0）
ALPHA_CANDIDATES = {"word": [0], "mask": [1], "wm": [i / 10 for i in range(11)]}


def search_best(candidates, score_fn, default=None):
    """スコアが 0 より大きく最大となる最初の候補を返す（なければ `default`）。"""
    best_score, best = 0, default
    for candidate in tqdm(candidates):
        score = score_fn(candidate)
        if best_score < score:
            best_score, best = score, candidate
    return best


def find_best_layers(input_dir, vec_types, layers, cluster_fn):
    """dev で BCubed F1 が最大になる層を、埋め込みの種類ごとに探す。

    `cluster_fn(df_vec, vec_array)` はクラスタリング結果の df を返す関数。
    """
    best_vec_type2layer = {}
    for vec_type in tqdm(vec_types):
        alpha = ALPHA_CANDIDATES[vec_type][0]

        def score_fn(layer):
            df_vec, vec_array = read_embedding(
                input_dir, "dev", {vec_type: layer}, alpha
            )
            return bcubed_f1(cluster_fn(df_vec, vec_array))

        best_layer = search_best(layers, score_fn)
        if best_layer is not None:
            best_vec_type2layer[vec_type] = best_layer
    return best_vec_type2layer
