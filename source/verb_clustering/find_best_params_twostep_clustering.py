import argparse
from functools import cache
from itertools import product
from pathlib import Path

from sfimwe2sc.f_induc.clustering_twostep import TwostepClustering
from sfimwe2sc.f_induc.embedding import read_embedding
from sfimwe2sc.f_induc.param_search import (
    ALPHA_CANDIDATES,
    find_best_layers,
    search_best,
)
from sfimwe2sc.modules.score_clustering import bcubed_f1
from sfimwe2sc.utils.data_utils import write_json
from sfimwe2sc.utils.model_utils import fix_seed

# "1st 用の埋め込み-2nd 用の埋め込み"。"wm" は 1st と 2nd で同じ alpha を用いる
VEC_TYPES = [
    "word",
    "mask",
    "wm",
    "word-mask",
    "mask-word",
    "word-wm",
    "mask-wm",
    "wm-word",
    "wm-mask",
    "wm-wm",
]


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clustering = TwostepClustering(args.clustering_method1, args.clustering_method2)

    def cluster(df_vec, vec_array):
        params = clustering.make_params(df_vec, vec_array)
        return clustering.step(df_vec, vec_array, vec_array, params)

    vec_types = (
        [args.vec_type] if args.vec_type in ["word", "mask"] else ["word", "mask"]
    )
    best_vec_type2layer = find_best_layers(
        args.input_dir, vec_types, args.layers, cluster
    )

    @cache
    def load(alpha):
        return read_embedding(args.input_dir, "dev", best_vec_type2layer, alpha)

    if args.vec_type == "wm":

        def score_fn(alpha):
            return bcubed_f1(cluster(*load(alpha)))

        best_alpha = search_best(ALPHA_CANDIDATES["wm"], score_fn, default=-1)
        best_alpha1, best_alpha2 = best_alpha, best_alpha
    else:
        vec_type1, vec_type2 = (
            [args.vec_type] * 2
            if args.vec_type in ["word", "mask"]
            else args.vec_type.split("-")
        )
        alphas1, alphas2 = ALPHA_CANDIDATES[vec_type1], ALPHA_CANDIDATES[vec_type2]
        if "wm" not in (vec_type1, vec_type2):
            best_alpha1, best_alpha2 = alphas1[0], alphas2[0]
        else:

            @cache
            def make_params(alpha1):
                df_vec, vec_array1 = load(alpha1)
                return clustering.make_params(df_vec, vec_array1)

            def score_fn(alphas):
                alpha1, alpha2 = alphas
                df_vec, vec_array1 = load(alpha1)
                _, vec_array2 = load(alpha2)
                params = make_params(alpha1)
                return bcubed_f1(
                    clustering.step(df_vec, vec_array1, vec_array2, params)
                )

            best_alpha1, best_alpha2 = search_best(
                list(product(alphas1, alphas2)), score_fn, default=(-1, -1)
            )

    best_params = {
        "alpha1": best_alpha1,
        "alpha2": best_alpha2,
        "vec_type2layer": best_vec_type2layer,
        "clustering_method1": args.clustering_method1,
        "clustering_method2": args.clustering_method2,
    }
    write_json(best_params, args.output_dir / "best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--vec_type", type=str, choices=VEC_TYPES, required=False)
    parser.add_argument("--layers", type=str, nargs="*")

    parser.add_argument(
        "--clustering_method1",
        type=str,
        choices=["xmeans", "average", "1cpv"],
        required=False,
    )
    parser.add_argument(
        "--clustering_method2",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args)
