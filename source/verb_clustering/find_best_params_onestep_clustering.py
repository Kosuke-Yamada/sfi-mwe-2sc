import argparse
from pathlib import Path

from sfimwe2sc.f_induc.clustering_onestep import OnestepClustering
from sfimwe2sc.f_induc.embedding import read_embedding
from sfimwe2sc.f_induc.param_search import (
    ALPHA_CANDIDATES,
    find_best_layers,
    search_best,
)
from sfimwe2sc.modules.score_clustering import bcubed_f1
from sfimwe2sc.utils.data_utils import write_json
from sfimwe2sc.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clustering = OnestepClustering(args.clustering_method)

    def cluster(df_vec, vec_array):
        params = clustering.make_params(df_vec, vec_array)
        return clustering.step(df_vec, vec_array, params)

    vec_types = ["word", "mask"] if args.vec_type == "wm" else [args.vec_type]
    best_vec_type2layer = find_best_layers(
        args.input_dir, vec_types, args.layers, cluster
    )

    alphas = ALPHA_CANDIDATES[args.vec_type]
    if args.vec_type == "wm":

        def score_fn(alpha):
            df_vec, vec_array = read_embedding(
                args.input_dir, "dev", best_vec_type2layer, alpha
            )
            return bcubed_f1(cluster(df_vec, vec_array))

        best_alpha = search_best(alphas, score_fn)
    else:
        best_alpha = alphas[0]

    best_params = {
        "alpha": best_alpha,
        "vec_type2layer": best_vec_type2layer,
        "clustering_method": args.clustering_method,
    }
    write_json(best_params, args.output_dir / "best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--vec_type", type=str, choices=["word", "mask", "wm"])
    parser.add_argument("--layers", type=str, nargs="*")

    parser.add_argument(
        "--clustering_method",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args)
