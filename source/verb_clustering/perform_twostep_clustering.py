import argparse
from pathlib import Path

from sfimwe2sc.f_induc.clustering_twostep import TwostepClustering
from sfimwe2sc.f_induc.embedding import make_vec_type2layer, read_embedding
from sfimwe2sc.utils.data_utils import read_json, write_json, write_jsonl
from sfimwe2sc.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_params_file is not None:
        for key, value in read_json(args.input_params_file).items():
            setattr(args, key, value)
    else:
        args.vec_type2layer = make_vec_type2layer(
            [args.alpha1, args.alpha2], args.layer
        )

    clustering = TwostepClustering(args.clustering_method1, args.clustering_method2)

    def load(split):
        df_vec, vec_array1 = read_embedding(
            args.input_dir, split, args.vec_type2layer, args.alpha1
        )
        _, vec_array2 = read_embedding(
            args.input_dir, split, args.vec_type2layer, args.alpha2
        )
        return df_vec, vec_array1, vec_array2

    df_vec, vec_array1, vec_array2 = load("dev")
    params = clustering.make_params(df_vec, vec_array1)
    df_clu_dev = clustering.step(df_vec, vec_array1, vec_array2, params)
    df_clu_test = clustering.step(*load("test"), params)

    write_jsonl(df_clu_dev.to_dict("records"), args.output_dir / "exemplars_dev.jsonl")
    write_jsonl(
        df_clu_test.to_dict("records"), args.output_dir / "exemplars_test.jsonl"
    )
    write_json(vars(args), args.output_dir / "params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--input_params_file", type=Path, required=False)

    parser.add_argument("--alpha1", type=float, required=False)
    parser.add_argument("--alpha2", type=float, required=False)
    parser.add_argument("--layer", type=str, required=False)

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
