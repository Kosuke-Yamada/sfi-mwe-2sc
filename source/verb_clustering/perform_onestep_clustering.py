import argparse
from pathlib import Path

from sfimwe2sc.f_induc.clustering_onestep import OnestepClustering
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
        args.vec_type2layer = make_vec_type2layer([args.alpha], args.layer)

    clustering = OnestepClustering(args.clustering_method)

    df_vec, vec_array = read_embedding(
        args.input_dir, "dev", args.vec_type2layer, args.alpha
    )
    params = clustering.make_params(df_vec, vec_array)
    df_clu_dev = clustering.step(df_vec, vec_array, params)

    df_vec, vec_array = read_embedding(
        args.input_dir, "test", args.vec_type2layer, args.alpha
    )
    if args.oracle:
        # 論文 Table 3 の設定: テストセットでも正解フレーム数で打ち切る
        df_clu_test = clustering.step_oracle(df_vec, vec_array)
    else:
        df_clu_test = clustering.step(df_vec, vec_array, params)

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

    parser.add_argument("--alpha", type=float, required=False)
    parser.add_argument("--layer", type=str, required=False)

    parser.add_argument(
        "--clustering_method",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="テストセットのクラスタ数を正解フレーム数に合わせる（論文の設定）",
    )
    args = parser.parse_args()
    print(args)
    main(args)
