import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from sfimwe2sc.utils.data_utils import read_jsonl, write_jsonl

COLUMNS = [
    "ex_id",
    "ex_idx",
    "verb",
    "frame",
    "verb_frame",
    "lu_id",
    "lu_name",
    "text_widx",
    "target_widx",
]


def split_dev_test(items, test_ratio=0.8):
    n_test = int(len(items) * test_ratio)
    return items[:-n_test], items[-n_test:]


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))
    df = df.sample(frac=1, random_state=args.random_state)
    df = df[df["lu_name"].str.split(".").str[-1] == "v"].reset_index(drop=True)

    df["target_widx"] = df["target_widx"].apply(lambda x: x[0])
    df["frame"] = df["frame_name"]
    df["verb"] = df["lu_name"].apply(lambda x: x.split(".v")[0])
    df["verb_frame"] = df["verb"].str.cat(df["frame"], sep=":")
    df = df[COLUMNS]

    # 用例数が min_n_examples 以上の LU から、それぞれ最大 max_n_examples 件を使う
    lu_counts = df["lu_id"].value_counts()
    ex_idx_list = []
    for lu_id in lu_counts[lu_counts >= args.min_n_examples].index:
        df_lu = df[df["lu_id"] == lu_id]
        ex_idx_list += df_lu["ex_idx"].to_list()[: args.max_n_examples]
    df_ex = df[df["ex_idx"].isin(ex_idx_list)]

    # 動詞を、LU が 1 つのもの (v1) と複数のもの (v2) に分けてそれぞれ分割する
    n_lus = df_ex.groupby("lu_name")["lu_id"].nunique()
    v1_list = sorted(n_lus[n_lus == 1].index)
    v2_list = sorted(n_lus[n_lus >= 2].index)

    random.seed(args.random_state)
    random.shuffle(v1_list)
    random.shuffle(v2_list)
    v1_dev, v1_test = split_dev_test(v1_list)
    v2_dev, v2_test = split_dev_test(v2_list)

    for split, lu_names in [("dev", v1_dev + v2_dev), ("test", v1_test + v2_test)]:
        df_split = df_ex[df_ex["lu_name"].isin(lu_names)].sort_values("ex_idx")
        write_jsonl(
            df_split.to_dict("records"),
            args.output_dir / f"exemplars_{split}.jsonl",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--min_n_examples", type=int, default=20)
    parser.add_argument("--max_n_examples", type=int, default=100)

    args = parser.parse_args()
    print(args)
    main(args)
