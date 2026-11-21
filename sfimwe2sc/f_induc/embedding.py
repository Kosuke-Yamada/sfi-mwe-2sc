import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from sfimwe2sc.f_induc.dataset import collate_fn, make_inputs
from sfimwe2sc.utils.data_utils import read_jsonl


def assign_vec_id(df):
    """埋め込み行列の行番号を `vec_id` 列として先頭に付与する。"""
    return df.reset_index(drop=True).reset_index().rename(columns={"index": "vec_id"})


def run_inference(inputs, forward, batch_size, collate_fn):
    """`inputs` をバッチに分けて `forward` に通し、出力を 1 つの配列にまとめる。"""
    dl = DataLoader(inputs, batch_size=batch_size, collate_fn=collate_fn)
    vec_list = []
    with torch.no_grad():
        for batch in tqdm(dl):
            vec_list += list(forward(batch).cpu().numpy())
    return np.array(vec_list)


class BaseEmbedding:
    def __init__(self, model, pretrained_model_name, vec_type, batch_size):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.vec_type = vec_type
        self.batch_size = batch_size

    def get_embedding(self, df):
        inputs = make_inputs(df, self.tokenizer, self.vec_type)
        vec_array = run_inference(inputs, self.model, self.batch_size, collate_fn)
        return assign_vec_id(df), vec_array


def make_vec_type2layer(alphas, layer):
    """alpha と層の指定（word と mask を混ぜる場合は "word層-mask層"）から
    `read_embedding` に渡す `vec_type2layer` を作る。"""
    if all(alpha == 0 for alpha in alphas):
        return {"word": layer}
    if all(alpha == 1 for alpha in alphas):
        return {"mask": layer}
    layer_word, layer_mask = layer.split("-")
    return {"word": layer_word, "mask": layer_mask}


def _load_vec(vec_dir, vec_type, layer, split):
    vec_type_dir = vec_dir / vec_type / layer
    vec_array = np.load(vec_type_dir / f"vec_{split}.npz", allow_pickle=True)["vec"]
    return vec_type_dir, vec_array


def read_embedding(vec_dir, split, vec_type2layer, alpha):
    """word / mask の埋め込みを `alpha` で重み付けして読み込む。

    alpha = 0 なら word のみ、alpha = 1 なら mask のみ、それ以外は
    word * (1 - alpha) + mask * alpha。
    """
    if alpha == 1:
        ex_dir, vec_array = _load_vec(vec_dir, "mask", vec_type2layer["mask"], split)
    else:
        ex_dir, vec_array = _load_vec(vec_dir, "word", vec_type2layer["word"], split)
        if alpha != 0:
            _, mask_array = _load_vec(vec_dir, "mask", vec_type2layer["mask"], split)
            vec_array = vec_array * (1 - alpha) + mask_array * alpha
    df_vec = pd.DataFrame(read_jsonl(ex_dir / f"exemplars_{split}.jsonl"))
    return df_vec, vec_array
