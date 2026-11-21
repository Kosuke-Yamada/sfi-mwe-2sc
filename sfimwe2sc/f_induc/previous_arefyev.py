import nltk
import pandas as pd
import torch
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM

from sfimwe2sc.f_induc.dataset import MODEL_INPUT_KEYS, collate_fn
from sfimwe2sc.f_induc.embedding import assign_vec_id, run_inference
from sfimwe2sc.f_induc.hierarchical import cut_linkage
from sfimwe2sc.f_induc.model import BaseNet, select_target
from sfimwe2sc.modules.score_clustering import bcubed_f1

TFIDF_PREFIX = "tfidf_"


def _wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


class Lemmatizer:
    """品詞タグを考慮して単語列を見出し語化する。"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, words):
        return [
            self.lemmatizer.lemmatize(word, _wordnet_pos(tag))
            for word, tag in nltk.pos_tag(words)
        ]


class ArefyevEmbedding:
    """BERT の第 6 層の埋め込みと、MLM による置換語の TF-IDF を求める。"""

    mlm_tokens = ["and", "then"]

    def __init__(self, pretrained_model_name, batch_size, device):
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = BaseNet(pretrained_model_name, "false", device, layer=6).eval()
        self.mlm = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.mlm = self.mlm.to(device).eval()

        self.lemmatize = Lemmatizer()
        self.topk = 40

    def _tokenize(self, words, target_widx):
        inputs = self.tokenizer(words, is_split_into_words=True)
        inputs["target_tidx"] = inputs.word_ids().index(target_widx)
        return inputs

    def get_embedding(self, df):
        """埋め込みと、置換語の TF-IDF 列 (`tfidf_*`) を追加した df を返す。"""
        inputs = [
            self._tokenize(self.lemmatize(text_widx.split()), target_widx)
            for text_widx, target_widx in zip(df["text_widx"], df["target_widx"])
        ]
        vec_array = run_inference(inputs, self.model, self.batch_size, collate_fn)
        return self._add_substitutions(assign_vec_id(df)), vec_array

    def _predict_substitutions(self, df):
        # 対象語の直後に "and then [MASK]" を挿入し、[MASK] の予測上位語を得る
        mlm_tokens = self.mlm_tokens + [self.tokenizer.mask_token]
        inputs = []
        for text_widx, target_widx in zip(df["text_widx"], df["target_widx"]):
            words = text_widx.split()
            words = words[: target_widx + 1] + mlm_tokens + words[target_widx + 1 :]
            inputs.append(self._tokenize(words, target_widx + len(mlm_tokens)))

        sub_list = []
        dl = DataLoader(inputs, batch_size=self.batch_size, collate_fn=collate_fn)
        for batch in dl:
            with torch.no_grad():
                logits = self.mlm(
                    **{k: batch[k].to(self.device) for k in MODEL_INPUT_KEYS}
                ).logits
            topk_ids = select_target(logits, batch["target_tidx"]).topk(self.topk)
            for ids in topk_ids.indices.cpu():
                subs = self.tokenizer.convert_ids_to_tokens(ids)
                sub_list.append(" ".join(self.lemmatize(subs)))
        return sub_list

    def _add_substitutions(self, df):
        sub_list = self._predict_substitutions(df)
        n_vocab = len({s for sub in sub_list for s in sub.split()})
        tfidf = TfidfVectorizer(max_features=n_vocab)
        df_tfidf = pd.DataFrame(
            tfidf.fit_transform(sub_list).toarray(),
            columns=[TFIDF_PREFIX + n for n in tfidf.get_feature_names_out()],
        )
        return pd.merge(df, df_tfidf, left_index=True, right_index=True)


class ArefyevClustering:
    def __init__(self):
        # 元実装は metric="cosine" を指定していたが、実際には使われていない
        # （sfimwe2sc.f_induc.hierarchical.make_linkage を参照）
        self.method = "average"
        self.min_th = 20
        self.max_th = 25

    def _clustering_1st(self, vec_array, params):
        return cut_linkage(vec_array, self.method, params["th"])

    def _clustering_2nd(self, df):
        """1st の各クラスタを、置換語の TF-IDF で最大 2 つに分割する。"""
        df2_list = []
        for frame_cluster_1st in tqdm(sorted(set(df["frame_cluster_1st"]))):
            df_cluster = df[df["frame_cluster_1st"] == frame_cluster_1st].copy()
            if len(df_cluster) == 1:
                df_cluster["frame_cluster_2nd"] = [1]
            else:
                tfidf_array = df_cluster.filter(regex=f"^{TFIDF_PREFIX}").values
                df_cluster["frame_cluster_2nd"] = list(
                    cut_linkage(tfidf_array, self.method, 2, criterion="maxclust")
                )
            df2_list.append(df_cluster[["frame_cluster_1st", "frame_cluster_2nd"]])

        df2 = pd.concat(df2_list).reset_index(drop=True)
        labels = (
            df2["frame_cluster_1st"]
            .astype(str)
            .str.cat(df2["frame_cluster_2nd"].astype(str), sep="")
        )
        map_1to2 = {c: i + 1 for i, c in enumerate(set(labels))}
        # NOTE: 返り値の index はクラスタ順に並べ替えた後の連番のため、呼び出し側で
        # df に代入すると元の行とはずれた対応になる。元実装（論文の再現値）と
        # 結果を一致させるため、この挙動をそのまま残している。
        return labels.map(map_1to2)

    def make_params(self, df, vec_array):
        """dev で BCubed F1 が最大になる距離の閾値を探す（同点なら大きい方）。"""
        best_bcf = 0
        for th in tqdm(range(self.min_th, self.max_th + 1)):
            df["frame_cluster_1st"] = self._clustering_1st(vec_array, {"th": th})
            df["frame_cluster"] = self._clustering_2nd(df)
            bcf = bcubed_f1(df)
            if best_bcf <= bcf:
                best_bcf = bcf
                params = {"th": th}
        return params

    def step(self, df, vec_array, params):
        df["frame_cluster_1st"] = self._clustering_1st(vec_array, params)
        df["frame_cluster"] = self._clustering_2nd(df)
        return df.filter(regex=f"^(?!{TFIDF_PREFIX})", axis=1)
