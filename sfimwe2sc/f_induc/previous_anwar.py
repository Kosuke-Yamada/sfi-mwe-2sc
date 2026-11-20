import numpy as np
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from sfimwe2sc.f_induc.embedding import assign_vec_id
from sfimwe2sc.f_induc.hierarchical import cut_linkage, make_linkage
from sfimwe2sc.modules.score_clustering import bcubed_f1


class AnwarEmbedding:
    """対象語の word2vec と、文脈語の TF-IDF 重み付き平均を連結したベクトル。"""

    def __init__(self, w2v_file):
        self.w2v = KeyedVectors.load_word2vec_format(w2v_file, binary=True)

    def get_embedding(self, df):
        self.w2v.init_sims(True)

        tfidf = TfidfVectorizer(vocabulary=self.w2v.index_to_key)
        tfidf_array = tfidf.fit_transform(list(df["text_widx"]))
        norm_tfidf_array = normalize(tfidf_array, norm="l2", axis=1, copy=False)

        sum_tfidf = np.sum(norm_tfidf_array, axis=1)
        sum_tfidf[np.where(sum_tfidf == 0)] = 1
        context_vecs = np.array(norm_tfidf_array.dot(self.w2v.vectors) / sum_tfidf)
        context_vecs = normalize(context_vecs, norm="l2", axis=1)

        vec_list = []
        for text_widx, target_widx, context_vec in zip(
            df["text_widx"], df["target_widx"], context_vecs
        ):
            target = text_widx.split()[target_widx]
            if target in self.w2v.key_to_index:
                target_vec = self.w2v.get_vector(target, norm=False)
            else:
                target_vec = np.zeros(300)
            vec = np.concatenate([target_vec, context_vec], axis=0).reshape(-1, 1)
            vec_list.append(normalize(vec, norm="l2", axis=0).flatten())

        return assign_vec_id(df), np.array(vec_list)


class AnwarClustering:
    def __init__(self):
        # 元実装は metric="cityblock" を指定していたが、実際には使われていない
        # （sfimwe2sc.f_induc.hierarchical.make_linkage を参照）
        self.method = "average"
        self.th_min = 100
        self.th_max = 110

    def make_params(self, df, vec_array):
        """dev で BCubed F1 が最大になる距離の閾値を探す（同点なら大きい方）。"""
        z = make_linkage(vec_array, self.method)
        best_bcf = 0
        for th in tqdm([i / 100 for i in range(self.th_min, self.th_max + 1)]):
            df["frame_cluster"] = fcluster(z, t=th, criterion="distance")
            bcf = bcubed_f1(df)
            if best_bcf <= bcf:
                best_bcf = bcf
                params = {"th": th}
        return params

    def step(self, df, vec_array, params):
        df["frame_cluster"] = cut_linkage(vec_array, self.method, params["th"])
        return df
