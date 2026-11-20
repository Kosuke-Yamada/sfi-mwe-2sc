import warnings
from collections import Counter
from math import comb

import numpy as np
import pandas as pd
from pyclustering.cluster.xmeans import kmeans_plusplus_initializer, xmeans
from scipy.cluster.hierarchy import fcluster

from sfimwe2sc.f_induc.hierarchical import cut_linkage, make_linkage

# pyclustering 0.10.1 は NumPy 1.24 で削除された numpy.warnings（標準ライブラリの
# warnings の別名）を参照するため、互換のために補う
if not hasattr(np, "warnings"):
    np.warnings = warnings


class TwostepClustering:
    """動詞ごとのクラスタリング (1st) と動詞横断のクラスタリング (2nd)。"""

    def __init__(self, clustering_method1, clustering_method2):
        self.clustering_method1 = clustering_method1
        self.clustering_method2 = clustering_method2

    def _clustering_1st(self, vec_array, params):
        if self.clustering_method1 == "average":
            if len(vec_array) < 2:
                return np.array([1])
            return cut_linkage(vec_array, "average", params["lth"])
        elif self.clustering_method1 == "xmeans":
            init_center = kmeans_plusplus_initializer(vec_array, 1).initialize()
            xm = xmeans(
                vec_array,
                init_center,
                ccore=False,
                kmax=params["kmax"],
                random_state=0,
            )
            xm.process()
            cluster_array = np.array([-1] * len(vec_array))
            for idx, clusters in enumerate(xm.get_clusters()):
                cluster_array[clusters] = idx + 1
            return cluster_array
        elif self.clustering_method1 == "1cpv":
            return np.array([1] * len(vec_array))

    def _make_vec_2nd(self, df, vec_array, count):
        """1st の各クラスタの平均ベクトルを作り、通し番号 `plu_global` を振る。"""
        vec_list, df_list = [], []
        for plu in sorted(set(df["plu_local"])):
            df_cluster = df[df["plu_local"] == plu].copy()
            vec_list.append(np.average(vec_array[df_cluster["vec_id"]], axis=0))
            count += 1
            df_cluster.loc[:, "plu_global"] = count
            df_list.append(df_cluster)
        return vec_list, pd.concat(df_list, axis=0)

    def _clustering_2nd(self, vec_array, params):
        z = make_linkage(vec_array, self.clustering_method2)
        return fcluster(z, t=self._decide_t(z, params["gth"]), criterion="distance")

    @staticmethod
    def _decide_t(z, gth):
        """同一クラスタに属する点のペアの割合が `gth` 以上になる距離を返す。"""
        n_points = len(z) + 1
        n_all_pairs = comb(n_points, 2)
        n_pairs = 0
        for child1, child2, distance, size in z:
            for child in (child1, child2):
                if child >= n_points:
                    n_pairs -= comb(int(z[int(child - n_points), 3]), 2)
            n_pairs += comb(int(size), 2)
            if n_pairs / n_all_pairs >= gth:
                return distance
        raise ValueError(f"No threshold reaches gth={gth}")

    def _make_lth(self, df, vec_array):
        """全動詞の連結距離のうち、(動詞-フレーム数 - 動詞数 + 1) 番目に大きい値。"""
        lth_list = []
        for verb in sorted(set(df["verb"])):
            verb_vec_array = vec_array[df[df["verb"] == verb]["vec_id"]]
            if len(verb_vec_array) >= 2:
                z = make_linkage(verb_vec_array, self.clustering_method1)
                lth_list += list(z[:, 2])
            else:
                lth_list.append(0)
        lth_list = pd.Series(lth_list, dtype=float).sort_values(ascending=False)
        n_splits = len(set(df["verb_frame"])) - len(set(df["verb"])) + 1
        return lth_list[:n_splits].values[-1] + 1e-6

    def make_params(self, df, vec_array):
        params = {}
        if self.clustering_method1 == "average":
            params["lth"] = self._make_lth(df, vec_array)
        elif self.clustering_method1 == "xmeans":
            params["kmax"] = df.groupby("verb")["frame"].nunique().max()

        # 動詞-フレームのペアのうち、同じフレームを持つペアの割合
        vf2f = dict(zip(df["verb_frame"], df["frame"]))
        n_same_frame_pairs = sum(comb(n, 2) for n in Counter(vf2f.values()).values())
        params["gth"] = n_same_frame_pairs / comb(len(vf2f), 2)
        return params

    def step(self, df, vec_array1, vec_array2, params):
        vec_array_2nd, df_2nd_list = [], []
        for verb in sorted(set(df["verb"])):
            df_1st = df[df["verb"] == verb].copy()
            df_1st.loc[:, "plu_local"] = self._clustering_1st(
                vec_array1[df_1st["vec_id"]], params
            )
            _vec_array_2nd, _df_2nd = self._make_vec_2nd(
                df_1st, vec_array2, len(vec_array_2nd)
            )
            vec_array_2nd += _vec_array_2nd
            df_2nd_list.append(_df_2nd)
        df_2nd = pd.concat(df_2nd_list, axis=0)

        cluster_array = self._clustering_2nd(np.array(vec_array_2nd), params)
        map_1to2 = {c1 + 1: c2 for c1, c2 in enumerate(cluster_array)}
        df_2nd["frame_cluster"] = df_2nd["plu_global"].map(map_1to2)
        return df_2nd
