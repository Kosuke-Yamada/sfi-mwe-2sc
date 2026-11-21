import networkx as nx
import numpy as np
import torch
from chinese_whispers import aggregate_clusters, chinese_whispers
from sklearn.metrics.pairwise import cosine_similarity

from sfimwe2sc.f_induc.elmo import Elmo, batch_to_ids
from sfimwe2sc.f_induc.embedding import assign_vec_id, run_inference
from sfimwe2sc.f_induc.model import select_target


def ribeiro_collate_fn(batch):
    return {
        "input_ids": batch_to_ids([b["input_words"] for b in batch]),
        "target_tidx": torch.LongTensor([b["target_tidx"] for b in batch]),
    }


class RibeiroEmbedding:
    """ELMo の 2 つの出力表現の和から対象語のベクトルを取り出す。"""

    def __init__(self, elmo_options_file, elmo_weights_file, batch_size, device):
        self.model = Elmo(elmo_options_file, elmo_weights_file, 2, dropout=0)
        self.model = self.model.to(device).eval()
        self.batch_size = batch_size
        self.device = device

    def _forward(self, batch):
        reps = self.model(batch["input_ids"].to(self.device))["elmo_representations"]
        return select_target(reps[0] + reps[1], batch["target_tidx"])

    def get_embedding(self, df):
        inputs = [
            {"input_words": text_widx.split(), "target_tidx": target_widx}
            for text_widx, target_widx in zip(df["text_widx"], df["target_widx"])
        ]
        vec_array = run_inference(
            inputs, self._forward, self.batch_size, ribeiro_collate_fn
        )
        return assign_vec_id(df), vec_array


class RibeiroClustering:
    """コサイン距離のグラフに対する Chinese Whispers によるクラスタリング。"""

    def __init__(self):
        self.weighting = "top"
        self.seed = 1337
        self.iterations = 20

    def _clustering(self, vec_array):
        n = len(vec_array)
        cosdist_matrix = 1 - cosine_similarity(vec_array, vec_array)
        cosdist_matrix *= np.triu(np.ones([n, n]), 1)

        nonzero = cosdist_matrix[cosdist_matrix.nonzero()]
        t = (nonzero.mean() + nonzero.std()) / 2
        cosdist_matrix = np.where(cosdist_matrix >= t, 0, cosdist_matrix)

        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(
            (i, j, {"weight": cosdist_matrix[i, j]})
            for i, j in zip(*np.nonzero(cosdist_matrix))
        )
        chinese_whispers(
            g, weighting=self.weighting, seed=self.seed, iterations=self.iterations
        )

        cluster_array = np.zeros(n, dtype=int)
        for i, nodes in enumerate(aggregate_clusters(g).values()):
            cluster_array[list(nodes)] = i + 1
        return cluster_array

    def make_params(self, df, vec_array):
        return {}

    def step(self, df, vec_array, params):
        df["frame_cluster"] = self._clustering(vec_array)
        return df
