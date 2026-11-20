from sfimwe2sc.f_induc.hierarchical import cut_linkage, make_linkage


class OnestepClustering:
    def __init__(self, clustering):
        self.clustering = clustering

    def make_params(self, df, vec_array):
        # 正解フレーム数と同じ数のクラスタになる距離を閾値とする
        z = make_linkage(vec_array, self.clustering)
        return {"th": z[-len(set(df["frame"])) + 1][2] + 1e-6}

    def step(self, df, vec_array, params):
        df["frame_cluster"] = cut_linkage(vec_array, self.clustering, params["th"])
        return df
