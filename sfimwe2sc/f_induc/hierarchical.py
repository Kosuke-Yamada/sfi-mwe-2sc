from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist


def make_linkage(vec_array, method):
    """ユークリッド距離で階層的クラスタリングの連結行列を作る。

    fastcluster は距離行列を渡すと `metric` 引数を無視するため、以前の手法の
    実装で指定されていた cosine / cityblock は実際には使われていなかった。
    結果を変えないよう、ここでは常にユークリッド距離を用いる。
    """
    return linkage(pdist(vec_array), method=method, preserve_input=False)


def cut_linkage(vec_array, method, t, criterion="distance"):
    return fcluster(make_linkage(vec_array, method), t=t, criterion=criterion)
