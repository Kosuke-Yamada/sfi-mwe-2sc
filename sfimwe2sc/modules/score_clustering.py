import itertools
from collections import defaultdict


def to_clusters(df, column):
    """`column` の値ごとに `ex_idx` をまとめたクラスタのリストを返す。"""
    return df.groupby(column)["ex_idx"].agg(list).tolist()


def true_and_pred_clusters(df):
    """正解フレーム (`frame`) と推定クラスタ (`frame_cluster`) を返す。"""
    return to_clusters(df, "frame"), to_clusters(df, "frame_cluster")


def bcubed_f1(df):
    return calculate_bcubed(*true_and_pred_clusters(df))[2]


def calculate_clustering_scores(true, pred):
    bcp, bcr, bcf = calculate_bcubed(true, pred)
    pu, ipu, puf = calculate_purity(true, pred)
    return {
        "n_pred_clusters": len(pred),
        "n_true_clusters": len(true),
        "n_pred_instances": sum(len(c) for c in pred),
        "n_true_instances": sum(len(c) for c in true),
        "bcp": bcp,
        "bcr": bcr,
        "bcf": bcf,
        "pu": pu,
        "ipu": ipu,
        "puf": puf,
    }


def calculate_bcubed(true, pred):
    true_map = defaultdict(set)
    true_map.update({e: set(c) for c in true for e in c})
    pred_map = defaultdict(set)
    pred_map.update({e: set(c) for c in pred for e in c})

    instances = set(itertools.chain(*true, *pred))
    sum_precision, sum_recall = 0, 0
    for instance in instances:
        n_commons = len(true_map[instance] & pred_map[instance])
        n_preds = len(pred_map[instance])
        n_trues = len(true_map[instance])
        if n_preds != 0:
            sum_precision += n_commons / n_preds
        if n_trues != 0:
            sum_recall += n_commons / n_trues

    avg_precision = sum_precision / len(instances)
    avg_recall = sum_recall / len(instances)
    f_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    return avg_precision, avg_recall, f_score


def _purity(clusters, references):
    """各クラスタについて最も重なる参照クラスタとの共通要素数を合計し、割合を返す。"""
    reference_sets = [set(r) for r in references]
    n_matched = sum(max(len(set(c) & r) for r in reference_sets) for c in clusters)
    return n_matched / sum(len(c) for c in clusters)


def calculate_purity(true, pred):
    purity = _purity(pred, true)
    inverse_purity = _purity(true, pred)
    f_score = (
        (2 * purity * inverse_purity) / (purity + inverse_purity)
        if (purity + inverse_purity) > 0
        else 0
    )
    return purity, inverse_purity, f_score
