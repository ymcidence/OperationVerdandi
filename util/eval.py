from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def purity(feat, semantic, pred):
    """
    computing the purity of clustering result
    :param feat:
    :param semantic:
    :param pred:
    :return:
    """

    return 0


def acc(semantic, pred):
    """
    unsupervised clustering accuracy
    :param semantic: [N]
    :param pred: [N]
    :return:
    """
    assert pred.size == semantic.size
    d = max(pred.max(), semantic.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(pred.size):
        w[pred[i], semantic[i]] += 1
    ind = linear_assignment(np.max(w) - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / pred.size, w


def nmi(semantic, pred):
    """
    NMI
    :param semantic:
    :param pred:
    :return:
    """
    return normalized_mutual_info_score(semantic, pred)


def ari(semantic, pred):
    """
    ARI
    :param semantic:
    :param pred:
    :return:
    """
    return adjusted_rand_score(semantic, pred)


def silhouette_coefficient(feat, pred, metric='euclidean'):
    return silhouette_score(feat, pred, metric=metric)
