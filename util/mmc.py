import numpy as np


def mmc(k, d_model):
    """
    python implementation of the MMCenters:
        Rethinking softmax cross-entropy loss for adversarial robustness
    :param k:
    :param d_model:
    :return:
    """
    rslt = np.zeros([k, d_model], dtype=np.float)
    rslt[0, 0] = 1
    for i in range(1, k):
        for j in range(i):
            rslt[i, j] = -(1 / (k - 1) + np.dot(rslt[i, :], rslt[j, :])) / rslt[j, j]

        rslt[i, i] = np.sqrt(np.abs(1 - np.sum(rslt[i, :] ** 2)))

    return rslt


if __name__ == '__main__':
    a = mmc(10, 128)
    print(a)
