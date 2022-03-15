import numpy as np


def get_ap(ap, dim0, dim1):
    if ap == 'PCA':
        a = np.identity(dim0)
        return np.pad(a, ((0, 0), (0, dim1 - dim0)))
    elif ap == 'RP':
        return np.random.random((dim0, dim1))
    else:
        return np.ones((dim0, dim1))
