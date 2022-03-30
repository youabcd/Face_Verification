import numpy as np


def get_ap(ap, dim0, dim1):
    if ap == 'PCA':
        a = np.identity(dim0)
        return np.pad(a, ((0, 0), (0, dim1 - dim0)))
    elif ap == 'RP':
        return np.random.random((dim0, dim1))
    elif ap == 'WPCA':
        feature = np.load("/home/chenzhentao/fgfv_data/pca_feature_500.npz", allow_pickle=True)['feature'].item()
        f_v = feature['feature_value'][:dim0]
        f_v = np.sqrt(f_v)
        a = np.diag(f_v)
        return np.pad(a, ((0, 0), (0, dim1 - dim0)))
    else:
        return np.ones((dim0, dim1))
