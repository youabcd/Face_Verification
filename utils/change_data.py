import numpy as np


def change_data(path):
    pca = np.load(path, allow_pickle=True)['pca'].item()
    data = pca['ux']
    data = np.transpose(data)
    pos = []
    neg = []
    for i in range(data.shape[0] // 2):
        if i % 2 == 0:
            sample = [data[i], data[i + 1]]
            pos.append(np.array(sample))
    for j in range(data.shape[0] // 2, data.shape[0]):
        if j % 2 == 0:
            sample = [data[j], data[j + 1]]
            neg.append(np.array(sample))
    pos = np.array(pos)
    neg = np.array(neg)
    return pos, neg


if __name__ == '__main__':
    positive, negative = change_data("/home/chenzhentao/fgfv_data/LBP_r1_pca.npz")
