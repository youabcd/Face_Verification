import numpy as np


def change_data(path, dim):
    pca = np.load(path, allow_pickle=True)['pca'].item()
    data = pca['ux']
    data = np.transpose(data)
    pos = []
    neg = []
    t = []
    for i in range(data.shape[0] // 2):
        if i % 2 == 0:
            sample = [data[i][:dim], data[i + 1][:dim]]
            val = [data[i][:dim], data[i + 1][:dim], 1]
            pos.append(np.array(sample))
            t.append(np.array(val))
    for j in range(data.shape[0] // 2, data.shape[0]):
        if j % 2 == 0:
            sample = [data[j][:dim], data[j + 1][:dim]]
            val = [data[j][:dim], data[j + 1][:dim], 0]
            neg.append(np.array(sample))
            t.append(np.array(val))
    pos = np.array(pos)
    neg = np.array(neg)
    t = np.array(t)
    return pos, neg, t


if __name__ == '__main__':
    positive, negative, test = change_data("/home/chenzhentao/fgfv_data/LBP_r1_pca.npz", 400)
