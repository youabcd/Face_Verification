import numpy as np


def change_data(path, dim):
    pca = np.load(path, allow_pickle=True)['pca'].item()
    data = pca['ux']
    data = np.transpose(data)
    for j in range(data.shape[0]):
        a = data[j]
        data[j] = a / np.linalg.norm(a)
    all_pos = data[:data.shape[0] // 2, :]
    all_neg = data[data.shape[0] // 2:, :]
    arr = np.array(range(all_pos.shape[0]))
    np.random.shuffle(arr)
    all_pos = all_pos[arr, :]
    arr_neg = np.array(range(all_neg.shape[0]))
    np.random.shuffle(arr)
    all_neg = all_neg[arr_neg, :]
    pos = []
    neg = []
    t = []
    test = []
    for i in range(all_pos.shape[0]):
        if i % 2 == 0:
            if i < int(all_pos.shape[0]*0.8):
                sample = [all_pos[i][:dim], all_pos[i + 1][:dim]]
                pos.append(np.array(sample))
                val = [all_pos[i][:dim], all_pos[i + 1][:dim], 1]
                t.append(np.array(val))
            else:
                val = [all_pos[i][:dim], all_pos[i + 1][:dim], 1]
                test.append(np.array(val))
    for j in range(all_neg.shape[0]):
        if j % 2 == 0:
            if j < int(all_neg.shape[0]*0.8):
                sample = [all_neg[j][:dim], all_neg[j + 1][:dim]]
                neg.append(np.array(sample))
                val = [all_neg[j][:dim], all_neg[j + 1][:dim], 0]
                t.append(np.array(val))
            else:
                val = [all_neg[j][:dim], all_neg[j + 1][:dim], 0]
                test.append(np.array(val))
    pos = np.array(pos)
    neg = np.array(neg)
    t = np.array(t)
    test = np.array(test)
    return pos, neg, t, test


if __name__ == '__main__':
    positive, negative, ts, tests = change_data("/home/chenzhentao/fgfv_data/LBP_r1_pca.npz", 400)
