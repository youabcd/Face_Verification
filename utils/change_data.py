import numpy as np


def change_data(path, dim):
    if path[-7:-4] == 'pca':
        pca = np.load(path, allow_pickle=True)['pca'].item()
        data = pca['ux']
        data = np.transpose(data)
    elif path[-7:-4] == '500':
        data = np.load(path, allow_pickle=True)['pca']
    else:
        data = np.random.random((100, 50))
    for j in range(data.shape[0]):
        a = data[j]
        data[j] = a / np.linalg.norm(a)
    use_pos = data[:data.shape[0] // 2, :]
    use_neg = data[data.shape[0] // 2:, :]
    all_pos = []
    all_neg = []
    for i in range(len(use_pos)):
        if i % 2 == 0:
            p = [use_pos[i][:dim], use_pos[i + 1][:dim]]
            all_pos.append(np.array(p))
            n = [use_neg[i][:dim], use_neg[i + 1][:dim]]
            all_neg.append(n)
    all_pos = np.array(all_pos)
    all_neg = np.array(all_neg)
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
        if i < int(all_pos.shape[0] * 0.8):
            sample = all_pos[i]
            pos.append(np.array(sample))
            val = [all_pos[i][0], all_pos[i][1], 1]
            t.append(np.array(val))
        else:
            val = [all_pos[i][0], all_pos[i][1], 1]
            test.append(np.array(val))
    for j in range(all_neg.shape[0]):
        if j < int(all_neg.shape[0] * 0.8):
            sample = all_neg[j]
            neg.append(np.array(sample))
            val = [all_neg[j][0], all_neg[j][1], 0]
            t.append(np.array(val))
        else:
            val = [all_neg[j][0], all_neg[j][1], 0]
            test.append(np.array(val))
    pos = np.array(pos)
    neg = np.array(neg)
    t = np.array(t)
    test = np.array(test)
    return pos, neg, t, test


if __name__ == '__main__':
    positive, negative, ts, tests = change_data("/home/chenzhentao/fgfv_data/LBP_r1_pca.npz", 400)
