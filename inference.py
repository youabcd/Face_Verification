import numpy as np
from utils.change_data import change_data
from data_loader.data_set import local_bp
import os
from utils.error_computation import cosine_similarity


def inference(img1, img2, parameter, feature):
    if isinstance(img1, str):
        img1 = local_bp(img1)
        img2 = local_bp(img2)
        feature = np.load(feature, allow_pickle=True)['feature'].item()
        feature_pca = feature['feature_pca']
        img1 = (feature_pca @ img1[:, np.newaxis]).reshape(-1)
        img2 = (feature_pca @ img2[:, np.newaxis]).reshape(-1)
    img1 = img1 / np.linalg.norm(img1)
    img2 = img2 / np.linalg.norm(img2)
    parameter = np.load(parameter, allow_pickle=True)['parameter'].item()
    a = parameter['a0_s'][0]
    theta = parameter['best_theta'][0]
    img1 = img1[:a.shape[1]]
    img2 = img2[:a.shape[1]]
    cs = cosine_similarity(img1.reshape((1, -1)), img2.reshape((1, -1)), a)[0]
    # print(cs)
    if cs >= theta:
        print("same")
        return 1
    else:
        print("twins")
        return 0


def compute_acc_path(data_path, pca_dim, parameter_path):
    _, _, _, test = change_data(data_path, pca_dim)
    parameter = np.load(parameter_path, allow_pickle=True)['parameter'].item()
    theta = parameter['best_theta'][0]
    a = parameter['a0_s'][0]
    csml = cosine_similarity(test[:, 0], test[:, 1], a)
    error = 0
    for i in range(csml.shape[0]):
        if (csml[i] < theta and test[i][2] == 1) or (csml[i] >= theta and test[i][2] == 0):
            error += 1
    return error, 1 - error / csml.shape[0]


def compute_acc(test_data, theta, a):
    test = test_data
    theta = theta
    a = a
    csml = cosine_similarity(test[:, 0], test[:, 1], a)
    same_err = 0
    twin_err = 0
    for i in range(csml.shape[0]):
        if csml[i] < theta and test[i][2] == 1:
            same_err += 1
        elif csml[i] >= theta and test[i][2] == 0:
            twin_err += 1
    error = same_err + twin_err
    return error, 1 - error / csml.shape[0], 1 - same_err / (csml.shape[0] / 2), 1 - twin_err / (csml.shape[0] / 2)


def main1():
    total_acc = 0
    size = 100
    max_acc = 0
    min_acc = 1
    for i in range(size):
        err, acc = compute_acc_path('E:\\FGFV_data\\demo_code\\data\\LBP_r1_pca.npz', 200,
                                    'E:\\Face_Verification\\experiment\\gradient_descent\\parameter_200_120.npz')
        print("acc: ", acc)
        total_acc = total_acc + acc
        if acc > max_acc:
            max_acc = acc
        if acc < min_acc:
            min_acc = acc
    print("avg acc: ", total_acc / size)
    print("min acc: ", min_acc)
    print("max acc: ", max_acc)


def main2():
    same_root = "/home/chenzhentao/fgfv_data/aligned/same/"
    twin_root = "/home/chenzhentao/fgfv_data/aligned/twins/"
    sames = os.listdir(same_root)
    twins = os.listdir(twin_root)
    twins.sort(key=lambda x: int(x[:-6]) * 10 + int(x[-5:-4]) if x.endswith('.jpg') else 0)
    pos_path = []
    neg_path = []
    for i in range(len(sames)):
        if sames[i].endswith('.jpg'):
            pos_path.append(same_root + '/' + sames[i])
    for i in range(len(twins)):
        if twins[i].endswith('.jpg'):
            neg_path.append(twin_root + '/' + twins[i])
    pos_acc = 0
    neg_err = 0
    for i in range(0, len(pos_path), 2):
        pos_acc += inference(pos_path[i], pos_path[i + 1],
                             "/home/chenzhentao/Face_Verification/experiment/parameter_200_100.npz",
                             "/home/chenzhentao/fgfv_data/LBP_pca_feature_500.npz")
    for i in range(0, len(neg_path), 2):
        neg_err += inference(neg_path[i], neg_path[i + 1],
                             "/home/chenzhentao/Face_Verification/experiment/parameter_200_100.npz",
                             "/home/chenzhentao/fgfv_data/LBP_pca_feature_500.npz")
    same_acc = pos_acc / (len(pos_path) // 2)
    twin_acc = (len(neg_path) // 2 - neg_err) / (len(neg_path) // 2)
    acc = (same_acc + twin_acc) / 2
    print("same_acc: ", same_acc)
    print("twin_acc: ", twin_acc)
    print("acc: ", acc)


def main3():
    data1 = np.load("/home/chenzhentao/fgfv_data/LBP_r1_pca.npz", allow_pickle=True)['pca'].item()
    data2 = np.load("/home/chenzhentao/fgfv_data/LBP_pca_500.npz", allow_pickle=True)['pca']
    data1 = data1['ux']
    data1 = np.transpose(data1)
    all_pos = data2[:data2.shape[0] // 2, :]
    all_neg = data2[data2.shape[0] // 2:, :]
    pos_acc = 0
    neg_err = 0
    for i in range(0, len(all_pos), 2):
        pos_acc += inference(all_pos[i], all_pos[i + 1],
                             "/home/chenzhentao/Face_Verification/experiment/parameter_200_100.npz",
                             "/home/chenzhentao/fgfv_data/LBP_pca_feature_500.npz")
    for i in range(0, len(all_neg), 2):
        neg_err += inference(all_neg[i], all_neg[i + 1],
                             "/home/chenzhentao/Face_Verification/experiment/gradient_descent/parameter_200_100.npz",
                             "/home/chenzhentao/fgfv_data/LBP_pca_feature_500.npz")
    same_acc = pos_acc / (len(all_pos) // 2)
    twin_acc = (len(all_neg) // 2 - neg_err) / (len(all_neg) // 2)
    acc = (same_acc + twin_acc) / 2
    print("same_acc: ", same_acc)
    print("twin_acc: ", twin_acc)
    print("acc: ", acc)


if __name__ == '__main__':
    main2()
    # main3()
