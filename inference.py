import numpy as np
from utils.change_data import change_data
from utils.error_computation import cosine_similarity


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
    error = 0
    for i in range(csml.shape[0]):
        if (csml[i] < theta and test[i][2] == 1) or (csml[i] >= theta and test[i][2] == 0):
            error += 1
    return error, 1 - error / csml.shape[0]


if __name__ == '__main__':
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
