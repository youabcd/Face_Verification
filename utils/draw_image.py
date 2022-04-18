import matplotlib.pyplot as plt
import numpy as np


def draw_img_1(x, y, img):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.xlabel('β', fontsize=20)
    plt.ylabel('avg_acc', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.plot(x, y)
    plt.savefig("E:\\Face_Verification\\experiment\\images\\" + img + ".jpg")
    plt.show()


def draw_img_2(x1, y1, x2, y2, x3, y3, img, legend):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.xlabel('m', fontsize=20)
    plt.ylabel('avg_acc', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x3, y3)
    plt.legend(legend)
    plt.savefig("E:\\Face_Verification\\experiment\\images\\" + img + ".jpg")
    plt.show()


def draw_img_3(t1, t2, acc, img):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.xlabel('t2', fontsize=20)
    plt.ylabel('avg_acc', fontsize=20)
    plt.tick_params(labelsize=15)
    legend = []
    for i in range(len(t1)):
        x = t2[i]
        y = acc[i]
        legend.append("t1=" + str(t1[i]))
        plt.plot(x, y)
    plt.legend(legend)
    plt.savefig("/home/chenzhentao/Face_Verification/experiment/images/" + img + ".jpg")
    plt.show()


if __name__ == '__main__':
    # img1
    # m = [50, 100, 200, 300, 400]
    # acc = [0.6898, 0.7025, 0.7143, 0.7173, 0.7124]
    # draw_img_1(x=m, y=acc, img="d_m_25")
    # img2
    # d = [20, 40, 60, 80, 90, 100, 110, 120, 140, 160, 180, 200]
    # acc = [0.7099, 0.6912, 0.7121, 0.7143, 0.7146, 0.7247, 0.7148, 0.7175, 0.7110, 0.7104, 0.7077, 0.7093]
    # draw_img_1(x=d, y=acc, img="d_m200")
    # img3
    # m = [50, 100, 200, 300, 400]
    # acc = [0.6964, 0.7124, 0.7247, 0.7168, 0.7038]
    # draw_img_1(x=m, y=acc, img="d_m_12")
    # img4
    # m = [50, 100, 200, 300, 400]
    # acc = [0.6956, 0.7044, 0.7175, 0.7118, 0.6984]
    # draw_img_1(x=m, y=acc, img="d_m_35")
    # img5
    # beta = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # acc = [0.7161, 0.7121, 0.7077, 0.7247, 0.7055, 0.7234, 0.7179, 0.7162, 0.7187, 0.7095, 0.7037, 0.7077]
    # draw_img_1(x=beta, y=acc, img="beta")
    # img6
    # m = [50, 100, 200, 300, 400]
    # pca = [0.6964, 0.7124, 0.7247, 0.7173, 0.7124]
    # wpca = [0.5764, 0.5904, 0.5885, 0.5889, 0.5865]
    # rp = [0.5129, 0.5330, 0.5522, 0.5519, 0.5472]
    # label = ['PCA', 'WPCA', 'RP']
    # draw_img_2(m, pca, m, wpca, m, rp, 'a0', label)
    # img7
    data = np.load("/home/chenzhentao/Face_Verification/experiment/t1_t2_test.npz", allow_pickle=True)['test'].item()
    t1 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    length = 0
    t2 = [[] for i in range(len(t1))]
    acc = [[] for i in range(len(t1))]
    t1_data = data['t1']
    t2_data = data['t2']
    acc_data = data['acc']
    for i in range(len(t1_data)):
        if i == len(t1_data)-1:
            t2[length].append(t2_data[i])
            acc[length].append(acc_data[i])
            break
        if t1_data[i] != t1_data[i+1]:
            t2[length].append(t2_data[i])
            acc[length].append(acc_data[i])
            length += 1
        else:
            t2[length].append(t2_data[i])
            acc[length].append(acc_data[i])
    draw_img_3(t1=t1, t2=t2, acc=acc, img='t1_t2')
