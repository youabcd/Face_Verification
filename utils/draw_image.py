import matplotlib.pyplot as plt
import numpy as np


def draw_img_1(x, y, img, x_label, y_label='avg_acc'):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
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


def draw_img_3(t1, t2, acc, img, label, x_label, y_label='avg_acc'):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.tick_params(labelsize=15)
    legend = []
    for i in range(len(t1)):
        x = t2[i]
        y = acc[i]
        legend.append("s=" + str(t1[i]))
        plt.plot(x, y)
    plt.legend(label)
    plt.savefig("/home/chenzhentao/Face_Verification/experiment/images/" + img + ".jpg")
    plt.show()


def draw_img_4(x_all, y_all, img, label, x_label, y_label='avg_acc'):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.tick_params(labelsize=15)
    for i in range(len(x_all)):
        x = x_all[i]
        y = y_all[i]
        plt.plot(x, y)
    plt.legend(label)
    plt.savefig("E:\\Face_Verification\\experiment\\images\\" + img + ".jpg")
    plt.show()


if __name__ == '__main__':
    # img1
    # label = ['d:m=2:5', 'd:m=1:2', 'd:m=3:5']
    # m = [[50, 100, 200, 300, 400] for i in range(3)]
    # acc = [[0.6898, 0.7025, 0.7143, 0.7173, 0.7124],
    #        [0.6964, 0.7124, 0.7247, 0.7168, 0.7038],
    #        [0.6956, 0.7044, 0.7175, 0.7118, 0.6984]]
    # draw_img_4(x_all=m, y_all=acc, img="d_m_s", label=label, x_label="m")
    # img2
    d = [20, 40, 60, 80, 90, 100, 110, 120, 140, 160, 180, 200]
    acc = [0.7099, 0.6912, 0.7121, 0.7150, 0.7146, 0.7247, 0.7148, 0.7175, 0.7110, 0.7104, 0.7077, 0.7093]
    draw_img_1(x=d, y=acc, img="d_m200", x_label="d")
    # img3
    # beta = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # acc = [0.7161, 0.7121, 0.7077, 0.7247, 0.7055, 0.7234, 0.7179, 0.7162, 0.7187, 0.7095, 0.7037, 0.7077]
    # draw_img_1(x=beta, y=acc, img="beta", x_label="β")
    # img4
    # label = ['Uniform LBP', 'Intensity', 'VGG-16', 'HOG']
    # m = [[50, 100, 200, 300, 400] for i in range(4)]
    # acc = [[0.6964, 0.7124, 0.7247, 0.7173, 0.7124],
    #        [0.5776, 0.5708, 0.5879, 0.5927, 0.6004],
    #        [0.5967, 0.5923, 0.5805, 0.6096, 0.6423],
    #        [0.6816, 0.6791, 0.7165, 0.7201, 0.7132]]
    # draw_img_4(x_all=m, y_all=acc, img="feature", label=label, x_label="m")
    # img5
    # label = ['PCA', 'WPCA', 'RP']
    # m = [[50, 100, 200, 300, 400] for i in range(3)]
    # acc = [[0.6964, 0.7124, 0.7247, 0.7173, 0.7124],
    #        [0.5764, 0.5904, 0.5885, 0.5889, 0.5865],
    #        [0.5129, 0.5330, 0.5522, 0.5519, 0.5472]]
    # draw_img_4(x_all=m, y_all=acc, img="A0", label=label, x_label="m")
    # img6
    # data = np.load("/home/chenzhentao/Face_Verification/experiment/new_t1_t2_test.npz", allow_pickle=True)['test'].item()
    # t1 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # length = 0
    # t2 = [[] for i in range(len(t1))]
    # acc = [[] for i in range(len(t1))]
    # t1_data = data['t1']
    # t2_data = data['t2']
    # acc_data = data['acc']
    # for i in range(len(t1_data)):
    #     if i == len(t1_data)-1:
    #         t2[length].append(t2_data[i])
    #         acc[length].append(acc_data[i])
    #         break
    #     if t1_data[i] != t1_data[i+1]:
    #         t2[length].append(t2_data[i])
    #         acc[length].append(acc_data[i])
    #         length += 1
    #     else:
    #         t2[length].append(t2_data[i])
    #         acc[length].append(acc_data[i])
    # draw_img_3(t1=t1, t2=t2, acc=acc, img='new_t1_t2')
    # img7
    # d = [20, 40, 60, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200]
    # acc = [0.7626, 0.7610, 0.7604, 0.7619, 0.7563, 0.7695, 0.7624, 0.7692, 0.7668, 0.7651, 0.7667, 0.7626, 0.7695,
    #        0.7657]
    # draw_img_1(x=d, y=acc, img="new_d_m200", x_label="d")
    # img8
    # label = ['d:m=9:10', 'd:m=3:5', 'd:m=1:2', 'd:m=2:5']
    # m = [[50, 100, 200, 300, 400] for i in range(4)]
    # acc = [[0.7151, 0.7453, 0.7695, 0.7560, 0.7663],
    #        [0.6997, 0.7511, 0.7692, 0.7511, 0.7665],
    #        [0.7022, 0.7343, 0.7695, 0.7502, 0.7555],
    #        [0.6989, 0.7481, 0.7619, 0.7591, 0.7615]]
    # draw_img_4(x_all=m, y_all=acc, img="new_d_m_s", label=label, x_label="m")
    # img9
    # label = ['Uniform LBP', 'Intensity', 'VGG-16', 'HOG']
    # m = [[50, 100, 200, 300, 400] for i in range(4)]
    # acc = [[0.7151, 0.7511, 0.7695, 0.7591, 0.7665],
    #        [0.6936, 0.7068, 0.6848, 0.7218, 0.6574],
    #        [0.5651, 0.5347, 0.5673, 0.6156, 0.5978],
    #        [0.7031, 0.7279, 0.7455, 0.7424, 0.7589]]
    # draw_img_4(x_all=m, y_all=acc, img="new_feature", label=label, x_label="m")
    # img10
    # label = ['PCA', 'WPCA', 'RP']
    # m = [[50, 100, 200, 300, 400] for i in range(3)]
    # acc = [[0.7151, 0.7511, 0.7695, 0.7591, 0.7665],
    #        [0.7185, 0.7475, 0.7653, 0.7527, 0.7459],
    #        [0.6897, 0.7283, 0.7147, 0.7136, 0.7330]]
    # draw_img_4(x_all=m, y_all=acc, img="new_A0", label=label, x_label="m")
