import numpy as np
from utils.change_data import change_data


# test data
# t = [[[1,2,3,4,5],[5,4,3,2,1],0],[[4,1,3,2,5],[4,2,3,2,5],1],[[1,2,3,4,5],[1,2,3,4,4],1],[[4,1,3,2,5],[5,2,4,1,3],0]]
def cs(data):
    ans = []
    for i in range(len(data)):
        x_norm = np.linalg.norm(data[i][0])
        y_norm = np.linalg.norm(data[i][1])
        ans.append(np.dot(data[i][0], data[i][1]) / (x_norm * y_norm))
    return ans


# 余弦相似度
def cosine_similarity(x, y, a):
    x = np.array(x.tolist())
    y = np.array(y.tolist())
    ax = np.dot(a, x.T)
    ay = np.dot(a, y.T)
    ax_norm = np.linalg.norm(ax, axis=0)
    ay_norm = np.linalg.norm(ay, axis=0)
    return np.diagonal(np.dot(ax.T, ay)) / (ax_norm * ay_norm)


def compute_error(t, a, k):
    """
    :param t: t = [[img1, img2, 1(same)], [img3, img4, 0(twins)]]
    :param a:
    :param k:
    :return: mean error
    """
    # for p in t:
    #     p[0] = np.dot(a, p[0])
    #     p[1] = np.dot(a, p[1])
    cs_sm = cosine_similarity(t[:, 0], t[:, 1], a)
    total_error = 0
    total_theta = 0
    t_split = np.array_split(t, k)
    cos_sim = np.array_split(cs_sm, k)
    # for i in range(k):
    #     cos_sim.append(cs(t_split[i]))
    for i in range(k):
        pri_theta = 1.
        max_cnt = 0
        test_error = 0
        for theta in np.arange(-1, 1, 0.005):
            cnt = 0
            for j in range(k):
                if j != i:
                    for sub in range(len(t_split[j])):
                        if (cos_sim[j][sub] >= theta and t_split[j][sub][2] == 1) or (
                                cos_sim[j][sub] < theta and t_split[j][sub][2] == 0):
                            cnt += 1
            if cnt >= max_cnt:
                max_cnt = cnt
                pri_theta = theta
        total_theta = total_theta + pri_theta
        # print("pri theta: ", pri_theta)
        for sub in range(len(t_split[i])):
            if (cos_sim[i][sub] < pri_theta and t_split[i][sub][2] == 1) or (
                    cos_sim[i][sub] >= pri_theta and t_split[i][sub][2] == 0):
                test_error += 1
        # print("test error: ", test_error)
        total_error = total_error + test_error
    return total_error / k, total_theta / k


if __name__ == '__main__':
    parameter = np.load('E:\Face_Verification\experiment\parameter_200_100_fl.npz', allow_pickle=True)[
        'parameter'].item()
    idx = np.where(parameter['min_cve_s'] == np.min(parameter['min_cve_s']))
    print("old error: ", np.min(parameter['min_cve_s']))
    a = parameter['a0_s'][idx[0][0]]
    # pos, neg, t = change_data('E:\毕设\demo_code\data\LBP_r1_pca.npz', a.shape[1])
    total_err = 0
    max_err = 0
    min_err = 91
    size = 100
    for i in range(size):
        pos, neg, t = change_data('E:\毕设\demo_code\data\LBP_r1_pca.npz', a.shape[1])
        e, theta = compute_error(t.copy(), a, 10)
        print("err: ", e)
        total_err = total_err + e
        if e > max_err:
            max_err = e
        if e < min_err:
            min_err = e
    print("avg err: ", total_err / size)
    print("max err: ", max_err)
    print("min err: ", min_err)
    print("ave: ", 1 - (total_err / size) / 91, " min: ", 1 - max_err / 91, " max: ", 1 - min_err / 91)
