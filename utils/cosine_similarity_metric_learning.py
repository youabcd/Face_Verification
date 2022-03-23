import numpy as np
from utils.error_computation import compute_error
import time
import scipy.optimize as optimize
from utils.change_data import change_data
from utils.get_ap import get_ap


# 余弦相似度
def cosine_similarity(x, y, a):
    # x = [1,2,3,3,2]
    # y = [4,5,6,5,6]
    # a = [[1,2,3,1,2],[4,5,6,4,5]] m = 500 and d = 200
    ax = np.dot(a, x.T)
    ay = np.dot(a, y.T)
    ax_norm = np.linalg.norm(ax, axis=0)
    ay_norm = np.linalg.norm(ay, axis=0)
    return np.diagonal(np.dot(ax.T, ay)) / (ax_norm * ay_norm)


# 余弦相似度梯度
def grad_cs(x, y, a):
    ax = np.dot(a, x)
    ay = np.dot(a, y)
    ax_norm = np.linalg.norm(ax)
    ay_norm = np.linalg.norm(ay)
    ua = np.dot(ax, ay)
    grad_u = np.dot(a, (2 * np.dot(x, y)))
    va = ax_norm * ay_norm
    grad_v = ((ay_norm / ax_norm) * ax[:, np.newaxis]) * x - ((ax_norm / ay_norm) * ay[:, np.newaxis]) * y
    return grad_u / va - (ua / np.square(va)) * grad_v


# def grad_cs_pal(x, y, a):
#     ax = np.dot(a, x.T)
#     ay = np.dot(a, y.T)
#     ax_norm = np.linalg.norm(ax, axis=0)
#     ay_norm = np.linalg.norm(ay, axis=0)
#     ua = np.diagonal(np.dot(ax.T, ay))
#     va = ax_norm * ay_norm
#     a1 = a.reshape((1, a.shape[0], a.shape[1]))
#     xy = np.diagonal(x @ y.T)
#     grad_u_va = np.tile(a1, (x.shape[0], 1, 1)) * np.tile((2 * xy / va).reshape((-1, 1, 1)),
#                                                           (1, a.shape[0], a.shape[1]))
#     grad_v = np.tile((ay_norm / ax_norm).reshape((-1, 1, 1)), (1, a.shape[0], a.shape[1])) * np.einsum('ij, jk->jik',
#                                                                                                        ax, x) - np.tile(
#         (ax_norm / ay_norm).reshape((-1, 1, 1)), (1, a.shape[0], a.shape[1])) * np.einsum('ij, jk->jik', ay, y)
#     return grad_u_va - np.tile((ua / np.square(va)).reshape((-1, 1, 1)), (1, a.shape[0], a.shape[1])) * grad_v


def grad_cs_pal(x, y, a):
    ax = np.einsum('ij, kj->ik', a, x)
    ay = np.einsum('ij, kj->ik', a, y)
    ax_norm = np.linalg.norm(ax, axis=0)
    ay_norm = np.linalg.norm(ay, axis=0)
    ua = np.einsum('ij,ij->j', ax, ay)
    va = ax_norm * ay_norm
    a1 = a.reshape((1, a.shape[0], a.shape[1]))
    xy = np.einsum('ij,ij->i', x, y)
    grad_u_va = a1 * (2 * xy / va).reshape([-1, 1, 1])
    t1_time = time.time()
    test1 = np.einsum('ij, jk->jik', ax, x)
    test2 = ((ay_norm / ax_norm) * (ua / np.square(va))).reshape((-1, 1, 1))
    test3 = test1 * test2
    time1 = time.time()
    print("test1: ", time1 - t1_time)
    test4 = ((ay_norm / ax_norm) * (ua / np.square(va))).reshape((-1, 1, 1)) * np.einsum('ij, jk->jik', ax, x)
    t2_time = time.time()
    print("test2: ", t2_time - time1)
    grad_v_ua = ((ay_norm / ax_norm) * (ua / np.square(va))).reshape((-1, 1, 1)) * np.einsum('ij, jk->jik', ax, x) - (
            (ax_norm / ay_norm) * (ua / np.square(va))).reshape((-1, 1, 1)) * np.einsum('ij, jk->jik', ay, y)
    time2 = time.time()
    print("grad_v: ", time2 - time1)
    return grad_u_va - grad_v_ua


# 目标函数
def obj_func(a, pos, neg, a0, alpha, beta):
    a = a.reshape(a0.shape)
    pos_sum = cosine_similarity(x=pos[:, 0, :], y=pos[:, 1, :], a=a)
    neg_sum = cosine_similarity(x=neg[:, 0, :], y=neg[:, 1, :], a=a)
    pos_sum = pos_sum.sum()
    neg_sum = neg_sum.sum()
    return -(pos_sum - (alpha * neg_sum) - (beta * np.square(np.linalg.norm(a - a0))))
    # return -(pos_sum - (alpha * neg_sum))


# 目标函数梯度
def grad_func(a, pos, neg, a0, alpha, beta):
    a = a.reshape(a0.shape)
    pos_sum = np.zeros(a0.shape)
    neg_sum = np.zeros(a0.shape)
    for i in range(len(pos)):
        # time1 = time.time()
        pos_sum = pos_sum + grad_cs(x=pos[i][0], y=pos[i][1], a=a)
        # print("one: ", time.time() - time1)
    for i in range(len(neg)):
        neg_sum = neg_sum + grad_cs(x=neg[i][0], y=neg[i][1], a=a)
    return -((pos_sum - (alpha * neg_sum) - (2 * beta * (a - a0))).reshape(-1))
    # return -((pos_sum - (alpha * neg_sum)).reshape(-1))


# def grad_func(pos, neg, a, a0, alpha, beta):
#     pos_all = grad_cs_pal(x=pos[:, 0, :], y=pos[:, 1, :], a=a)
#     neg_all = grad_cs_pal(x=neg[:, 0, :], y=neg[:, 1, :], a=a)
#     pos_sum = np.sum(pos_all, axis=0)
#     neg_sum = np.sum(neg_all, axis=0)
#     return pos_sum - (alpha * neg_sum) - (2 * beta * (a - a0))


# 最速下降法
def lower_fast(pos, neg, t, a0, alpha, beta, a_shape):
    min_func = 0
    min_err = 91
    min_k = 0
    max_k = 1000
    k = 0
    step = 0.01
    epsilon = 1e-6
    at = a0
    # while True:
    while k < max_k:
        g = grad_func(a=at.reshape(-1), pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
        a = at + (step * (-g)).reshape(a_shape)
        # step = step * 0.95
        print(k, " g_norm", np.linalg.norm(g))
        # if np.linalg.norm(g) < epsilon:
        #     break
        func = obj_func(a=a.reshape(-1), pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
        if func < min_func:
            min_func = func
            min_k = k
        print("func: ", func)
        distance = np.linalg.norm((a - at).diagonal())
        print("distance: ", distance)
        err, _ = compute_error(t=t.copy(), a=a.reshape(a_shape), k=10)
        if err < min_err:
            min_err = err
            min_k = k
        print("err: ", err)
        if distance < epsilon:
            break
        at = a
        k += 1
    print("min_func: ", min_func, " k: ", min_k)
    print("min_err: ", min_err, " k: ", min_k)
    return a


# 共轭梯度算法
# def cg(pos, neg, a0, alpha, beta):
#     max_iter = 5000
#     epsilon = 1e-4
#     a0_line = a0.reshape(-1)
#     r0 = grad_func(pos=pos, neg=neg, a=a0, a0=a0, alpha=alpha, beta=beta)
#     p0 = -r0
#     for i in range(max_iter):
#         step_alpha = np.linalg.norm(r0.reshape(-1))


# 采用Armijo准测的共轭梯度算法 传入a0为2维数组:shape=[d,m]
def cg_arm(pos, neg, t, a0, alpha, beta, a_shape, rho):
    min_func = 0
    min_err = 91
    min_k = 0
    max_k = 529
    rho = rho
    print("rho: ", rho)
    sigma = .4
    k = 0
    best_a = a0
    epsilon = 1e-6
    a = a0.reshape(-1)
    n = len(a)
    g0 = grad_func(a=a, pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
    d0 = -g0
    # d0 = g0
    print("func: ", obj_func(a=a, pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta))
    while True:
        # while k < max_k:
        g = grad_func(a=a, pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
        item = k % n
        if item == 0:
            d = -g
            # d = g
        else:
            theta = np.linalg.norm(g) / np.linalg.norm(g0)
            d = -g + theta * d0
            # d = g + theta * d0
            gd = np.dot(g, d)
            if gd >= 0:
                d = -g
            # if gd >= 0:
            #     d = g
        print(k, " g_norm", np.linalg.norm(g))
        if np.linalg.norm(g) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if obj_func(a=a + rho ** m * d, pos=pos,
                        neg=neg, a0=a0, alpha=alpha, beta=beta) < obj_func(a=a, pos=pos, neg=neg, a0=a0, alpha=alpha,
                                                                           beta=beta) + sigma * rho ** m * g.T @ d:
                mk = m
                break
            m += 1
        a = a + rho ** mk * d
        # func = obj_func(a=a, pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
        # if func < min_func:
        #     min_func = func
        #     min_k = k
        # print("func: ", func)
        err, _ = compute_error(t=t.copy(), a=a.reshape(a_shape), k=10)
        if err < min_err:
            min_err = err
            min_k = k
            best_a = a
        print("err: ", err)
        g0 = g
        d0 = d
        k += 1
    # print("min_func: ", min_func, " k: ", min_k)
    print("min_err: ", min_err, " k: ", min_k)
    print("rho: ", rho)
    # return a.reshape(a_shape)
    return best_a.reshape(a_shape)


# 余弦相似度度量学习
def cs_ml(pos, neg, t, d, ap, k, repeat, rho):
    """
    :param pos: 正样本 size 样本对数量*2*图像维数
    :param neg: 负样本
    :param t: 验证样本
    :param d: dimension of the data after applying CSML
    :param ap: 预先设定的矩阵A
    :param k: 将验证样本拆分成k个子样本
    :param repeat: 寻找最佳A的最大次数
    :param rho
    :return: A_csml
    """
    a0 = ap
    a_next = ap
    alpha = len(pos) / len(neg)
    best_beta = []
    beta_next = 0
    min_cve_s = []
    best_theta = []
    theta_next = 1
    best_a = []
    for i in range(repeat):
        min_cve = np.finfo(np.float32).max
        for beta in np.arange(0.1, 0.2, 0.1):
            time_cg = time.time()
            # a1 = lower_fast(pos=pos, neg=neg, t=t, a0=a0, alpha=alpha, beta=beta, a_shape=a0.shape)
            a1 = cg_arm(pos=pos, neg=neg, t=t, a0=a0, alpha=alpha, beta=beta, a_shape=a0.shape, rho=rho)
            # a1 = (
            #     optimize.fmin_cg(obj_func, a0.reshape(-1), fprime=grad_func, args=(pos, neg, a0, alpha, beta))).reshape(
            #     a0.shape)
            cve, pri_theta = compute_error(t=t.copy(), a=a1, k=k)
            # time_err = time.time()
            # print("compute error: ", time_err - time_cg_end)
            time_cg_end = time.time()
            print("finish a epoch: ", time_cg_end - time_cg)
            print("beta: ", beta, " cve: ", cve)
            if cve < min_cve:
                min_cve = cve
                a_next = a1
                beta_next = beta
                theta_next = pri_theta
        min_cve_s.append(min_cve)
        best_beta.append(beta_next)
        best_theta.append(theta_next)
        best_a.append(a_next)
        a0 = a_next
        print("i: ", i, " min_cve_s: ", min_cve_s, " beta: ", best_beta)
        if min_cve < 25:
            break
    return np.array(best_a), np.array(min_cve_s), np.array(best_theta), np.array(best_beta)


if __name__ == '__main__':
    parameter = np.load('E:\Face_Verification\experiment\parameter_400_200.npz', allow_pickle=True)[
        'parameter'].item()
    idx = np.where(parameter['min_cve_s'] == np.min(parameter['min_cve_s']))
    print("old error: ", np.min(parameter['min_cve_s']))
    a = parameter['a0_s'][idx[0][0]]
    pos, neg, t = change_data('E:\毕设\demo_code\data\LBP_r1_pca.npz', a.shape[1])
    print(obj_func(a, pos, neg, get_ap('PCA', a.shape[0], a.shape[1]), 1, 0.1))
