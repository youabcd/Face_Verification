import numpy as np
from utils.error_computation import compute_error
import time
import scipy.optimize as optimize
from utils.change_data import change_data
from utils.get_ap import get_ap
import matplotlib.pyplot as plt
from utils.object_function import ObjFunc1, ObjFunc2


# 最速下降法
# def lower_fast(pos, neg, t, a0, alpha, beta, a_shape):
    # min_func = 0
    # all_func = []
    # min_err = 91
    # min_k = 0
    # max_k = 1000
    # k = 0
    # step = 0.01
    # epsilon = 1e-6
    # at = a0
    # all_func.append(obj_func(a=at, pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta))
    # while True:
    #     # while k < max_k:
    #     g = grad_func(a=at.reshape(-1), pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
    #     a = at + (step * (-g)).reshape(a_shape)
    #     step = step * 0.95
    #     print(k, " g_norm", np.linalg.norm(g))
    #     # if np.linalg.norm(g) < epsilon:
    #     #     break
    #     func = obj_func(a=a.reshape(-1), pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
    #     all_func.append(func)
    #     # if func < min_func:
    #     #     min_func = func
    #     #     min_k = k
    #     # print("func: ", func)
    #     distance = np.linalg.norm((a - at).diagonal())
    #     print("distance: ", distance)
    #     # err, _ = compute_error(t=t.copy(), a=a.reshape(a_shape), k=10)
    #     # if err < min_err:
    #     #     min_err = err
    #     #     min_k = k
    #     # print("err: ", err)
    #     if distance < epsilon:
    #         break
    #     at = a
    #     k += 1
    # print("min_func: ", min_func, " k: ", min_k)
    # # print("min_err: ", min_err, " k: ", min_k)
    # return a, all_func


# 最速下降法
def lower_fast(function, a0, a_shape):
    min_func = 0
    all_func = []
    min_k = 0
    k = 0
    step = 0.01
    epsilon = 1e-6
    at = a0
    all_func.append(function.function(a=at))
    while True:
        g = function.grad(a=at.reshape(-1))
        a = at + (step * (-g)).reshape(a_shape)
        step = step * 0.95
        print(k, " g_norm", np.linalg.norm(g))
        func = function.function(a=a.reshape(-1))
        all_func.append(func)
        distance = np.linalg.norm((a - at).diagonal())
        print("distance: ", distance)
        if distance < epsilon:
            break
        at = a
        k += 1
    print("min_func: ", min_func, " k: ", min_k)
    return a, all_func


# 共轭梯度算法
# 采用Armijo准测的共轭梯度算法 传入a0为2维数组:shape=[d,m]
def cg_arm(function, a0, a_shape, rho):
    min_func = 0
    all_func = []
    min_err = 91
    min_k = 0
    max_k = 250
    rho = rho
    # print("rho: ", rho)
    print("a shape: ", a0.shape)
    sigma = .4
    k = 0
    best_a = a0
    epsilon = 1e-6
    a = a0.reshape(-1)
    n = len(a)
    g0 = function.grad(a=a)
    d0 = -g0
    all_func.append(-function.function(a=a))
    # while True:
    while k < max_k:
        g = function.grad(a=a)
        item = k % n
        if item == 0:
            d = -g
        else:
            theta = np.linalg.norm(g) / np.linalg.norm(g0)
            d = -g + theta * d0
            gd = np.dot(g, d)
            if gd >= 0:
                d = -g
        print(k, " g_norm", np.linalg.norm(g))
        if np.linalg.norm(g) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if function.function(a=a + rho ** m * d) < function.function(a=a) + sigma * rho ** m * g.T @ d:
                mk = m
                break
            m += 1
        distance = np.linalg.norm(((a + rho ** mk * d).reshape(a_shape) - a.reshape(a_shape)).diagonal())
        print("distance: ", distance)
        a = a + rho ** mk * d
        # if distance < epsilon:
        #     break
        # err, _ = compute_error(t=t.copy(), a=a.reshape(a_shape), k=10)
        func = function.function(a=a)
        all_func.append(-func)
        # if err < min_err:
        #     min_err = err
        #     min_k = k
        #     best_a = a
        #     min_func = func
        # print("err: ", err)
        if k % 500 == 0:
            # print("rho: ", rho)
            print("a shape: ", a0.shape)
            print("best err in 500: ", min_err)
            print("best err k: ", min_k)
            print("min_func: ", min_func)
        g0 = g
        d0 = d
        rho = 0.95 * rho
        k += 1
    print("min_func: ", min_func, " k: ", min_k)
    print("min_err: ", min_err, " k: ", min_k)
    # print("rho: ", rho)
    print("a shape: ", a0.shape)
    return best_a.reshape(a_shape), all_func


# 余弦相似度度量学习
def cs_ml(pos, neg, t, d, ap, k, repeat, rho, t1, t2):
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
        for beta in np.arange(0.1, 0.15, 0.1):
            print("beta: ", beta)
            time_cg = time.time()
            # new_func = ObjFunc1(pos=pos, neg=neg, t1=t1, t2=t2, a0=a0)
            # a1, all_func = lower_fast(new_func, a0=a0, a_shape=a0.shape)
            # a1 = (optimize.fmin_cg(new_func.function, a0.reshape(-1), fprime=new_func.grad,
            #                        gtol=1e-6)).reshape(a0.shape)
            new_func = ObjFunc2(pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
            # a1, all_func = lower_fast(new_func, a0=a0, a_shape=a0.shape)
            a1 = (optimize.fmin_cg(new_func.function, a0.reshape(-1), fprime=new_func.grad,
                                   gtol=1e-6)).reshape(a0.shape)
            cve, pri_theta = compute_error(t=t.copy(), a=a1, k=k)
            time_cg_end = time.time()
            print("finish a epoch: ", time_cg_end - time_cg)
            print("beta: ", beta, " cve: ", cve)
            if cve < min_cve:
                min_cve = cve
                a_next = a1
                beta_next = beta
                theta_next = pri_theta
            # plt.figure(figsize=(12, 8), dpi=80)
            # plt.plot(range(len(all_func)), all_func)
            # plt.savefig("/home/chenzhentao/Face_Verification/experiment/100_200.png")
            # plt.show()
        min_cve_s.append(min_cve)
        best_beta.append(beta_next)
        best_theta.append(theta_next)
        best_a.append(a_next)
        a0 = a_next
        print("i: ", i, " min_cve_s: ", min_cve_s, " beta: ", best_beta)
        if min_cve < 0:
            break
    return np.array(best_a), np.array(min_cve_s), np.array(best_theta), np.array(best_beta)


if __name__ == '__main__':
    parameter = np.load('E:\Face_Verification\experiment\\009\parameter_200_20.npz', allow_pickle=True)[
        'parameter'].item()
    idx = np.where(parameter['min_cve_s'] == np.min(parameter['min_cve_s']))
    print("old error: ", np.min(parameter['min_cve_s']))
    a = parameter['a0_s'][idx[0][0]]
    pos, neg, t, _ = change_data('E:\毕设\demo_code\data\LBP_r1_pca.npz', a.shape[1])
    print("a shape: ", a.shape)
