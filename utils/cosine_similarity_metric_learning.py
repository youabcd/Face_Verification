import numpy as np
from utils.error_computation import compute_error


# 余弦相似度
def cosine_similarity(x, y, a):
    # x = [1,2,3,3,2]
    # y = [4,5,6,5,6]
    # a = [[1,2,3,1,2],[4,5,6,4,5]] m = 500 and d = 200
    ax = np.dot(a, x)
    ay = np.dot(a, y)
    ax_norm = np.linalg.norm(ax)
    ay_norm = np.linalg.norm(ay)
    return np.dot(ax, ay) / (ax_norm * ay_norm)


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


# 目标函数
def obj_func(pos, neg, a, a0, alpha, beta):
    pos_sum = 0
    neg_sum = 0
    for i in len(pos):
        pos_sum = pos_sum + cosine_similarity(x=pos[i][0], y=pos[i][1], a=a)
    for i in len(neg):
        neg_sum = neg_sum + cosine_similarity(x=neg[i][0], y=neg[i][1], a=a)
    return pos_sum - (alpha * neg_sum) - (beta * np.square(np.linalg.norm(a - a0)))


# 目标函数梯度
def grad_func(pos, neg, a, a0, alpha, beta):
    pos_sum = np.zeros(a0.shape)
    neg_sum = np.zeros(a0.shape)
    for i in len(pos):
        pos_sum = pos_sum + grad_cs(x=pos[i][0], y=pos[i][1], a=a)
    for i in len(neg):
        neg_sum = neg_sum + grad_cs(x=neg[i][0], y=neg[i][1], a=a)
    return pos_sum - (alpha * neg_sum) - (2 * beta * (a - a0))


# 共轭梯度算法
def cg(pos, neg, a0, alpha, beta):
    max_iter = 5000
    epsilon = 1e-4
    a0_line = a0.reshape(-1)
    r0 = grad_func(pos=pos, neg=neg, a=a0, a0=a0, alpha=alpha, beta=beta)
    p0 = -r0
    for i in range(max_iter):
        step_alpha = np.linalg.norm(r0.reshape(-1))


# 采用Armijo准测的共轭梯度算法 传入a0为一维数组:shape=[n,]
def cg_arm(pos, neg, a0, alpha, beta, a_shape=[200, 300]):
    max_k = 5000
    rho = .6
    sigma = .4
    k = 0
    epsilon = 1e-4
    n = len(a0)
    a = a0
    g0 = grad_func(pos=pos, neg=neg, a=a.reshape(a_shape), a0=a0.reshape(a_shape), alpha=alpha, beta=beta)
    g0 = g0.reshape(-1)
    d0 = -g0
    while k < max_k:
        g = grad_func(pos=pos, neg=neg, a=a.reshape(a_shape), a0=a0.reshape(a_shape), alpha=alpha, beta=beta)
        g = g.reshape(-1)
        item = k % n
        if item == 0:
            d = -g
        else:
            theta = np.linalg.norm(g) / np.linalg.norm(g0)
            d = -g + theta * d0
            gd = np.dot(g, d)
            if gd >= 0:
                d = -g
        if np.linalg.norm(g) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if obj_func(pos=pos, neg=neg, a=(a + rho ** m * d).reshape(a_shape), a0=a0.reshape(a_shape), alpha=alpha,
                        beta=beta) > obj_func(pos=pos, neg=neg, a=a.reshape(a_shape), a0=a0.reshape(a_shape),
                                              alpha=alpha, beta=beta) + sigma * rho ** m * g.T @ d:
                mk = m
                break
            m += 1
        a = a + rho ** mk * d
        g0 = g
        d0 = d
        k += 1
    return a.reshape(a_shape)


# 余弦相似度度量学习
def cs_ml(pos, neg, t, d, ap, k, repeat):
    """
    :param pos: 正样本 size 样本对数量*2*图像维数
    :param neg: 负样本
    :param t: 验证样本
    :param d: dimension of the data after applying CSML
    :param ap: 预先设定的矩阵A
    :param k: 将验证样本拆分成k个子样本
    :param repeat: 寻找最佳A的最大次数
    :return: A_csml
    """
    a0 = ap
    a_next = ap
    alpha = len(pos) / len(neg)
    for i in range(repeat):
        min_cve = np.finfo(np.float32).max
        for beta in np.arange(0.5, 5, 0.1):
            a1 = cg_arm(pos=pos, neg=neg, a0=a0.reshape(-1), alpha=alpha, beta=beta, a_shape=a0.shape)
            cve = compute_error(t=t, a=a1, k=k)
            if cve < min_cve:
                min_cve = cve
                a_next = a1
        a0 = a_next
        if min_cve < 1e-6:
            break
